#!/usr/bin/env python3
"""
Feature assembly and normalization for LLM ABR policy.
"""

import json
import math
import os
import warnings
from typing import Dict, List, Optional

try:
    from ai_llm.metadata_loader import LADDER_LEVELS
except ImportError:
    from metadata_loader import LADDER_LEVELS

try:
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER
except ImportError:
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER


NETWORK_FEATURE_NAMES: List[str] = [
    "throughput_estimate_bps",
    "buffer_level_s",
    "buffer_differential_s",
    "segment_fetch_time_s",
    # Actual encoded bitrate (VBR) of the last downloaded segment - replaces the
    # nominal "current_bitrate_bps", which was a 1:1 restatement of the rung index.
    "last_segment_bitrate_bps",
    "smoothed_rtt_ms",
    "loss_rate_recent",
]

SEMANTIC_FEATURE_NAMES: List[str] = [
    "motion_score",
    "saliency_score",
    "scene_cut_count",
    "object_score",
    "quality_sensitivity",
]

SENTINEL_REPLACEMENT_FEATURES = {"smoothed_rtt_ms", "loss_rate_recent"}

# Fixed order of the missing-indicator (0/1) channels appended after the z-scored network
# features when masks are enabled. Order is frozen for train/inference parity; the model's
# network_dim grows by len(SENTINEL_MASK_ORDER) (7 -> 9) when these are fed in.
SENTINEL_MASK_ORDER = ["smoothed_rtt_ms", "loss_rate_recent"]


class ModelInput:
    """
    Fixed-order model input payload for train/inference parity.
    """

    def __init__(
        self,
        network_norm: List[float],
        semantic_norm: List[float],
        ladder_context: List[List[float]],
        missing_masks: Dict[str, float],
    ) -> None:
        self.network_norm = network_norm
        self.semantic_norm = semantic_norm
        self.ladder_context = ladder_context
        self.missing_masks = missing_masks

    def to_dict(self) -> Dict[str, object]:
        return {
            "network_norm": list(self.network_norm),
            "semantic_norm": list(self.semantic_norm),
            "ladder_context": [list(row) for row in self.ladder_context],
            "missing_masks": dict(self.missing_masks),
        }


def _load_scaler(path: Optional[str]) -> Optional[Dict[str, List[float]]]:
    if not path:
        return None
    if not os.path.exists(path):
        # Schema v2: extraction/build runs before scalers are fit (Stage 4c), and normalization
        # happens at load time. A missing scaler must degrade to identity (raw), never crash -
        # this is what let the whole trace campaign fail before. (Inference, where the scaler is
        # deployed alongside the checkpoint, still loads it normally.)
        warnings.warn("Scaler not found at %s; using identity (raw) normalization." % path)
        return None
    with open(path, "r", encoding="utf-8") as handle:
        scaler = json.load(handle)
    required = {"feature_names", "mean", "std"}
    missing = required - set(scaler.keys())
    if missing:
        raise ValueError("Scaler file missing required keys: %s" % sorted(missing))
    return scaler


def _safe_float(value: float, fallback: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:  # pylint: disable=broad-except
        return fallback
    if math.isfinite(x):
        return x
    return fallback


def _zscore(value: float, mean: float, std: float) -> float:
    denom = std if abs(std) > 1e-12 else 1.0
    return (value - mean) / denom


def _normalize_network(
    observation: Dict[str, float],
    network_scaler: Optional[Dict[str, List[float]]],
) -> (List[float], Dict[str, float]):
    names = NETWORK_FEATURE_NAMES
    log_features: set = set()
    if network_scaler is not None:
        names = list(network_scaler["feature_names"])
        log_features = set(network_scaler.get("log_features", []) or [])

    means = [0.0] * len(names)
    stds = [1.0] * len(names)
    if network_scaler is not None:
        means = [float(x) for x in network_scaler["mean"]]
        stds = [float(x) for x in network_scaler["std"]]
        if len(names) != len(means) or len(names) != len(stds):
            raise ValueError("Network scaler vectors have mismatched lengths.")

    output: List[float] = []
    missing_masks: Dict[str, float] = {}
    for idx, name in enumerate(names):
        raw = _safe_float(observation.get(name, 0.0), fallback=0.0)
        missing = 0.0
        if name in SENTINEL_REPLACEMENT_FEATURES and raw < 0.0:
            # Sentinel: impute to the fitted (already-transformed-space) mean and flag.
            # Skip the log transform - means[idx] is in the scaler's target space.
            value = means[idx]
            missing = 1.0
        else:
            # Apply the same log transform used when fitting the scaler, BEFORE z-score.
            if name in log_features:
                raw = math.log1p(max(raw, 0.0))
            value = raw
        output.append(_zscore(value, means[idx], stds[idx]))
        missing_masks[name] = missing
    return output, missing_masks


def _normalize_semantic(
    semantic_raw: Dict[str, float],
    semantic_norm: Optional[Dict[str, float]],
    semantic_scaler: Optional[Dict[str, List[float]]],
) -> List[float]:
    if semantic_norm is not None and len(semantic_norm) > 0:
        return [_safe_float(semantic_norm.get(name, 0.0), 0.0) for name in SEMANTIC_FEATURE_NAMES]

    if semantic_scaler is None:
        return [_safe_float(semantic_raw.get(name, 0.0), 0.0) for name in SEMANTIC_FEATURE_NAMES]

    names = semantic_scaler["feature_names"]
    means = semantic_scaler["mean"]
    stds = semantic_scaler["std"]
    if len(names) != len(means) or len(names) != len(stds):
        raise ValueError("Semantic scaler vectors have mismatched lengths.")

    norm_by_name: Dict[str, float] = {}
    for name, mean, std in zip(names, means, stds):
        raw = _safe_float(semantic_raw.get(name, 0.0), 0.0)
        norm_by_name[name] = _zscore(raw, float(mean), float(std))
    return [_safe_float(norm_by_name.get(name, 0.0), 0.0) for name in SEMANTIC_FEATURE_NAMES]


def _build_observation_dict(state) -> Dict[str, float]:
    return {
        "throughput_estimate_bps": _safe_float(getattr(state, "throughput_estimate_bps", 0.0), 0.0),
        "buffer_level_s": _safe_float(getattr(state, "buffer_level_s", 0.0), 0.0),
        "buffer_differential_s": _safe_float(getattr(state, "buffer_differential_s", 0.0), 0.0),
        "segment_fetch_time_s": _safe_float(getattr(state, "segment_fetch_time_s", 0.0), 0.0),
        "last_segment_bitrate_bps": _safe_float(getattr(state, "last_segment_bitrate_bps", 0.0), 0.0),
        "smoothed_rtt_ms": _safe_float(getattr(state, "smoothed_rtt_ms", -1.0), -1.0),
        "loss_rate_recent": _safe_float(getattr(state, "loss_rate_recent", -1.0), -1.0),
    }


def _build_ladder_context(segment_meta) -> List[List[float]]:
    # Per rung: [size_bytes, vmaf]. actual_bitrate_kbps is dropped - it is a fixed
    # rescale of size_bytes (= size_bytes * 8 / 1000 / segment_duration, duration = 2s),
    # so it is perfectly collinear and adds no information. size_bytes is kept because
    # download_time = 8 * size_bytes / throughput uses it directly.
    rows: List[List[float]] = []
    for level in LADDER_LEVELS:
        rows.append(
            [
                _safe_float(segment_meta.size_bytes.get(level, 0.0), 0.0),
                _safe_float(segment_meta.vmaf.get(level, 0.0), 0.0),
            ]
        )
    return rows


def build_model_input(
    state,
    segment_meta,
    network_scaler_path: Optional[str] = None,
    semantic_scaler_path: Optional[str] = None,
) -> ModelInput:
    """
    Single entry point for converting raw state + metadata into model tensors.
    """
    network_scaler = _load_scaler(network_scaler_path or DEFAULT_NETWORK_SCALER)
    semantic_scaler = _load_scaler(semantic_scaler_path or DEFAULT_SEMANTIC_SCALER)

    observation = _build_observation_dict(state)
    network_norm, missing_masks = _normalize_network(observation, network_scaler)
    # TRAIN/SERVE PARITY: always re-derive semantic from raw + the CURRENT scaler. The metadata's
    # baked `semantic_norm` was computed with a stale scaler, so passing it (the previous behavior)
    # fed the live model different values than training/eval, collapsing it to a degenerate output.
    # This mirrors build_model_input_from_row() (the offline twin), which already forces None.
    semantic_norm = _normalize_semantic(
        semantic_raw=getattr(segment_meta, "semantic_raw", {}),
        semantic_norm=None,
        semantic_scaler=semantic_scaler,
    )
    ladder_context = _build_ladder_context(segment_meta)

    return ModelInput(
        network_norm=network_norm,
        semantic_norm=semantic_norm,
        ladder_context=ladder_context,
        missing_masks=missing_masks,
    )


def _ladder_context_from_row(row: Dict) -> List[List[float]]:
    """Recover the raw 6x2 [size_bytes, vmaf] ladder from a stored JSONL row.

    Prefers an explicit raw `ladder_context` (schema v2), then `model_input.ladder_context`
    (any schema), then reconstructs from the human-readable `ladder` dict.
    """
    lc = row.get("ladder_context")
    if isinstance(lc, list) and lc:
        return [[_safe_float(v, 0.0) for v in r] for r in lc]
    mi = row.get("model_input") or {}
    lc = mi.get("ladder_context")
    if isinstance(lc, list) and lc:
        return [[_safe_float(v, 0.0) for v in r] for r in lc]
    ladder = row.get("ladder") or {}
    size_bytes = ladder.get("size_bytes", {}) or {}
    vmaf = ladder.get("vmaf", {}) or {}
    rows: List[List[float]] = []
    for level in LADDER_LEVELS:
        rows.append(
            [
                _safe_float(size_bytes.get(level, 0.0), 0.0),
                _safe_float(vmaf.get(level, 0.0), 0.0),
            ]
        )
    return rows


# --- Public wrappers so callers (e.g. the train loader) can load a scaler once and
# --- apply it many times (per window timestep) without re-reading JSON each call.

def load_scaler(path: Optional[str]) -> Optional[Dict[str, List[float]]]:
    """Load a scaler JSON (or None)."""
    return _load_scaler(path)


def normalize_network_vector(observation: Dict[str, float], scaler: Optional[Dict]) -> List[float]:
    """Z-score (+ log on log_features) one raw network observation. `scaler` is a loaded dict."""
    return _normalize_network(observation, scaler)[0]


def normalize_network_vector_with_masks(
    observation: Dict[str, float], scaler: Optional[Dict]
) -> List[float]:
    """Mask-aware variant: the z-scored network vector plus one 0/1 missing-indicator per
    sentinel feature (in `SENTINEL_MASK_ORDER`), appended *after* z-scoring (masks are
    categorical, not scaled). Returns length len(features) + len(SENTINEL_MASK_ORDER)
    (7 + 2 = 9). Shared by the train loader and live inference so the channel order is
    identical on both paths.
    """
    values, masks = _normalize_network(observation, scaler)
    return list(values) + [float(masks.get(name, 0.0)) for name in SENTINEL_MASK_ORDER]


def normalize_semantic_vector(semantic_raw: Dict[str, float], scaler: Optional[Dict]) -> List[float]:
    """Z-score one raw semantic_raw dict. `scaler` is a loaded dict."""
    return _normalize_semantic(semantic_raw, None, scaler)


def ladder_context_from_row(row: Dict) -> List[List[float]]:
    """Raw 6x2 [size_bytes, vmaf] ladder from a stored row."""
    return _ladder_context_from_row(row)


def network_observation(state) -> Dict[str, float]:
    """Extract the raw 7-field network observation dict from a live IPC state object.

    Used at inference to buffer per-UE history for the time-series window.
    """
    return _build_observation_dict(state)


def build_model_input_from_row(
    row: Dict,
    network_scaler_path: Optional[str] = None,
    semantic_scaler_path: Optional[str] = None,
) -> ModelInput:
    """Build a ModelInput from a stored (raw) training row.

    This is the offline twin of build_model_input(): same normalization code
    (`_normalize_network` / `_normalize_semantic`), so the train loader and live
    inference produce identical tensors from identical inputs. Reads the raw
    `observation` (7 network features) and `semantic_raw` (5 semantic features) and the
    raw `ladder_context`; the scaler is applied here, not baked into the dataset.
    """
    network_scaler = _load_scaler(network_scaler_path or DEFAULT_NETWORK_SCALER)
    semantic_scaler = _load_scaler(semantic_scaler_path or DEFAULT_SEMANTIC_SCALER)

    observation = row.get("observation", {}) or {}
    network_norm, missing_masks = _normalize_network(observation, network_scaler)
    semantic_norm = _normalize_semantic(
        semantic_raw=row.get("semantic_raw", {}) or {},
        semantic_norm=None,  # always re-derive from raw + scaler for train/infer parity
        semantic_scaler=semantic_scaler,
    )
    ladder_context = _ladder_context_from_row(row)

    return ModelInput(
        network_norm=network_norm,
        semantic_norm=semantic_norm,
        ladder_context=ladder_context,
        missing_masks=missing_masks,
    )
