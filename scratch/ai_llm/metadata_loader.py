#!/usr/bin/env python3
"""
Utilities for loading and normalizing semantic ABR metadata at inference time.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ai_llm.paths import DEFAULT_SEMANTIC_SCALER
except ImportError:
    from paths import DEFAULT_SEMANTIC_SCALER


DEFAULT_SEMANTIC_FEATURES: List[str] = [
    "motion_score",
    "saliency_score",
    "scene_cut_count",
    "object_score",
    "quality_sensitivity",
]

LADDER_LEVELS: List[str] = ["250k", "750k", "1500k", "3000k", "6000k", "12000k"]


class SegmentMetadata:
    def __init__(
        self,
        segment_index: int,
        segment_file: str,
        size_bytes: Dict[str, float],
        actual_bitrate_kbps: Dict[str, float],
        vmaf: Dict[str, float],
        semantic_raw: Dict[str, float],
        semantic_norm: Dict[str, float],
    ) -> None:
        self.segment_index = segment_index
        self.segment_file = segment_file
        self.size_bytes = size_bytes
        self.actual_bitrate_kbps = actual_bitrate_kbps
        self.vmaf = vmaf
        self.semantic_raw = semantic_raw
        self.semantic_norm = semantic_norm


class MetadataIndex:
    def __init__(self, rows: Dict[int, SegmentMetadata]) -> None:
        self._rows = rows

    def get(self, segment_index: int) -> Optional[SegmentMetadata]:
        return self._rows.get(segment_index)

    def __len__(self) -> int:
        return len(self._rows)


def _load_scaler(scaler_path: Optional[str]) -> Optional[dict]:
    if not scaler_path:
        return None
    with open(scaler_path, "r", encoding="utf-8") as handle:
        scaler = json.load(handle)
    required_keys = {"feature_names", "mean", "std"}
    missing = required_keys - set(scaler.keys())
    if missing:
        raise ValueError(f"Scaler file missing keys: {sorted(missing)}")
    return scaler


def _normalize_features(raw_features: Dict[str, float], scaler: Optional[dict]) -> Dict[str, float]:
    if scaler is None:
        return dict(raw_features)

    names = scaler["feature_names"]
    means = scaler["mean"]
    stds = scaler["std"]
    if len(names) != len(means) or len(names) != len(stds):
        raise ValueError("Scaler vectors have mismatched lengths.")

    norm: Dict[str, float] = {}
    for name, mean, std in zip(names, means, stds):
        value = raw_features.get(name)
        if value is None:
            continue
        denom = std if abs(std) > 1e-12 else 1.0
        norm[name] = (value - mean) / denom

    # Preserve any semantic features that are not present in the scaler.
    for key, value in raw_features.items():
        if key not in norm:
            norm[key] = value
    return norm


def load_metadata_index(
    metadata_csv_path: str,
    scaler_json_path: Optional[str] = None,
    semantic_features: Optional[List[str]] = None,
) -> MetadataIndex:
    csv_path = Path(metadata_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    semantic_features = semantic_features or list(DEFAULT_SEMANTIC_FEATURES)
    scaler = _load_scaler(scaler_json_path)
    rows: Dict[int, SegmentMetadata] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"segment_index", "segment_file"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata CSV missing required columns: {sorted(missing)}")

        for row in reader:
            seg_id = int(row["segment_index"])
            size_bytes = {lvl: float(row[f"size_bytes_{lvl}"]) for lvl in LADDER_LEVELS}
            actual_bitrate_kbps = {lvl: float(row[f"actual_bitrate_kbps_{lvl}"]) for lvl in LADDER_LEVELS}
            vmaf = {lvl: float(row[f"vmaf_{lvl}"]) for lvl in LADDER_LEVELS}

            semantic_raw = {name: float(row[name]) for name in semantic_features}
            semantic_norm = _normalize_features(semantic_raw, scaler)

            rows[seg_id] = SegmentMetadata(
                segment_index=seg_id,
                segment_file=row["segment_file"],
                size_bytes=size_bytes,
                actual_bitrate_kbps=actual_bitrate_kbps,
                vmaf=vmaf,
                semantic_raw=semantic_raw,
                semantic_norm=semantic_norm,
            )

    return MetadataIndex(rows)
