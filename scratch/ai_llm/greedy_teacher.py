#!/usr/bin/env python3
"""Self-contained GREEDY-REWARD causal teacher for the ABR dataset.

This module OWNS the greedy-teacher logic end-to-end so the dataset build has no dependency on
`scripts/dp_oracle.py` (the abandoned DP-oracle) or `benchmark/relabel_greedy.py`. The functions
below are COPIED (not imported) from those files:
  - from scripts/dp_oracle.py:            `_minmax`, `compute_semantic_weights`, `load_reward_config`
  - from benchmark/relabel_greedy.py:     `_ladder`, `greedy_rung`
plus a new `assign_greedy_labels` glue that writes the teacher block onto built records.

Greedy label = per-segment argmax of the IMMEDIATE QoE reward, present-only (causal):
  reward(r) = w_vmaf * semantic_weight * VMAF(r)
            - w_stall * max(0, 8*size(r)/(thr_safety*thr) - buffer)     # myopic download vs buffer
            - w_switch * |VMAF(r) - VMAF_prev| * (1 - scene_cut_discount * is_scene_cut)
This is NOT the DP-oracle: single-segment (myopic) + present-only (causal), so a model can
reproduce it online; and it depends on content (VMAF, semantic_weight, scene-cut) + buffer.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ai_llm.metadata_loader import LADDER_LEVELS
except ImportError:  # when scratch/ai_llm is directly on sys.path
    from metadata_loader import LADDER_LEVELS

DEFAULT_REWARD_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "oracle_reward.json"
)


# --------------------------------------------------------------------------------------------
# Copied from scripts/dp_oracle.py (semantic weight = per-video min-max mixture of semantic_raw)
# --------------------------------------------------------------------------------------------
def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / rng for v in values]


def compute_semantic_weights(metadata, mixture: Dict[str, float]) -> Dict[int, float]:
    """Min-max normalize each mixture component across this video's segments -> {seg_id: weight}."""
    seg_ids = sorted(metadata._rows.keys())  # type: ignore[attr-defined]
    components = list(mixture.keys())
    raw_by_component: Dict[str, List[float]] = {c: [] for c in components}
    for seg_id in seg_ids:
        meta = metadata.get(seg_id)
        for c in components:
            raw_by_component[c].append(float(meta.semantic_raw.get(c, 0.0)))
    norm_by_component = {c: _minmax(raw_by_component[c]) for c in components}
    weights: Dict[int, float] = {}
    for i, seg_id in enumerate(seg_ids):
        w = 0.0
        for c in components:
            w += mixture[c] * norm_by_component[c][i]
        weights[seg_id] = w
    return weights


def load_reward_config(path: Optional[str] = None) -> Dict:
    """Load the QoE reward config (config/oracle_reward.json). Requires reward_weights + mixture."""
    cfg_path = Path(path) if path else Path(DEFAULT_REWARD_CONFIG)
    if not cfg_path.is_file():
        raise FileNotFoundError("reward config not found: {}".format(cfg_path))
    with open(str(cfg_path), "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for section in ("reward_weights", "semantic_weight_mixture"):
        if section not in cfg:
            raise ValueError("reward config missing section: {}".format(section))
    cfg["__path__"] = str(cfg_path.resolve())
    return cfg


# --------------------------------------------------------------------------------------------
# Copied from scratch/ai_llm/benchmark/relabel_greedy.py (per-decision greedy rung)
# --------------------------------------------------------------------------------------------
def _ladder(row):
    lad = row.get("ladder", {}) or {}
    sb = lad.get("size_bytes", {}) or {}
    vm = lad.get("vmaf", {}) or {}
    abk = lad.get("actual_bitrate_kbps", {}) or {}
    sizes = [float(sb.get(lv, 0.0)) for lv in LADDER_LEVELS]
    vmafs = [float(vm.get(lv, 0.0)) for lv in LADDER_LEVELS]
    brs = [float(abk.get(lv, 0.0)) * 1000.0 for lv in LADDER_LEVELS]
    return sizes, vmafs, brs


def greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W) -> int:
    best, bi = -1e18, 0
    # thr_safety<1 discounts the (causal) throughput estimate -> larger predicted download time
    # -> more conservative exactly when bandwidth is the binding constraint (scarcity).
    thr_eff = max(thr * W.get("thr_safety", 1.0), 1.0)
    for r in range(len(LADDER_LEVELS)):
        dl = 8.0 * sizes[r] / thr_eff
        rebuf = max(0.0, dl - buf)
        # switch_mode=down_only: DIRECTIONAL smoothness - penalize only quality DROPS (up-switches free),
        # removing the symmetric under-ask lock-in. Must match dp_oracle._switch_penalty (single QoE def).
        if prev_vmaf is None:
            sw = 0.0
        else:
            mag = max(0.0, prev_vmaf - vmafs[r]) if W.get("switch_mode") == "down_only" \
                else abs(vmafs[r] - prev_vmaf)
            sw = W["w_switch"] * mag * (1.0 - W["scene_cut_discount"] * cut)
        q = W["w_vmaf"] * sem * vmafs[r]
        q += W.get("w_quality_sensitive", 0.0) * sem * vmafs[r]  # off by default (=0)
        rew = q - W["w_stall"] * rebuf - sw
        if rew > best:
            best, bi = rew, r
    return bi


# --------------------------------------------------------------------------------------------
# New glue: assign greedy labels onto a video's built records
# --------------------------------------------------------------------------------------------
def assign_greedy_labels(records: List[Dict], weights: Dict[int, float], W: Dict) -> Dict[int, int]:
    """Overwrite each record's `teacher` block with the greedy-reward causal label (in place).

    records : dataset records (extract_trace.build_records output) for ONE video, any # of runs.
    weights : {segment_id -> semantic_weight} from compute_semantic_weights(metadata, mixture).
    W       : reward_weights (from load_reward_config()['reward_weights']).

    Grouped per (run_id, node_id) session, ordered by segment_id, tracking the previously chosen
    VMAF for the switch term. Returns the rung distribution {rung: count} for diagnostics.
    """
    # Nominal ladder bps per level (from "250k".."12000k") to map the DELIVERED bitrate -> rung so
    # throughput_bps_replay = the realized channel rate = 8*delivered_bytes/fetch_time. This is REQUIRED
    # by perceptual_qoe.replay_session (reads teacher["throughput_bps_replay"]); the oracle path sets it
    # from dp_oracle, so the greedy path must set it too or replay divides by 0 -> infinite phantom stall.
    nominal_bps = [int(str(lv).rstrip("k")) * 1000 for lv in LADDER_LEVELS]
    MIN_FETCH_S = 0.05  # guard buffer-full near-instant fetches (matches decisions_to_throughput)
    dist: Dict[int, int] = {}
    groups: Dict = defaultdict(list)
    for r in records:
        groups[(r.get("run_id"), r.get("node_id"))].append(r)
    for _, sess in groups.items():
        sess.sort(key=lambda x: int(x.get("segment_id", 0)))
        prev_vmaf = None
        for r in sess:
            obs = r.get("observation", {}) or {}
            thr = float(obs.get("throughput_estimate_bps", 0.0))
            buf = float(obs.get("buffer_level_s", 0.0))
            sizes, vmafs, brs = _ladder(r)
            # Realized channel throughput for replay: delivered rung's actual bytes / realized fetch time.
            # Delivered bitrate is on the pre-overwrite teacher; fetch time on observation/trace_meta.
            old_t = r.get("teacher") or {}
            delivered_bps = float(old_t.get("chosen_bitrate_bps", 0.0))
            fetch = float(obs.get("segment_fetch_time_s",
                                  (r.get("trace_meta") or {}).get("segment_fetch_time_s", 0.0)))
            thr_replay = 0.0
            if delivered_bps > 0 and sizes:
                di = min(range(len(nominal_bps)), key=lambda i: abs(nominal_bps[i] - delivered_bps))
                thr_replay = 8.0 * sizes[di] / max(fetch, MIN_FETCH_S)
            seg_id = int(r.get("segment_id", 0))
            sem = float(weights.get(seg_id, 0.0))
            cut = 1 if float((r.get("semantic_raw") or {}).get("scene_cut_count", 0.0)) > 0 else 0
            g = greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W)
            r["teacher"] = {
                "name": "greedy_reward",
                "chosen_bitrate_class": g,
                "oracle_bitrate_idx": g,  # kept for downstream loaders that read this field
                "chosen_bitrate_bps": int(round(brs[g])),
                "throughput_bps_replay": thr_replay,  # realized channel rate for replay_session
                "vmaf_chosen": vmafs[g],
                "semantic_weight": sem,
                "is_scene_cut": cut,
            }
            dist[g] = dist.get(g, 0) + 1
            prev_vmaf = vmafs[g]
    return dist
