#!/usr/bin/env python3
"""Closed-loop QoE from live ns-3 runs (Phase 5 harness).

Consumes the per-segment `decisions.jsonl` emitted by `run_ai_dash_llm.py` during a live
AiDash run and computes the three core QoE pillars - delivered bitrate, rebuffering, and
switching - per UE (`node_id`) and pooled. Unlike the offline `evaluate.py` replay (which
models the player from a trace), these numbers come from the **actual ns-3 playout**: the
chosen rung per segment and the cumulative stall (`interruption_time_s`) the simulator
recorded, because the network genuinely reacted to the policy's choices.

Pass one labelled decisions file per variant to compare V0/V1/V2 on identical runs:

    eval_qoe.py v0:path/to/v0/decisions.jsonl v1:.../v1/decisions.jsonl v2:.../v2/decisions.jsonl

Torch-free: pure-Python aggregation, runs anywhere (incl. the 3.6 interpreter, so no
`from __future__ import annotations`).
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

try:
    from ai_llm.paths import NS3_RATE_LADDER_BPS
    from ai_llm.trace_schema import MPEG_SEGMENT_DURATION_S
except ImportError:  # run directly from scratch/ai_llm
    from paths import NS3_RATE_LADDER_BPS
    from trace_schema import MPEG_SEGMENT_DURATION_S


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _load_decisions(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_ue_qoe(rows: Sequence[Dict]) -> Dict[int, Dict[str, float]]:
    """Group decisions by node_id and compute QoE pillars per UE."""
    by_node: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_node[int(r.get("node_id", 0))].append(r)

    out: Dict[int, Dict[str, float]] = {}
    for node, recs in by_node.items():
        recs = sorted(recs, key=lambda r: int(r.get("segment_id", 0)))
        rungs = [int(r.get("chosen_rung", 0)) for r in recs]
        n = len(recs)
        bitrate_mbps = sum(NS3_RATE_LADDER_BPS[c] for c in rungs) / 1e6 / max(n, 1)
        # interruption_time_s is cumulative total stall at decision time -> the max is the total.
        total_stall = max((float(r.get("interruption_time_s", 0.0)) for r in recs), default=0.0)
        play_time = n * MPEG_SEGMENT_DURATION_S
        rebuffer_ratio = total_stall / (play_time + total_stall) if (play_time + total_stall) > 0 else 0.0
        switches = sum(1 for a, b in zip(rungs, rungs[1:]) if a != b)
        lat = [float(r.get("inference_latency_ms", 0.0)) for r in recs]
        out[node] = {
            "segments": float(n),
            "mean_bitrate_mbps": bitrate_mbps,
            "total_stall_s": total_stall,
            "rebuffer_ratio": rebuffer_ratio,
            "switch_count": float(switches),
            "lat_p50_ms": _percentile(lat, 0.50),
            "lat_p95_ms": _percentile(lat, 0.95),
            "fallbacks": float(sum(1 for r in recs if str(r.get("reason", "")).startswith("fallback"))),
        }
    return out


def pooled(per_ue: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Average each pillar across UEs (latency pooled as the max p95 / mean p50)."""
    if not per_ue:
        return {}
    keys = ["segments", "mean_bitrate_mbps", "total_stall_s", "rebuffer_ratio",
            "switch_count", "lat_p50_ms", "lat_p95_ms", "fallbacks"]
    n = len(per_ue)
    agg = {k: sum(u[k] for u in per_ue.values()) / n for k in keys}
    agg["n_ues"] = float(n)
    agg["lat_p95_ms"] = max(u["lat_p95_ms"] for u in per_ue.values())  # worst-case across UEs
    return agg


def _parse_labelled(items: Sequence[str]) -> List[Tuple[str, str]]:
    """Parse `label:path` (or bare `path`) arguments."""
    out = []
    for it in items:
        if ":" in it and not it.split(":", 1)[1].startswith("//"):
            label, path = it.split(":", 1)
        else:
            label, path = it, it
        out.append((label, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", nargs="+", help="One or more `label:decisions.jsonl` (or bare path).")
    ap.add_argument("--per-ue", action="store_true", help="Also print every UE's row.")
    args = ap.parse_args()

    runs = _parse_labelled(args.decisions)
    results: List[Tuple[str, Dict[str, float], Dict[int, Dict[str, float]]]] = []
    for label, path in runs:
        rows = _load_decisions(path)
        ue = per_ue_qoe(rows)
        results.append((label, pooled(ue), ue))

    hdr = ("variant", "UEs", "segs", "bitrate_Mbps", "rebuf_ratio", "stall_s", "switches",
           "lat_p50", "lat_p95", "fallbk")
    print("=== Closed-loop QoE (live ns-3, actual playout) ===")
    print("%-8s %4s %5s %12s %11s %8s %8s %8s %8s %6s" % hdr)
    for label, agg, ue in results:
        if not agg:
            print("%-8s  (no decisions)" % label)
            continue
        print("%-8s %4d %5.1f %12.3f %11.4f %8.2f %8.1f %8.1f %8.1f %6.1f" % (
            label, int(agg["n_ues"]), agg["segments"], agg["mean_bitrate_mbps"],
            agg["rebuffer_ratio"], agg["total_stall_s"], agg["switch_count"],
            agg["lat_p50_ms"], agg["lat_p95_ms"], agg["fallbacks"]))
        if args.per_ue:
            for node in sorted(ue):
                u = ue[node]
                print("    node %-3d segs=%-4d bitrate=%.3f Mbps rebuf=%.4f stall=%.2fs switches=%d" % (
                    node, int(u["segments"]), u["mean_bitrate_mbps"], u["rebuffer_ratio"],
                    u["total_stall_s"], int(u["switch_count"])))


if __name__ == "__main__":
    main()
