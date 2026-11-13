#!/usr/bin/env python3
"""Offline w_stall sweep gate (no sims): relabel train_jsonl with the greedy-reward
causal teacher at several w_stall values and report, per value:
  - rung distribution (counts + %) and mean rung
  - predicted-stall rate: % of rows whose greedy choice has rebuffer_est > 0
  - agreement with the existing DP-oracle label + mean signed gap (greedy - dp)
  - % of rows where greedy differs from the throughput-only safe policy

This is the plan's "inspect rung-dist + offline gate" pre-select step. Higher w_stall
should shift the teacher conservative (lower mean rung, lower predicted-stall rate,
more-negative gap vs the acausal DP-oracle). Final pick is validated online in Stage C.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))                         # -> scratch/
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "scripts")))
from ai_llm.metadata_loader import LADDER_LEVELS                            # noqa: E402
from dp_oracle import load_reward_config                                    # noqa: E402
from relabel_greedy import _ladder, greedy_rung, thr_only_rung             # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "..", "..", "train_jsonl", "train")
W_STALL_GRID = [30.0, 45.0, 65.0]
NR = len(LADDER_LEVELS)


def rebuf_for(thr, buf, sizes, r, thr_safety):
    thr_eff = max(thr * thr_safety, 1.0)
    dl = 8.0 * sizes[r] / thr_eff
    return max(0.0, dl - buf)


def run_one(files, W):
    thr_safety = W.get("thr_safety", 1.0)
    dist = {i: 0 for i in range(NR)}
    n = n_stall = n_diff = 0
    gap_sum = 0
    agree = 0
    for fp in files:
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        groups = {}
        for r in rows:
            groups.setdefault((r.get("run_id"), r.get("node_id")), []).append(r)
        for _, sess in groups.items():
            sess.sort(key=lambda x: int(x.get("segment_id", 0)))
            prev_vmaf = None
            for r in sess:
                obs = r.get("observation", {}) or {}
                t = r.get("teacher", {}) or {}
                thr = float(obs.get("throughput_estimate_bps", 0.0))
                buf = float(obs.get("buffer_level_s", 0.0))
                sizes, vmafs, brs = _ladder(r)
                sem = float(t.get("semantic_weight", 0.0))
                cut = int(t.get("is_scene_cut", 0))
                g = greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W)
                dp = int(t.get("oracle_bitrate_idx", g))
                dist[g] += 1
                n += 1
                if rebuf_for(thr, buf, sizes, g, thr_safety) > 0:
                    n_stall += 1
                if g != thr_only_rung(thr, brs):
                    n_diff += 1
                gap_sum += (g - dp)
                if g == dp:
                    agree += 1
                prev_vmaf = vmafs[g]
    mean_rung = sum(i * c for i, c in dist.items()) / max(n, 1)
    return {
        "n": n, "dist": dist, "mean_rung": mean_rung,
        "stall_pct": 100.0 * n_stall / max(n, 1),
        "diff_pct": 100.0 * n_diff / max(n, 1),
        "dp_agree_pct": 100.0 * agree / max(n, 1),
        "mean_gap_vs_dp": gap_sum / max(n, 1),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Offline w_stall sweep gate on a train dir.")
    ap.add_argument("--data-dir", default=DATA_DIR, help="Dir of *.jsonl shards (default: train_jsonl/train).")
    args = ap.parse_args()
    data_dir = args.data_dir
    files = sorted(f for f in glob.glob(os.path.join(data_dir, "*.jsonl"))
                   if os.path.basename(f) != "all_runs.jsonl")
    cfg = load_reward_config(os.path.join(HERE, "..", "config", "oracle_reward.json"))
    base = dict(cfg["reward_weights"])
    print("data: %d files (%s)" % (len(files), os.path.normpath(data_dir)))
    print("base weights: w_vmaf=%.3f w_switch=%.3f scene_cut=%.3f thr_safety=%.2f"
          % (base["w_vmaf"], base["w_switch"], base["scene_cut_discount"], base.get("thr_safety", 1.0)))
    print("=" * 92)
    hdr = "%-8s | %-30s | %6s | %7s | %7s | %8s | %8s"
    print(hdr % ("w_stall", "rung-dist % [r0..r5]", "mean", "stall%", "diff%", "dp_agree%", "gap_vs_dp"))
    print("-" * 92)
    results = {}
    for ws in W_STALL_GRID:
        W = dict(base)
        W["w_stall"] = ws
        res = run_one(files, W)
        results[ws] = res
        pct = [100.0 * res["dist"][i] / max(res["n"], 1) for i in range(NR)]
        diststr = " ".join("%4.1f" % p for p in pct)
        print(hdr % ("%.0f" % ws, diststr, "%.2f" % res["mean_rung"],
                     "%.1f" % res["stall_pct"], "%.1f" % res["diff_pct"],
                     "%.1f" % res["dp_agree_pct"], "%+.2f" % res["mean_gap_vs_dp"]))
    print("=" * 92)
    print("Read: higher w_stall -> lower mean rung, lower stall%, more-negative gap_vs_dp")
    print("      (greedy more conservative than the acausal DP-oracle = the desired direction).")


if __name__ == "__main__":
    main()
