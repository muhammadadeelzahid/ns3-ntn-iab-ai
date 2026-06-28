#!/usr/bin/env python3
"""Relabel a train_jsonl dataset with a CAUSAL throughput-sustainable expert.

Root cause (06-26, [[acausal-oracle-misspecified-expert]]): the acausal DP-oracle labels bitrate ABOVE
current throughput on 87% of failure states, so imitating it = online over-ask -> collapse. This rewrites
each row's teacher label to a CAUSAL rule that never asks above what's available NOW:

    chosen_bitrate_class = max rung r with actual_bitrate[r] <= observation.throughput_estimate_bps * sf
                           (floor 0 if none sustainable)

Only the teacher label is changed (chosen_bitrate_class, chosen_bitrate_bps, oracle_bitrate_idx kept
consistent); observation / model_input / ladder / semantic are untouched, so the existing train.py reads it
unchanged. Relabels every *.jsonl in <in_dir> -> <out_dir>.

Usage: relabel_causal.py <in_dir> <out_dir> [safety_factor=0.9]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # -> scratch/
from ai_llm.metadata_loader import LADDER_LEVELS

LADDER_BPS_NOMINAL = [250000, 750000, 1500000, 3000000, 6000000, 12000000]


def causal_rung(thr_bps, bitrates_bps, sf):
    budget = thr_bps * sf
    r = 0
    for i, b in enumerate(bitrates_bps):
        if b <= budget:
            r = i
    return r


def main():
    in_dir, out_dir = sys.argv[1:3]
    sf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    os.makedirs(out_dir, exist_ok=True)
    n_rows = n_files = 0
    dist = {}
    for fp in glob.glob(os.path.join(in_dir, "*.jsonl")):
        out_lines = []
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            obs = r.get("observation", {}) or {}
            thr = float(obs.get("throughput_estimate_bps", 0.0))
            lad = (r.get("ladder", {}) or {}).get("actual_bitrate_kbps", {}) or {}
            if lad:
                brs = [float(lad.get(lv, LADDER_BPS_NOMINAL[i] / 1000.0)) * 1000.0
                       for i, lv in enumerate(LADDER_LEVELS)]
            else:
                brs = list(LADDER_BPS_NOMINAL)
            cr = causal_rung(thr, brs, sf)
            teacher = r.setdefault("teacher", {})
            teacher["name"] = "causal_throughput_sustainable"
            teacher["chosen_bitrate_class"] = cr
            teacher["oracle_bitrate_idx"] = cr
            teacher["chosen_bitrate_bps"] = int(round(brs[cr]))
            teacher["safety_factor"] = sf
            dist[cr] = dist.get(cr, 0) + 1
            out_lines.append(json.dumps(r))
        if out_lines:
            with open(os.path.join(out_dir, os.path.basename(fp)), "w") as fh:
                fh.write("\n".join(out_lines) + "\n")
            n_files += 1
            n_rows += len(out_lines)
    print("relabeled %d rows across %d files -> %s (sf=%.2f)" % (n_rows, n_files, out_dir, sf))
    print("causal rung-dist:", {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main()
