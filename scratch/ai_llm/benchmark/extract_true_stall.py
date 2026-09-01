#!/usr/bin/env python3
"""Extract TRUE closed-loop ns-3 rebuffering per (arm, video, condition) for a paper-safe Fig-1.

The perceptual_qoe REPLAY stall is an artifact for the model arm (it re-simulates downloads and ignores
the serving buffer-guard, inflating the model's stall ~10-100x). This reads the ACTUAL ns-3 MPEG-player
interruption instead:

  MODEL (bridge runs, Quic_artifacts/AiDash/bench_down_scored/<video>/run_<cond>/decisions.jsonl):
    each decision logs cumulative `interruption_time_s` per node -> per-node stall = max over its decisions;
    per-run stall = mean over the run's nodes (mean per-UE rebuffering).

  HEURISTICS (ns-3 runs, Quic_artifacts/<ARM>/<video>/run_<seed>/sim.log):
    the MPEG player prints `FinalizeInterruptionTime ... current interruption_time: X s` per UE. X is the
    REAL mid-playback rebuffer; the end-of-run `ue-id: N InterruptionTime: M` is CONTAMINATED by idle time
    after the video finishes (player DONE, idling to sim-stop) so it is NOT used. per-run stall = mean of
    the per-UE current interruption_time. Odd seeds {1,3,5,7,9} map to conditions {1..5} (same as the model).

Writes results/true_stall.csv (arm,video,cond,stall_s) + prints per-arm/per-cond means.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
Q = os.path.join(_REPO, "Quic_artifacts")
VIDS = ["big_buck_bunny", "chef_cooking", "city_night_traffic", "hiking", "peace_autumn_lake",
        "pexels_28887170", "pexels_30886101", "pexels_34254266", "pexels_34797065", "pexels_35132716"]
SEED2COND = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5}
CUR_RE = re.compile(r"current interruption_time:\s*([0-9.]+)\s*s")


def model_stall():
    """{(video,cond): (mean-per-UE, sum-over-UEs) real rebuffer}."""
    out = {}
    for v in VIDS:
        for cond in range(1, 6):
            dec = os.path.join(Q, "AiDash", "bench_down_scored", v, "run_%d" % cond, "decisions.jsonl")
            if not os.path.isfile(dec):
                continue
            per_node = defaultdict(float)
            for line in open(dec):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # pylint: disable=broad-except
                    continue
                n = d.get("node_id")
                per_node[n] = max(per_node[n], float(d.get("interruption_time_s", 0.0)))
            if per_node:
                vals = list(per_node.values())
                out[(v, cond)] = (mean(vals), sum(vals))
    return out


def heuristic_stall(arm):
    """{(video,cond): (mean-per-UE, sum-over-UEs)} from sim.log current interruption_time lines."""
    out = {}
    for v in VIDS:
        for seed, cond in SEED2COND.items():
            log = os.path.join(Q, arm, v, "run_%d" % seed, "sim.log")
            if not os.path.isfile(log):
                continue
            vals = [float(m) for m in CUR_RE.findall(open(log, errors="ignore").read())]
            if vals:
                out[(v, cond)] = (mean(vals), sum(vals))   # one entry per UE (the finalize call per player)
    return out


def main():
    arms = {"AiDash": model_stall(),
            "Bola": heuristic_stall("Bola"),
            "Raahs": heuristic_stall("Raahs"),
            "Aaash": heuristic_stall("Aaash")}
    outcsv = os.path.join(_HERE, "results", "true_stall.csv")
    with open(outcsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "video", "cond", "stall_s", "stall_sum_s"])  # stall_s=mean per-UE (Fig 1); stall_sum_s=sum over UEs
        for arm, d in arms.items():
            for (v, c), (m, s) in sorted(d.items()):
                w.writerow([arm, v, c, "%.4f" % m, "%.4f" % s])
    print("wrote", outcsv)
    print("\n%-8s | %6s %6s %6s %6s %6s | %7s | n   (mean per-UE stall, s)" %
          ("arm", "100G", "36M", "18M", "9M", "4.5M", "overall"))
    for arm in ("AiDash", "Raahs", "Aaash", "Bola"):
        d = arms[arm]
        per_c = {c: [m for (v, cc), (m, s) in d.items() if cc == c] for c in range(1, 6)}
        cells = [mean(per_c[c]) if per_c[c] else float("nan") for c in range(1, 6)]
        allv = [m for (m, s) in d.values()]
        print("%-8s | %6.2f %6.2f %6.2f %6.2f %6.2f | %7.3f | %d" %
              (arm, cells[0], cells[1], cells[2], cells[3], cells[4], mean(allv), len(allv)))


if __name__ == "__main__":
    main()
