#!/usr/bin/env python3
"""Multi-seed benchmark statistics (model v3, Phase F): per-arm means with 95 % CIs and paired tests vs a reference arm.

Each arm is a list of scored STEMS, one per seed set (the scoring chain renames run dirs to run_<cond>, so seed
sets cannot share a stem); every stem contributes results/<STEM>.txt (per-cell %teacher, mean VMAF; written by
score_bridge_arm.sh / score_v2_perceptual) and, optionally, rows of results/true_stall.csv (per-UE true stall,
extract_true_stall.py) under the stem's stall key. Cells = video x rate x seed set.

  mean +- CI   : cluster bootstrap over VIDEOS (a resampled video brings all its rates/seeds), 2000 draws
  vs reference : paired difference per cell, cluster-bootstrap CI of the mean difference, and a Wilcoxon
                 signed-rank test on the per-video means (n = videos) - rates pooled and per rate

Usage (re-run 2, one seed set per arm):
  python3.11 multiseed_stats.py --videos test_videos_26.txt --ref Qmpc \
      --arm Ours=AiDashClean:AiDash --arm Qmpc=Qmpc:Qmpc --arm Mpc=Mpc:Mpc --arm BOLA=BolaFull_dscore:BolaFull
  (--arm LABEL=STEM[:STALLKEY][,STEM[:STALLKEY]...]; omit :STALLKEY when the stem has no true-stall rows)
"""
import argparse
import csv
import os
import random
import re
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
LINE = re.compile(r"^(\S+)/run_(\d)\s+\d+\s+([0-9.]+)%\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)")


def load_stem(stem, stall_key, videos):
    """{(video, cond): {pct, vmaf, stall}} for one seed set."""
    cells = {}
    for line in open(os.path.join(RES, stem + ".txt")):
        m = LINE.match(line)
        if m and m.group(1) in videos:
            cells[(m.group(1), int(m.group(2)))] = {"pct": float(m.group(3)), "vmaf": float(m.group(5))}
    if stall_key:
        for r in csv.DictReader(open(os.path.join(RES, "true_stall.csv"))):
            k = (r["video"], int(r["cond"]))
            if r["arm"] == stall_key and k in cells:
                cells[k]["stall"] = float(r["stall_s"])
    return cells


def load_arm(spec, videos):
    """{(video, cond, seedset): metrics}"""
    out = {}
    for i, part in enumerate(spec.split(",")):
        stem, _, key = part.partition(":")
        for (v, c), m in load_stem(stem, key or None, videos).items():
            out[(v, c, i)] = m
    return out


def by_video(cells, metric, conds):
    """{video: [values over (cond in conds) x seedset]}"""
    d = defaultdict(list)
    for (v, c, s), m in cells.items():
        if c in conds and metric in m:
            d[v].append(m[metric])
    return d


def cluster_boot(per_video, draws, rng, stat=mean):
    vids = list(per_video)
    vals = []
    for _ in range(draws):
        samp = [x for v in rng.choices(vids, k=len(vids)) for x in per_video[v]]
        vals.append(stat(samp))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def fmt_ci(per_video, draws, rng, unit=""):
    xs = [x for v in per_video.values() for x in v]
    if not xs:
        return "%14s" % "-"
    lo, hi = cluster_boot(per_video, draws, rng)
    return "%6.2f [%5.2f,%5.2f]%s" % (mean(xs), lo, hi, unit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=STEM[:STALLKEY][,...]")
    ap.add_argument("--videos", required=True, help="video list (one per line), e.g. test_videos_26.txt")
    ap.add_argument("--ref", required=True, help="arm LABEL used as the paired reference")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    videos = set(open(args.videos if os.path.exists(args.videos) else os.path.join(HERE, args.videos)).read().split())
    arms = {}
    for a in args.arm:
        label, _, spec = a.partition("=")
        arms[label] = load_arm(spec, videos)
    groups = [("overall", list(COND))] + [(COND[c], [c]) for c in COND]

    for metric, unit in (("pct", " %"), ("vmaf", ""), ("stall", " s")):
        print("\n== %s (mean [95%% cluster-bootstrap CI over %d videos])" % (
            {"pct": "% of greedy teacher", "vmaf": "mean VMAF", "stall": "true stall per UE"}[metric], len(videos)))
        print("  %-10s " % "arm" + " ".join("%22s" % g for g, _ in groups))
        for label, cells in arms.items():
            n = len({s for (_, _, s) in cells})
            print("  %-10s " % ("%s(x%d)" % (label, n)) + " ".join(
                "%22s" % fmt_ci(by_video(cells, metric, conds), args.draws, rng, unit) for _, conds in groups))

        ref = arms[args.ref]
        print("  -- paired vs %s: mean diff [CI], Wilcoxon p on per-video means" % args.ref)
        for label, cells in arms.items():
            if label == args.ref:
                continue
            row = []
            for g, conds in groups:
                diff = defaultdict(list)
                for k, m in cells.items():
                    if k[1] in conds and metric in m and k in ref and metric in ref[k]:
                        diff[k[0]].append(m[metric] - ref[k][metric])
                if len(diff) < 5:
                    row.append("%22s" % "-")
                    continue
                pv = [mean(x) for x in diff.values()]
                lo, hi = cluster_boot(diff, args.draws, rng)
                try:
                    p = wilcoxon(pv).pvalue if any(abs(x) > 1e-12 for x in pv) else 1.0
                except ValueError:
                    p = float("nan")
                row.append("%+6.2f [%+5.2f,%+5.2f] p=%.3f" % (mean([x for v in diff.values() for x in v]), lo, hi, p))
            print("  %-10s " % label + " ".join("%22s" % r for r in row))


if __name__ == "__main__":
    main()
