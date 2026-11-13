#!/usr/bin/env python3
"""Absolute QoE/seg with TRUE stall (0.65*VMAF - 65*stall_sum/segs, no switch term) per arm and rate, seed sets
averaged per cell; paired vs the first arm: cluster bootstrap over the 26 test videos (2000 draws) + Wilcoxon."""
import csv, re, sys, random
from collections import defaultdict
from statistics import mean
from scipy.stats import wilcoxon
R = "scratch/ai_llm/benchmark/results"
TEST = sorted(set(open("scratch/ai_llm/benchmark/test_videos_26.txt").read().split()))
ST = defaultdict(dict)
for r in csv.DictReader(open(R + "/true_stall.csv")):
    ST[r["arm"]][(r["video"], int(r["cond"]))] = float(r["stall_sum_s"])
COND = {1: "100G", 2: "36M", 3: "18M", 4: "9M", 5: "4.5M"}
arms = {}
for spec in sys.argv[1:]:
    lab, stems = spec.split("=")
    per = defaultdict(list)
    for s in stems.split(","):
        stem, key = s.split(":")
        for line in open("%s/%s.txt" % (R, stem)):
            m = re.match(r"^(\S+)/run_(\d)\s+(\d+)\s+\S+\s+\S+\s+([0-9.]+)", line)
            if m and m.group(1) in TEST:
                k = (m.group(1), int(m.group(2)))
                per[k].append(0.65 * float(m.group(4)) - 65.0 * ST[key][k] / int(m.group(3)))
    arms[lab] = {k: mean(v) for k, v in per.items()}
def vid(a, cs):  # per-video mean over the given conditions
    return [mean(arms[a][(v, c)] for c in cs) for v in TEST]
groups = [("overall", list(COND))] + [(COND[c], [c]) for c in COND]
print("%-10s" % "QoE/seg" + "".join("%22s" % g for g, _ in groups))
for a in arms:
    print("%-10s" % a + "".join("%22.2f" % mean(vid(a, cs)) for _, cs in groups))
ref = list(arms)[0]; rnd = random.Random(0)
print("-- paired vs %s: diff [95%% CI], Wilcoxon p" % ref)
for a in list(arms)[1:]:
    cells = []
    for _, cs in groups:
        d = [x - y for x, y in zip(vid(a, cs), vid(ref, cs))]
        bs = sorted(mean(rnd.choice(d) for _ in d) for _ in range(2000))
        p = wilcoxon(d).pvalue if any(d) else 1.0
        cells.append("%+.2f [%+.2f,%+.2f] p=%.3f" % (mean(d), bs[50], bs[1949], p))
    print("%-10s" % a + "".join("%22s" % c[:21] if False else "  " + c for c in cells))
