#!/usr/bin/env python3
"""Full Eq.(1) absolute QoE per cell for the Phase F arms, with the TRUE ns-3 stall.

Same method as absolute_qoe_table.py: perceptual_qoe.score_run replays the delivered rungs through the reward of
config/oracle_reward.json (content weight omega_n * VMAF quality term, stall term, down-only switching term with
the scene-cut discount); its replay stall is then swapped for the stall measured by ns-3:
    abs_true = policy_qoe + w_stall * replay_stall_sum - w_stall * true_stall_sum      (per cell, summed over UEs)
    qoe_full = abs_true / segments
Writes results/full_qoe_cells.csv: stem, video, cond, segments, qoe_full, qoe_q_stall (0.65*VMAF - 65*stall/seg)
Usage: python3.11 full_qoe_cells.py
"""
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratch"))
from ai_llm.perceptual_qoe import score_run          # noqa: E402
from scripts.dp_oracle import load_reward_config     # noqa: E402

Q = os.path.join(_REPO, "Quic_artifacts")
RES = os.path.join(_HERE, "results")
VIDS = open(os.path.join(_HERE, "test_videos_26.txt")).read().split()
BRIDGE = ["OursV3_s1", "OursV3_s2", "OursV3_s3", "Qmpc", "Qmpc_s2", "Qmpc_s3", "Mpc", "Mpc_s2", "Mpc_s3"]
BOLA = ["BolaFull", "BolaFull_s2", "BolaFull_s3"]

cfg = load_reward_config(os.path.join(_REPO, "scratch", "ai_llm", "config", "oracle_reward.json"))
w_stall = float(cfg["reward_weights"]["w_stall"])
w_vmaf = float(cfg["reward_weights"]["w_vmaf"])
ts = {}
for r in csv.DictReader(open(os.path.join(RES, "true_stall.csv"))):
    ts[(r["arm"], r["video"], int(r["cond"]))] = float(r["stall_sum_s"])


def vmaf_of(results_stem):
    out = {}
    for line in open(os.path.join(RES, results_stem + ".txt")):
        m = re.match(r"^(\S+)/run_(\d)\s+(\d+)\s+\S+\s+\S+\s+([0-9.]+)", line)
        if m:
            out[(m.group(1), int(m.group(2)))] = (int(m.group(3)), float(m.group(4)))
    return out


rows = []
for stem in BRIDGE + BOLA:
    bola = stem in BOLA
    vm = vmaf_of(stem + "_dscore" if bola else stem)
    for v in VIDS:
        for c in range(1, 6):
            if bola:
                log = os.path.join(Q, stem + "_dscore", v, "run_%d" % c, "sim.log")
                dec = None
                shard = os.path.join(Q, stem + "_teacher_down", "%s_run_%d.jsonl" % (v, c))
            else:
                log = None
                dec = os.path.join(Q, stem, v, "run_%d" % c, "decisions.jsonl")
                shard = os.path.join(Q, stem + "_shards", "%s_run_%d.jsonl" % (v, c))
            res = score_run(log, shard, cfg, label="%s/%s/%d" % (stem, v, c), decisions_jsonl=dec)
            if res is None:
                print("MISSING", stem, v, c)
                continue
            segs = res["segments"]
            true = ts[(stem, v, c)]
            full = (res["_qoe"]["policy"] + w_stall * res["stall_s"] - w_stall * true) / max(segs, 1)
            s2, vmaf = vm[(v, c)]
            simple = w_vmaf * vmaf - w_stall * true / s2
            rows.append((stem, v, c, segs, full, simple, s2))
    print(stem, "done", sum(1 for r in rows if r[0] == stem))

with open(os.path.join(RES, "full_qoe_cells.csv"), "w") as fh:
    w = csv.writer(fh)
    w.writerow(["stem", "video", "cond", "segments", "qoe_full", "qoe_q_stall", "segments_results"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], "%.4f" % r[4], "%.4f" % r[5], r[6]])
print("wrote", len(rows), "cells; segment-count mismatches vs results files:",
      sum(1 for r in rows if r[3] != r[6]))
