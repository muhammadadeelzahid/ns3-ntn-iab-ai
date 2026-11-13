#!/usr/bin/env python3
"""Jain's fairness index (Eq. in the paper) across the 3 concurrent UEs, per policy + feeder rate.

Reproduces TABLE II(b) of latex-article/slm_abr.tex and extends it to the Pensieve arm. For each run
(= one video x one feeder rate, 3 UEs) we form each UE's delivered bitrate and delivered VMAF from
its chosen rungs + the per-segment metadata, then J = (sum x)^2 / (n sum x^2) over the 3 UEs; the
per-condition value is the mean over the 10 test videos. Bitrate fairness reproduces the published
Ours column; VMAF is a chosen-rung approximation (see note in the paper pipeline).

Per-UE source (chosen rung per node): AiDash  Quic_artifacts/AiDash/bench_down_scored/<v>/run_<c>/decisions.jsonl
                                      Pensieve Quic_artifacts/Pensieve/<v>/run_<c>/decisions.jsonl
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "scratch"))
from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS  # noqa: E402
import model_arm  # noqa: E402  (MODEL_STEM switch for the model arm's layout)

META = "/project/6030214/adeel/abr/output/metadata"
Q = os.path.join(_REPO, "Quic_artifacts")
COND = {1: "100\\,Gb/s", 2: "36\\,Mb/s", 3: "18\\,Mb/s", 4: "9\\,Mb/s", 5: "4.5\\,Mb/s"}
COND_ORDER = [1, 2, 3, 4, 5]
_HERE = os.path.dirname(os.path.abspath(__file__))
# Full 26-video stage-B test split (BENCH_VIDS=<file> selects another list); missing cells are skipped.
VIDS = [v for v in open(os.environ.get("BENCH_VIDS", os.path.join(_HERE, "test_videos_26.txt"))).read().split() if v]
POLICIES = [("AiDash", "Ours"), ("Qmpc", "Quality-MPC"), ("Mpc", "RobustMPC"),
            ("BolaFull", "BOLA")]   # 4-arm full-length benchmark (paper_style.ALGOS)
BRIDGE_ARMS = ("Pensieve", "Mpc", "Qmpc", "Greedy")   # standard-layout bridge runs with chosen_rung
# Degenerate 4.5 Mb/s cell (single-UE scheduler starvation) excluded for all policies, matching the
# QoE table; it is analyzed separately as a closed-loop fairness finding.
EXCLUDE_CELLS = {("pexels_34254266", 5)}

_mcache = {}


def _meta(v):
    if v not in _mcache:
        _mcache[v] = load_metadata_index(os.path.join(META, "%s_segments.csv" % v))
    return _mcache[v]


def _jain(xs):
    xs = [float(x) for x in xs if np.isfinite(x) and x >= 0]
    if not xs or sum(x * x for x in xs) <= 0:
        return None
    return (sum(xs) ** 2) / (len(xs) * sum(x * x for x in xs))


# Per-UE rung source per arm. Bridge arms (AiDash/Pensieve) log decisions.jsonl with `chosen_rung`;
# heuristics log <ARM>_dscore/.../trace_decisions.jsonl with teacher.chosen_bitrate_class. All arms
# use the SAME uniform method (chosen rung -> metadata actual_bitrate/VMAF) so J is comparable.
def _dec_path(arm, v, c):
    if arm == "AiDash":
        return model_arm.decisions(v, c)
    if arm in BRIDGE_ARMS:
        return os.path.join(Q, arm, v, "run_%d" % c, "decisions.jsonl")
    return os.path.join(Q, "%s_dscore" % arm, v, "run_%d" % c, "trace_decisions.jsonl")


def _rung_of(arm, d):
    if arm == "AiDash" or arm in BRIDGE_ARMS:
        return d.get("chosen_rung")
    return (d.get("teacher") or {}).get("chosen_bitrate_class")


def _per_ue(arm, v, c):
    """{node: (mean delivered bitrate kbps, mean delivered VMAF)} over the node's segments."""
    p = _dec_path(arm, v, c)
    if not os.path.isfile(p):
        return None
    idx = _meta(v)
    n_seg = len(idx.segment_ids())
    acc = defaultdict(lambda: [[], []])
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rung = _rung_of(arm, d)
        if rung is None:
            continue
        sm = idx.get((int(d.get("segment_id", 1)) - 1) % n_seg)  # 1-indexed -> 0-indexed, looped
        if sm is None:
            continue
        lv = LADDER_LEVELS[int(rung)]
        acc[d["node_id"]][0].append(float(sm.actual_bitrate_kbps[lv]))
        acc[d["node_id"]][1].append(float(sm.vmaf[lv]))
    return {n: (np.mean(b[0]), np.mean(b[1])) for n, b in acc.items() if b[0]}


def fairness():
    """{policy: {cond: (J_bitrate, J_vmaf)}, 'overall': ...}."""
    out = {}
    for arm, _ in POLICIES:
        out[arm] = {}
        allb, allv = [], []
        for c in COND_ORDER:
            jb, jv = [], []
            for v in VIDS:
                if (v, c) in EXCLUDE_CELLS:
                    continue
                pu = _per_ue(arm, v, c)
                if not pu or len(pu) < 2:
                    continue
                b = _jain([x[0] for x in pu.values()])
                vm = _jain([x[1] for x in pu.values()])
                if b is not None:
                    jb.append(b); allb.append(b)
                if vm is not None:
                    jv.append(vm); allv.append(vm)
            out[arm][c] = (np.mean(jb) if jb else float("nan"),
                           np.mean(jv) if jv else float("nan"))
        out[arm]["overall"] = (np.mean(allb) if allb else float("nan"),
                               np.mean(allv) if allv else float("nan"))
    return out


if __name__ == "__main__":
    f = fairness()
    name = dict(POLICIES)
    for arm, _ in POLICIES:
        print("=== %s ===" % name[arm])
        print("  %-10s %8s %8s" % ("feeder", "Jbitrate", "Jvmaf"))
        for c in COND_ORDER:
            b, v = f[arm][c]
            print("  %-10s %8.2f %8.2f" % (COND[c].replace("\\,", ""), b, v))
        b, v = f[arm]["overall"]
        print("  %-10s %8.2f %8.2f" % ("Overall", b, v))
