#!/usr/bin/env python3
"""Per-video variance / CI for the headline absolute VMAF-QoE (Ours=9.55, RAAHS=7.42).

Reuses the EXACT scoring path of absolute_qoe_table.py (ai_llm.perceptual_qoe.score_run +
results/true_stall.csv), but aggregates PER VIDEO instead of per condition:

  per run (video x cond):  abs_true = pol_qoe + w_stall*replay_stall - w_stall*true_stall
                           abs_per_segment = abs_true / segments
  per video             :  mean of abs_per_segment over the 5 conditions
  overall               :  mean over all (video x cond) cells  (== mean of 10 per-video means,
                           balanced 5 conds each)

No re-simulation: reads only saved decisions.jsonl / sim.log / teacher shards / true_stall.csv.
"""
import collections
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratch"))
sys.path.insert(0, _HERE)
from ai_llm.perceptual_qoe import score_run                     # noqa: E402
from scripts.dp_oracle import load_reward_config                # noqa: E402

COND_ORDER = [1, 2, 3, 4, 5]
Q = os.path.join(_REPO, "Quic_artifacts")
RESULTS = os.path.join(_HERE, "results")
CFG = os.path.join(_REPO, "scratch", "ai_llm", "config", "oracle_reward.json")
VIDS = ["big_buck_bunny", "chef_cooking", "city_night_traffic", "hiking", "peace_autumn_lake",
        "pexels_28887170", "pexels_30886101", "pexels_34254266", "pexels_34797065", "pexels_35132716"]
ARMS = [("AiDash", "Ours/LLM-ABR"), ("Raahs", "RAAHS")]


def true_stall_sums():
    out = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "true_stall.csv"))):
        out[(r["arm"], r["video"], int(r["cond"]))] = float(r["stall_sum_s"])
    return out


def run_inputs(arm, video, cond):
    if arm == "AiDash":
        dec = os.path.join(Q, "AiDash", "bench_down_scored", video, "run_%d" % cond, "decisions.jsonl")
        shard = os.path.join(Q, "AiDash", "shards_down", "%s_run_%d.jsonl" % (video, cond))
        return None, dec, shard
    log = os.path.join(Q, "%s_dscore" % arm, video, "run_%d" % cond, "sim.log")
    shard = os.path.join(Q, "%s_teacher_down" % arm, "%s_run_%d.jsonl" % (video, cond))
    return log, None, shard


def stats(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    ci = 1.96 * sd / math.sqrt(n)
    return m, sd, ci, min(xs), max(xs)


def main():
    cfg = load_reward_config(CFG)
    w_stall = float(cfg["reward_weights"]["w_stall"])
    tstall = true_stall_sums()

    # per-video per-cond abs_per_segment
    cell = {arm: collections.defaultdict(dict) for arm, _ in ARMS}  # arm -> video -> {cond: val}
    for arm, _ in ARMS:
        for v in VIDS:
            for c in COND_ORDER:
                sim_log, dec, shard = run_inputs(arm, v, c)
                if not os.path.isfile(shard) or not os.path.isfile(dec or sim_log):
                    continue
                res = score_run(sim_log, shard, cfg, label="%s/%s/run_%d" % (arm, v, c), decisions_jsonl=dec)
                if res is None:
                    continue
                pol = res["_qoe"]["policy"]
                abs_true = pol + w_stall * res["stall_s"] - w_stall * tstall.get((arm, v, c), 0.0)
                cell[arm][v][c] = abs_true / max(res["segments"], 1)

    per_video = {}  # arm -> [ (video, mean-over-conds) ]
    for arm, _ in ARMS:
        pv = []
        for v in VIDS:
            vals = [cell[arm][v][c] for c in COND_ORDER if c in cell[arm][v]]
            pv.append((v, sum(vals) / len(vals), len(vals)))
        per_video[arm] = pv

    for arm, name in ARMS:
        print("\n=== %s (%s) : per-video absolute VMAF-QoE (mean over 5 conditions) ===" % (name, arm))
        for v, mv, nc in per_video[arm]:
            print("  %-22s %8.4f   (%d conds)" % (v, mv, nc))
        means = [mv for _, mv, _ in per_video[arm]]
        m, sd, ci, lo, hi = stats(means)
        # overall per-cell mean (balanced == mean of per-video means)
        allcells = [cell[arm][v][c] for v in VIDS for c in COND_ORDER if c in cell[arm][v]]
        overall = sum(allcells) / len(allcells)
        print("  ---- mean(per-video)=%.4f  overall(per-cell)=%.4f  std=%.4f  95%%CI=+/-%.4f  "
              "[%.4f, %.4f]  min=%.4f max=%.4f" % (m, overall, sd, ci, m - ci, m + ci, lo, hi))

    # paired comparison across videos
    print("\n=== Paired Ours vs RAAHS across 10 videos ===")
    diffs = []
    wins = 0
    o = dict((v, mv) for v, mv, _ in per_video["AiDash"])
    r = dict((v, mv) for v, mv, _ in per_video["Raahs"])
    for v in VIDS:
        d = o[v] - r[v]
        diffs.append(d)
        wins += int(d > 0)
        print("  %-22s Ours=%8.4f  RAAHS=%8.4f  diff=%+8.4f" % (v, o[v], r[v], d))
    md, sdd, cid, lod, hid = stats(diffs)
    tstat = md / (sdd / math.sqrt(len(diffs)))
    print("  ---- mean diff=%.4f  std=%.4f  95%%CI=+/-%.4f  [%.4f, %.4f]" % (md, sdd, cid, md - cid, md + cid))
    print("  ---- Ours wins %d/10 videos; paired t (df=9) = %.4f" % (wins, tstat))
    try:
        from scipy import stats as sp
        t, p = sp.ttest_rel([o[v] for v in VIDS], [r[v] for v in VIDS])
        print("  ---- scipy paired t=%.4f  p=%.6g" % (t, p))
    except Exception as e:
        print("  ---- scipy unavailable (%s); report t-stat only" % type(e).__name__)


if __name__ == "__main__":
    main()
