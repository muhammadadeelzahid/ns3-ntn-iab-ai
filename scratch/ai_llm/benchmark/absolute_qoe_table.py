#!/usr/bin/env python3
"""Table 2 - ABSOLUTE VMAF-QoE per chunk (+ %-improvement over the best heuristic).

Field-standard companion to Table 1's %-of-oracle: Pensieve/NADA report absolute per-chunk QoE and
improvement over a baseline, not an oracle-normalized percentage. We report each arm's own achieved
VMAF-QoE (per-segment mean) using the TRUE ns-3 rebuffer (not the perceptual_qoe replay stall, which
inflates the model arm - same artifact fixed in Fig 1), plus LLM-ABR's %-improvement over the best heuristic.

Per run (= one video × one condition, summed over the 3 UEs): `replay_session` returns `combined_qoe` and
`stall_s`; the stall term in `_segment_reward` (dp_oracle.py) is exactly `-w_stall·rebuffer_s` (w_rung=0), so
    abs_qoe_true    = combined_qoe + w_stall·replay_stall_sum - w_stall·true_stall_sum   # swap replay->true
    abs_per_segment = abs_qoe_true / segments                                            # per-chunk mean
Then per condition = mean over the 10 videos; %-improvement(cond) = 100·(LLM-ABR - max_heur)/max_heur.

Reuses `ai_llm.perceptual_qoe.score_run` (identical QoE path as Table 1) + `results/true_stall.csv`
(`stall_sum_s`, from extract_true_stall.py). Teacher shards persisted off /tmp:
  MODEL      : Quic_artifacts/AiDash/bench_down_scored/<v>/run_<c>/decisions.jsonl + AiDash/shards_down/<v>_run_<c>.jsonl
  HEURISTICS : Quic_artifacts/<ARM>_dscore/<v>/run_<c>/sim.log             + <ARM>_teacher_down/<v>_run_<c>.jsonl
"""
import collections
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratch"))
sys.path.insert(0, _HERE)
from ai_llm.perceptual_qoe import score_run                     # noqa: E402
from scripts.dp_oracle import load_reward_config                # noqa: E402

# local copies of paper_style constants (paper_style imports matplotlib, which is absent from the .venv this
# scoring script runs under - and this script writes tables only, no plots). Keep in sync with paper_style.
ALGOS = [("AiDash", {"name": "LLM-ABR"}), ("Raahs", {"name": "RAAHS"}),
         ("Aaash", {"name": "AAASH"}), ("Bola", {"name": "BOLA"})]
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
COND_ORDER = [1, 2, 3, 4, 5]
PAPER_DIR = os.path.join(_HERE, "paper")

Q = os.path.join(_REPO, "Quic_artifacts")
RESULTS = os.path.join(_HERE, "results")
CFG = os.path.join(_REPO, "scratch", "ai_llm", "config", "oracle_reward.json")
VIDS = ["big_buck_bunny", "chef_cooking", "city_night_traffic", "hiking", "peace_autumn_lake",
        "pexels_28887170", "pexels_30886101", "pexels_34254266", "pexels_34797065", "pexels_35132716"]
HEURS = ["Raahs", "Aaash", "Bola"]


def true_stall_sums():
    """{(arm,video,cond): sum-over-UEs true stall (s)} from results/true_stall.csv."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "true_stall.csv"))):
        out[(r["arm"], r["video"], int(r["cond"]))] = float(r["stall_sum_s"])
    return out


def run_inputs(arm, video, cond):
    """(sim_log, decisions_jsonl, teacher_shard) for one run; sim_log/decisions_jsonl are mutually exclusive."""
    if arm == "AiDash":
        dec = os.path.join(Q, "AiDash", "bench_down_scored", video, "run_%d" % cond, "decisions.jsonl")
        shard = os.path.join(Q, "AiDash", "shards_down", "%s_run_%d.jsonl" % (video, cond))
        return None, dec, shard
    log = os.path.join(Q, "%s_dscore" % arm, video, "run_%d" % cond, "sim.log")
    shard = os.path.join(Q, "%s_teacher_down" % arm, "%s_run_%d.jsonl" % (video, cond))
    return log, None, shard


def main():
    cfg = load_reward_config(CFG)
    w_stall = float(cfg["reward_weights"]["w_stall"])
    tstall = true_stall_sums()
    stems = [s for s, _ in ALGOS]                     # AiDash, Raahs, Aaash, Bola (registry order)

    # per (arm, cond) -> list of abs_per_segment over videos
    cell = collections.defaultdict(list)
    n_run = 0
    for arm in stems:
        for v in VIDS:
            for c in COND_ORDER:
                sim_log, dec, shard = run_inputs(arm, v, c)
                if not os.path.isfile(shard) or not os.path.isfile(dec or sim_log):
                    continue
                res = score_run(sim_log, shard, cfg, label="%s/%s/run_%d" % (arm, v, c), decisions_jsonl=dec)
                if res is None:
                    continue
                pol = res["_qoe"]["policy"]
                replay_stall = res["stall_s"]
                segs = res["segments"]
                ts = tstall.get((arm, v, c), 0.0)
                abs_true = pol + w_stall * replay_stall - w_stall * ts   # swap replay stall -> true ns-3 stall
                cell[(arm, c)].append(abs_true / max(segs, 1))
                n_run += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    absq = {(a, c): mean(cell[(a, c)]) for a in stems for c in COND_ORDER}
    overall = {a: mean([x for c in COND_ORDER for x in cell[(a, c)]]) for a in stems}

    # LLM-ABR %-improvement over best heuristic, per condition + overall
    def best_heur(c):
        return max(absq[(h, c)] for h in HEURS)
    impr = {c: 100.0 * (absq[("AiDash", c)] - best_heur(c)) / best_heur(c) for c in COND_ORDER}
    bh_overall = max(overall[h] for h in HEURS)
    impr_overall = 100.0 * (overall["AiDash"] - bh_overall) / bh_overall

    # ---- stdout review ----
    name = dict(ALGOS)
    print("Absolute VMAF-QoE per chunk (true ns-3 stall), mean over 10 videos:  (%d runs)" % n_run)
    hdr = "  %-16s " % "Algorithm" + " ".join("%8s" % COND[c] for c in COND_ORDER) + " %8s" % "Overall"
    print(hdr)
    for a in stems:
        print("  %-16s " % name[a]["name"] + " ".join("%8.2f" % absq[(a, c)] for c in COND_ORDER)
              + " %8.2f" % overall[a])
    print("  %-16s " % "LLM-ABR impr%" + " ".join("%+7.1f%%" % impr[c] for c in COND_ORDER)
          + " %+7.1f%%" % impr_overall)

    # ---- paper floats: OVERALL-only (per-condition absolute QoE is stall-dominated/heavy-tailed; the overall
    #      is the robust, Table-1-consistent number). Arms sorted by absolute QoE descending. ----
    os.makedirs(PAPER_DIR, exist_ok=True)
    order = sorted(stems, key=lambda a: overall[a], reverse=True)

    md = ["| Algorithm | Absolute VMAF-QoE (per chunk) |", "|---|---|"]
    for a in order:
        md.append("| %s | %.2f |" % (name[a]["name"], overall[a]))
    md.append("")
    md.append("LLM-ABR improvement over best heuristic (overall): **%+.1f%%**." % impr_overall)
    open(os.path.join(PAPER_DIR, "table2_absqoe.md"), "w").write("\n".join(md) + "\n")

    # ---- latex (booktabs, mirrors make_paper_table). Built by concatenation to avoid %-operator vs literal-% ----
    imp = "%+.1f" % impr_overall  # e.g. "+28.7"
    tex = ["% Table 2 -- absolute VMAF-QoE per chunk (true ns-3 stall). "
           "LLM-ABR overall improvement over best heuristic = " + imp + r"\%.",
           r"\begin{table}[t]\centering\footnotesize",
           r"\caption{Absolute perceptual QoE (VMAF-based) per video chunk, averaged over the 10 test videos and "
           r"five bottleneck link rates (true ns-3 rebuffering). LLM-ABR improves over the best heuristic by "
           + imp + r"\%. Higher is better.}",
           r"\label{tab:absqoe}",
           r"\begin{tabular}{lr}",
           r"\toprule",
           r"Algorithm & Absolute VMAF-QoE \\",
           r"\midrule"]
    for a in order:
        nm, val = name[a]["name"], ("%.2f" % overall[a])
        if a == "AiDash":
            tex.append(r"\textbf{" + nm + r"} & \textbf{" + val + r"} \\")
        else:
            tex.append(nm + " & " + val + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(PAPER_DIR, "table2_absqoe.tex"), "w").write("\n".join(tex) + "\n")
    print("\nwrote paper/table2_absqoe.{md,tex}  (overall-only; full per-condition breakdown above is stdout-only)")


if __name__ == "__main__":
    main()
