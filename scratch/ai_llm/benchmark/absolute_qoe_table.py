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
import model_arm                                                # noqa: E402  (MODEL_STEM switch)

# local copies of paper_style constants (paper_style imports matplotlib, which is absent from the .venv this
# scoring script runs under - and this script writes tables only, no plots). Keep in sync with paper_style.
# RAAHS removed from the paper benchmark (2026-10). Pensieve is the strongest/learned baseline.
ALGOS = [("AiDash", {"name": "LLM-ABR"}), ("Qmpc", {"name": "Quality-MPC"}), ("Mpc", {"name": "RobustMPC"}),
         ("BolaFull", {"name": "BOLA"})]   # BolaFull = BOLA re-run on whole videos (paper_style.ALGOS)
BRIDGE_ARMS = ("Pensieve", "Mpc", "Qmpc", "Greedy")   # standard-layout bridge runs (score_bridge_arm.sh)
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
COND_ORDER = [1, 2, 3, 4, 5]
PAPER_DIR = os.path.join(_HERE, "paper")

Q = os.path.join(_REPO, "Quic_artifacts")
RESULTS = os.path.join(_HERE, "results")
CFG = os.path.join(_REPO, "scratch", "ai_llm", "config", "oracle_reward.json")
# Benchmark video set: the full 26-video stage-B test split (BENCH_VIDS=<file> to select another list,
# e.g. the old 10-video shrunk matrix). Missing cells for an arm are skipped, so partial arms still tabulate.
VIDS = [v for v in open(os.environ.get("BENCH_VIDS", os.path.join(_HERE, "test_videos_26.txt"))).read().split() if v]
HEURS = ["BolaFull"]
# Cells excluded from the QoE means for ALL policies. The old 10-video benchmark excluded
# ("pexels_34254266", 5) (Pensieve starved one UE for 83 s at 4.5 Mb/s on the pre-fix binary). On the fixed
# binary (re-run 2, 2026-10-31) that cell has 0 s stall in every arm, so nothing is excluded any more.
EXCLUDE_CELLS = set()


def true_stall_sums():
    """{(arm,video,cond): sum-over-UEs true stall (s)} from results/true_stall.csv."""
    out = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "true_stall.csv"))):
        out[(r["arm"], r["video"], int(r["cond"]))] = float(r["stall_sum_s"])
    return out


def run_inputs(arm, video, cond):
    """(sim_log, decisions_jsonl, teacher_shard) for one run; sim_log/decisions_jsonl are mutually exclusive."""
    if arm == "AiDash":   # model arm: layout selected by MODEL_STEM (see model_arm.py)
        return None, model_arm.decisions(video, cond), model_arm.shard(video, cond)
    if arm in BRIDGE_ARMS:   # standard-layout bridge runs (run_<cond>) + persisted teacher shards
        dec = os.path.join(Q, arm, video, "run_%d" % cond, "decisions.jsonl")
        shard = os.path.join(Q, "%s_shards" % arm, "%s_run_%d.jsonl" % (video, cond))
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
                if (v, c) in EXCLUDE_CELLS:
                    continue
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

    def agg(xs):
        # MEAN over videos (the paper statistic). The one starved-UE cell that would swing the mean is
        # excluded above (EXCLUDE_CELLS) for all policies, so the mean is well-behaved here.
        return sum(xs) / len(xs) if xs else float("nan")

    absq = {(a, c): agg(cell[(a, c)]) for a in stems for c in COND_ORDER}
    overall = {a: agg([x for c in COND_ORDER for x in cell[(a, c)]]) for a in stems}

    # LLM-ABR %-improvement over best heuristic, per condition + overall
    def best_heur(c):
        return max(absq[(h, c)] for h in HEURS)
    impr = {c: 100.0 * (absq[("AiDash", c)] - best_heur(c)) / best_heur(c) for c in COND_ORDER}
    bh_overall = max(overall[h] for h in HEURS)
    impr_overall = 100.0 * (overall["AiDash"] - bh_overall) / bh_overall

    # ---- stdout review ----
    name = dict(ALGOS)
    print("Absolute VMAF-QoE per chunk (true ns-3 stall), mean over %d videos:  (%d runs)" % (len(VIDS), n_run))
    for a in stems:
        n_a = sum(len(cell[(a, c)]) for c in COND_ORDER)
        if n_a < len(VIDS) * len(COND_ORDER) - len(EXCLUDE_CELLS):
            print("  NOTE: %s has %d of %d cells" % (a, n_a, len(VIDS) * len(COND_ORDER) - len(EXCLUDE_CELLS)))
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

    # per-(arm,cond) absolute QoE for Fig 2 (absolute VMAF-QoE vs link rate). Keyed by results-file
    # stem so make_paper_figs.fig2_curve can map stem -> ALGOS style.
    with open(os.path.join(PAPER_DIR, "absqoe_by_cond.csv"), "w") as fh:
        fh.write("stem,cond,cond_label,absqoe\n")
        for a in stems:
            for c in COND_ORDER:
                fh.write("%s,%d,%s,%.4f\n" % (a, c, COND[c], absq[(a, c)]))

    order = sorted(stems, key=lambda a: overall[a], reverse=True)

    # ---- per-condition raw-QoE table (raw-QoE analog of Table 1: arms x link rate + overall) ----
    def _best_in_col(c):
        return max(absq[(a, c)] for a in stems)
    best_overall = max(overall[a] for a in stems)
    rows_tex = []
    for a in order:
        cells = []
        for c in COND_ORDER:
            val = "%.2f" % absq[(a, c)]
            cells.append(r"\textbf{" + val + "}" if absq[(a, c)] == _best_in_col(c) else val)
        ov = "%.2f" % overall[a]
        ov = r"\textbf{" + ov + "}" if overall[a] == best_overall else ov
        rows_tex.append(name[a]["name"] + " & " + " & ".join(cells) + " & " + ov + r" \\")
    tex_bc = [r"% Table -- ABSOLUTE per-chunk VMAF-QoE by algorithm x bottleneck link rate (true ns-3 stall). Auto-generated.",
              r"\begin{table}[t]\centering\footnotesize",
              r"\caption{Absolute perceptual QoE (VMAF-based) per video chunk by ABR algorithm and bottleneck "
              r"XPLACEHOLDERX; best per column in bold.}",
              r"\label{tab:absqoe_bycond}",
              r"\begin{tabular}{lrrrrrr}", r"\toprule",
              "Algorithm & " + " & ".join(COND[c] for c in COND_ORDER) + r" & Overall \\", r"\midrule"]
    tex_bc += rows_tex
    tex_bc += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(PAPER_DIR, "table_rawqoe_bycond.tex"), "w").write("\n".join(tex_bc) + "\n")

    md_bc = ["| Algorithm | " + " | ".join(COND[c] for c in COND_ORDER) + " | Overall |",
             "|" + "---|" * (len(COND_ORDER) + 2)]
    for a in order:
        md_bc.append("| %s | " % name[a]["name"]
                     + " | ".join("%.2f" % absq[(a, c)] for c in COND_ORDER)
                     + " | %.2f |" % overall[a])
    open(os.path.join(PAPER_DIR, "table_rawqoe_bycond.md"), "w").write("\n".join(md_bc) + "\n")

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
           r"\caption{Absolute perceptual QoE (VMAF-based) per video chunk, mean over the 26 test videos and "
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
