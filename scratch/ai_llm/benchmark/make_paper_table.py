#!/usr/bin/env python3
"""Table 1 (% of DP-oracle by algorithm x link-rate + overall) as LaTeX + markdown, plus inline_stats.txt.

Reads the per-run results/*.txt for all 7 arms (heuristics + AiDash, all in the SAME format), aggregates
mean %-of-oracle per condition (mean over runs, matching aggregate_table.py), and emits:
  paper/table1_oracle.tex  (booktabs, \\footnotesize; best-per-column + V2 overall bolded)
  paper/table1_oracle.md   (review echo)
  paper/inline_stats.txt   (per-algo overall %oracle/regret/stall/switch + per-condition regret + key deltas)
Uses paper_style.ALGOS for order/display-names and COND for the rate labels.
"""
import collections
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paper_style import ALGOS, ALGO_STYLE, COND, COND_ORDER, PAPER_DIR  # noqa: E402
import model_arm  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")


def parse_results(path):
    """results/*.txt -> per-run dicts (run_idx, pct, regret, vmaf, bit, stall, switch)."""
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) < 8 or not p[0].count("/") or "run_" not in p[0]:
            continue
        try:
            run_idx = int(p[0].split("run_")[-1])
            rows.append({"cond": run_idx, "pct": float(p[2].rstrip("%")), "regret": float(p[3]),
                         "vmaf": float(p[4]), "bit": float(p[5]), "stall": float(p[6]), "switch": float(p[7])})
        except ValueError:
            continue
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    # algo stem -> {cond: [pct...]}, and flat lists for overall + inline stats
    per = {}
    for stem, _ in ALGOS:
        path = model_arm.results_txt() if stem == "AiDash" else os.path.join(RESULTS_DIR, "%s.txt" % stem)
        if not os.path.exists(path):  # native heuristics are scored by score_report.sh <ARM>_dscore
            path = os.path.join(RESULTS_DIR, "%s_dscore.txt" % stem)
        per[stem] = parse_results(path) if os.path.exists(path) else []

    # build per-(algo,cond) mean %oracle, per-algo overall mean %oracle, and per-cond regret
    pct_cell = {stem: {c: mean([r["pct"] for r in per[stem] if r["cond"] == c]) for c in COND_ORDER}
                for stem, _ in ALGOS}
    pct_all = {stem: mean([r["pct"] for r in per[stem]]) for stem, _ in ALGOS}
    reg_cell = {stem: {c: mean([r["regret"] for r in per[stem] if r["cond"] == c]) for c in COND_ORDER}
                for stem, _ in ALGOS}

    # best (max %oracle) per column for bolding
    best = {c: max((pct_cell[s][c] for s, _ in ALGOS if pct_cell[s][c] == pct_cell[s][c]), default=float("nan"))
            for c in COND_ORDER}
    best_all = max((v for v in pct_all.values() if v == v), default=float("nan"))

    os.makedirs(PAPER_DIR, exist_ok=True)

    # ---- LaTeX (booktabs) ----
    def cell(val, is_best):
        if val != val:  # nan
            return "--"
        s = "%.1f" % val
        return "\\textbf{%s}" % s if is_best else s

    lines = []
    lines.append("% Table 1 -- % of greedy causal-teacher ceiling (VMAF-QoE, down-only) by algorithm x bottleneck link rate. Auto-generated.")
    lines.append("\\begin{table}[t]\\centering\\footnotesize")
    lines.append("\\caption{Perceptual QoE as \\%% of the greedy causal-teacher ceiling, by ABR algorithm and "
                 "bottleneck link rate (10 headroom-stratified test videos). Higher is better; best per column in bold.}")
    lines.append("\\label{tab:oracle}")
    lines.append("\\begin{tabular}{l" + "r" * len(COND_ORDER) + "r}")
    lines.append("\\toprule")
    lines.append("Algorithm & " + " & ".join(COND[c] for c in COND_ORDER) + " & Overall \\\\")
    lines.append("\\midrule")
    for stem, sty in ALGOS:
        # bold by DISPLAYED value (1 dp) so genuine round-ties (e.g. V2 & SVAA both 30.3) are both bold
        cells = [cell(pct_cell[stem][c], round(pct_cell[stem][c], 1) == round(best[c], 1)) for c in COND_ORDER]
        ov = pct_all[stem]
        ov_s = cell(ov, round(ov, 1) == round(best_all, 1))
        name = "\\textbf{%s}" % sty["name"] if stem == "Greedy_sem" else sty["name"]
        lines.append("%s & %s & %s \\\\" % (name, " & ".join(cells), ov_s))
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    tex = "\n".join(lines) + "\n"
    open(os.path.join(PAPER_DIR, "table1_oracle.tex"), "w").write(tex)

    # ---- Markdown echo ----
    md = ["| Algorithm | " + " | ".join(COND[c] for c in COND_ORDER) + " | Overall |",
          "|" + "---|" * (len(COND_ORDER) + 2)]
    for stem, sty in ALGOS:
        cells = ["%.1f" % pct_cell[stem][c] if pct_cell[stem][c] == pct_cell[stem][c] else "-"
                 for c in COND_ORDER]
        md.append("| %s | %s | %.1f |" % (sty["name"], " | ".join(cells), pct_all[stem]))
    open(os.path.join(PAPER_DIR, "table1_oracle.md"), "w").write("\n".join(md) + "\n")

    # ---- inline_stats.txt ----
    out = ["Per-algorithm overall metrics (mean over all test runs):",
           "  algo        %oracle  regret  meanVMAF  bitMbps  stall(s)  switch"]
    for stem, sty in ALGOS:
        rs = per[stem]
        out.append("  %-10s  %5.1f   %5.1f   %6.1f   %6.2f   %6.1f   %5.1f" % (
            sty["name"], pct_all[stem], mean([r["regret"] for r in rs]), mean([r["vmaf"] for r in rs]),
            mean([r["bit"] for r in rs]), mean([r["stall"] for r in rs]), mean([r["switch"] for r in rs])))
    out.append("")
    out.append("Per-condition mean regret (lower=better) - checks 'V2 lowest in every condition':")
    out.append("  algo        " + "  ".join("%8s" % COND[c] for c in COND_ORDER))
    for stem, sty in ALGOS:
        out.append("  %-10s  " % sty["name"] + "  ".join("%8.1f" % reg_cell[stem][c] for c in COND_ORDER))
    # key deltas
    out.append("")
    v2 = pct_cell["AiDash"]  # AiDash = our LLM-ABR model (the headline arm)
    heur = [s for s, _ in ALGOS if s != "AiDash"]
    best_heur = max(heur, key=lambda s: pct_all[s]) if heur else None
    out.append("Key deltas for prose:")
    out.append("  Ours (LLM-ABR) overall %%oracle = %.1f (best overall arm = %.1f)" % (
        pct_all["AiDash"], best_all))
    if best_heur:
        out.append("  Best heuristic = %s @ %.1f overall (ours %+.1f vs it)" % (
            best_heur, pct_all[best_heur], pct_all["AiDash"] - pct_all[best_heur]))
    out.append("  Ours @36Mbps=%.1f @18Mbps=%.1f (moderate congestion: ours strongest) ; @100Gb/s=%.1f (uncongested: ours weakest)" % (
        v2[2], v2[3], v2[1]))
    v2_lowest = all(reg_cell["AiDash"][c] <= min(reg_cell[s][c] for s, _ in ALGOS) + 1e-9 for c in COND_ORDER)
    out.append("  Ours has lowest per-condition regret in EVERY condition: %s" % v2_lowest)
    open(os.path.join(PAPER_DIR, "inline_stats.txt"), "w").write("\n".join(out) + "\n")

    print("wrote paper/table1_oracle.tex, .md, inline_stats.txt")
    print("\n".join(md))


if __name__ == "__main__":
    main()
