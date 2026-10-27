#!/usr/bin/env python3
"""Paper figures (uniform via paper_style). Default: Figure 1 only.

  Figure 1 (default)         -- quality-rebuffering Pareto: mean delivered VMAF vs rebuffering, 7 arms.
  --fig2 curve               -- absolute per-chunk VMAF-QoE vs link rate (one line per algorithm).

Reads the per-run results/*.txt (all 7 arms, same format). Run make_paper_table.py first is not required,
but results/AiDash.txt must exist (from score_v2_perceptual.py).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paper_style import (ALGOS, COND, COND_ORDER, COL_SINGLE, PAPER_DIR, apply_style, save_fig)  # noqa: E402
from make_paper_table import parse_results, mean, RESULTS_DIR  # reuse the parser  # noqa: E402
import model_arm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _results_path(stem):
    return model_arm.results_txt() if stem == "AiDash" else os.path.join(RESULTS_DIR, "%s.txt" % stem)


def load_all():
    return {stem: parse_results(_results_path(stem)) for stem, _ in ALGOS if os.path.exists(_results_path(stem))}


def _true_stall_by_arm():
    """Per-arm mean of the TRUE closed-loop ns-3 rebuffering (results/true_stall.csv from
    extract_true_stall.py). The perceptual_qoe REPLAY stall inflates the model arm ~10-100x (it
    re-simulates downloads and ignores the serving buffer-guard), so Fig-1's y-axis MUST use this
    actual ns-3 MPEG-player interruption instead."""
    import csv
    path = os.path.join(RESULTS_DIR, "true_stall.csv")
    # Exclude the degenerate single-UE starvation cell (pexels_34254266 @ 4.5 Mb/s) for all arms,
    # matching the QoE table; it is reported separately as a closed-loop fairness finding.
    exclude = {("pexels_34254266", 5)}
    by_arm = {}
    for row in csv.DictReader(open(path)):
        if (row["video"], int(row["cond"])) in exclude:
            continue
        by_arm.setdefault(row["arm"], []).append(float(row["stall_s"]))
    return {a: mean(v) for a, v in by_arm.items()}


def fig1_pareto(data):
    """x = mean delivered VMAF (higher better), y = TRUE ns-3 rebuffering (lower better). Best = lower-right."""
    apply_style()
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    # y = actual closed-loop ns-3 rebuffering (NOT the perceptual_qoe replay, which is a model-arm artifact).
    stalls = _true_stall_by_arm()
    vmafs = {s: mean([r["vmaf"] for r in rs]) for s, rs in data.items()}
    comp = [stalls[s] for s, _ in ALGOS if s in stalls and s != "Osmp"]
    ytop = max(comp) * 1.25 if comp else 1.0
    for stem, sty in ALGOS:
        if stem not in data or stem not in stalls:
            continue
        x, y = vmafs[stem], stalls[stem]
        ours = stem == "AiDash"
        ax.scatter([x], [y], c=sty["color"], marker=sty["marker"],
                   s=52 if ours else 34, edgecolors="black", linewidths=0.5 if ours else 0.3,
                   zorder=5 if ours else 3, label=sty["name"])
    ax.set_xlabel("Mean delivered VMAF")
    ax.set_ylabel("Rebuffering (s)")
    ax.set_ylim(-0.03, ytop * 1.12)
    ax.legend(loc="upper center", ncol=2, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    pdf, png = save_fig(fig, "fig1_pareto")
    print("wrote", pdf, png)


def fig2_curve(data):
    """ABSOLUTE per-chunk VMAF-QoE vs bottleneck link rate (one line per arm).

    Reads paper/absqoe_by_cond.csv (produced by absolute_qoe_table.py); no % of teacher anywhere.
    """
    import csv as _csv
    apply_style()
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    path = os.path.join(PAPER_DIR, "absqoe_by_cond.csv")
    absq = {(r["stem"], int(r["cond"])): float(r["absqoe"]) for r in _csv.DictReader(open(path))}
    # ASCENDING link-rate x-axis (4.5 -> 9 -> 18 -> 36 -> 100Gb/s) for readability.
    cond_axis = list(reversed(COND_ORDER))
    x = list(range(len(cond_axis)))
    for stem, sty in ALGOS:
        y = [absq.get((stem, c)) for c in cond_axis]
        if any(v is None for v in y):
            continue
        ours = stem == "AiDash"
        ax.plot(x, y, color=sty["color"], marker=sty["marker"], linewidth=1.6 if ours else 1.0,
                markersize=5 if ours else 4, label=sty["name"], zorder=5 if ours else 3)
    ax.set_xticks(x)
    ax.set_xticklabels([COND[c] for c in cond_axis], rotation=20)
    ax.set_xlabel("Bottleneck link rate")
    ax.set_ylabel("Per-segment QoE reward")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    pdf, png = save_fig(fig, "fig2_curve")
    print("wrote", pdf, png)


def main():
    ap = argparse.ArgumentParser()
    # Both paper figures are emitted by default (Fig 1 Pareto + Fig 2 %-oracle-vs-rate curve).
    ap.add_argument("--no-fig2", action="store_true", help="skip Figure 2 (the link-rate curve)")
    args = ap.parse_args()
    data = load_all()
    fig1_pareto(data)
    if not args.no_fig2:
        fig2_curve(data)


if __name__ == "__main__":
    main()
