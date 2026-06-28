#!/usr/bin/env python3
"""Paper figures (uniform via paper_style). Default: Figure 1 only.

  Figure 1 (default)         -- quality-rebuffering Pareto: mean delivered VMAF vs rebuffering, 7 arms.
  --fig2 curve               -- % of DP-oracle vs link rate (one line per algorithm).

Reads the per-run results/*.txt (all 7 arms, same format). Run make_paper_table.py first is not required,
but results/AiDash.txt must exist (from score_v2_perceptual.py).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paper_style import (ALGOS, COND, COND_ORDER, COL_SINGLE, apply_style, save_fig)  # noqa: E402
from make_paper_table import parse_results, mean, RESULTS_DIR  # reuse the parser  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def load_all():
    return {stem: parse_results(os.path.join(RESULTS_DIR, "%s.txt" % stem))
            for stem, _ in ALGOS if os.path.exists(os.path.join(RESULTS_DIR, "%s.txt" % stem))}


def fig1_pareto(data):
    """x = mean delivered VMAF (higher better), y = rebuffering time (lower better). Best = lower-right."""
    apply_style()
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    # competitive y-range from non-degenerate arms; clamp outliers (e.g. OSMP ~761s) to top + annotate.
    stalls = {s: mean([r["stall"] for r in rs]) for s, rs in data.items()}
    vmafs = {s: mean([r["vmaf"] for r in rs]) for s, rs in data.items()}
    comp = [stalls[s] for s, _ in ALGOS if s in data and s != "Osmp"]
    ytop = max(comp) * 1.25 if comp else 20.0
    for stem, sty in ALGOS:
        if stem not in data:
            continue
        x, y = vmafs[stem], stalls[stem]
        big = stem == "Greedy_sem"
        clamped = y > ytop
        yp = ytop if clamped else y
        ax.scatter([x], [yp], c=sty["color"], marker=sty["marker"],
                   s=70 if big else 40, edgecolors="black", linewidths=0.6 if big else 0.3,
                   zorder=5 if big else 3, label=sty["name"])
        if clamped:
            ax.annotate("%s (%.0fs)" % (sty["name"], y), (x, yp), textcoords="offset points",
                        xytext=(0, -8), ha="center", fontsize=6, color=sty["color"])
    ax.set_xlabel("Mean delivered VMAF (higher better)")
    ax.set_ylabel("Rebuffering time, s (lower better)")
    ax.set_ylim(-0.5, ytop * 1.18)   # headroom so the clamped-outlier annotation clears the top
    ax.annotate("better", xy=(0.985, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, style="italic", color="0.3")
    ax.legend(loc="center left", ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.3,
              fontsize=6)
    fig.tight_layout(pad=0.3)
    pdf, png = save_fig(fig, "fig1_pareto")
    print("wrote", pdf, png)


def fig2_curve(data):
    apply_style()
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    x = list(range(len(COND_ORDER)))
    for stem, sty in ALGOS:
        if stem not in data:
            continue
        y = [mean([r["pct"] for r in data[stem] if r["cond"] == c]) for c in COND_ORDER]
        lw = 1.8 if stem == "Greedy_sem" else 1.0
        ax.plot(x, y, color=sty["color"], marker=sty["marker"], linewidth=lw,
                markersize=5 if stem == "Greedy_sem" else 4, label=sty["name"], zorder=5 if stem == "Greedy_sem" else 3)
    ax.set_xticks(x)
    ax.set_xticklabels([COND[c] for c in COND_ORDER], rotation=20)
    ax.set_xlabel("S1-U feeder rate (more congested $\\rightarrow$)")
    ax.set_ylabel("% of DP-oracle (VMAF-QoE)")
    ax.legend(loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
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
