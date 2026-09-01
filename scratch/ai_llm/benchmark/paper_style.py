#!/usr/bin/env python3
"""Single source of visual uniformity for all paper tables/figures.

Every figure imports apply_style() + save_fig() and uses the ALGOS registry, so an algorithm has the SAME
display name / color / marker in every figure and all figures share font, size, and column width. Tables use
the same ALGOS order + display names. Output goes to scratch/ai_llm/benchmark/paper/.
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless cluster
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(_HERE, "paper")

# IEEE/ACM 2-column layout widths (inches) - keep ALL figures one of these two sizes for uniformity.
COL_SINGLE = (3.4, 2.3)   # one column
COL_DOUBLE = (7.0, 2.6)   # full width

# run-index -> S1-U feeder rate (matches build_job_matrix / aggregate_table.py)
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
COND_ORDER = [1, 2, 3, 4, 5]

# Algorithm registry: results-file stem -> (display name, color, marker). Order = legend/table order.
# Fixed-binary benchmark (2026-07-05): "AiDash" IS our content+network LLM-ABR model (greedy-trained,
# checkpoints_bola) - the highlighted headline (bold red diamond); the 3 classics BOLA/RAAHS/AAASH are
# the baselines. (Older arms Greedy_sem/Svaa/Sftm/Fdash/Osmp are not in this benchmark.)
ALGOS = [
    ("AiDash", {"name": "LLM-ABR", "color": "#d62728", "marker": "D"}),
    ("Raahs",  {"name": "RAAHS", "color": "#ff7f0e", "marker": "^"}),
    ("Aaash",  {"name": "AAASH", "color": "#8c564b", "marker": "P"}),
    ("Bola",   {"name": "BOLA",  "color": "#1f77b4", "marker": "o"}),
]
ALGO_STYLE = dict(ALGOS)
ORACLE_STYLE = {"name": "DP-oracle", "color": "#000000", "marker": "*"}


def apply_style():
    """Set global rcParams so every figure shares font/size/look (IEEE minimalist). Call once before plotting.

    IEEE conventions: Times serif ~8pt, single-column ~3.4in, no chartjunk - top/right spines off, thin
    axes, outward ticks, and a faint dashed HORIZONTAL-only reference grid (no vertical rules)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],   # DejaVu always available fallback
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # minimalist reference grid: faint dashed, horizontal only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": (0, (4, 4)),
        "grid.alpha": 0.35,
        "grid.linewidth": 0.4,
        # thin frame, only left+bottom, outward short ticks
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
    })


def save_fig(fig, name):
    """Write paper/<name>.pdf (vector, for the paper) + paper/<name>.png (preview), same dimensions."""
    os.makedirs(PAPER_DIR, exist_ok=True)
    pdf = os.path.join(PAPER_DIR, name + ".pdf")
    png = os.path.join(PAPER_DIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png
