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
# Fixed-binary benchmark (2026-07-05): "AiDash" IS our content-aware SLM policy (greedy-trained,
# checkpoints_bola) - the highlighted headline (bold red diamond); the 3 classics BOLA/RAAHS/AAASH are
# the baselines. (Older arms Greedy_sem/Svaa/Sftm/Fdash/Osmp are not in this benchmark.)
# Paper label is "Ours" - no coined method name; the teacher is the training target, not an oracle
# (see docs/paper-outline.md).
# 26-video full-length benchmark (2026-10-27): four arms - Ours, the two RobustMPC controllers and BOLA
# (the only classic heuristic retained; RAAHS/AAASH/FDASH/Pensieve dropped from the paper tables).
# "BolaFull" = BOLA re-run natively on whole videos (Quic_artifacts/BolaFull), same session rule as the
# bridge arms; the old truncated 14-segment collection runs stay under Quic_artifacts/Bola.
ALGOS = [
    ("AiDash",   {"name": "Ours", "color": "#d62728", "marker": "D"}),
    ("Qmpc",     {"name": "Quality-MPC", "color": "#ff7f0e", "marker": "v"}),   # RobustMPC on the scoring QoE (content-aware control)
    ("Mpc",      {"name": "RobustMPC", "color": "#9467bd", "marker": "^"}),     # RobustMPC on QoE_lin (Yin'15; Pensieve's objective)
    ("BolaFull", {"name": "BOLA",  "color": "#1f77b4", "marker": "o"}),
]
# Bridge arms (decisions.jsonl under Quic_artifacts/<stem>/<video>/run_<cond>/ + <stem>_shards teacher
# shards, produced by benchmark/score_bridge_arm.sh). AiDash keeps its own legacy layout for now.
BRIDGE_ARMS = ("Pensieve", "Mpc", "Qmpc", "Greedy")
ALGO_STYLE = dict(ALGOS)


def apply_style():
    """Set global rcParams so every figure shares font/size/look (IEEE minimalist). Call once before plotting.

    IEEE conventions: Times serif ~8pt, single-column ~3.4in, no chartjunk - top/right spines off, thin
    axes, outward ticks, and a faint dashed HORIZONTAL-only reference grid (no vertical rules)."""
    plt.rcParams.update({
        "font.family": "serif",
        # Prefer a Times-metric serif (IEEE uses Times); fall through metric-compatible substitutes;
        # DejaVu Serif is the always-available last resort.
        # Liberation Serif first: a real TrueType, metrically identical to Times New Roman, so it
        # embeds cleanly under pdf.fonttype=42 (Nimbus Roman is OTF/CFF and triggers a type mismatch).
        "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes",
                       "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-compatible math (matches the serif better than dejavuserif)
        # IEEE min figure text ~8pt: keep NOTHING below 8pt (ticks/legend were 7pt).
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        # IEEE PDF eXpress REJECTS Type 3 fonts (matplotlib's default). fonttype 42 embeds TrueType
        # (Type 42), which is accepted; set it for PS too so any EPS export is compliant.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # minimalist reference grid: faint dashed, horizontal only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": (0, (4, 4)),
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
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
