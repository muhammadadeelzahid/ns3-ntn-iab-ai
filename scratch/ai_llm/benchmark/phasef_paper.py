#!/usr/bin/env python3
"""Paper tables + figures for the Phase F benchmark (26 test videos x 5 rates x 3 seed sets per arm).

Read-only over scored results. Per cell (video, rate) each arm's seed sets are averaged first; CIs are a
cluster bootstrap over the 26 test videos (2000 draws), paired tests are Wilcoxon signed-rank on per-video
means (same convention as multiseed_stats.py). Absolute QoE per segment uses the TRUE ns-3 stall:
    QoE/seg = 0.65 * meanVMAF - 65 * stall_sum_s / segs        (switch term omitted)
Val-only design table (model selection) uses the 26 val videos at 9 and 4.5 Mb/s.

Outputs (scratch/ai_llm/benchmark/paper/):
  tab_v3_main.tex, tab_v3_rate.tex, tab_v3_design.tex, tab_v3_fairness.tex
  fig_v3_qoe_rate, fig_v3_pareto, fig_v3_paired, fig_v3_rung, fig_v3_content, fig_v3_trace (.pdf + .png)
Usage: ~/abr-llm-venv/bin/python phasef_paper.py
"""
import csv
import glob
import json
import os
import random
import re
from collections import defaultdict
from statistics import mean

import numpy as np
from scipy.stats import wilcoxon

from paper_style import COL_DOUBLE, COL_SINGLE, PAPER_DIR, apply_style, save_fig
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
Q = os.path.join(REPO, "Quic_artifacts")
RES = os.path.join(HERE, "results")
META = "/project/6030214/adeel/abr/output/metadata"
TEST = sorted(open(os.path.join(HERE, "test_videos_26.txt")).read().split())
VAL = sorted(open(os.path.join(HERE, "val_videos_26.txt")).read().split())
COND = {1: "100 Gb/s", 2: "36 Mb/s", 3: "18 Mb/s", 4: "9 Mb/s", 5: "4.5 Mb/s"}
TEXCOND = {1: "100\\,Gb/s", 2: "36\\,Mb/s", 3: "18\\,Mb/s", 4: "9\\,Mb/s", 5: "4.5\\,Mb/s"}
LADDER = ["250k", "750k", "1500k", "3000k", "6000k", "12000k"]
RUNG_MBPS = [0.25, 0.75, 1.5, 3, 6, 12]
W_Q, W_S = 0.65, 65.0

# label -> (style, [(results stem, true-stall key, decisions stem, kind)])
ARMS = [
    ("Ours", {"color": "#d62728", "marker": "D"},
     [("OursV3_s%d" % n, "OursV3_s%d" % n, "OursV3_s%d" % n, "bridge") for n in (1, 2, 3)]),
    ("Quality-MPC", {"color": "#ff7f0e", "marker": "v"},
     [(s, s, s, "bridge") for s in ("Qmpc", "Qmpc_s2", "Qmpc_s3")]),
    ("RobustMPC", {"color": "#9467bd", "marker": "^"},
     [(s, s, s, "bridge") for s in ("Mpc", "Mpc_s2", "Mpc_s3")]),
    ("BOLA", {"color": "#1f77b4", "marker": "o"},
     [(s + "_dscore", s, s + "_dscore", "bola") for s in ("BolaFull", "BolaFull_s2", "BolaFull_s3")]),
]
PREV = ("Ours, previous model", [("AiDashClean", "AiDash", None, "bridge")])
STYLE = {a[0]: a[1] for a in ARMS}
RNG = random.Random(0)


# ---------------------------------------------------------------- loading
def load_results(stem):
    out = {}
    for line in open(os.path.join(RES, stem + ".txt")):
        m = re.match(r"^(\S+)/run_(\d)\s+(\d+)\s+([0-9.]+)%\s+\S+\s+([0-9.]+)", line)
        if m:
            out[(m.group(1), int(m.group(2)))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)))
    return out


TS = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(RES, "true_stall.csv"))):
    TS[r["arm"]][(r["video"], int(r["cond"]))] = (float(r["stall_s"]), float(r["stall_sum_s"]))


def cell_metrics(seedsets, videos):
    """{(video,cond): dict(qoe, vmaf, stall, pct)} averaged over seed sets."""
    acc = defaultdict(lambda: defaultdict(list))
    for stem, key, _, _ in seedsets:
        for k, (segs, pct, vmaf) in load_results(stem).items():
            if k[0] not in videos or k not in TS[key]:
                continue
            st, ss = TS[key][k]
            a = acc[k]
            a["qoe"].append(W_Q * vmaf - W_S * ss / segs)
            a["vmaf"].append(vmaf)
            a["stall"].append(st)
            a["pct"].append(pct)
    return {k: {m: mean(v) for m, v in d.items()} for k, d in acc.items()}


def per_video(cells, metric, conds):
    return np.array([mean(cells[(v, c)][metric] for c in conds) for v in TEST])


def boot_ci(x, draws=2000):
    n = len(x)
    bs = sorted(float(np.mean([x[RNG.randrange(n)] for _ in range(n)])) for _ in range(draws))
    return bs[int(0.025 * draws)], bs[int(0.975 * draws) - 1]


def paired(a, b):
    d = a - b
    lo, hi = boot_ci(d)
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return float(d.mean()), lo, hi, p


def decisions(stem, kind, video, cond):
    """[(node, t, rung, buffer_s)] for one cell."""
    out = []
    if kind == "bridge":
        f = os.path.join(Q, stem, video, "run_%d" % cond, "decisions.jsonl")
        if not os.path.exists(f):
            return out
        for line in open(f):
            r = json.loads(line)
            if r.get("chosen_rung") is not None:
                out.append((int(r["node_id"]), float(r["sim_time_s"]), int(r["chosen_rung"]),
                            float(r.get("buffer_level_s", 0.0))))
    else:
        f = os.path.join(Q, stem, video, "run_%d" % cond, "trace_decisions.jsonl")
        if not os.path.exists(f):
            return out
        for line in open(f):
            r = json.loads(line)
            out.append((int(r["node_id"]), float(r["sim_time_s"]), int(r["teacher"]["chosen_bitrate_class"]),
                        float(r["observation"]["buffer_level_s"])))
    return out


_META = {}


def video_meta(video):
    """(mean VMAF per rung, VMAF headroom = mean(top) - mean(bottom))."""
    if video not in _META:
        rows = list(csv.DictReader(open(os.path.join(META, video + "_segments.csv"))))
        vm = [mean(float(r["vmaf_" + lv]) for r in rows) for lv in LADDER]
        _META[video] = (vm, vm[-1] - vm[0])
    return _META[video]


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


# ---------------------------------------------------------------- tables
def tab_main(cells):
    """Table II: raw mean per arm (QoE, VMAF, stall) + Ours' relative difference vs each baseline, Wilcoxon p."""
    mets = (("qoe", "QoE/seg"), ("vmaf", "VMAF"), ("stall", "Stall/UE (s)"))
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "Policy & " + " & ".join(h for _, h in mets) + r" \\", r"\midrule"]
    for lab, _, _ in ARMS:
        vals = ["%.2f" % per_video(cells[lab], m, COND).mean() for m, _ in mets]
        if lab == "Ours":
            vals = [r"\textbf{%s}" % v for v in vals]
        lines.append(("\\textbf{Ours}" if lab == "Ours" else lab) + " & " + " & ".join(vals) + r" \\")
    lines += [r"\midrule", r"\multicolumn{4}{l}{\textit{Ours relative to baseline (Wilcoxon $p$)}} \\"]
    for lab in ("Quality-MPC", "RobustMPC", "BOLA"):
        row = ["vs " + lab]
        for m, _ in mets:
            o, b = per_video(cells["Ours"], m, COND), per_video(cells[lab], m, COND)
            d, lo, hi, p = paired(o, b)
            row.append(r"%+.1f\%% {\scriptsize(%s)}" % (100 * d / b.mean(), fmt_p(p)))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_DIR, "tab_v3_main.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

def tab_rate(cells):
    lines = [r"\begin{tabular}{l" + "c" * 5 + "}", r"\toprule",
             "Policy & " + " & ".join(TEXCOND[c] for c in COND) + r" \\", r"\midrule"]
    for lab, _, _ in ARMS:
        vals = [per_video(cells[lab], "qoe", [c]).mean() for c in COND]
        best = [max(per_video(cells[a[0]], "qoe", [c]).mean() for a in ARMS) for c in COND]
        s = [(r"\textbf{%.2f}" if abs(v - b) < 1e-9 else "%.2f") % v for v, b in zip(vals, best)]
        lines.append(lab + " & " + " & ".join(s) + r" \\")
    lines.append(r"\midrule")
    row = []
    for c in COND:
        d, lo, hi, p = paired(per_video(cells["Ours"], "qoe", [c]), per_video(cells["Quality-MPC"], "qoe", [c]))
        rel = 100 * d / per_video(cells["Quality-MPC"], "qoe", [c]).mean()
        row.append(r"%+.1f\%% {\scriptsize($p$=%s)}" % (rel, fmt_p(p)))
    lines.append(r"Ours vs Q-MPC & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_DIR, "tab_v3_rate.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def val_cell_stats(stem):
    """val, rates 9M(4)/4.5M(5): {cond: (vmaf, stall/UE, qoe/seg)} from results + decisions true stall."""
    res = {k: v for k, v in load_results(stem).items() if k[0] in VAL and k[1] in (4, 5)}
    stall = {}
    for dec in glob.glob(os.path.join(Q, stem, "*", "run_*", "decisions.jsonl")):
        p = dec.split(os.sep)
        m = re.match(r"run_(\d+)_(.+)$", p[-2])
        c = {"9Mbps": 4, "4.5Mbps": 5}.get(m.group(2)) if m else int(p[-2].split("_")[1])
        per = defaultdict(float)
        for line in open(dec):
            r = json.loads(line)
            per[r["node_id"]] = max(per[r["node_id"]], float(r.get("interruption_time_s", 0.0)))
        if per:
            stall[(p[-3], c)] = (mean(per.values()), sum(per.values()))
    out = {}
    for c in (4, 5):
        ks = [k for k in res if k[1] == c and k in stall]
        out[c] = (mean(res[k][2] for k in ks), mean(stall[k][0] for k in ks),
                  mean(W_Q * res[k][2] - W_S * stall[k][1] / res[k][0] for k in ks), len(ks))
    return out


def tab_design():
    rows = [
        ("Quality-MPC", ["V3Qmpc"]),
        ("Previous model", ["ValGuard_avg"]),
        ("New data, 1-step greedy labels", ["ValV3_greedy"]),
        ("\\quad + DAgger round", ["ValV3_greedy_dag"]),
        ("New data, 5-step lookahead labels", ["ValV3_k5estimate", "ValV3_k5estimate_s43"]),
        ("\\quad + DAgger round", ["ValV3_k5estimate_dag"]),
        ("\\quad + DAgger, throughput factor 0.85", ["ValV3_k5est_pf085_dag", "ValV3_k5est_pf085_dag_s43"]),
        ("\\quad + DAgger, throughput factor 0.7 (\\textbf{selected})", ["ValV3_k5est_pf07_dag", "ValV3_k5est_pf07_dag_s43"]),
    ]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{9\,Mb/s} & \multicolumn{3}{c}{4.5\,Mb/s} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Variant (seeds) & VMAF & Stall & QoE & VMAF & Stall & QoE \\", r"\midrule"]
    out = []
    for lab, stems in rows:
        st = [val_cell_stats(s) for s in stems]
        vals = []
        for c in (4, 5):
            vals += [mean(x[c][i] for x in st) for i in range(3)]
        lines.append("%s (%d) & " % (lab, len(stems)) + " & ".join("%.2f" % v for v in vals) + r" \\")
        out.append((lab, len(stems), vals))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_DIR, "tab_v3_design.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def jain(x):
    x = np.asarray(x, float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum())) if (x ** 2).sum() > 0 else 1.0


def tab_fairness():
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Policy & $J$ bitrate & $J$ VMAF \\", r"\midrule"]
    res = {}
    for lab, _, seedsets in ARMS:
        jb, jv = [], []
        for _, _, dstem, kind in seedsets:
            for v in TEST:
                vm, _ = video_meta(v)
                for c in COND:
                    per = defaultdict(list)
                    for node, _, rung, _ in decisions(dstem, kind, v, c):
                        per[node].append(rung)
                    if len(per) < 2:
                        continue
                    jb.append(jain([mean(RUNG_MBPS[r] for r in rs) for rs in per.values()]))
                    jv.append(jain([mean(vm[r] for r in rs) for rs in per.values()]))
        res[lab] = (mean(jb), mean(jv), len(jb))
        lines.append("%s & %.3f & %.3f \\\\" % (lab, res[lab][0], res[lab][1]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER_DIR, "tab_v3_fairness.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


# ---------------------------------------------------------------- figures
def fig_qoe_rate(cells):
    """Absolute QoE per segment vs feeder-link rate (mean over 26 test videos x 3 seed sets), no bands; the
    y-axis is bounded to the data range so the gaps between arms stay visible."""
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    order = [5, 4, 3, 2, 1]
    xs = np.arange(5)
    allv = []
    for lab, st, _ in ARMS:
        m = [per_video(cells[lab], "qoe", [c]).mean() for c in order]
        allv += m
        ax.plot(xs, m, color=st["color"], marker=st["marker"], lw=1.8 if lab == "Ours" else 1.0,
                ms=5 if lab == "Ours" else 4, label=lab, zorder=3 if lab == "Ours" else 2)
    ax.set_ylim(np.floor(min(allv) / 5) * 5, np.ceil(max(allv) / 5) * 5)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND[c] for c in order])
    ax.set_xlabel("Feeder-link rate")
    ax.set_ylabel("QoE per segment")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, "fig_v3_qoe_rate")

def fig_vmaf_rate(cells):
    """Mean delivered VMAF vs feeder-link rate (mean over 26 test videos x 3 seed sets); same style as
    fig_qoe_rate, y-axis bounded to the data range."""
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    order = [5, 4, 3, 2, 1]
    xs = np.arange(5)
    allv = []
    for lab, st, _ in ARMS:
        m = [per_video(cells[lab], "vmaf", [c]).mean() for c in order]
        allv += m
        ax.plot(xs, m, color=st["color"], marker=st["marker"], lw=1.8 if lab == "Ours" else 1.0,
                ms=5 if lab == "Ours" else 4, label=lab, zorder=3 if lab == "Ours" else 2)
    ax.set_ylim(np.floor(min(allv) / 5) * 5, np.ceil(max(allv) / 5) * 5)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND[c] for c in order])
    ax.set_xlabel("Feeder-link rate")
    ax.set_ylabel("Mean delivered VMAF")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, "fig_v3_vmaf_rate")


def fig_rung_rate():
    """Rung usage per feeder-link rate: one stacked bar per policy within each rate group (full width)."""
    order = [5, 4, 3, 2, 1]
    short = {"Ours": "Ours", "Quality-MPC": "Quality-MPC", "RobustMPC": "RobustMPC", "BOLA": "BOLA"}
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(COL_DOUBLE[0], 1.6))
    w, gap = 0.8, 1.2
    xt, xl, gx = [], [], []
    for g, c in enumerate(order):
        for i, (lab, st, ss) in enumerate(ARMS):
            cnt = np.zeros(6)
            for _, _, d, k in ss:
                for v in TEST:
                    for _, _, r, _ in decisions(d, k, v, c):
                        cnt[r] += 1
            sh = 100 * cnt / cnt.sum()
            x = g * (len(ARMS) + gap) + i
            bottom = 0.0
            for r in range(6):
                ax.bar(x, sh[r], w, bottom=bottom, color=cmap(r / 5), edgecolor="white", lw=0.3,
                       label="%g" % RUNG_MBPS[r] if (g == 0 and i == 0) else None)
                bottom += sh[r]
            if lab == "Ours":
                ax.bar(x, 100, w, fill=False, edgecolor=st["color"], lw=1.2)
            xt.append(x); xl.append(short[lab])
        gx.append(g * (len(ARMS) + gap) + (len(ARMS) - 1) / 2)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=90, fontsize=6)
    for x, c in zip(gx, order):
        ax.annotate(COND[c], xy=(x, 1.02), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Decisions at rung (%)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], title="Rung (Mb/s)", bbox_to_anchor=(1.005, 1), loc="upper left", fontsize=7,
              title_fontsize=7)
    fig.subplots_adjust(bottom=0.40, top=0.87)
    save_fig(fig, "fig_v3_rung_rate")


def fig_results_combined(cells):
    """Two panels, full width: (a) QoE per segment vs feeder-link rate, (b) mean VMAF vs stall per UE over all
    rates; one shared legend on top. Same data and styling as fig_qoe_rate / fig_pareto."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_DOUBLE[0], 2.1))
    order = [5, 4, 3, 2, 1]
    xs = np.arange(5)
    allv = []
    for lab, st, _ in ARMS:
        m = [per_video(cells[lab], "qoe", [c]).mean() for c in order]
        allv += m
        a.plot(xs, m, color=st["color"], marker=st["marker"], lw=1.8 if lab == "Ours" else 1.0,
               ms=5 if lab == "Ours" else 4, label=lab, zorder=3 if lab == "Ours" else 2)
        b.scatter(per_video(cells[lab], "vmaf", COND).mean(), per_video(cells[lab], "stall", COND).mean(),
                  marker=st["marker"], color=st["color"], s=60 if lab == "Ours" else 42, edgecolor="k", lw=0.5,
                  zorder=3)
    a.set_ylim(np.floor(min(allv) / 5) * 5, np.ceil(max(allv) / 5) * 5)
    a.set_xticks(xs)
    a.set_xticklabels([COND[c] for c in order])
    a.set_xlabel("Feeder-link rate")
    a.set_ylabel("QoE per segment")
    a.set_title("(a) QoE vs. feeder-link rate")
    b.set_xlabel("Mean delivered VMAF")
    b.set_ylabel("Rebuffering per user (s)")
    b.set_ylim(0, 1.1)
    b.set_title("(b) Quality vs. rebuffering, all rates")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.subplots_adjust(wspace=0.28)
    save_fig(fig, "fig_v3_results")


def fig_pareto(cells):
    """Mean delivered VMAF vs stall per UE over all rates, one marker per policy (no error bars)."""
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    for lab, st, _ in ARMS:
        xv = per_video(cells[lab], "vmaf", COND); yv = per_video(cells[lab], "stall", COND)
        ax.scatter(xv.mean(), yv.mean(), marker=st["marker"], color=st["color"], s=60 if lab == "Ours" else 42,
                   edgecolor="k", lw=0.5, label=lab, zorder=3)
    ax.set_xlabel("Mean delivered VMAF")
    ax.set_ylabel("Stall per UE (s)")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    save_fig(fig, "fig_v3_pareto")

def fig_paired(cells):
    fig, axs = plt.subplots(1, 2, figsize=COL_DOUBLE, sharey=True)
    for ax, lab in zip(axs, ("Quality-MPC", "RobustMPC")):
        d = per_video(cells["Ours"], "qoe", COND) - per_video(cells[lab], "qoe", COND)
        idx = np.argsort(d)
        cols = ["#d62728" if x > 0 else "#7f7f7f" for x in d[idx]]
        ax.bar(range(len(d)), d[idx], color=cols, width=0.8)
        ax.axhline(d.mean(), color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        lo, hi = boot_ci(d)
        ax.set_title("(%s) Ours $-$ %s" % ("a" if lab == "Quality-MPC" else "b", lab))
        ax.text(0.03, 0.95, "mean %+.2f [%+.2f, %+.2f]\n%d of 26 videos above zero"
                % (d.mean(), lo, hi, int((d > 0).sum())), transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel("Test videos (sorted)")
    axs[0].set_ylabel("$\\Delta$ QoE per segment")
    save_fig(fig, "fig_v3_paired")


def fig_rung():
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    cmap = plt.get_cmap("viridis")
    labs = [a[0] for a in ARMS]
    shares = {}
    for lab, _, seedsets in ARMS:
        cnt = np.zeros(6)
        for _, _, dstem, kind in seedsets:
            for v in TEST:
                for c in COND:
                    for _, _, r, _ in decisions(dstem, kind, v, c):
                        cnt[r] += 1
        shares[lab] = 100 * cnt / cnt.sum()
    bottom = np.zeros(len(labs))
    for r in range(6):
        vals = np.array([shares[l][r] for l in labs])
        ax.bar(labs, vals, bottom=bottom, color=cmap(r / 5), edgecolor="white", lw=0.4, label="%g" % RUNG_MBPS[r])
        bottom += vals
    ax.set_ylabel("Decisions at rung (%)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], title="Rung (Mb/s)", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    save_fig(fig, "fig_v3_rung")
    return shares


def fig_content():
    """Per video at 100 Gb/s (no network limit): delivered bitrate vs the video's VMAF headroom."""
    fig, axs = plt.subplots(1, 2, figsize=COL_DOUBLE)
    stats = {}
    for lab, st, seedsets in ARMS:
        hx, br, vq = [], [], []
        for v in TEST:
            vm, head = video_meta(v)
            rs = []
            for _, _, dstem, kind in seedsets:
                rs += [r for _, _, r, _ in decisions(dstem, kind, v, 1)]
            if not rs:
                continue
            hx.append(head); br.append(mean(RUNG_MBPS[r] for r in rs)); vq.append(mean(vm[r] for r in rs))
        hx, br, vq = map(np.array, (hx, br, vq))
        r_br = float(np.corrcoef(hx, br)[0, 1])
        eff = vq / br
        stats[lab] = (r_br, float(br.mean()), float(vq.mean()), float(eff.mean()))
        axs[0].scatter(hx, br, color=st["color"], marker=st["marker"], s=14 if lab == "Ours" else 10,
                       edgecolor="k", lw=0.3, label="%s (r=%+.2f)" % (lab, r_br), zorder=3 if lab == "Ours" else 2)
        z = np.polyfit(hx, br, 1)
        xx = np.linspace(hx.min(), hx.max(), 10)
        axs[0].plot(xx, np.polyval(z, xx), color=st["color"], lw=0.8)
        axs[1].bar(lab, eff.mean(), color=st["color"])
    axs[0].set_xlabel("Video VMAF headroom (top rung $-$ bottom rung)")
    axs[0].set_ylabel("Mean requested bitrate (Mb/s)")
    axs[0].set_title("(a) Bitrate vs content headroom, 100 Gb/s")
    axs[0].legend(loc="upper left", fontsize=7)
    axs[1].set_ylabel("VMAF per Mb/s requested")
    axs[1].set_title("(b) Quality per bit, 100 Gb/s")
    save_fig(fig, "fig_v3_content")
    return stats


def fig_trace(cells):
    """4.5 Mb/s: rung + buffer of one UE, Ours vs Quality-MPC, seed set 1; video = median paired dQoE."""
    d = {v: cells["Ours"][(v, 5)]["qoe"] - cells["Quality-MPC"][(v, 5)]["qoe"] for v in TEST}
    nseg = {v: len(list(csv.DictReader(open(os.path.join(META, v + "_segments.csv"))))) for v in TEST}
    srt = sorted((v for v in d if nseg[v] >= 40), key=d.get)   # full-length sessions only
    video = srt[len(srt) // 2]   # the median paired difference: a typical, not a cherry-picked, case
    fig, axs = plt.subplots(2, 1, figsize=(COL_DOUBLE[0], 3.2), sharex=True)
    for lab, stem, kind in (("Ours", "OursV3_s1", "bridge"), ("Quality-MPC", "Qmpc", "bridge")):
        dec = decisions(stem, kind, video, 5)
        node = min(n for n, _, _, _ in dec)
        pts = sorted((t, r, b) for n, t, r, b in dec if n == node)
        t = [p[0] for p in pts]
        st = STYLE[lab]
        axs[0].step(t, [RUNG_MBPS[p[1]] for p in pts], where="post", color=st["color"], label=lab,
                    lw=1.4 if lab == "Ours" else 1.0)
        axs[1].plot(t, [max(0.0, p[2]) for p in pts], color=st["color"], lw=1.4 if lab == "Ours" else 1.0)
    axs[0].set_ylabel("Rung (Mb/s)")
    axs[0].legend(loc="upper right", ncol=2)
    axs[1].set_ylabel("Buffer (s)")
    axs[1].set_xlabel("Time (s)")
    axs[0].set_title("%s, 4.5 Mb/s, first UE, seed set 1 (QoE: Ours %.1f, Quality-MPC %.1f)"
                     % (video, cells["Ours"][(video, 5)]["qoe"], cells["Quality-MPC"][(video, 5)]["qoe"]))
    save_fig(fig, "fig_v3_trace")
    return video


def main():
    apply_style()
    os.makedirs(PAPER_DIR, exist_ok=True)
    cells = {lab: cell_metrics(ss, set(TEST)) for lab, _, ss in ARMS}
    for lab in cells:
        assert len(cells[lab]) == 130, (lab, len(cells[lab]))
    print("== main"); tab_main(cells)
    print("== rate"); tab_rate(cells)
    print("== design (val)"); tab_design()
    print("== fairness"); print(tab_fairness())
    fig_qoe_rate(cells); fig_pareto(cells); fig_paired(cells)
    print("== rung shares", {k: [round(x, 1) for x in v] for k, v in fig_rung().items()})
    print("== content (r(headroom,bitrate), mean Mb/s, mean VMAF, VMAF per Mb/s) @100G", fig_content())
    print("== trace video", fig_trace(cells))
    prev = cell_metrics(PREV[1], set(TEST))
    print("== previous model (1 seed): QoE %.2f VMAF %.2f stall %.2f pct %.2f" % tuple(
        per_video(prev, m, COND).mean() for m in ("qoe", "vmaf", "stall", "pct")))


if __name__ == "__main__":
    main()
