#!/usr/bin/env python3
"""Fig-3 by-content cut: does the model's advantage over the best heuristic depend on video headroom?

Reads results/{AiDash,Bola,Raahs,Aaash}.txt (per-(video,run) %oracle), computes per-video mean %oracle
for each arm (mean over the 5 conditions), then per video:
  best_heuristic = max(Bola, Raahs, Aaash) mean %oracle
  advantage      = AiDash - best_heuristic
Joins per-video headroom from catalog_selection/stage_b_subset/manifest.csv and reports
Pearson corr(headroom, advantage) + a low/high-headroom split. Emits a scatter to paper/fig3_headroom.{png,pdf}.

The OLD (symmetric-teacher) verdict was NO-GO: the model LOST on low-headroom (corr(headroom, advantage) large POS
= advantage grows with headroom). This re-runs it under the DOWN-ONLY QoE to check whether the collapse is cured.
"""
import csv
import glob
import os
import re
import sys
from statistics import mean

_HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(_HERE, "results")
MANIFEST = os.path.join(_HERE, "catalog_selection", "stage_b_subset", "manifest.csv")
PAPER = os.path.join(_HERE, "paper")
ROW_RE = re.compile(r"^(\S+)/run_(\d+)\s+\d+\s+([\d.]+)%")
HEURS = ["Bola", "Raahs", "Aaash"]


def load_arm(stem):
    """results/<stem>.txt -> {video: [pct per run]}."""
    path = os.path.join(RESDIR, "%s.txt" % stem)
    by_video = {}
    for line in open(path):
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        video, _run, pct = m.group(1), int(m.group(2)), float(m.group(3))
        by_video.setdefault(video, []).append(pct)
    return {v: mean(p) for v, p in by_video.items()}


def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def main():
    model = load_arm("AiDash")
    heur = {h: load_arm(h) for h in HEURS}
    headroom = {}
    for r in csv.DictReader(open(MANIFEST)):
        headroom[r["video_id"]] = float(r["headroom"])

    videos = sorted(v for v in model if v in headroom)
    rows = []
    for v in videos:
        best_h = max(heur[h].get(v, float("nan")) for h in HEURS)
        best_arm = max(HEURS, key=lambda h: heur[h].get(v, -1))
        adv = model[v] - best_h
        rows.append((headroom[v], v, model[v], best_h, best_arm, adv))
    rows.sort()

    print("\n=== Fig-3 by-content cut (DOWN-ONLY QoE) ===")
    print("%-22s %7s %8s %10s %8s %8s" % ("video", "headrm", "model%", "bestHeur%", "(arm)", "adv"))
    for hr, v, m, bh, ba, adv in rows:
        print("%-22s %7.1f %8.1f %10.1f %8s %+8.1f" % (v, hr, m, bh, ba, adv))

    hrs = [r[0] for r in rows]
    advs = [r[5] for r in rows]
    corr = pearson(hrs, advs)
    med = sorted(hrs)[len(hrs) // 2]
    lo = [a for h, a in zip(hrs, advs) if h < med]
    hi = [a for h, a in zip(hrs, advs) if h >= med]
    print("\ncorr(headroom, model advantage) = %+.3f  (n=%d)" % (corr, len(rows)))
    print("mean advantage  LOW-headroom (<%.1f):  %+.1f pts  (n=%d)" % (med, mean(lo), len(lo)))
    print("mean advantage HIGH-headroom (>=%.1f): %+.1f pts  (n=%d)" % (med, mean(hi), len(hi)))
    n_win = sum(1 for a in advs if a > 0)
    print("model beats best heuristic on %d/%d videos" % (n_win, len(rows)))
    lo_win = sum(1 for h, a in zip(hrs, advs) if h < med and a > 0)
    print("  ... of which LOW-headroom wins: %d/%d" % (lo_win, len(lo)))

    # scatter (shared IEEE-minimalist style via paper_style)
    try:
        sys.path.insert(0, _HERE)
        from paper_style import apply_style, save_fig, COL_SINGLE  # noqa: E402
        import matplotlib.pyplot as plt  # noqa: E402
        apply_style()
        fig, ax = plt.subplots(figsize=COL_SINGLE)
        ax.axhline(0, color="0.55", lw=0.6, zorder=1)
        colors = ["#1b7837" if a > 0 else "#b2182b" for a in advs]
        ax.scatter(hrs, advs, c=colors, s=30, zorder=3, edgecolor="k", linewidth=0.4)
        ax.set_xlabel("Video VMAF headroom")
        ax.set_ylabel("Advantage (pts)")
        fig.tight_layout(pad=0.2)
        save_fig(fig, "fig3_headroom")
        print("\nwrote %s/fig3_headroom.{png,pdf}  (corr=%+.2f - report in caption)" % (PAPER, corr))
    except Exception as e:  # pylint: disable=broad-except
        print("scatter skipped: %s" % e, file=sys.stderr)


if __name__ == "__main__":
    main()
