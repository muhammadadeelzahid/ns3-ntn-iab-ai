#!/usr/bin/env python3
"""Behavioral figures from the SCORED benchmark decisions (down-only re-benchmark). No new sims.

Fig 4 (smoothness)  : per-arm bitrate switches per segment, split into DOWN vs UP switches (stacked).
                      Down-switches are what the down-only QoE penalizes and what viewers notice most.
Fig 5 (rung mix)    : per-arm distribution over the 6 ladder rungs (stacked % of all decisions).

Rung sources (the SAME odd-seed->condition sets that produced Table 1, all 3 UEs, 5 conditions):
  MODEL (AiDash) : Quic_artifacts/AiDash/bench_down_scored/<v>/run_<c>/decisions.jsonl  -> chosen_rung
  HEURISTICS     : Quic_artifacts/<ARM>_dscore/<v>/run_<c>/trace_decisions.jsonl        -> teacher.chosen_bitrate_class
Per (arm,video,cond,node) the rungs are ordered by segment_id; a switch = a change between consecutive segments.
"""
import glob
import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
from paper_style import ALGOS, apply_style, save_fig, COL_SINGLE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
Q = os.path.join(_REPO, "Quic_artifacts")
NRUNG = 6
LADDER_MBPS = ["0.25", "0.75", "1.5", "3", "6", "12"]   # rung 0..5 nominal bitrate (Mbps) for the legend


def _sessions_decisions(dec_glob):
    """Sessions from bridge decisions.jsonl (chosen_rung) - used by the AiDash + Pensieve arms."""
    for dec in glob.glob(dec_glob):
        by_node = defaultdict(list)
        for l in open(dec):
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            by_node[d["node_id"]].append((int(d.get("segment_id", 0)), int(d["chosen_rung"])))
        for seq in by_node.values():
            yield [r for _, r in sorted(seq)]


def _sessions_model():
    return _sessions_decisions(os.path.join(Q, "AiDash", "bench_down_scored", "*", "run_*", "decisions.jsonl"))


def _sessions_heuristic(arm):
    for dec in glob.glob(os.path.join(Q, "%s_dscore" % arm, "*", "run_*", "trace_decisions.jsonl")):
        by_node = defaultdict(list)
        for l in open(dec):
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            rung = (d.get("teacher") or {}).get("chosen_bitrate_class")
            if rung is None:
                continue
            by_node[d["node_id"]].append((int(d.get("segment_id", 0)), int(rung)))
        for seq in by_node.values():
            yield [r for _, r in sorted(seq)]


def sessions(stem):
    if stem == "AiDash":
        return list(_sessions_model())
    if stem in ("Pensieve", "Mpc", "Qmpc", "Greedy"):   # standard-layout bridge runs with chosen_rung
        return list(_sessions_decisions(os.path.join(Q, stem, "*", "run_*", "decisions.jsonl")))
    return list(_sessions_heuristic(stem))


def main():
    data = {stem: sessions(stem) for stem, _ in ALGOS}
    stems = [s for s, _ in ALGOS]
    names = [dict(ALGOS)[s]["name"] for s in stems]

    # ---- rung histogram (fraction of decisions at each rung, per arm) ----
    hist = {}
    xpos = range(len(stems))
    for s in stems:
        h = [0] * NRUNG
        for seq in data[s]:
            for r in seq:
                if 0 <= r < NRUNG:
                    h[r] += 1
        tot = sum(h) or 1
        hist[s] = [x / tot for x in h]

    print("arm        rung-mix(%)")
    for s in stems:
        print("%-9s  %s" % (dict(ALGOS)[s]["name"], " ".join("%2.0f" % (100 * x) for x in hist[s])))

    # ---- Fig 5: rung-selection distribution (stacked % per arm) ----
    apply_style()
    fig, ax = plt.subplots(figsize=COL_SINGLE)
    cmap = plt.cm.viridis
    bottoms = [0.0] * len(stems)
    for r in range(NRUNG):
        vals = [100 * hist[s][r] for s in stems]
        ax.bar(xpos, vals, bottom=bottoms, color=cmap(r / (NRUNG - 1)),
               edgecolor="white", linewidth=0.3, label="%s" % LADDER_MBPS[r])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Decisions at rung (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Rung (Mbps)", loc="center left", bbox_to_anchor=(1.0, 0.5),
              ncol=1, fontsize=6, title_fontsize=6)
    fig.tight_layout(pad=0.2)
    save_fig(fig, "fig5_rung_dist")
    print("wrote fig5_rung_dist")


if __name__ == "__main__":
    main()
