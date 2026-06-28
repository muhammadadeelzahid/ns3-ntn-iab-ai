#!/usr/bin/env python3
"""Perceptual (VMAF) QoE benchmark metric for ANY ABR algorithm's ns-3 run.

The headline number is **% of the DP-oracle ceiling**, which is already perceptual: the oracle's
reward (config/oracle_reward.json) is VMAF-weighted (w_vmaf*VMAF - w_stall*stall - w_switch*|dVMAF|).
So "% of oracle = policy VMAF-QoE / oracle VMAF-QoE * 100" = what fraction of the achievable perceptual
quality the algorithm captured. We reuse `evaluate.py:replay_session` verbatim so the math is identical
to the existing pipeline.

Inputs per run:
  - sim.log               -> the ALGORITHM's actual chosen rung per (node, segment), via
                             decision_parser.parse_decisions (`new_bitrate_bps`).
  - <video>_run_N.jsonl   -> the full-teacher records (oracle label `chosen_bitrate_class`, realized
                             throughput `throughput_bps_replay`, per-rung VMAF ladder, semantic weight),
                             i.e. a train_jsonl-style shard built for that run.

Metrics (per the approved plan):
  A. % of DP-oracle (VMAF-QoE)   -- HEADLINE ranking (condition-fair, cross-video comparable)
  B. QoE pillars                 -- stall, switches, mean delivered bitrate
  C. regret (oracle - policy)/seg + confusion(policy vs oracle rung) + rung histogram  -- decision quality
     (F1/accuracy optional, caveated -- the oracle is acausal so F1 is NOT a headline; regret is.)
  descriptive: mean delivered VMAF (0-100), interpretability anchor only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from ai_llm.evaluate import replay_session
    from ai_llm.features import ladder_context_from_row
    from ai_llm.paths import NS3_RATE_LADDER_BPS, NUM_LADDER_RUNGS
    from ai_llm.trace_schema import LABEL_FIELD_CLASS
    from ai_llm.decision_parser import parse_decisions
    from scripts.dp_oracle import load_reward_config  # type: ignore
except ImportError:  # run directly from scratch/ai_llm
    from evaluate import replay_session
    from features import ladder_context_from_row
    from paths import NS3_RATE_LADDER_BPS, NUM_LADDER_RUNGS
    from trace_schema import LABEL_FIELD_CLASS
    from decision_parser import parse_decisions
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from dp_oracle import load_reward_config  # type: ignore


def bitrate_to_rung(bps: float) -> int:
    """Map a delivered bitrate (bps) to its ladder rung index (nearest ladder entry)."""
    return min(range(NUM_LADDER_RUNGS), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))


def _algo_rungs_from_log(sim_log: str) -> Dict[Tuple[int, int], int]:
    """(node_id, segment_id) -> the rung the ABR client actually chose, from sim.log."""
    out: Dict[Tuple[int, int], int] = {}
    for d in parse_decisions(sim_log):
        seg = d.get("segment_id")
        if seg is None:
            continue
        out[(int(d["node_id"]), int(seg))] = bitrate_to_rung(float(d["new_bitrate_bps"]))
    return out


def _algo_rungs_from_decisions(decisions_jsonl: str) -> Dict[Tuple[int, int], int]:
    """(node_id, segment_id0) -> the rung the bridge ABR actually delivered, from decisions.jsonl.

    For the V2 (LLM) bridge arm there is no usable sim.log, but decisions.jsonl records the served
    `chosen_rung` (post-guard) per decision. Segment ids are 1-indexed there; shift to the 0-indexed
    teacher/metadata convention so the (node, seg) join matches the Converter-B shard exactly.
    """
    out: Dict[Tuple[int, int], int] = {}
    with open(decisions_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seg = d.get("segment_id")
            node = d.get("node_id")
            if seg is None or node is None:
                continue
            out[(int(node), int(seg) - 1)] = int(d.get("chosen_rung", 0))
    return out


def _teacher_rows(jsonl: str) -> Dict[Tuple[int, int], Dict]:
    """(node_id, segment_id) -> full-teacher record (oracle label + realized throughput + VMAF ladder)."""
    out: Dict[Tuple[int, int], Dict] = {}
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            node = r.get("node_id")
            seg = r.get("segment_id")
            if node is None or seg is None:
                continue
            out[(int(node), int(seg))] = r
    return out


def score_run(sim_log: str, teacher_jsonl: str, reward_cfg: Dict, label: str = "",
              decisions_jsonl: Optional[str] = None) -> Optional[Dict]:
    """Compute the perceptual-QoE metric set for one algorithm run (one video).

    Delivered rungs come from sim.log (classic C++ arms) or, for the V2 bridge arm, from
    `decisions_jsonl` (chosen_rung) — everything downstream (oracle/floor replay, per-run
    floor-normalized %-of-oracle) is identical, so the arms are scored the SAME way.
    """
    algo = _algo_rungs_from_decisions(decisions_jsonl) if decisions_jsonl else _algo_rungs_from_log(sim_log)
    rows_by_key = _teacher_rows(teacher_jsonl)
    # keep only decision points present in BOTH the log and the teacher records, grouped by node
    by_node: Dict[int, List[Tuple[int, Dict, int]]] = defaultdict(list)
    for (node, seg), row in rows_by_key.items():
        if (node, seg) in algo:
            by_node[node].append((seg, row, algo[(node, seg)]))
    if not by_node:
        return None

    # The oracle reward is signed (heavy stall/rung penalties), so the raw policy/oracle ratio is
    # ill-defined under congestion. We normalize against a deterministic FLOOR (all-lowest-rung policy,
    # which the oracle could have chosen but didn't, so oracle_qoe >= floor_qoe always):
    #   % of oracle = 100 * (policy - floor) / (oracle - floor)   in ~[0,100]
    # 100% = optimal, 0% = no better than always-min-quality, <0% = worse (e.g. over-asking -> stalls).
    pol_qoe = ora_qoe = flr_qoe = 0.0
    stall = switches = seg_total = bitrate_mbps = 0.0
    vmaf_sum = 0.0
    confusion = [[0] * NUM_LADDER_RUNGS for _ in range(NUM_LADDER_RUNGS)]
    rung_hist: Counter = Counter()
    n_match = 0
    for node, items in by_node.items():
        items.sort(key=lambda x: x[0])  # by segment_id
        rows = [it[1] for it in items]
        ladders = [ladder_context_from_row(r) for r in rows]
        pol_rungs = [it[2] for it in items]
        ora_rungs = [int(r["teacher"][LABEL_FIELD_CLASS]) for r in rows]
        flr_rungs = [0] * len(rows)  # conservative all-lowest-rung floor
        pol = replay_session(pol_rungs, rows, ladders, reward_cfg)
        ora = replay_session(ora_rungs, rows, ladders, reward_cfg)
        flr = replay_session(flr_rungs, rows, ladders, reward_cfg)
        pol_qoe += pol["combined_qoe"]
        ora_qoe += ora["combined_qoe"]
        flr_qoe += flr["combined_qoe"]
        stall += pol["stall_s"]
        switches += pol["switch_count"]
        seg_total += pol["segments"]
        bitrate_mbps += pol["bitrate_mbps_sum"]
        for t, (p, o) in enumerate(zip(pol_rungs, ora_rungs)):
            confusion[o][p] += 1
            rung_hist[p] += 1
            vmaf_sum += float(ladders[t][p][1])
            n_match += int(p == o)
    n = int(seg_total)
    # Bounded % of oracle: clip policy into [floor, oracle] before normalizing, so an unbounded stall
    # penalty can't blow it up. Range [0,100]: 100% = optimal, 0% = no better than always-min (incl.
    # catastrophic stalling — REGRET carries the magnitude of how-much-worse below floor).
    span = ora_qoe - flr_qoe
    if span > 1e-6:
        clipped = min(ora_qoe, max(flr_qoe, pol_qoe))
        pct_oracle = 100.0 * (clipped - flr_qoe) / span
    else:
        # oracle ~= all-min floor (e.g. easy content: rung0 optimal). 100% only if the policy also
        # achieved that trivial optimum; a policy that over-asks/stalls here is worse -> 0%.
        pct_oracle = 100.0 if pol_qoe >= ora_qoe - 1e-6 else 0.0
    return {
        "label": label,
        "segments": n,
        "pct_of_oracle": pct_oracle,                 # A: HEADLINE (floor-normalized, bounded)
        "regret_per_seg": (ora_qoe - pol_qoe) / max(n, 1),  # C: decision quality (QoE units, robust)
        "mean_vmaf": vmaf_sum / max(n, 1),           # descriptive anchor
        "mean_bitrate_mbps": bitrate_mbps / max(n, 1),  # B
        "stall_s": stall,                            # B
        "switches": int(switches),                   # B
        "oracle_agreement": n_match / max(n, 1),     # C (optional/caveated)
        "rung_hist": dict(sorted(rung_hist.items())),
        "confusion": confusion,                      # C diagnostic (rows=oracle, cols=policy)
        "_qoe": {"policy": pol_qoe, "oracle": ora_qoe, "floor": flr_qoe},  # for pooled aggregation
    }


def _discover_runs(run_dir: str, teacher_dir: str, video: Optional[str]) -> List[Tuple[str, str, str]]:
    """Find (label, sim_log, teacher_jsonl) triples. teacher_jsonl matched by video name + run id."""
    triples = []
    for sl in sorted(glob.glob(os.path.join(run_dir, "**", "sim.log"), recursive=True)):
        parts = sl.split(os.sep)
        # .../<video>/run_<seed>/sim.log
        run = parts[-2] if len(parts) >= 2 else "?"
        vid = parts[-3] if len(parts) >= 3 else "?"
        if video and vid != video:
            continue
        cands = glob.glob(os.path.join(teacher_dir, "%s_%s.jsonl" % (vid, run))) or \
            glob.glob(os.path.join(teacher_dir, "%s_run_*.jsonl" % vid))
        if cands:
            triples.append(("%s/%s" % (vid, run), sl, cands[0]))
    return triples


def main() -> None:
    ap = argparse.ArgumentParser(description="Perceptual (VMAF) %-of-oracle QoE metric for an ABR run.")
    ap.add_argument("--sim-log", help="single run sim.log")
    ap.add_argument("--teacher-jsonl", help="matching full-teacher records (train_jsonl-style shard)")
    ap.add_argument("--run-dir", help="root of <video>/run_*/sim.log to batch-score")
    ap.add_argument("--teacher-dir", help="dir of <video>_run_N.jsonl full-teacher shards")
    ap.add_argument("--video", help="restrict batch to one video name")
    ap.add_argument("--reward-config", default=None, help="oracle_reward.json (default: config/)")
    args = ap.parse_args()

    cfg = load_reward_config(args.reward_config)
    results = []
    if args.sim_log and args.teacher_jsonl:
        r = score_run(args.sim_log, args.teacher_jsonl, cfg, label=os.path.basename(os.path.dirname(args.sim_log)))
        if r:
            results.append(r)
    elif args.run_dir and args.teacher_dir:
        for label, sl, tj in _discover_runs(args.run_dir, args.teacher_dir, args.video):
            r = score_run(sl, tj, cfg, label=label)
            if r:
                results.append(r)
    else:
        ap.error("give --sim-log+--teacher-jsonl OR --run-dir+--teacher-dir")

    # Per-run: bounded % of oracle (headline) + REGRET (magnitude of failure) + QoE pillars.
    print("%-26s %5s %8s %8s %8s %8s %6s %6s" % (
        "run", "segs", "%oracle", "regret", "meanVMAF", "bitMbps", "stall", "switch"))
    for r in results:
        print("%-26s %5d %7.1f%% %8.3f %8.1f %8.2f %6.1f %6d" % (
            r["label"], r["segments"], r["pct_of_oracle"], r["regret_per_seg"],
            r["mean_vmaf"], r["mean_bitrate_mbps"], r["stall_s"], r["switches"]))
    if results:
        import statistics as st
        mean_pct = st.mean(r["pct_of_oracle"] for r in results)     # bounded [0,100] -> safe to average
        mean_regret = st.mean(r["regret_per_seg"] for r in results)
        mean_vmaf = st.mean(r["mean_vmaf"] for r in results)
        tot_stall = sum(r["stall_s"] for r in results)
        print("-- POOLED over %d runs: %% of oracle = %.1f%%  |  mean regret/seg = %.3f  |  "
              "mean VMAF = %.1f  |  total stall = %.0fs --" % (
                  len(results), mean_pct, mean_regret, mean_vmaf, tot_stall))


if __name__ == "__main__":
    main()
