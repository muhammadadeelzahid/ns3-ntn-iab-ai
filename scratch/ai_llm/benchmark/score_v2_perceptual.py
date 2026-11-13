#!/usr/bin/env python3
"""Score the V2 (LLM bridge) arm with the IDENTICAL perceptual_qoe.score_run path as the heuristics, so
its %-of-oracle is directly comparable to the SVAA/SFTM/.../FDASH headline.

Heuristics: sim.log delivered rungs + per-run teacher shard -> per-(video,run) floor-normalized %-oracle
(perceptual_qoe.score_run) -> mean over runs per condition (aggregate_table). This does the same for V2 but
sources delivered rungs from decisions.jsonl (chosen_rung) and the teacher shard from Converter-B
(decisions_to_trainrows.py). Same score_run, same per-run floor-normalization, same per-condition mean -
the ONLY change is where delivered rungs come from. PREREQ: decisions_to_throughput -> dp_oracle
(ALGO_SHORT=AiDash) -> decisions_to_trainrows already run so the teacher shards exist.

Usage: score_v2_perceptual.py <AiDash_root> <teacher_shards_dir> <video_list> <reward_config>
"""
import collections
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)                              # -> repo root (enables `import scripts.dp_oracle`)
sys.path.insert(0, os.path.join(_REPO, "scratch"))     # -> scratch/ (enables `import ai_llm.*`)
from ai_llm.perceptual_qoe import score_run
from scripts.dp_oracle import load_reward_config  # type: ignore

COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}


def main():
    root, shards_dir, vlist, reward_cfg_path = sys.argv[1:5]
    # Optional 5th arg: path to write a per-run results file uniform with the heuristic results/*.txt.
    results_out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "AiDash.txt")
    keep = set(open(vlist).read().split())
    cfg = load_reward_config(reward_cfg_path)
    per_cond = collections.defaultdict(list)      # cond -> [pct_of_oracle per (video,run)]
    vmaf_cond = collections.defaultdict(list)
    run_rows = []                                  # (label, res) for the per-run results dump
    n_runs = 0
    for dec in sorted(glob.glob(os.path.join(root, "*", "run_*", "decisions.jsonl"))):
        video = dec.split("/")[-3]
        if video not in keep:
            continue
        run_id = dec.split("/")[-2]                # run_N
        cond = int(run_id.split("_")[1])
        shard = os.path.join(shards_dir, "%s_%s.jsonl" % (video, run_id))
        if not os.path.exists(shard):
            continue
        label = "%s/%s" % (video, run_id)
        res = score_run(None, shard, cfg, label=label, decisions_jsonl=dec)
        if res is None:
            continue
        per_cond[cond].append(res["pct_of_oracle"])
        vmaf_cond[cond].append(res["mean_vmaf"])
        run_rows.append((label, res))
        n_runs += 1

    # Per-run dump in the SAME format as the heuristic results/*.txt (run, segs, %oracle, regret,
    # meanVMAF, bitMbps, stall, switch) so make_paper_table/figs treat all 7 arms identically.
    os.makedirs(os.path.dirname(results_out), exist_ok=True)
    with open(results_out, "w") as fh:
        fh.write("%-26s %5s %8s %8s %8s %8s %6s %6s\n" % (
            "run", "segs", "%oracle", "regret", "meanVMAF", "bitMbps", "stall", "switch"))
        for label, r in sorted(run_rows):
            fh.write("%-26s %5d %7.1f%% %8.3f %8.1f %8.2f %6.1f %6d\n" % (
                label, r["segments"], r["pct_of_oracle"], r["regret_per_seg"],
                r["mean_vmaf"], r["mean_bitrate_mbps"], r["stall_s"], r["switches"]))
    print("wrote per-run results -> %s (%d runs)" % (results_out, len(run_rows)))

    print("V2 (causal+guard) perceptual %-of-oracle - same path as heuristics (mean over runs per cond):")
    print("  cond     | runs | %oracle | meanVMAF")
    all_pct = []
    for c in sorted(per_cond):
        p = per_cond[c]
        v = vmaf_cond[c]
        all_pct.extend(p)
        print("  %-8s | %4d | %6.1f%% | %7.1f" % (COND[c], len(p), sum(p) / len(p), sum(v) / len(v)))
    if all_pct:
        print("  OVERALL  | %4d | %6.1f%% |" % (len(all_pct), sum(all_pct) / len(all_pct)))
    print("(%d runs scored)" % n_runs)


if __name__ == "__main__":
    main()
