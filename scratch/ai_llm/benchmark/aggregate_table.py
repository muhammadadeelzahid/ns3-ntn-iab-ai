#!/usr/bin/env python3
"""Aggregate per-arm perceptual_qoe outputs into a per-condition benchmark table.

Reads scratch/ai_llm/benchmark/results/<ALGO>.txt (the per-run tables written by
perceptual_qoe via score_report.sh), groups each run by its link-rate condition
(run-index 1..5 = the swept S1-U feeder rates), and emits a markdown table of
mean %-of-oracle per (algorithm x condition) plus the overall mean. The run-index ->
linkRate map matches build_job_matrix / bench_v2_run.sbatch (run = link_idx + 1).

Usage:  python3 aggregate_table.py [results_dir]   (default: ./results next to this file)
"""
import glob
import os
import re
import sys

# run-index -> condition label (order = the LINK_RATES sweep)
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
# preferred arm display order (classics by strength, V2 last); unknown arms appended
ORDER = ["AiDash", "Svaa", "Sftm", "Raahs", "Fdash", "Aaash", "Osmp"]

ROW_RE = re.compile(r"^(\S+)/run_(\d+)\s+\d+\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)")


def parse_arm(path):
    """Return {cond_idx: [pct_oracle, ...]} and {cond_idx: [vmaf,...]} for one arm."""
    pct, vmaf = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = ROW_RE.match(line.strip())
            if not m:
                continue
            run_idx = int(m.group(2))
            if run_idx not in COND:
                continue
            pct.setdefault(run_idx, []).append(float(m.group(3)))
            vmaf.setdefault(run_idx, []).append(float(m.group(5)))
    return pct, vmaf


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "results")
    files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(results_dir, "*.txt"))}
    arms = [a for a in ORDER if a in files] + [a for a in sorted(files) if a not in ORDER]
    if not arms:
        raise SystemExit("no results/*.txt found in %s" % results_dir)

    conds = sorted(COND)
    print("# Benchmark: mean %-of-oracle (VMAF-QoE) by ABR algorithm x network condition\n")
    print("Held-out 14-video test split. Each cell = mean over videos at that S1-U feeder rate.\n")
    hdr = "| Algorithm | " + " | ".join(COND[c] for c in conds) + " | **all** |"
    sep = "|" + "---|" * (len(conds) + 2)
    print(hdr)
    print(sep)
    for arm in arms:
        pct, _ = parse_arm(files[arm])
        cells = []
        allvals = []
        for c in conds:
            vals = pct.get(c, [])
            allvals += vals
            cells.append("%.1f" % mean(vals) if vals else "-")
        label = "**%s (V2/LLM)**" % arm if arm == "AiDash" else arm
        print("| %s | %s | **%.1f** |" % (label, " | ".join(cells), mean(allvals)))
    print("\n(Condition = run-index: run_1=100Gb/s ... run_5=4.5Mbps. '-' = arm not yet scored.)")


if __name__ == "__main__":
    main()
