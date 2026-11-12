#!/usr/bin/env python3
"""Catalog selection by VMAF-headroom coverage.

Reads every processed metadata CSV (size_bytes_* / vmaf_* per rung), computes a
per-video VMAF headroom profile, splits videos into ALREADY-IN-TRAINING vs NEW
candidates, and proposes ~TARGET_ADD new videos that best fill the headroom-coverage
gap (stratified toward an even final distribution, with emphasis on the
under-represented early-saturating / low-headroom end the Fig-3 story needs).

Outputs: an analysis printout + a proposed keep-list written to --out (markdown+csv).
Offline only; reads metadata CSVs. Final list is for review before Stage B.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

LADDER = ["250k", "750k", "1500k", "3000k", "6000k", "12000k"]
METADATA_GLOB = "/home/adeel/projects/def-ekram/adeel/abr/output/metadata/*_segments.csv"
TRAIN_JSONL = "/project/6030214/adeel/ns3-ntn-iab/train_jsonl"
SAT_EPS = 1.0  # VMAF within this of the top-rung mean => "saturated" at that rung


def existing_video_names():
    names = set()
    for split in ("train", "val", "test"):
        for fp in glob.glob(os.path.join(TRAIN_JSONL, split, "*.jsonl")):
            base = os.path.basename(fp)
            if "_run_" in base:
                names.add(base.split("_run_")[0])
    return names


def video_profile(csv_path):
    """Mean VMAF per rung across segments -> headroom + saturation rung."""
    sums = defaultdict(float)
    n = 0
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            ok = True
            vals = {}
            for lv in LADDER:
                key = "vmaf_%s" % lv
                if key not in row or row[key] == "":
                    ok = False
                    break
                vals[lv] = float(row[key])
            if not ok:
                continue
            for lv in LADDER:
                sums[lv] += vals[lv]
            n += 1
    if n == 0:
        return None
    mean = {lv: sums[lv] / n for lv in LADDER}
    top = mean[LADDER[-1]]
    headroom = top - mean[LADDER[0]]
    sat_rung = len(LADDER) - 1
    for i, lv in enumerate(LADDER):
        if mean[lv] >= top - SAT_EPS:
            sat_rung = i
            break
    return {"n_seg": n, "mean": mean, "headroom": headroom,
            "sat_rung": sat_rung, "vmaf_top": top, "vmaf_bot": mean[LADDER[0]]}


def hist(values, lo, hi, bins):
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        for b in range(bins):
            if (v >= edges[b] and v < edges[b + 1]) or (b == bins - 1 and v >= edges[b]):
                counts[b] += 1
                break
    return edges, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-add", type=int, default=70)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "catalog_selection"))
    args = ap.parse_args()

    existing = existing_video_names()
    profiles = {}
    for csv_path in sorted(glob.glob(METADATA_GLOB)):
        name = os.path.basename(csv_path)[:-len("_segments.csv")]
        p = video_profile(csv_path)
        if p:
            p["existing"] = name in existing
            profiles[name] = p

    new = {k: v for k, v in profiles.items() if not v["existing"]}
    old = {k: v for k, v in profiles.items() if v["existing"]}
    print("profiles: %d total | %d existing | %d new candidates"
          % (len(profiles), len(old), len(new)))

    all_hr = [p["headroom"] for p in profiles.values()]
    lo, hi = min(all_hr), max(all_hr)
    print("headroom range: %.1f .. %.1f VMAF" % (lo, hi))

    def show(label, d):
        hrs = [p["headroom"] for p in d.values()]
        edges, counts = hist(hrs, lo, hi, args.bins)
        print("\n%s (n=%d): headroom histogram" % (label, len(d)))
        for b in range(args.bins):
            print("  [%5.1f,%5.1f) %s %d" % (edges[b], edges[b + 1], "#" * counts[b], counts[b]))
        return counts, edges

    old_counts, edges = show("EXISTING", old)
    new_counts, _ = show("NEW candidates", new)

    # Stratified selection: aim for an even final distribution across bins.
    # target per bin = ceil(total_final / bins); deficit = target - existing.
    target_final = len(old) + args.target_add
    per_bin_target = -(-target_final // args.bins)  # ceil
    # bucket new candidates by bin
    new_by_bin = defaultdict(list)
    for name, p in new.items():
        for b in range(args.bins):
            if (p["headroom"] >= edges[b] and p["headroom"] < edges[b + 1]) or \
               (b == args.bins - 1 and p["headroom"] >= edges[b]):
                new_by_bin[b].append((name, p))
                break

    selected = []
    # pass 1: fill each bin's deficit (most under-represented first)
    deficits = sorted(range(args.bins), key=lambda b: (per_bin_target - old_counts[b]), reverse=True)
    for b in deficits:
        need = max(0, per_bin_target - old_counts[b])
        cand = sorted(new_by_bin[b], key=lambda x: x[1]["headroom"])
        take = cand[:need]
        selected.extend(take)
        for nm, _ in take:
            new_by_bin[b] = [c for c in new_by_bin[b] if c[0] != nm]
    # pass 2: if under target, top up from remaining new (prefer low-headroom = early-saturating)
    if len(selected) < args.target_add:
        rest = [c for b in range(args.bins) for c in new_by_bin[b]]
        rest.sort(key=lambda x: x[1]["headroom"])
        selected.extend(rest[:args.target_add - len(selected)])
    selected = selected[:args.target_add]

    # report final coverage
    final = dict(old)
    for nm, p in selected:
        final[nm] = p
    print("\n=> selected %d new videos (target %d). FINAL catalog = %d videos"
          % (len(selected), args.target_add, len(final)))
    _, fcounts = hist([p["headroom"] for p in final.values()], lo, hi, args.bins)
    print("FINAL headroom histogram:")
    for b in range(args.bins):
        print("  [%5.1f,%5.1f) %s %d" % (edges[b], edges[b + 1], "#" * fcounts[b], fcounts[b]))

    # write proposed keep-list
    os.makedirs(args.out, exist_ok=True)
    sel_sorted = sorted(selected, key=lambda x: x[1]["headroom"])
    md = os.path.join(args.out, "proposed_additions.md")
    cs = os.path.join(args.out, "proposed_additions.csv")
    with open(md, "w") as f, open(cs, "w") as c:
        c.write("video,headroom,sat_rung,vmaf_bot,vmaf_top,n_seg\n")
        f.write("# Proposed catalog additions (%d new videos)\n\n" % len(selected))
        f.write("| Video | Headroom (VMAF) | Saturates @ rung | VMAF bot->top | segs |\n")
        f.write("|---|--:|--|--|--:|\n")
        for nm, p in sel_sorted:
            f.write("| %s | %.1f | r%d (%s) | %.1f -> %.1f | %d |\n"
                    % (nm, p["headroom"], p["sat_rung"], LADDER[p["sat_rung"]],
                       p["vmaf_bot"], p["vmaf_top"], p["n_seg"]))
            c.write("%s,%.2f,%d,%.2f,%.2f,%d\n"
                    % (nm, p["headroom"], p["sat_rung"], p["vmaf_bot"], p["vmaf_top"], p["n_seg"]))
    print("\nwrote: %s\n       %s" % (md, cs))


if __name__ == "__main__":
    main()
