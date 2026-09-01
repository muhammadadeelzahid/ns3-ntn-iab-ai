#!/usr/bin/env python3
"""Build the headroom-BALANCED Stage-B subset (~180) + a stratified split.

From all processed metadata (270 videos), cap each VMAF-headroom bin at --cap so the
over-represented high-headroom bins are trimmed while every low/mid video is kept. Then
assign a group-aware (per-video) train/val/test split STRATIFIED within each headroom bin
so the low-headroom bucket is represented in test (the Fig-3 requirement).

Within an over-capped bin, existing (already-in-train_jsonl) videos are kept first for
continuity, then others; selection is deterministic (sorted by name). Output: a manifest
(video_id, headroom, bin, split) for Stage-B trace collection + Stage-C split routing.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from select_catalog import LADDER, METADATA_GLOB, existing_video_names, video_profile  # noqa: E402
import glob  # noqa: E402

NBINS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=30, help="max videos per headroom bin")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(HERE, "catalog_selection", "stage_b_subset"))
    args = ap.parse_args()

    existing = existing_video_names()
    profs = {}
    for csv_path in sorted(glob.glob(METADATA_GLOB)):
        name = os.path.basename(csv_path)[:-len("_segments.csv")]
        p = video_profile(csv_path)
        if p:
            p["existing"] = name in existing
            profs[name] = p
    hrs = [p["headroom"] for p in profs.values()]
    lo, hi = min(hrs), max(hrs)
    edges = [lo + (hi - lo) * i / NBINS for i in range(NBINS + 1)]

    def binof(h):
        for b in range(NBINS):
            if h < edges[b + 1] or b == NBINS - 1:
                return b
        return NBINS - 1

    by_bin = {b: [] for b in range(NBINS)}
    for name, p in profs.items():
        by_bin[binof(p["headroom"])].append((name, p))

    # select + split per bin
    manifest = []
    print("bin  headroom-range        pool  kept | train val test")
    for b in range(NBINS):
        # existing first, then new; deterministic by name
        items = sorted(by_bin[b], key=lambda x: (not x[1]["existing"], x[0]))
        kept = items[:args.cap]
        n = len(kept)
        # stratified split within the bin
        test_n = max(1, round(args.test_frac * n)) if n else 0
        val_n = max(1, round(args.val_frac * n)) if n else 0
        if test_n + val_n >= n:  # keep at least 1 train for tiny bins
            test_n = min(test_n, max(0, n - 2))
            val_n = min(val_n, max(0, n - 1 - test_n))
        # assign: spread so each split samples the bin (stride)
        names = [k[0] for k in kept]
        test = set(names[:test_n])
        val = set(names[test_n:test_n + val_n])
        for name, p in kept:
            split = "test" if name in test else ("val" if name in val else "train")
            manifest.append({"video_id": name, "headroom": round(p["headroom"], 1),
                             "bin": b, "split": split, "existing": p["existing"]})
        tr = n - test_n - val_n
        print("%2d   [%5.1f,%5.1f)   %4d  %4d | %4d %4d %4d"
              % (b, edges[b], edges[b + 1], len(items), n, tr, val_n, test_n))

    total = len(manifest)
    from collections import Counter
    sc = Counter(m["split"] for m in manifest)
    print("\nSubset total: %d videos (cap=%d) | train=%d val=%d test=%d"
          % (total, args.cap, sc["train"], sc["val"], sc["test"]))
    # low-headroom (bins 0-2) per split - the Fig-3 anchor
    low = [m for m in manifest if m["bin"] <= 2]
    lc = Counter(m["split"] for m in low)
    print("low-headroom (<%0.0f) videos: %d  -> train=%d val=%d test=%d"
          % (edges[3], len(low), lc["train"], lc["val"], lc["test"]))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"cap": args.cap, "edges": edges, "videos": manifest}, f, indent=2)
    with open(os.path.join(args.out, "manifest.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "headroom", "bin", "split", "existing"])
        for m in manifest:
            w.writerow([m["video_id"], m["headroom"], m["bin"], m["split"], m["existing"]])
    # plain id list for Stage B
    with open(os.path.join(args.out, "stage_b_video_ids.txt"), "w") as f:
        f.write("\n".join(m["video_id"] for m in manifest) + "\n")
    print("\nwrote manifest.json / manifest.csv / stage_b_video_ids.txt under %s" % args.out)


if __name__ == "__main__":
    main()
