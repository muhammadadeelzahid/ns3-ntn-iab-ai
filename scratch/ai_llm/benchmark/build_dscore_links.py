#!/usr/bin/env python3
"""(Re)build the per-condition symlink views Quic_artifacts/<ARM>_dscore/<video>/run_<cond> for the
native-heuristic arms, for ALL test videos, mapping each raw run to its condition from the run's own
run_manifest.json `link_rate` - NOT from the seed number.

Why: the paper's _dscore dirs were hand-made for 10 videos assuming seed 2c-1 <-> cond c (true for the
Bola/Raahs/Aaash 10-seed campaigns) but Fdash/Osmp used a different seed->rate order (Fdash run_3 =
18 Mbps, Osmp run_2 = 9 Mbps). The 26-video re-scoring (absolute_qoe_table / jain_fairness /
plot_behavior / extract_true_stall all read <ARM>_dscore/<v>/run_<cond>) needs one correct, uniform view.

run_<cond> -> the LOWEST seed whose manifest link_rate maps to that condition; further seeds of the same
condition are exposed as run_<cond>_s<seed> (extra replicates, ignored by the run_<cond> readers).
Idempotent: existing symlinks are re-pointed, real directories are never touched (reported instead).

Usage: build_dscore_links.py [--arms Bola,Raahs,Aaash,Fdash,Osmp] [--videos test_videos_26.txt] [--dry-run]
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
Q = os.path.join(_REPO, "Quic_artifacts")
COND = {"100Gb/s": 1, "36Mbps": 2, "18Mbps": 3, "9Mbps": 4, "4.5Mbps": 5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="Bola,Raahs,Aaash,Fdash,Osmp")
    ap.add_argument("--videos", default=os.path.join(_HERE, "test_videos_26.txt"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    videos = [v for v in open(args.videos).read().split() if v]

    for arm in args.arms.split(","):
        src_root = os.path.join(Q, arm)
        dst_root = os.path.join(Q, "%s_dscore" % arm)
        n_links = n_missing = n_skipped = 0
        missing_cells = []
        for v in videos:
            runs = {}
            for rd in sorted(glob.glob(os.path.join(src_root, v, "run_*"))):
                mf = os.path.join(rd, "run_manifest.json")
                if not os.path.isfile(mf) or not os.path.isfile(os.path.join(rd, "sim.log")):
                    continue
                m = json.load(open(mf))
                cond = COND.get(str(m.get("link_rate", "")))
                if cond is None:
                    print("  ? %s: unknown link_rate %r" % (rd, m.get("link_rate")), file=sys.stderr)
                    continue
                seed = int(m.get("run_seed", os.path.basename(rd).split("_")[1]))
                runs.setdefault(cond, []).append((seed, rd))
            for cond in range(1, 6):
                seeds = sorted(runs.get(cond, []))
                if not seeds:
                    n_missing += 1
                    missing_cells.append("%s/run_%d" % (v, cond))
                    continue
                for i, (seed, rd) in enumerate(seeds):
                    name = "run_%d" % cond if i == 0 else "run_%d_s%d" % (cond, seed)
                    dst = os.path.join(dst_root, v, name)
                    if os.path.lexists(dst) and not os.path.islink(dst):
                        n_skipped += 1
                        print("  ! %s is a real directory - left untouched" % dst, file=sys.stderr)
                        continue
                    if os.path.islink(dst) and os.readlink(dst) == rd:
                        n_links += 1
                        continue
                    if not args.dry_run:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if os.path.islink(dst):
                            os.unlink(dst)
                        os.symlink(rd, dst)
                    n_links += 1
        print("%s: %d run_<cond> links for %d videos, %d missing cells, %d real dirs skipped%s" % (
            arm, n_links, len(videos), n_missing, n_skipped, " (dry-run)" if args.dry_run else ""))
        if missing_cells:
            print("   missing:", ", ".join(missing_cells[:12]), "..." if len(missing_cells) > 12 else "")


if __name__ == "__main__":
    main()
