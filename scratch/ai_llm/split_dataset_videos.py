#!/usr/bin/env python3
"""Persisted, genre-stratified, **whole-video** train/val/test split for the ABR dataset.

Why this exists
---------------
Training must never be evaluated on data it learned from. `train.py`'s in-memory split is
*by session* (so the same video lands in both train and val) and `evaluate.py` otherwise scores
the entire corpus. Standard ML methodology requires three *disjoint* sets where validation/test
are never trained on, the split is fixed *before* feature scaling, and — for time-series — no
sequence (here, a whole video) appears in more than one set.

What it does
------------
1. Discovers every `video_name` present in the dataset shards (`<video>_run_<id>.jsonl`).
2. Derives a coarse **genre** per video from the abr catalog (query in `notes:` for discovered
   Pexels videos; descriptive `tags:` for the original handful).
3. Stratifies a 70/15/15 train/val/test split **within each genre** (deterministic by seed), so
   each content category is represented across splits. Genres too small to span three sets go to
   train.
4. Materializes `train/`, `val/`, `test/` subdirectories of **symlinks** to the per-video shards
   and writes `split_manifest.json` recording the assignment, genres, seed and counts. Idempotent.

Train/eval then point at the subdirectories:
    train.py     --data-dir train_jsonl/train --val-data-dir train_jsonl/val
    evaluate.py  --data-dir train_jsonl/test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_CATALOG = "/home/adeel/projects/def-ekram/adeel/abr/config/video_catalog.yaml"
SPLITS = ("train", "val", "test")

# Coarse canonical genres. Both the Pexels discovery queries and the original videos' tags map
# into this set so the two sources are stratified consistently.
QUERY_GENRE = [
    ("nature forest", "nature"),
    ("city night", "urban"),
    ("ocean", "water"),
    ("waterfall", "water"),
    ("wildlife", "animals"),
    ("sports", "sports"),
    ("mountain", "mountain"),
]
TAG_GENRE = {
    "nature": "nature", "forest": "nature", "lake": "nature", "hiking": "nature",
    "adventure": "nature", "sunset": "nature",
    "ocean": "water", "water": "water", "underwater": "water", "waterfall": "water",
    "urban": "urban", "city": "urban", "traffic": "urban", "night": "urban",
    "food": "food", "cooking": "food",
    "sports": "sports", "fitness": "sports",
    "wildlife": "animals", "animals": "animals",
    "weather": "weather", "storm": "weather", "rain": "weather",
    "people": "people", "dancing": "people",
    "animation": "animation", "blender": "animation", "live_action": "animation", "vfx": "animation",
}


def genre_for(entry: Dict) -> str:
    """Derive a coarse genre from a catalog entry (query in notes, else tags, else 'other')."""
    notes = (entry.get("notes") or "").lower()
    m = re.search(r"query:\s*([^)\']+)", notes)
    if m:
        q = m.group(1).strip()
        for key, g in QUERY_GENRE:
            if key in q:
                return g
    for tag in (entry.get("tags") or []):
        t = str(tag).lower()
        if t in TAG_GENRE:
            return TAG_GENRE[t]
    return "other"


def load_catalog_genres(catalog_path: str) -> Dict[str, str]:
    import yaml  # provided by the project .venv

    with open(catalog_path, "r", encoding="utf-8") as fh:
        cat = yaml.safe_load(fh)
    entries = cat.get("videos", cat) if isinstance(cat, dict) else cat
    return {e.get("id"): genre_for(e) for e in entries if e.get("id")}


def discover_video_shards(data_dir: Path, glob_pattern: str) -> Dict[str, List[Path]]:
    """Map video_name -> [shard paths]. video_name is read from each shard's first row
    (robust to underscores/`run` in video ids). Subdirectories and merged files are skipped."""
    videos: Dict[str, List[Path]] = defaultdict(list)
    for shard in sorted(data_dir.glob(glob_pattern)):
        if not shard.is_file() or shard.name == "all_runs.jsonl":
            continue
        vid = None
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    vid = json.loads(line).get("video_name")
                except json.JSONDecodeError:
                    vid = None
                break
        if vid is None:
            print("  WARN: no video_name in %s — skipped" % shard.name, file=sys.stderr)
            continue
        videos[vid].append(shard)
    return videos


def stratified_split(
    videos_by_genre: Dict[str, List[str]],
    seed: int,
    val_frac: float,
    test_frac: float,
    min_genre_to_span: int,
) -> Dict[str, str]:
    """Return {video_name: split}. Within each genre, deterministically assign val/test/train.
    Genres with fewer than `min_genre_to_span` videos go entirely to train."""
    import random

    assignment: Dict[str, str] = {}
    for genre in sorted(videos_by_genre):
        vids = sorted(videos_by_genre[genre])
        rng = random.Random("%s:%d" % (genre, seed))  # genre-scoped, reproducible
        rng.shuffle(vids)
        n = len(vids)
        if n < min_genre_to_span:
            for v in vids:
                assignment[v] = "train"
            continue
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val >= n:  # keep at least one training video
            n_test, n_val = 1, 1
        for v in vids[:n_test]:
            assignment[v] = "test"
        for v in vids[n_test:n_test + n_val]:
            assignment[v] = "val"
        for v in vids[n_test + n_val:]:
            assignment[v] = "train"
    return assignment


def materialize(
    out_dir: Path,
    shards_by_video: Dict[str, List[Path]],
    assignment: Dict[str, str],
    splits: List[str],
) -> Dict[str, int]:
    """Create split subdirs of relative symlinks to the per-video shards. Idempotent.

    Symlink targets are computed relative to each link so the split dirs work whether they live
    under the shard directory (``out_dir == data_dir``) or in a separate holdout tree
    (``out_dir == train_jsonl_holdout``, shards still in ``train_jsonl``).
    """
    for sp in splits:
        d = out_dir / sp
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    counts = {sp: 0 for sp in splits}
    for vid, shards in shards_by_video.items():
        sp = assignment.get(vid, "train")
        for shard in shards:
            link = out_dir / sp / shard.name
            os.symlink(os.path.relpath(shard, start=link.parent), link)  # relative target
            counts[sp] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="train_jsonl", help="Directory with per-video shards.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to materialize the split subdirs (default: --data-dir). Use a "
                    "separate dir (e.g. train_jsonl_holdout) to keep the existing split untouched.")
    ap.add_argument("--glob", dest="glob_pattern", default="*.jsonl", help="Shard glob.")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG, help="abr video_catalog.yaml for genres.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--min-genre-to-span", type=int, default=4,
                    help="Genres with fewer videos than this go entirely to train.")
    ap.add_argument("--holdout-genre", default=None,
                    help="Route ALL videos of this genre into a separate test_genre/ split (never "
                    "seen in train/val) for the unseen-content experiment. The remaining genres are "
                    "stratified into train/val/test as usual.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir
    shards_by_video = discover_video_shards(data_dir, args.glob_pattern)
    if not shards_by_video:
        raise SystemExit("No video shards found under %s/%s" % (data_dir, args.glob_pattern))

    genres = load_catalog_genres(args.catalog)
    # When a genre is held out, those videos bypass the stratified split entirely and land in
    # test_genre/. The remaining genres are stratified into train/val/test as usual.
    holdout_genre = args.holdout_genre
    splits = list(SPLITS) + (["test_genre"] if holdout_genre else [])

    videos_by_genre: Dict[str, List[str]] = defaultdict(list)
    unmapped: List[str] = []
    holdout_videos: List[str] = []
    for vid in shards_by_video:
        g = genres.get(vid, "other")
        if vid not in genres:
            unmapped.append(vid)
        if holdout_genre and g == holdout_genre:
            holdout_videos.append(vid)
            continue
        videos_by_genre[g].append(vid)

    if holdout_genre and not holdout_videos:
        raise SystemExit("--holdout-genre %r matched no videos (genres present: %s)" % (
            holdout_genre, sorted({genres.get(v, "other") for v in shards_by_video})))

    assignment = stratified_split(
        videos_by_genre, args.seed, args.val_frac, args.test_frac, args.min_genre_to_span
    )
    for vid in holdout_videos:
        assignment[vid] = "test_genre"
    counts_shards = materialize(out_dir, shards_by_video, assignment, splits)

    # Per-split video counts and per-genre breakdown for the manifest + console.
    split_videos = defaultdict(list)
    for vid, sp in assignment.items():
        split_videos[sp].append(vid)
    genre_split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for vid, sp in assignment.items():
        genre_split_counts[genres.get(vid, "other")][sp] += 1

    manifest = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "min_genre_to_span": args.min_genre_to_span,
        "catalog": args.catalog,
        "holdout_genre": holdout_genre,
        "out_dir": str(out_dir),
        "n_videos": len(shards_by_video),
        "video_count_per_split": {sp: len(split_videos[sp]) for sp in splits},
        "shard_count_per_split": counts_shards,
        "genre_split_counts": {
            g: {sp: genre_split_counts[g].get(sp, 0) for sp in splits}
            for g in sorted(genre_split_counts)
        },
        "videos": {
            vid: {"genre": genres.get(vid, "other"), "split": assignment.get(vid, "train")}
            for vid in sorted(shards_by_video)
        },
    }
    manifest_path = out_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    print("=== video split (seed=%d, val=%.2f, test=%.2f%s) ===" % (
        args.seed, args.val_frac, args.test_frac,
        ", holdout-genre=%s" % holdout_genre if holdout_genre else ""))
    print("videos: %d  |  per-split videos: %s  |  per-split shards: %s" % (
        len(shards_by_video),
        {sp: len(split_videos[sp]) for sp in splits}, counts_shards))
    print("genre breakdown (videos per split):")
    for g in sorted(genre_split_counts):
        cols = "  ".join("%s=%-3d" % (sp, genre_split_counts[g].get(sp, 0)) for sp in splits)
        print("  %-10s %s" % (g, cols))
    if unmapped:
        print("NOTE: %d videos not in catalog (-> 'other' -> train): %s" % (
            len(unmapped), unmapped[:10]))
    print("manifest: %s" % manifest_path)


if __name__ == "__main__":
    main()
