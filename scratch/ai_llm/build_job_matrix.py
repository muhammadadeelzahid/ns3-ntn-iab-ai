#!/usr/bin/env python3
"""
Build job_matrix.csv for dash_trace_launch.slurm (any ns-3 DASH algorithm).

One row per (metadata_csv, run_seed) combination.
"""

import argparse
import csv
from pathlib import Path


def discover_metadata_files(metadata_dir: Path, video_glob: str, max_videos: int) -> list:
    files = sorted(metadata_dir.glob(video_glob))
    files = [f for f in files if f.is_file() and f.name.endswith("_segments.csv")]
    if max_videos > 0:
        files = files[:max_videos]
    return files


def video_name_from_csv(path: Path) -> str:
    name = path.stem
    if name.endswith("_segments"):
        return name[: -len("_segments")]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--output", required=True, help="job_matrix.csv path")
    parser.add_argument("--runs-per-metadata", type=int, default=1)
    parser.add_argument("--run-offset", type=int, default=1)
    parser.add_argument("--num-relay", type=int, default=1)
    parser.add_argument("--num-ues", type=int, default=10)
    parser.add_argument("--algorithm", default="ns3::FdashClient")
    parser.add_argument("--video-glob", default="*_segments.csv")
    parser.add_argument("--max-videos", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir).resolve()
    files = discover_metadata_files(metadata_dir, args.video_glob, args.max_videos)
    if not files:
        raise SystemExit(f"No *_segments.csv under {metadata_dir}")

    rows = []
    task_id = 1
    for meta_path in files:
        video_name = video_name_from_csv(meta_path)
        for i in range(args.runs_per_metadata):
            run_seed = args.run_offset + i
            rows.append(
                {
                    "task_id": task_id,
                    "metadata_csv": str(meta_path),
                    "video_name": video_name,
                    "run_seed": run_seed,
                    "num_relay": args.num_relay,
                    "num_ues": args.num_ues,
                    "algorithm": args.algorithm,
                }
            )
            task_id += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "metadata_csv",
        "video_name",
        "run_seed",
        "num_relay",
        "num_ues",
        "algorithm",
    ]
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tasks to {out}")


if __name__ == "__main__":
    main()
