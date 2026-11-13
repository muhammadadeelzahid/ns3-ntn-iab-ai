#!/usr/bin/env python3
"""
Build job_matrix.csv for dash_trace_launch.slurm (any ns-3 DASH algorithm).

One row per (metadata_csv, run_seed, link_rate) combination. The S1-U feeder rate
(``--link-rates``) is swept to collect congestion-diverse traces; each (run x link_rate)
gets a UNIQUE run_seed within a video so the per-run artifact dir never collides.
"""

import argparse
import csv
from pathlib import Path


def discover_metadata_files(metadata_dir: Path, video_glob: str, max_videos: int,
                            video_list: list = None) -> list:
    files = sorted(metadata_dir.glob(video_glob))
    files = [f for f in files if f.is_file() and f.name.endswith("_segments.csv")]
    if video_list:  # restrict to a named set (e.g. the held-out test/ split for the benchmark)
        keep = set(video_list)
        files = [f for f in files if video_name_from_csv(f) in keep]
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
    parser.add_argument("--video-list", default=None,
                        help="File with one video_name per line; restrict the matrix to these "
                        "(e.g. the held-out test/ split for the benchmark).")
    parser.add_argument(
        "--link-rates",
        default="100Gb/s",
        help="Comma-separated S1-U feeder rates to sweep, e.g. '100Gb/s,18Mbps,9Mbps'. "
        "Each rate x run gets a unique run_seed within a video.",
    )
    args = parser.parse_args()

    link_rates = [r.strip() for r in args.link_rates.split(",") if r.strip()]
    if not link_rates:
        raise SystemExit("--link-rates produced an empty list")

    metadata_dir = Path(args.metadata_dir).resolve()
    vlist = None
    if args.video_list:
        with open(args.video_list, encoding="utf-8") as fh:
            vlist = [ln.strip() for ln in fh if ln.strip()]
    files = discover_metadata_files(metadata_dir, args.video_glob, args.max_videos, vlist)
    if not files:
        raise SystemExit(f"No *_segments.csv under {metadata_dir}")

    rows = []
    task_id = 1
    for meta_path in files:
        video_name = video_name_from_csv(meta_path)
        for link_idx, link_rate in enumerate(link_rates):
            for i in range(args.runs_per_metadata):
                # Unique run_seed per (link_rate, run) within a video -> distinct artifact dir
                # (run_artifacts_dir is keyed by algo/video/run_seed) AND distinct RNG realization.
                run_seed = args.run_offset + link_idx * args.runs_per_metadata + i
                rows.append(
                    {
                        "task_id": task_id,
                        "metadata_csv": str(meta_path),
                        "video_name": video_name,
                        "run_seed": run_seed,
                        "num_relay": args.num_relay,
                        "num_ues": args.num_ues,
                        "algorithm": args.algorithm,
                        "link_rate": link_rate,
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
        "link_rate",
    ]
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tasks to {out}")


if __name__ == "__main__":
    main()
