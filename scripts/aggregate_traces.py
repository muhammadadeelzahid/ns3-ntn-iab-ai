#!/usr/bin/env python3
"""
Aggregate per-run throughput_trace.jsonl into per-video segment summaries.

Input (many per run): Quic_artifacts/<AlgoShort>/<video>/run_*/throughput_trace.jsonl
Output (one per video): Quic_artifacts/<AlgoShort>/<video>/aggregated_throughput.jsonl

This script pools across all run seeds and all UEs for each (video_name, segment_id).
It is used as an intermediate step for dp_oracle.py (future).
"""

from __future__ import print_function

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median, stdev
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from ai_llm.metadata_loader import load_metadata_index  # noqa: E402
from ai_llm.trace_schema import (  # noqa: E402
    AGGREGATED_THROUGHPUT_OPTIONAL_FIELDS,
    AGGREGATED_THROUGHPUT_REQUIRED_FIELDS,
    THROUGHPUT_TRACE_REQUIRED_FIELDS,
)


def _validate_raw_row(row: Dict) -> Optional[str]:
    for field in THROUGHPUT_TRACE_REQUIRED_FIELDS:
        if field not in row:
            return "missing field: {}".format(field)
    if row["throughput_bps"] <= 0:
        return "non-positive throughput_bps"
    if row["segment_fetch_time_s"] <= 0:
        return "non-positive segment_fetch_time_s"
    if row["segment_bytes"] <= 0:
        return "non-positive segment_bytes"
    return None


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(str(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def discover_throughput_files(
    artifacts_root: Path,
    algo_short: str,
    video: str,
    run_glob: str,
) -> List[Path]:
    base = artifacts_root / algo_short / video
    if not base.is_dir():
        return []
    import glob as glob_mod

    pattern = str(base / run_glob / "throughput_trace.jsonl")
    return [Path(p) for p in sorted(glob_mod.glob(pattern))]


def discover_videos(artifacts_root: Path, algo_short: str) -> List[str]:
    base = artifacts_root / algo_short
    if not base.is_dir():
        return []
    videos = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            videos.append(child.name)
    return videos


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(stdev(values))


def _aggregate_group(rows: List[Dict]) -> Dict:
    throughput = [float(r["throughput_bps"]) for r in rows]
    fetch = [float(r["segment_fetch_time_s"]) for r in rows]
    seg_bytes = [float(r["segment_bytes"]) for r in rows]

    first = rows[0]
    run_ids = sorted(set(str(r["run_id"]) for r in rows))
    node_ids = sorted(set(int(r["node_id"]) for r in rows))

    record = {
        "video_name": first.get("video_name", ""),
        "segment_id": int(first["segment_id"]),
        "sample_count": int(len(rows)),
        "throughput_bps_median": float(median(throughput)),
        "throughput_bps_std": _std(throughput),
        "segment_fetch_time_s_median": float(median(fetch)),
        "segment_fetch_time_s_std": _std(fetch),
        "segment_bytes_median": float(median(seg_bytes)),
        "segment_bytes_std": _std(seg_bytes),
        "throughput_bps_conservative": float(min(throughput)),
        "segment_fetch_time_s_conservative": float(max(fetch)),
        "run_ids": run_ids,
        "node_count": int(len(node_ids)),
        "run_count": int(len(run_ids)),
    }

    for field in ("metadata_csv", "algo_short", "segment_duration_s", "segment_convention"):
        if field in first:
            record[field] = first[field]

    return record


def aggregate_video(
    artifacts_root: Path,
    algo_short: str,
    video: str,
    run_glob: str,
    output: Optional[Path],
    strict: bool,
    min_samples: int,
    metadata_dir: Optional[str],
) -> Dict:
    files = discover_throughput_files(artifacts_root, algo_short, video, run_glob)
    if not files:
        msg = "No throughput_trace.jsonl found for {}".format(video)
        if strict:
            raise RuntimeError(msg)
        return {"video": video, "segments_written": 0, "files": 0, "skipped_rows": 0, "warning": msg}

    raw_rows: List[Dict] = []
    skipped_rows = 0

    for path in files:
        for row in load_jsonl(path):
            err = _validate_raw_row(row)
            if err:
                if strict:
                    raise RuntimeError("Invalid row in {}: {}".format(path, err))
                skipped_rows += 1
                continue
            raw_rows.append(row)

    # The extraction contract is keyed for downstream joins, but in practice ns-3 can emit
    # multiple decision completions with the same (run_id, node_id, segment_id). Treat those
    # as independent samples. Only drop exact duplicates (same timestamps + metrics).
    seen: set = set()
    deduped_rows: List[Dict] = []
    for row in raw_rows:
        key = (
            str(row["run_id"]),
            int(row["node_id"]),
            int(row["segment_id"]),
            float(row.get("sim_time_s", 0.0)),
            float(row["throughput_bps"]),
            float(row["segment_fetch_time_s"]),
            int(row["segment_bytes"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)

    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for row in deduped_rows:
        grouped[(str(row.get("video_name", video)), int(row["segment_id"]))].append(row)

    aggregated: List[Dict] = []
    min_samples_observed = None
    for (_, seg_id), rows in grouped.items():
        if len(rows) < min_samples:
            if strict:
                raise RuntimeError(
                    "segment {} has only {} samples (<{})".format(seg_id, len(rows), min_samples)
                )
            continue
        aggregated.append(_aggregate_group(rows))
        if min_samples_observed is None:
            min_samples_observed = len(rows)
        else:
            min_samples_observed = min(min_samples_observed, len(rows))

    aggregated.sort(key=lambda r: int(r["segment_id"]))

    if metadata_dir:
        csv_path = Path(metadata_dir) / "{}_segments.csv".format(video)
        if csv_path.is_file():
            meta = load_metadata_index(str(csv_path), scaler_json_path=None)
            expected_segments = set(meta._rows.keys())  # type: ignore[attr-defined]
            present_segments = set(int(r["segment_id"]) for r in aggregated)
            missing = sorted(expected_segments - present_segments)
            if missing:
                msg = "Missing {} segments vs metadata (first 5: {})".format(
                    len(missing), missing[:5]
                )
                if strict:
                    raise RuntimeError(msg)
                # Keep going, but report.
        else:
            if strict:
                raise RuntimeError("metadata CSV not found: {}".format(csv_path))

    out_path = (
        output
        if output
        else (artifacts_root / algo_short / video / "aggregated_throughput.jsonl")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as handle:
        for record in aggregated:
            for field in AGGREGATED_THROUGHPUT_REQUIRED_FIELDS:
                if field not in record:
                    raise RuntimeError("Missing aggregated field: {}".format(field))
            handle.write(json.dumps(record))
            handle.write("\n")

    return {
        "video": video,
        "output": str(out_path),
        "segments_written": int(len(aggregated)),
        "files": int(len(files)),
        "rows_in": int(len(raw_rows)),
        "rows_deduped": int(len(deduped_rows)),
        "skipped_rows": int(skipped_rows),
        "min_samples": int(min_samples_observed or 0),
    }


def _video_worker(payload: Tuple[str, str, str, str, Optional[str], bool, int]) -> Dict:
    artifacts_root_s, algo_short, video, run_glob, metadata_dir, strict, min_samples = payload
    return aggregate_video(
        artifacts_root=Path(artifacts_root_s),
        algo_short=algo_short,
        video=video,
        run_glob=run_glob,
        output=None,
        strict=strict,
        min_samples=min_samples,
        metadata_dir=metadata_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate throughput_trace.jsonl across runs.")
    parser.add_argument("--artifacts-root", default="Quic_artifacts")
    parser.add_argument("--algo-short", default="Fdash")
    parser.add_argument("--video", help="Aggregate only this video (else all videos).")
    parser.add_argument("--run-glob", default="run_*")
    parser.add_argument("--output", help="Single-video mode: override output JSONL path.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel per-video processes.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing/invalid data.")
    parser.add_argument("--min-samples", type=int, default=1, help="Min pooled rows per segment.")
    parser.add_argument(
        "--metadata-dir",
        help="If set, require coverage vs <video>_segments.csv (strict: fail; else warn).",
    )
    args = parser.parse_args()

    artifacts_root = (REPO_ROOT / args.artifacts_root).resolve()
    metadata_dir = str(Path(args.metadata_dir).resolve()) if args.metadata_dir else None

    if args.video:
        stats = aggregate_video(
            artifacts_root=artifacts_root,
            algo_short=args.algo_short,
            video=args.video,
            run_glob=args.run_glob,
            output=Path(args.output).resolve() if args.output else None,
            strict=args.strict,
            min_samples=args.min_samples,
            metadata_dir=metadata_dir,
        )
        print(
            "Wrote {segments_written} segments for {video} to {output} "
            "(files={files}, rows={rows_deduped}, skipped={skipped_rows}, min_samples={min_samples})".format(
                **stats
            )
        )
        return

    videos = discover_videos(artifacts_root, args.algo_short)
    if not videos:
        raise SystemExit("No videos found under {}".format(artifacts_root / args.algo_short))

    payloads = [
        (
            str(artifacts_root),
            args.algo_short,
            video,
            args.run_glob,
            metadata_dir,
            args.strict,
            int(args.min_samples),
        )
        for video in videos
    ]

    total_segments = 0
    failures = 0

    if args.workers <= 1:
        for payload in payloads:
            video = payload[2]
            try:
                stats = _video_worker(payload)
                total_segments += int(stats.get("segments_written", 0))
                print(
                    "OK {video}: {segments_written} segments (rows={rows_deduped})".format(**stats)
                )
            except Exception as exc:
                failures += 1
                print("FAIL {}: {}".format(video, exc), file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(_video_worker, p): p[2] for p in payloads}
            for future in as_completed(futures):
                video = futures[future]
                try:
                    stats = future.result()
                    total_segments += int(stats.get("segments_written", 0))
                    print(
                        "OK {video}: {segments_written} segments (rows={rows_deduped})".format(
                            **stats
                        )
                    )
                except Exception as exc:
                    failures += 1
                    print("FAIL {}: {}".format(video, exc), file=sys.stderr)

    print(
        "Batch done: {} videos, {} segments, {} failures".format(
            len(videos), total_segments, failures
        )
    )
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

