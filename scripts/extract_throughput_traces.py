#!/usr/bin/env python3
"""
Extract per-segment transport throughput traces from ns-3 run directories.

Output: throughput_trace.jsonl with one row per (run_id, node_id, segment_id).
Used by aggregate_traces.py (cross-seed collapse) and dp_oracle.py (label generation).

Does not replace extract_trace.py / trace_decisions.jsonl.
"""

from __future__ import print_function

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from ai_llm.decision_parser import parse_decisions  # noqa: E402
from ai_llm.extract_trace import (  # noqa: E402
    find_rx_file,
    find_sim_log,
    load_run_manifest,
    parse_dash_client_rx,
)
from ai_llm.trace_schema import (  # noqa: E402
    COMPLETED_SEGMENT_CONVENTION,
    MPEG_FRAMES_PER_SEGMENT,
    MPEG_SEGMENT_DURATION_S,
    THROUGHPUT_TRACE_OPTIONAL_FIELDS,
    THROUGHPUT_TRACE_REQUIRED_FIELDS,
)

# Default throughput floor is 0: very low throughput is valid data.
# Use --min-throughput-bps > 0 if you want to filter/fail on implausibly low values.
DEFAULT_MIN_THROUGHPUT_BPS = 0.0
DEFAULT_MIN_FETCH_S = 1e-6


def validate_throughput_row(row: Dict) -> Optional[str]:
    """Return error string if row violates v1 schema, else None."""
    for field in THROUGHPUT_TRACE_REQUIRED_FIELDS:
        if field not in row:
            return "missing field: {}".format(field)
    if row["segment_fetch_time_s"] <= 0:
        return "non-positive segment_fetch_time_s"
    if row["segment_bytes"] <= 0:
        return "non-positive segment_bytes"
    if row["throughput_bps"] <= 0:
        return "non-positive throughput_bps"
    return None


def find_tx_file(run_dir: Path, node_id: int) -> Optional[Path]:
    candidates = [
        run_dir / "DashClientTx_Node_{}.txt".format(node_id),
        run_dir / "DashClientTx_Node_{}.txt".format(node_id),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def parse_dash_client_tx(path: Path) -> List[Tuple[float, int]]:
    """Return list of (time_s, packet_size_bytes) for segment request events."""
    samples: List[Tuple[float, int]] = []
    if not path.is_file():
        return samples
    with open(str(path), "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    samples.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    continue
    return samples


def segment_bytes_from_rx(rx_samples: List[Tuple[float, int]], segment_id: int) -> int:
    start = segment_id * MPEG_FRAMES_PER_SEGMENT
    end = start + MPEG_FRAMES_PER_SEGMENT
    if start >= len(rx_samples):
        return 0
    end = min(end, len(rx_samples))
    return sum(size for _, size in rx_samples[start:end])


def resolve_metadata_path(
    run_dir: Path,
    manifest: Dict,
    metadata_arg: Optional[str],
    metadata_dir: Optional[str],
) -> str:
    if metadata_arg:
        return str(Path(metadata_arg).resolve())
    if manifest.get("metadata_csv"):
        return str(Path(manifest["metadata_csv"]).resolve())
    video_name = manifest.get("video_name") or run_dir.parent.name
    if metadata_dir:
        candidate = Path(metadata_dir) / "{}_segments.csv".format(video_name)
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(
        "metadata CSV not found for {}; pass --metadata or --metadata-dir".format(run_dir)
    )


def resolve_run_id(run_dir: Path, manifest: Dict, run_id_arg: Optional[str]) -> str:
    if run_id_arg:
        return run_id_arg
    if manifest.get("run_seed") is not None:
        return "run_{}".format(manifest["run_seed"])
    return run_dir.name


def build_throughput_records(
    run_dir: Path,
    metadata_csv: str,
    run_id: str,
    min_throughput_bps: float,
    min_fetch_s: float,
    strict: bool,
    include_estimate: bool,
) -> Tuple[List[Dict], Dict]:
    sim_log = find_sim_log(run_dir)
    manifest = load_run_manifest(run_dir)
    decisions = parse_decisions(str(sim_log))

    video_name = manifest.get("video_name") or run_dir.parent.name
    algo_short = manifest.get("algo_short", "")

    rx_by_node: Dict[int, List[Tuple[float, int]]] = {}
    tx_by_node: Dict[int, List[Tuple[float, int]]] = {}
    prev_completion: Dict[int, float] = {}

    records: List[Dict] = []
    skipped = 0
    missing_rx = 0

    for decision in sorted(decisions, key=lambda d: (d["node_id"], d.get("segment_id", -1), d["sim_time_s"])):
        segment_id = decision.get("segment_id")
        if segment_id is None:
            skipped += 1
            continue

        node_id = int(decision["node_id"])
        sim_time_s = float(decision["sim_time_s"])

        if node_id not in rx_by_node:
            rx_path = find_rx_file(run_dir, node_id)
            rx_by_node[node_id] = parse_dash_client_rx(rx_path) if rx_path else []
        if node_id not in tx_by_node:
            tx_path = find_tx_file(run_dir, node_id)
            tx_by_node[node_id] = parse_dash_client_tx(tx_path) if tx_path else []

        seg_bytes = segment_bytes_from_rx(rx_by_node[node_id], segment_id)
        if seg_bytes <= 0:
            missing_rx += 1
            if strict:
                raise RuntimeError(
                    "No Rx bytes for node={} segment={} in {}".format(node_id, segment_id, run_dir)
                )
            skipped += 1
            continue

        if node_id in prev_completion:
            fetch_s = max(min_fetch_s, sim_time_s - prev_completion[node_id])
            source = "decision_completion_delta"
        else:
            t0 = None
            if tx_by_node[node_id]:
                t0 = tx_by_node[node_id][0][0]
            elif rx_by_node[node_id]:
                start_frame = segment_id * MPEG_FRAMES_PER_SEGMENT
                if start_frame < len(rx_by_node[node_id]):
                    t0 = rx_by_node[node_id][start_frame][0]
            if t0 is None:
                fetch_s = min_fetch_s
                source = "fallback_min_fetch"
            else:
                fetch_s = max(min_fetch_s, sim_time_s - t0)
                source = "first_completion_delta"

        prev_completion[node_id] = sim_time_s
        throughput_bps = (8.0 * float(seg_bytes)) / fetch_s

        if min_throughput_bps > 0.0 and throughput_bps < min_throughput_bps:
            if strict:
                raise RuntimeError(
                    "Throughput below floor for node={} segment={}: {} bps (floor={})".format(
                        node_id, segment_id, throughput_bps, min_throughput_bps
                    )
                )
            skipped += 1
            continue

        video_id = decision.get("video_id")
        if video_id is None:
            video_id = manifest.get("video_id", 0)

        row = {
            "run_id": run_id,
            "node_id": node_id,
            "segment_id": int(segment_id),
            "throughput_bps": throughput_bps,
            "segment_fetch_time_s": fetch_s,
            "segment_bytes": int(seg_bytes),
            "video_id": int(video_id) if video_id is not None else 0,
            "video_name": video_name,
            "metadata_csv": metadata_csv,
            "sim_time_s": sim_time_s,
            "segment_duration_s": MPEG_SEGMENT_DURATION_S,
            "segment_convention": COMPLETED_SEGMENT_CONVENTION,
            "source": source,
            "algo_short": algo_short,
        }
        if include_estimate:
            row["throughput_estimate_bps"] = float(decision["throughput_estimate_bps"])
            row["chosen_bitrate_bps"] = int(decision["new_bitrate_bps"])
        err = validate_throughput_row(row)
        if err:
            if strict:
                raise RuntimeError(
                    "Invalid row node={} segment={}: {}".format(node_id, segment_id, err)
                )
            skipped += 1
            continue
        records.append(row)

    stats = {
        "decisions_parsed": len(decisions),
        "rows_written": len(records),
        "skipped": skipped,
        "missing_rx": missing_rx,
    }
    if strict and len(records) == 0:
        raise RuntimeError("No throughput rows extracted from {}".format(run_dir))
    return records, stats


def write_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as handle:
        for record in sorted(
            records,
            key=lambda r: (r["node_id"], r["segment_id"], r["sim_time_s"]),
        ):
            handle.write(json.dumps(record))
            handle.write("\n")


def summarize_records(records: List[Dict]) -> Dict:
    if not records:
        return {"min_bps": 0.0, "median_bps": 0.0, "max_bps": 0.0}
    throughputs = sorted(r["throughput_bps"] for r in records)
    n = len(throughputs)
    mid = throughputs[n // 2]
    return {
        "min_bps": throughputs[0],
        "median_bps": mid,
        "max_bps": throughputs[-1],
    }


def process_run(
    run_dir: Path,
    metadata: Optional[str],
    metadata_dir: Optional[str],
    output: Optional[Path],
    run_id: Optional[str],
    min_throughput_bps: float,
    min_fetch_s: float,
    strict: bool,
    include_estimate: bool,
) -> Dict:
    manifest = load_run_manifest(run_dir)
    metadata_csv = resolve_metadata_path(run_dir, manifest, metadata, metadata_dir)
    rid = resolve_run_id(run_dir, manifest, run_id)
    out = output if output else run_dir / "throughput_trace.jsonl"

    records, stats = build_throughput_records(
        run_dir,
        metadata_csv,
        rid,
        min_throughput_bps=min_throughput_bps,
        min_fetch_s=min_fetch_s,
        strict=strict,
        include_estimate=include_estimate,
    )
    write_jsonl(records, out)
    summary = summarize_records(records)
    stats.update(summary)
    stats["output"] = str(out)
    stats["run_dir"] = str(run_dir)
    return stats


def discover_run_dirs(
    artifacts_root: Path,
    algo_short: str,
    video: Optional[str],
    run_glob: str,
) -> List[Path]:
    base = artifacts_root / algo_short
    if video:
        base = base / video
    if not base.is_dir():
        return []
    pattern = str(base / "**" / run_glob)
    import glob as glob_mod

    dirs = []
    for path_str in sorted(glob_mod.glob(pattern, recursive=True)):
        path = Path(path_str)
        if path.is_dir() and (
            (path / "sim.log").is_file() or (path / "fdash_meta.log").is_file()
        ):
            dirs.append(path)
    return dirs


def _batch_worker(payload: Tuple[str, Optional[str], Optional[str], bool, bool, float, float]) -> Dict:
    run_dir_s, metadata_dir, metadata, strict, include_estimate, min_bps, min_fetch = payload
    return process_run(
        Path(run_dir_s),
        metadata=metadata,
        metadata_dir=metadata_dir,
        output=None,
        run_id=None,
        min_throughput_bps=min_bps,
        min_fetch_s=min_fetch,
        strict=strict,
        include_estimate=include_estimate,
    )


def run_batch(
    run_dirs: List[Path],
    metadata_dir: Optional[str],
    workers: int,
    min_throughput_bps: float,
    min_fetch_s: float,
    strict: bool,
    include_estimate: bool,
) -> Tuple[int, int]:
    total_rows = 0
    failures = 0
    payloads = [
        (
            str(run_dir),
            metadata_dir,
            None,
            strict,
            include_estimate,
            min_throughput_bps,
            min_fetch_s,
        )
        for run_dir in run_dirs
    ]

    if workers <= 1:
        for payload in payloads:
            run_dir = Path(payload[0])
            try:
                stats = _batch_worker(payload)
                total_rows += stats["rows_written"]
                print(
                    "OK {run_dir}: {rows_written} rows "
                    "(median {median_bps:.0f} bps)".format(**stats)
                )
            except Exception as exc:
                failures += 1
                print("FAIL {}: {}".format(run_dir, exc), file=sys.stderr)
        return total_rows, failures

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_batch_worker, p): p[0] for p in payloads}
        for future in as_completed(futures):
            run_dir_s = futures[future]
            try:
                stats = future.result()
                total_rows += stats["rows_written"]
                print(
                    "OK {run_dir}: {rows_written} rows "
                    "(median {median_bps:.0f} bps)".format(**stats)
                )
            except Exception as exc:
                failures += 1
                print("FAIL {}: {}".format(run_dir_s, exc), file=sys.stderr)
    return total_rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-segment throughput_trace.jsonl from ns-3 run dirs."
    )
    parser.add_argument("--run-dir", help="Single run_* directory with sim.log and traces.")
    parser.add_argument("--metadata", help="Path to *_segments.csv (default: run_manifest).")
    parser.add_argument(
        "--output",
        help="Output JSONL (default: <run-dir>/throughput_trace.jsonl).",
    )
    parser.add_argument("--run-id", help="Run label (default: run_<seed> or directory name).")
    parser.add_argument(
        "--artifacts-root",
        default="Quic_artifacts",
        help="Batch mode: artifacts root (with --video or all videos).",
    )
    parser.add_argument("--algo-short", default="Fdash", help="Algorithm folder under artifacts root.")
    parser.add_argument("--video", help="Batch mode: only this video name.")
    parser.add_argument("--run-glob", default="run_*", help="Batch mode: run directory glob.")
    parser.add_argument(
        "--metadata-dir",
        help="Batch mode: directory containing <video>_segments.csv files.",
    )
    parser.add_argument(
        "--min-throughput-bps",
        type=float,
        default=DEFAULT_MIN_THROUGHPUT_BPS,
        help="Minimum valid throughput_bps.",
    )
    parser.add_argument(
        "--min-fetch-s",
        type=float,
        default=DEFAULT_MIN_FETCH_S,
        help="Minimum segment_fetch_time_s denominator.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing segments or invalid throughput.",
    )
    parser.add_argument(
        "--include-estimate",
        action="store_true",
        help="Include estBitRate and chosen bitrate from sim.log.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Batch mode: parallel worker processes (default 1).",
    )
    args = parser.parse_args()

    if args.run_dir:
        stats = process_run(
            Path(args.run_dir).resolve(),
            metadata=args.metadata,
            metadata_dir=args.metadata_dir,
            output=Path(args.output).resolve() if args.output else None,
            run_id=args.run_id,
            min_throughput_bps=args.min_throughput_bps,
            min_fetch_s=args.min_fetch_s,
            strict=args.strict,
            include_estimate=args.include_estimate,
        )
        print(
            "Wrote {rows_written} rows to {output} "
            "(decisions={decisions_parsed}, skipped={skipped}, "
            "throughput_bps {min_bps:.0f}-{max_bps:.0f}, median={median_bps:.0f})".format(
                **stats
            )
        )
        return

    root = (REPO_ROOT / args.artifacts_root).resolve()
    if args.metadata_dir:
        metadata_dir = str(Path(args.metadata_dir).resolve())
    else:
        metadata_dir = None

    if args.video:
        run_dirs = discover_run_dirs(root, args.algo_short, args.video, args.run_glob)
    else:
        run_dirs = []
        algo_base = root / args.algo_short
        if algo_base.is_dir():
            for video_dir in sorted(algo_base.iterdir()):
                if video_dir.is_dir():
                    run_dirs.extend(
                        discover_run_dirs(root, args.algo_short, video_dir.name, args.run_glob)
                    )

    if not run_dirs:
        raise SystemExit("No run directories found under {}".format(root / args.algo_short))

    total_rows, failures = run_batch(
        run_dirs,
        metadata_dir=metadata_dir,
        workers=max(1, args.workers),
        min_throughput_bps=args.min_throughput_bps,
        min_fetch_s=args.min_fetch_s,
        strict=args.strict,
        include_estimate=args.include_estimate,
    )

    print(
        "Batch done: {} runs, {} rows, {} failures".format(
            len(run_dirs), total_rows, failures
        )
    )
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
