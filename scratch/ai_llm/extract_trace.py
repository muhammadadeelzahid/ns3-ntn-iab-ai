#!/usr/bin/env python3
"""
Extract per-decision training records from a single ns-3 run directory.

Reads sim.log, optional DASH/QUIC ascii traces, and metadata CSV; writes trace_decisions.jsonl.
"""

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from ai_llm.decision_parser import parse_decisions
    from ai_llm.features import build_model_input
    from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index
    from ai_llm.paths import (
        DEFAULT_NETWORK_SCALER,
        DEFAULT_SEMANTIC_SCALER,
        NS3_RATE_LADDER_BPS,
    )
    from ai_llm.trace_schema import SCHEMA_VERSION
except ImportError:
    from decision_parser import parse_decisions
    from features import build_model_input
    from metadata_loader import LADDER_LEVELS, load_metadata_index
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
    from trace_schema import SCHEMA_VERSION

MPEG_FRAMES_PER_SEGMENT = 100
LADDER_BPS_SET = set(NS3_RATE_LADDER_BPS)


class DecisionState:
    def __init__(self, decision: Dict) -> None:
        self.throughput_estimate_bps = decision["throughput_estimate_bps"]
        self.buffer_level_s = decision["buffer_level_s"]
        self.buffer_differential_s = decision["buffer_differential_s"]
        self.segment_fetch_time_s = decision.get("segment_fetch_time_s", 0.0)
        self.smoothed_rtt_ms = decision.get("smoothed_rtt_ms", -1.0)
        self.loss_rate_recent = decision.get("loss_rate_recent", -1.0)
        # Actual encoded bitrate of the just-downloaded segment (VBR), not the nominal
        # rung. Computed in build_records and attached to the decision dict; 0.0 at
        # startup (no previously downloaded segment).
        self.last_segment_bitrate_bps = decision.get("last_segment_bitrate_bps", 0.0)


def find_sim_log(run_dir: Path) -> Path:
    for name in ("sim.log", "fdash_meta.log"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    matches = sorted(run_dir.glob("*.log"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No sim.log in {run_dir}")


def load_run_manifest(run_dir: Path) -> Dict:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_server_trace(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not csv_path.is_file():
        return rows
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_time = (row.get("sim_time_s") or "").strip()
            if not raw_time or raw_time == "sim_time_s":
                continue
            try:
                sim_time_s = float(raw_time)
            except ValueError:
                continue
            rows.append(
                {
                    "sim_time_s": sim_time_s,
                    "video_id": int(row["video_id"]),
                    "segment_id": int(row["segment_id"]),
                    "bitrate_bps": int(row["bitrate_bps"]),
                    "target_bytes": int(row.get("target_bytes", 0)),
                    "total_bytes": int(row.get("total_bytes", 0)),
                    "source": row.get("source", ""),
                }
            )
    return rows


def parse_dash_client_rx(path: Path) -> List[Tuple[float, int]]:
    """Return list of (time_s, packet_size_bytes)."""
    samples: List[Tuple[float, int]] = []
    if not path.is_file():
        return samples
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
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


def parse_quic_rtt(path: Path) -> List[Tuple[float, float]]:
    """Return list of (time_s, rtt_ms) from QUIC RTT trace."""
    samples: List[Tuple[float, float]] = []
    if not path.is_file():
        return samples
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    # Column 2 is newRtt (smoothed) in seconds; convert to ms.
                    samples.append((float(parts[0]), float(parts[2]) * 1000.0))
                except ValueError:
                    continue
            elif len(parts) >= 2:
                try:
                    samples.append((float(parts[0]), float(parts[1]) * 1000.0))
                except ValueError:
                    continue
    return samples


def _value_at_or_before(series: List[Tuple[float, float]], t: float) -> Optional[float]:
    best: Optional[float] = None
    for ts, val in series:
        if ts <= t:
            best = val
        else:
            break
    return best


def estimate_segment_fetch_time(rx_samples: List[Tuple[float, int]], segment_id: int) -> float:
    """Approximate fetch duration from frame arrival gaps (100 frames per segment)."""
    if not rx_samples:
        return 0.0
    start_frame = segment_id * MPEG_FRAMES_PER_SEGMENT
    end_frame = start_frame + MPEG_FRAMES_PER_SEGMENT - 1
    if end_frame >= len(rx_samples):
        return 0.0
    t0 = rx_samples[start_frame][0]
    t1 = rx_samples[end_frame][0]
    return max(0.0, t1 - t0)


def find_rx_file(run_dir: Path, node_id: int) -> Optional[Path]:
    candidates = [
        run_dir / f"DashClientRx_Node_{node_id}.txt",
        run_dir / f"clientQUIC-rx-data{node_id}.txt",
        run_dir / f"client/client{node_id}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_rtt_file(run_dir: Path, node_id: int) -> Optional[Path]:
    candidates = [
        run_dir / f"clientQUIC-rtt{node_id}.txt",
        run_dir / f"client/client{node_id}QUIC-RTT{node_id}.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def bitrate_to_class(bps: int) -> int:
    for idx, rate in enumerate(NS3_RATE_LADDER_BPS):
        if bps == rate:
            return idx
    # Nearest ladder rung
    return min(range(len(NS3_RATE_LADDER_BPS)), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))


def build_records(
    run_dir: Path,
    metadata_csv: str,
    run_id: str,
    semantic_scaler: Optional[str],
    network_scaler: Optional[str],
) -> List[Dict]:
    sim_log = find_sim_log(run_dir)
    manifest = load_run_manifest(run_dir)
    metadata = load_metadata_index(metadata_csv, semantic_scaler)

    decisions = parse_decisions(str(sim_log))

    # Cache per-node auxiliary traces
    rx_by_node: Dict[int, List[Tuple[float, int]]] = {}
    rtt_by_node: Dict[int, List[Tuple[float, float]]] = {}

    records: List[Dict] = []
    per_node_index: Dict[int, int] = {}

    for decision in decisions:
        if decision["new_bitrate_bps"] not in LADDER_BPS_SET:
            continue

        node_id = decision["node_id"]
        segment_id = decision["segment_id"]
        if segment_id is None:
            continue

        segment_meta = metadata.get(segment_id)
        if segment_meta is None:
            continue

        if node_id not in rx_by_node:
            rx_path = find_rx_file(run_dir, node_id)
            rx_by_node[node_id] = parse_dash_client_rx(rx_path) if rx_path else []
        if node_id not in rtt_by_node:
            rtt_path = find_rtt_file(run_dir, node_id)
            rtt_by_node[node_id] = parse_quic_rtt(rtt_path) if rtt_path else []

        if "segment_fetch_time_s" in decision:
            fetch_s = float(decision["segment_fetch_time_s"])
        else:
            fetch_s = estimate_segment_fetch_time(rx_by_node[node_id], segment_id)
            decision["segment_fetch_time_s"] = fetch_s
        rtt = _value_at_or_before(rtt_by_node[node_id], decision["sim_time_s"])
        decision["smoothed_rtt_ms"] = rtt if rtt is not None else -1.0
        if "loss_rate_recent" not in decision:
            decision["loss_rate_recent"] = -1.0

        # Actual encoded bitrate of the segment that just finished downloading.
        # ffmpeg VBR makes this differ from the nominal rung (old_bitrate_bps): a
        # segment's real bytes depend on scene complexity, so this is the network-load
        # signal the policy needs, not the rung restatement. The completed segment is
        # the previous one for this node (segment_id - 1) at the rung given by
        # old_bitrate_bps. 0.0 at startup (no previously downloaded segment).
        last_segment_bitrate_bps = 0.0
        prev_meta = metadata.get(segment_id - 1)
        if prev_meta is not None:
            old_rung = bitrate_to_class(decision["old_bitrate_bps"])
            level = LADDER_LEVELS[old_rung]
            last_segment_bitrate_bps = float(
                prev_meta.actual_bitrate_kbps.get(level, 0.0)
            ) * 1000.0
        decision["last_segment_bitrate_bps"] = last_segment_bitrate_bps

        per_node_index[node_id] = per_node_index.get(node_id, 0) + 1
        decision_index = per_node_index[node_id]

        video_id = decision.get("video_id")
        if video_id is None:
            video_id = manifest.get("video_id", 0)

        chosen_bps = decision["new_bitrate_bps"]
        record = {
            "run_id": run_id,
            "node_id": node_id,
            "video_id": video_id,
            "segment_id": segment_id,
            "sim_time_s": decision["sim_time_s"],
            "decision_index": decision_index,
            "schema_version": SCHEMA_VERSION,
            "observation": {
                "throughput_estimate_bps": decision["throughput_estimate_bps"],
                "buffer_level_s": decision["buffer_level_s"],
                "buffer_differential_s": decision["buffer_differential_s"],
                "segment_fetch_time_s": fetch_s,
                "last_segment_bitrate_bps": last_segment_bitrate_bps,
                "smoothed_rtt_ms": decision["smoothed_rtt_ms"],
                "loss_rate_recent": decision["loss_rate_recent"],
            },
            "semantic_raw": segment_meta.semantic_raw,
            "ladder": {
                "levels": LADDER_LEVELS,
                "size_bytes": segment_meta.size_bytes,
                "actual_bitrate_kbps": segment_meta.actual_bitrate_kbps,
                "vmaf": segment_meta.vmaf,
            },
            "teacher": {
                "name": "fdash",
                "chosen_bitrate_bps": chosen_bps,
                "chosen_bitrate_class": bitrate_to_class(chosen_bps),
                "delay_s": decision["delay_s"],
            },
            "trace_meta": {
                "segment_fetch_time_s": fetch_s,
                "server_source": "metadata",
                "reject_invalid_nearby": False,
            },
        }

        state = DecisionState(decision)
        # Schema v2 (raw): do NOT bake normalized vectors into the dataset. Build the raw
        # ladder_context ([size_bytes, vmaf]) and missing_masks scaler-free; network_norm
        # and semantic_norm are derived at load time / inference from `observation` and
        # `semantic_raw` via the pinned scaler (features.build_model_input_from_row).
        # `network_scaler` is accepted for signature compatibility but no longer used here.
        model_input = build_model_input(
            state=state,
            segment_meta=segment_meta,
            network_scaler_path=None,
            semantic_scaler_path=None,
        )
        record["model_input"] = {
            "ladder_context": model_input.ladder_context,
            "missing_masks": model_input.missing_masks,
        }
        records.append(record)

    records, dropped = dedupe_records_by_segment(records)
    if dropped > 0:
        print(
            "Deduped {} duplicate (node_id, segment_id) rows (kept latest sim_time_s)".format(
                dropped
            )
        )
    return records


def dedupe_records_by_segment(records: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Keep one record per (node_id, segment_id).

    ns-3 can log multiple ABR decisions for the same completed segment (watchdog
    retries, extra frame-99 receives). Keep the latest by sim_time_s.
    """
    best: Dict[Tuple[int, int], Dict] = {}
    for record in records:
        key = (int(record["node_id"]), int(record["segment_id"]))
        prev = best.get(key)
        if prev is None or float(record["sim_time_s"]) >= float(prev["sim_time_s"]):
            best[key] = record

    deduped = sorted(
        best.values(),
        key=lambda r: (int(r["node_id"]), int(r["segment_id"]), float(r["sim_time_s"])),
    )
    per_node_index: Dict[int, int] = {}
    for record in deduped:
        node_id = int(record["node_id"])
        per_node_index[node_id] = per_node_index.get(node_id, 0) + 1
        record["decision_index"] = per_node_index[node_id]

    dropped = len(records) - len(deduped)
    return deduped, dropped


def write_jsonl(records: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def discover_run_dirs(
    artifacts_root: Path,
    algo_short: str,
    video: Optional[str] = None,
    run_glob: str = "run_*",
) -> Iterator[Tuple[str, Path]]:
    """Yield (video_name, run_dir) under <artifacts>/<algo>/<video>/run_*."""
    base = artifacts_root / algo_short
    if not base.is_dir():
        return
    if video:
        video_dirs = [base / video]
    else:
        video_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        for run_dir in sorted(video_dir.glob(run_glob)):
            if run_dir.is_dir():
                yield video_dir.name, run_dir


def process_run(
    run_dir: Path,
    metadata_csv: Path,
    output_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    semantic_scaler: Optional[str] = None,
    network_scaler: Optional[str] = None,
) -> Dict[str, object]:
    """Extract trace_decisions.jsonl for one run directory."""
    if not (run_dir / "sim.log").is_file() and not (run_dir / "fdash_meta.log").is_file():
        raise FileNotFoundError("no sim.log in {}".format(run_dir))
    if not metadata_csv.is_file():
        raise FileNotFoundError("metadata not found: {}".format(metadata_csv))

    out = output_path or (run_dir / "trace_decisions.jsonl")
    rid = run_id or run_dir.name
    records = build_records(
        run_dir,
        str(metadata_csv),
        rid,
        semantic_scaler,
        network_scaler,
    )
    if not records:
        raise RuntimeError("no records extracted from {}".format(run_dir))

    write_jsonl(records, out)
    return {
        "run_dir": str(run_dir),
        "output": str(out),
        "records": len(records),
    }


def _batch_worker(payload: Tuple) -> Dict[str, object]:
    (
        run_dir_s,
        metadata_csv_s,
        semantic_scaler,
        network_scaler,
    ) = payload
    return process_run(
        Path(run_dir_s),
        Path(metadata_csv_s),
        semantic_scaler=semantic_scaler,
        network_scaler=network_scaler,
    )


def run_batch(
    artifacts_root: Path,
    algo_short: str,
    metadata_dir: Path,
    video: Optional[str],
    run_glob: str,
    workers: int,
    semantic_scaler: Optional[str],
    network_scaler: Optional[str],
    strict: bool,
) -> Tuple[int, int, int]:
    """Extract trace_decisions.jsonl for all runs under artifacts_root/algo_short."""
    jobs: List[Tuple[str, str, Optional[str], Optional[str]]] = []
    skipped = 0
    for video_name, run_dir in discover_run_dirs(
        artifacts_root, algo_short, video=video, run_glob=run_glob
    ):
        metadata_csv = metadata_dir / "{}_segments.csv".format(video_name)
        if not metadata_csv.is_file():
            skipped += 1
            print(
                "SKIP {} {}: no metadata {}".format(video_name, run_dir.name, metadata_csv),
                file=sys.stderr,
            )
            continue
        if not (run_dir / "sim.log").is_file() and not (run_dir / "fdash_meta.log").is_file():
            skipped += 1
            print(
                "SKIP {} {}: no sim.log".format(video_name, run_dir.name),
                file=sys.stderr,
            )
            continue
        jobs.append(
            (str(run_dir), str(metadata_csv), semantic_scaler, network_scaler)
        )

    if not jobs:
        raise RuntimeError(
            "no runnable jobs under {}/{}".format(artifacts_root, algo_short)
        )

    total_records = 0
    failures = 0
    if workers <= 1:
        for payload in jobs:
            run_dir = Path(payload[0])
            try:
                stats = _batch_worker(payload)
                total_records += int(stats["records"])
                print(
                    "OK {run_dir}: {records} records -> {output}".format(**stats)
                )
            except Exception as exc:
                failures += 1
                print("FAIL {}: {}".format(run_dir, exc), file=sys.stderr)
        if strict and failures:
            raise SystemExit(1)
        return len(jobs) - failures, skipped, total_records

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_batch_worker, p): p[0] for p in jobs}
        for future in as_completed(futures):
            run_dir_s = futures[future]
            try:
                stats = future.result()
                total_records += int(stats["records"])
                print(
                    "OK {run_dir}: {records} records -> {output}".format(**stats)
                )
            except Exception as exc:
                failures += 1
                print("FAIL {}: {}".format(run_dir_s, exc), file=sys.stderr)

    if strict and failures:
        raise SystemExit(1)
    return len(jobs) - failures, skipped, total_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract trace_decisions.jsonl from a run directory.")
    parser.add_argument("--run-dir", help="Path to run_* folder with sim.log and traces.")
    parser.add_argument("--metadata", help="Path to *_segments.csv metadata file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL (default: <run-dir>/trace_decisions.jsonl).",
    )
    parser.add_argument("--run-id", default=None, help="Run id label (default: run directory name).")
    parser.add_argument("--semantic-scaler", default=DEFAULT_SEMANTIC_SCALER)
    parser.add_argument("--network-scaler", default=DEFAULT_NETWORK_SCALER)
    parser.add_argument(
        "--artifacts-root",
        help="Batch mode: artifact root (e.g. Quic_artifacts).",
    )
    parser.add_argument("--algo-short", default="Fdash", help="Batch mode: algo folder under artifacts-root.")
    parser.add_argument(
        "--metadata-dir",
        help="Batch mode: directory of <video>_segments.csv files.",
    )
    parser.add_argument("--video", help="Batch mode: only this video.")
    parser.add_argument("--run-glob", default="run_*", help="Batch mode: run directory glob.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Batch mode: parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Batch mode: exit non-zero if any run fails.",
    )
    args = parser.parse_args()

    if args.artifacts_root:
        if not args.metadata_dir:
            raise RuntimeError("Batch mode requires --metadata-dir")
        written, skipped, total_records = run_batch(
            artifacts_root=Path(args.artifacts_root).resolve(),
            algo_short=args.algo_short,
            metadata_dir=Path(args.metadata_dir).resolve(),
            video=args.video,
            run_glob=args.run_glob,
            workers=args.workers,
            semantic_scaler=args.semantic_scaler,
            network_scaler=args.network_scaler,
            strict=args.strict,
        )
        print(
            "Batch done: {written} runs written, {skipped} skipped, {total_records} total records".format(
                written=written,
                skipped=skipped,
                total_records=total_records,
            )
        )
        return

    if not args.run_dir or not args.metadata:
        raise RuntimeError("Single-run mode requires --run-dir and --metadata (or use --artifacts-root).")

    stats = process_run(
        run_dir=Path(args.run_dir).resolve(),
        metadata_csv=Path(args.metadata).resolve(),
        output_path=Path(args.output).resolve() if args.output else None,
        run_id=args.run_id,
        semantic_scaler=args.semantic_scaler,
        network_scaler=args.network_scaler,
    )
    print(
        "Wrote {records} records to {output}".format(
            records=stats["records"],
            output=stats["output"],
        )
    )


if __name__ == "__main__":
    main()
