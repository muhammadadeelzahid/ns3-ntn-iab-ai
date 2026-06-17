#!/usr/bin/env python3
"""
Build a Phase-1 ABR dataset from ns-3 run logs + metadata CSV, or from trace_decisions.jsonl.

Single run:
  python3 scratch/ai_llm/dataset_builder.py \\
    --run-dir Quic_artifacts/Fdash/hiking/run_1 \\
    --metadata .../hiking_segments.csv \\
    --output train_jsonl/hiking_run_1.jsonl \\
    --run-id run_1 --teacher-source oracle \\
    --oracle-labels Quic_artifacts/Fdash/hiking --strict

Catalog batch (all videos × runs under an algo):
  python3 scratch/ai_llm/dataset_builder.py \\
    --artifacts-root Quic_artifacts --algo-short Fdash \\
    --metadata-dir /path/to/metadata --output-dir train_jsonl \\
    --teacher-source oracle --strict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from ai_llm.decision_parser import parse_decisions
    from ai_llm.extract_trace import build_records, write_jsonl
    from ai_llm.features import build_model_input
    from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
    from ai_llm.trace_schema import ORACLE_LABELS_STATE_REQUIRED_FIELDS, SCHEMA_VERSION
except ImportError:
    from decision_parser import parse_decisions
    from extract_trace import build_records, write_jsonl
    from features import build_model_input
    from metadata_loader import LADDER_LEVELS, load_metadata_index
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
    from trace_schema import ORACLE_LABELS_STATE_REQUIRED_FIELDS, SCHEMA_VERSION


def _bitrate_to_class(bps: int) -> int:
    for idx, rate in enumerate(NS3_RATE_LADDER_BPS):
        if int(bps) == rate:
            return idx
    return min(
        range(len(NS3_RATE_LADDER_BPS)),
        key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - int(bps)),
    )


def _derive_video_name(metadata_csv: str, override: Optional[str]) -> str:
    if override:
        return override
    stem = Path(metadata_csv).stem
    if stem.endswith("_segments"):
        stem = stem[: -len("_segments")]
    return stem


def _resolve_oracle_files(oracle_path: Path) -> List[Path]:
    """Locate oracle_labels_state.jsonl under a file or directory path."""
    if oracle_path.is_file():
        return [oracle_path]
    if oracle_path.is_dir():
        files = sorted(oracle_path.glob("**/oracle_labels_state.jsonl"))
        if files:
            return files
        raise RuntimeError(
            "No oracle_labels_state.jsonl under {}".format(oracle_path)
        )
    raise RuntimeError("oracle path not found: {}".format(oracle_path))


def load_oracle_labels(
    oracle_path: Path,
    video_name: Optional[str],
) -> Dict[Tuple[str, int, int], Dict]:
    """Load oracle rows keyed by (run_id, node_id, segment_id).

    `oracle_path` may be a single .jsonl file or a directory containing per-video
    oracle_labels_state.jsonl files (e.g. <artifacts>/<algo>/<video>/oracle_labels_state.jsonl).
    """
    files = _resolve_oracle_files(oracle_path)
    table: Dict[Tuple[str, int, int], Dict] = {}
    for path in files:
        with open(str(path), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for field in ORACLE_LABELS_STATE_REQUIRED_FIELDS:
                    if field not in row:
                        raise RuntimeError(
                            "Oracle row missing {}: {}".format(field, path)
                        )
                key = (
                    str(row["run_id"]),
                    int(row["node_id"]),
                    int(row["segment_id"]),
                )
                table[key] = row

    if video_name and not table:
        raise RuntimeError(
            "No oracle rows found for video {} in {}".format(video_name, oracle_path)
        )
    return table


def attach_oracle_teacher(
    records: List[Dict],
    oracle_table: Dict[Tuple[str, int, int], Dict],
    video_name: str,
    strict: bool,
) -> Dict[str, object]:
    """Replace teacher block with oracle labels. Returns join diagnostics."""
    missing = 0
    attached = 0
    missing_keys: List[Tuple] = []

    for record in records:
        record["video_name"] = video_name
        key = (
            str(record["run_id"]),
            int(record["node_id"]),
            int(record["segment_id"]),
        )

        oracle = oracle_table.get(key)
        if oracle is None:
            missing += 1
            if len(missing_keys) < 5:
                missing_keys.append(key)
            if strict:
                raise RuntimeError("No oracle row for key {}".format(key))
            continue

        chosen_bps = int(oracle["oracle_bitrate_bps"])
        record["teacher"] = {
            "name": "dp_oracle",
            "chosen_bitrate_bps": chosen_bps,
            "chosen_bitrate_class": _bitrate_to_class(chosen_bps),
            "oracle_bitrate_idx": int(oracle["oracle_bitrate_idx"]),
            "delay_s": 0.0,
            "oracle_policy": oracle.get("oracle_policy", ""),
            "immediate_reward": float(oracle.get("immediate_reward", 0.0)),
            "future_value": float(oracle.get("future_value", 0.0)),
            "total_q_value": float(oracle.get("total_q_value", 0.0)),
            "predicted_buffer_before_s": float(oracle.get("buffer_before_s", 0.0)),
            "predicted_buffer_after_s": float(oracle.get("buffer_after_s", 0.0)),
            "predicted_rebuffer_s": float(oracle.get("predicted_rebuffer_s", 0.0)),
            "switch_penalty": float(oracle.get("switch_penalty", 0.0)),
            "semantic_weight": float(oracle.get("semantic_weight", 0.0)),
            "is_scene_cut": int(oracle.get("is_scene_cut", 0)),
            "vmaf_chosen": float(oracle.get("vmaf_chosen", 0.0)),
            "throughput_bps_replay": float(oracle.get("throughput_bps", 0.0)),
        }
        attached += 1

    return {
        "attached": attached,
        "missing": missing,
        "missing_keys": missing_keys,
    }


class DecisionState:
    def __init__(self, decision: Dict[str, float]) -> None:
        self.throughput_estimate_bps = decision["throughput_estimate_bps"]
        self.buffer_level_s = decision["buffer_level_s"]
        self.buffer_differential_s = decision["buffer_differential_s"]
        self.segment_fetch_time_s = decision.get("segment_fetch_time_s", 0.0)
        self.smoothed_rtt_ms = decision.get("smoothed_rtt_ms", -1.0)
        self.loss_rate_recent = decision.get("loss_rate_recent", -1.0)
        # Actual encoded bitrate (VBR) of the last downloaded segment, not the nominal rung.
        self.last_segment_bitrate_bps = decision.get("last_segment_bitrate_bps", 0.0)


def build_records_from_log(
    decisions: List[Dict],
    metadata_csv: str,
    scaler_json: Optional[str],
    run_id: str,
    scenario: str,
) -> List[Dict]:
    metadata = load_metadata_index(metadata_csv, scaler_json)
    records: List[Dict] = []
    per_node_index: Dict[int, int] = {}

    for decision in decisions:
        segment_id = decision.get("segment_id")
        if segment_id is None:
            continue
        segment_meta = metadata.get(segment_id)
        if segment_meta is None:
            continue

        node_id = decision["node_id"]
        per_node_index[node_id] = per_node_index.get(node_id, 0) + 1

        # Actual encoded bitrate (VBR) of the just-downloaded segment (segment_id - 1
        # at the old rung), not the nominal rung restatement. 0.0 at startup.
        last_segment_bitrate_bps = 0.0
        prev_meta = metadata.get(segment_id - 1)
        if prev_meta is not None:
            old_bps = decision["old_bitrate_bps"]
            old_rung = min(
                range(len(NS3_RATE_LADDER_BPS)),
                key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - old_bps),
            )
            level = LADDER_LEVELS[old_rung]
            last_segment_bitrate_bps = float(
                prev_meta.actual_bitrate_kbps.get(level, 0.0)
            ) * 1000.0
        decision["last_segment_bitrate_bps"] = last_segment_bitrate_bps

        record = {
            "run_id": run_id,
            "scenario": scenario,
            "decision_index": per_node_index[node_id],
            "node_id": node_id,
            "segment_id": segment_id,
            "sim_time_s": decision["sim_time_s"],
            "schema_version": SCHEMA_VERSION,
            "observation": {
                "throughput_estimate_bps": decision["throughput_estimate_bps"],
                "buffer_level_s": decision["buffer_level_s"],
                "buffer_differential_s": decision["buffer_differential_s"],
                "last_segment_bitrate_bps": last_segment_bitrate_bps,
                "player_state": 1,
                "interruption_time_s": decision["interruption_time_s"],
                "interruptions": 0,
            },
            "semantic_raw": segment_meta.semantic_raw,
            "ladder": {
                "levels": LADDER_LEVELS,
                "size_bytes": segment_meta.size_bytes,
                "actual_bitrate_kbps": segment_meta.actual_bitrate_kbps,
                "vmaf": segment_meta.vmaf,
            },
            "teacher": {
                "name": "fdash_native_or_ai_log",
                "chosen_bitrate_bps": decision["new_bitrate_bps"],
                "delay_s": decision["delay_s"],
            },
        }
        state = DecisionState(decision)
        # Schema v2 (raw): build ladder_context + masks scaler-free; no baked norms.
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
    return records


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


def build_dataset_for_run(
    run_dir: Path,
    metadata_csv: Path,
    output_path: Path,
    run_id: str,
    semantic_scaler: Optional[str],
    scenario: str,
    teacher_source: str,
    oracle_table: Optional[Dict[Tuple[str, int, int], Dict]],
    video_name: str,
    strict: bool,
    network_scaler: Optional[str] = None,
) -> Dict[str, object]:
    """Build one run JSONL. Returns summary stats."""
    if not (run_dir / "trace_decisions.jsonl").is_file() and not (run_dir / "sim.log").is_file():
        return {"skipped": True, "reason": "no trace_decisions.jsonl or sim.log"}

    records = build_records(
        run_dir,
        str(metadata_csv),
        run_id,
        semantic_scaler,
        network_scaler,
    )
    if not records:
        return {"skipped": True, "reason": "no records produced"}

    join_diag: Dict[str, object] = {}
    if teacher_source == "oracle":
        if oracle_table is None:
            raise RuntimeError("oracle_table required when teacher_source=oracle")
        join_diag = attach_oracle_teacher(
            records=records,
            oracle_table=oracle_table,
            video_name=video_name,
            strict=strict,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_path)
    return {
        "skipped": False,
        "records": len(records),
        "output": str(output_path),
        **join_diag,
    }


def _file_sha256(path: Optional[str]) -> Optional[str]:
    import hashlib
    import os

    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_dataset_manifest(
    output_dir: Path,
    rows: int,
    network_scaler: Optional[str],
    semantic_scaler: Optional[str],
) -> None:
    """Record schema version + the scalers the loader/inference should apply.

    The dataset is raw (schema v2): normalization happens at load time via these scalers,
    so pinning them here makes a dataset traceable to its exact preprocessing.
    """
    from datetime import datetime, timezone

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "normalization": "applied at load time from raw observation/semantic_raw "
        "(features.build_model_input_from_row); not baked into shards",
        "network_scaler": network_scaler,
        "network_scaler_sha256": _file_sha256(network_scaler),
        "semantic_scaler": semantic_scaler,
        "semantic_scaler_sha256": _file_sha256(semantic_scaler),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("Wrote dataset manifest -> {}".format(path))


def merge_jsonl_shards(output_dir: Path, merge_path: Path) -> int:
    """Concatenate shard JSONL files in output_dir into merge_path (excludes merge_path itself)."""
    merge_resolved = merge_path.resolve()
    shard_paths = sorted(
        p for p in output_dir.glob("*.jsonl") if p.resolve() != merge_resolved
    )
    if not shard_paths:
        return 0
    merge_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(merge_path, "w", encoding="utf-8") as out_handle:
        for shard in shard_paths:
            with open(shard, "r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    if not line.strip():
                        continue
                    out_handle.write(line if line.endswith("\n") else line + "\n")
                    total += 1
    return total


def build_catalog(
    artifacts_root: Path,
    algo_short: str,
    metadata_dir: Path,
    output_dir: Path,
    semantic_scaler: Optional[str],
    scenario: str,
    teacher_source: str,
    strict: bool,
    video: Optional[str],
    run_glob: str,
    merge_output: Optional[Path] = None,
    network_scaler: Optional[str] = None,
) -> Dict[str, object]:
    """Build JSONL for every run under artifacts_root/algo_short."""
    oracle_cache: Dict[str, Dict[Tuple[str, int, int], Dict]] = {}
    written = 0
    skipped = 0
    total_records = 0
    failures = 0
    missing_total = 0

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

        run_id = run_dir.name
        output_path = output_dir / "{}_{}.jsonl".format(video_name, run_id)

        oracle_table: Optional[Dict[Tuple[str, int, int], Dict]] = None
        if teacher_source == "oracle":
            if video_name not in oracle_cache:
                oracle_path = artifacts_root / algo_short / video_name
                oracle_cache[video_name] = load_oracle_labels(oracle_path, video_name)
            oracle_table = oracle_cache[video_name]

        try:
            stats = build_dataset_for_run(
                run_dir=run_dir,
                metadata_csv=metadata_csv,
                output_path=output_path,
                run_id=run_id,
                semantic_scaler=semantic_scaler,
                scenario=scenario,
                teacher_source=teacher_source,
                oracle_table=oracle_table,
                video_name=video_name,
                strict=strict,
                network_scaler=network_scaler,
            )
        except Exception as exc:
            failures += 1
            print(
                "FAIL {} {}: {}".format(video_name, run_id, exc),
                file=sys.stderr,
            )
            if strict:
                raise
            continue

        if stats.get("skipped"):
            skipped += 1
            print(
                "SKIP {} {}: {}".format(video_name, run_id, stats.get("reason", "")),
                file=sys.stderr,
            )
            continue

        written += 1
        total_records += int(stats.get("records", 0))
        missing_total += int(stats.get("missing", 0))
        if teacher_source == "oracle":
            print(
                "OK {video} {run}: {n} rows -> {out} "
                "(oracle attached={att} missing={mis})".format(
                    video=video_name,
                    run=run_id,
                    n=stats["records"],
                    out=stats["output"],
                    att=stats.get("attached", 0),
                    mis=stats.get("missing", 0),
                )
            )
        else:
            print(
                "OK {video} {run}: {n} rows -> {out}".format(
                    video=video_name,
                    run=run_id,
                    n=stats["records"],
                    out=stats["output"],
                )
            )

    summary = {
        "runs_written": written,
        "runs_skipped": skipped,
        "runs_failed": failures,
        "total_records": total_records,
        "oracle_missing_total": missing_total,
        "output_dir": str(output_dir),
    }
    print(
        "Catalog done: {runs_written} files, {total_records} records, "
        "{runs_skipped} skipped, {runs_failed} failures -> {output_dir}".format(
            **summary
        )
    )
    if merge_output is not None:
        merged_rows = merge_jsonl_shards(output_dir, merge_output)
        summary["merge_output"] = str(merge_output)
        summary["merge_rows"] = merged_rows
        print("Merged {} rows into {}".format(merged_rows, merge_output))

    # Pin schema + the scalers the loader/inference will apply (dataset is raw/v2).
    _write_dataset_manifest(
        output_dir,
        total_records,
        DEFAULT_NETWORK_SCALER,
        semantic_scaler or DEFAULT_SEMANTIC_SCALER,
    )

    if failures and strict:
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ABR JSONL dataset from run logs and metadata (single run or catalog batch)."
    )
    parser.add_argument("--trace-jsonl", help="Use pre-extracted trace_decisions.jsonl (copy/merge mode).")
    parser.add_argument("--run-dir", help="Run directory for extract_trace path (single-run mode).")
    parser.add_argument("--log", help="Path to ns-3 run log containing newBitRate lines.")
    parser.add_argument("--metadata", help="Path to pruned metadata CSV (single-run mode).")
    parser.add_argument(
        "--output",
        help="Output JSONL path (single-run mode). Omit when using --output-dir for catalog batch.",
    )
    parser.add_argument(
        "--artifacts-root",
        help="Catalog batch: artifact root (e.g. Quic_artifacts). Enables batch mode with --output-dir.",
    )
    parser.add_argument("--algo-short", default="Fdash", help="Catalog batch: algo folder under artifacts-root.")
    parser.add_argument(
        "--metadata-dir",
        help="Catalog batch: directory of <video>_segments.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="train_jsonl",
        help="Catalog batch: write <video>_<run_id>.jsonl here (default: train_jsonl).",
    )
    parser.add_argument(
        "--video",
        help="Catalog batch: only this video (default: all videos under algo).",
    )
    parser.add_argument(
        "--run-glob",
        default="run_*",
        help="Catalog batch: glob for run directories under each video.",
    )
    parser.add_argument(
        "--merge-output",
        help="Catalog batch: concatenate all shard JSONL files into this path "
        "(e.g. train_jsonl/all_runs.jsonl).",
    )
    parser.add_argument(
        "--semantic-scaler",
        "--scaler",
        dest="semantic_scaler",
        default=DEFAULT_SEMANTIC_SCALER,
    )
    parser.add_argument(
        "--network-scaler",
        dest="network_scaler",
        default=DEFAULT_NETWORK_SCALER,
        help="DEPRECATED / ignored. Schema v2 datasets are raw; normalization is applied "
        "at load time (train.py --network-scaler) and inference, not baked here. Kept for "
        "backward-compatible CLI invocations.",
    )
    parser.add_argument("--run-id", default="run-unknown")
    parser.add_argument("--scenario", default="ntn-iab-quic-dash-ai")
    parser.add_argument(
        "--teacher-source",
        choices=["fdash", "oracle"],
        default="fdash",
        help="Source of teacher labels: native FDASH decisions or dp_oracle.py labels.",
    )
    parser.add_argument(
        "--oracle-labels",
        help="Path to oracle_labels_state.jsonl or directory containing per-video oracle files.",
    )
    parser.add_argument(
        "--video-name",
        help="Video name for oracle join (default: derived from metadata CSV stem).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if oracle is missing any (run_id, node_id, segment_id) join key.",
    )
    args = parser.parse_args()

    if args.artifacts_root:
        if not args.metadata_dir:
            raise RuntimeError("Catalog batch requires --metadata-dir")
        merge_path = Path(args.merge_output).resolve() if args.merge_output else None
        build_catalog(
            artifacts_root=Path(args.artifacts_root).resolve(),
            algo_short=args.algo_short,
            metadata_dir=Path(args.metadata_dir).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            semantic_scaler=args.semantic_scaler,
            scenario=args.scenario,
            teacher_source=args.teacher_source,
            strict=args.strict,
            video=args.video,
            run_glob=args.run_glob,
            merge_output=merge_path,
            network_scaler=(args.network_scaler or None),
        )
        return

    if args.trace_jsonl:
        src = Path(args.trace_jsonl)
        dst = Path(args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        count = sum(1 for line in src.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"Copied {count} records from {src} to {dst}")
        return

    if not args.output:
        raise RuntimeError("Single-run mode requires --output (or use --artifacts-root for catalog batch).")

    if args.run_dir and args.metadata:
        video_name = _derive_video_name(args.metadata, args.video_name)
        oracle_table: Optional[Dict[Tuple[str, int, int], Dict]] = None
        if args.teacher_source == "oracle":
            if not args.oracle_labels:
                raise RuntimeError("--teacher-source oracle requires --oracle-labels")
            oracle_table = load_oracle_labels(
                Path(args.oracle_labels).resolve(),
                video_name,
            )
        stats = build_dataset_for_run(
            run_dir=Path(args.run_dir),
            metadata_csv=Path(args.metadata),
            output_path=Path(args.output),
            run_id=args.run_id,
            semantic_scaler=args.semantic_scaler,
            scenario=args.scenario,
            teacher_source=args.teacher_source,
            oracle_table=oracle_table,
            video_name=video_name,
            strict=args.strict,
            network_scaler=(args.network_scaler or None),
        )
        if stats.get("skipped"):
            raise RuntimeError("Skipped run: {}".format(stats.get("reason", "")))
        if args.teacher_source == "oracle":
            print(
                "Oracle join: attached={att} missing={mis} (e.g. {sample})".format(
                    att=stats.get("attached", 0),
                    mis=stats.get("missing", 0),
                    sample=stats.get("missing_keys", [])[:3],
                )
            )
        print("Wrote {} records to {}".format(stats["records"], args.output))
        return
    elif args.log and args.metadata:
        decisions = parse_decisions(args.log)
        if not decisions:
            raise RuntimeError("No ABR decisions found in the provided log file.")
        records = build_records_from_log(
            decisions, args.metadata, args.semantic_scaler, args.run_id, args.scenario
        )
    else:
        raise RuntimeError("Provide --trace-jsonl, or (--run-dir + --metadata), or (--log + --metadata).")

    if not records:
        raise RuntimeError("No records were produced.")

    if args.teacher_source == "oracle":
        if not args.oracle_labels:
            raise RuntimeError("--teacher-source oracle requires --oracle-labels")
        if not args.metadata:
            raise RuntimeError("--teacher-source oracle requires --metadata for video-name derivation")
        video_name = _derive_video_name(args.metadata, args.video_name)
        oracle_table = load_oracle_labels(
            Path(args.oracle_labels).resolve(),
            video_name,
        )
        diag = attach_oracle_teacher(
            records=records,
            oracle_table=oracle_table,
            video_name=video_name,
            strict=args.strict,
        )
        print(
            "Oracle join: attached={att} missing={mis} (e.g. {sample})".format(
                att=diag["attached"],
                mis=diag["missing"],
                sample=diag["missing_keys"][:3],
            )
        )

    write_jsonl(records, Path(args.output))
    print("Wrote {} records to {}".format(len(records), args.output))


if __name__ == "__main__":
    main()
