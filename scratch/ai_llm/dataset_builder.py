#!/usr/bin/env python3
"""
Build a Phase-1 ABR dataset from ns-3 run logs + metadata CSV.

The output is JSONL with one record per segment decision.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from metadata_loader import LADDER_LEVELS, load_metadata_index

try:
    from ai_llm.paths import DEFAULT_SEMANTIC_SCALER
except ImportError:
    from paths import DEFAULT_SEMANTIC_SCALER


DECISION_RE = re.compile(
    r"(?P<sim_time>[0-9eE+.\-]+)\s+ue-id:\s+(?P<node_id>\d+)\s+"
    r"newBitRate:\s+(?P<new_bitrate>\d+)\s+oldBitRate:\s+(?P<old_bitrate>\d+)\s+"
    r"estBitRate:\s+(?P<est_bitrate>[0-9eE+.\-]+)\s+interTime:\s+(?P<inter_time>[0-9eE+.\-]+)\s+"
    r"T:\s+(?P<buffer_level>[0-9eE+.\-]+)\s+dT:\s+(?P<buffer_diff>[0-9eE+.\-]+)\s+"
    r"del:\s+(?P<delay>[0-9eE+.\-]+)"
)


def parse_decisions(log_path: str) -> List[Dict[str, float]]:
    decisions: List[Dict[str, float]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = DECISION_RE.search(line)
            if not match:
                continue
            d = match.groupdict()
            decisions.append(
                {
                    "sim_time_s": float(d["sim_time"]),
                    "node_id": int(d["node_id"]),
                    "new_bitrate_bps": int(d["new_bitrate"]),
                    "old_bitrate_bps": int(d["old_bitrate"]),
                    "throughput_estimate_bps": float(d["est_bitrate"]),
                    "interruption_time_s": float(d["inter_time"]),
                    "buffer_level_s": float(d["buffer_level"]),
                    "buffer_differential_s": float(d["buffer_diff"]),
                    "delay_s": float(d["delay"]),
                }
            )
    return decisions


def build_records(
    decisions: List[Dict[str, float]],
    metadata_csv: str,
    scaler_json: Optional[str],
    run_id: str,
    scenario: str,
) -> List[Dict]:
    metadata = load_metadata_index(metadata_csv, scaler_json)
    records: List[Dict] = []

    for seg_idx, decision in enumerate(decisions):
        segment_meta = metadata.get(seg_idx)
        if segment_meta is None:
            continue

        record = {
            "run_id": run_id,
            "scenario": scenario,
            "decision_index": seg_idx,
            "node_id": decision["node_id"],
            "segment_id": seg_idx,
            "sim_time_s": decision["sim_time_s"],
            "observation": {
                "throughput_estimate_bps": decision["throughput_estimate_bps"],
                "buffer_level_s": decision["buffer_level_s"],
                "buffer_differential_s": decision["buffer_differential_s"],
                "current_bitrate_bps": decision["old_bitrate_bps"],
                "prev_bitrate_bps": decision["old_bitrate_bps"],
                "player_state": 1,
                "interruption_time_s": decision["interruption_time_s"],
                "interruptions": 0,
            },
            "semantic_raw": segment_meta.semantic_raw,
            "semantic_norm": segment_meta.semantic_norm,
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
        records.append(record)
    return records


def write_jsonl(records: List[Dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ABR JSONL dataset from run logs and metadata.")
    parser.add_argument("--log", required=True, help="Path to ns-3 run log containing newBitRate lines.")
    parser.add_argument("--metadata", required=True, help="Path to pruned metadata CSV.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--semantic-scaler",
        "--scaler",
        dest="semantic_scaler",
        default=DEFAULT_SEMANTIC_SCALER,
        help="Path to semantic feature scaler JSON (default: feature_scaler_semantic.json).",
    )
    parser.add_argument("--run-id", default="run-unknown", help="Run identifier for output records.")
    parser.add_argument("--scenario", default="ntn-iab-quic-dash-ai", help="Scenario label.")
    args = parser.parse_args()

    decisions = parse_decisions(args.log)
    if not decisions:
        raise RuntimeError("No ABR decisions found in the provided log file.")

    records = build_records(decisions, args.metadata, args.semantic_scaler, args.run_id, args.scenario)
    if not records:
        raise RuntimeError("No records were produced. Check metadata alignment with segment indexes.")

    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
