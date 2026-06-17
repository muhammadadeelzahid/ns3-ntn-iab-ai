#!/usr/bin/env python3
"""Validate trace_decisions.jsonl from extract_trace.py."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch"))
    from ai_llm.paths import NS3_RATE_LADDER_BPS
    from ai_llm.metadata_loader import load_metadata_index
except ImportError:
    from paths import NS3_RATE_LADDER_BPS  # type: ignore
    from metadata_loader import load_metadata_index  # type: ignore

LADDER_SET = set(NS3_RATE_LADDER_BPS)


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    metadata = load_metadata_index(args.metadata)
    max_seg = len(metadata) - 1
    errors = []
    keys = Counter()
    bitrates = set()

    for rec in load_jsonl(Path(args.jsonl)):
        seg = rec.get("segment_id")
        node = rec.get("node_id")
        keys[(node, seg)] += 1
        chosen = rec.get("teacher", {}).get("chosen_bitrate_bps")
        if chosen not in LADDER_SET:
            errors.append(f"invalid bitrate {chosen} node={node} seg={seg}")
        if seg is None or seg < 0 or seg > max_seg:
            errors.append(f"segment_id out of range: {seg} (max {max_seg})")

    dupes = [k for k, c in keys.items() if c > 1]
    warnings = []
    if dupes:
        warnings.append(
            "duplicate (node_id, segment_id) keys: {} (re-run extract_trace.py to dedupe)".format(
                dupes[:5]
            )
        )
        if len(dupes) > 5:
            warnings[-1] += " ... ({} total)".format(len(dupes))

    if errors:
        print("VALIDATION FAILED:")
        for err in errors:
            print("  - {}".format(err))
        sys.exit(1)

    print(
        "OK: {} records, segments 0..{}, ladder bitrates only".format(
            sum(keys.values()), max_seg
        )
    )
    for warn in warnings:
        print("WARN: {}".format(warn))


if __name__ == "__main__":
    main()
