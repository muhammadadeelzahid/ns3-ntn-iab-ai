#!/usr/bin/env python3
"""
Parse ABR decision lines from ns-3 simulation stdout (sim.log).
"""

import re
from typing import Dict, List, Optional

# Extended format with videoId and segmentId (dash-client.cc).
_TAIL_RE = (
    r"(?:\s+fetch:\s+(?P<segment_fetch_time_s>[0-9eE+.\-]+))?"
    r"(?:\s+loss:\s+(?P<loss_rate_recent>[0-9eE+.\-]+))?"
)

DECISION_RE = re.compile(
    r"(?P<sim_time>[0-9eE+.\-]+)\s+ue-id:\s+(?P<node_id>\d+)\s+"
    r"videoId:\s+(?P<video_id>\d+)\s+segmentId:\s+(?P<segment_id>\d+)\s+"
    r"newBitRate:\s+(?P<new_bitrate>\d+)\s+oldBitRate:\s+(?P<old_bitrate>\d+)\s+"
    r"estBitRate:\s+(?P<est_bitrate>[0-9eE+.\-]+)\s+interTime:\s+(?P<inter_time>[0-9eE+.\-]+)\s+"
    r"T:\s+(?P<buffer_level>[0-9eE+.\-]+)\s+dT:\s+(?P<buffer_diff>[0-9eE+.\-]+)\s+"
    r"del:\s+(?P<delay>[0-9eE+.\-]+)"
    + _TAIL_RE
)

# Legacy format without videoId/segmentId (backward compatible).
DECISION_RE_LEGACY = re.compile(
    r"(?P<sim_time>[0-9eE+.\-]+)\s+ue-id:\s+(?P<node_id>\d+)\s+"
    r"newBitRate:\s+(?P<new_bitrate>\d+)\s+oldBitRate:\s+(?P<old_bitrate>\d+)\s+"
    r"estBitRate:\s+(?P<est_bitrate>[0-9eE+.\-]+)\s+interTime:\s+(?P<inter_time>[0-9eE+.\-]+)\s+"
    r"T:\s+(?P<buffer_level>[0-9eE+.\-]+)\s+dT:\s+(?P<buffer_diff>[0-9eE+.\-]+)\s+"
    r"del:\s+(?P<delay>[0-9eE+.\-]+)"
    + _TAIL_RE
)


def _row_from_match(match, video_id_default: Optional[int] = None) -> Dict:
    d = match.groupdict()
    row = {
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
    if "video_id" in d and d["video_id"] is not None:
        row["video_id"] = int(d["video_id"])
        row["segment_id"] = int(d["segment_id"])
    else:
        row["video_id"] = video_id_default
        row["segment_id"] = None
    if d.get("segment_fetch_time_s") is not None:
        row["segment_fetch_time_s"] = float(d["segment_fetch_time_s"])
    if d.get("loss_rate_recent") is not None:
        row["loss_rate_recent"] = float(d["loss_rate_recent"])
    return row


def parse_decisions(log_path: str, video_id_default: Optional[int] = None) -> List[Dict]:
    """
    Parse all ABR decision lines from a log file.
    Returns rows with video_id/segment_id when present in log.
    """
    decisions: List[Dict] = []
    per_node_segment_counter: Dict[int, int] = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = DECISION_RE.search(line) or DECISION_RE_LEGACY.search(line)
            if not match:
                continue
            row = _row_from_match(match, video_id_default=video_id_default)
            if row["segment_id"] is None:
                node = row["node_id"]
                seg = per_node_segment_counter.get(node, 0)
                row["segment_id"] = seg
                per_node_segment_counter[node] = seg + 1
            decisions.append(row)
    return decisions
