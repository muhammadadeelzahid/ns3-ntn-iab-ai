#!/usr/bin/env python3
"""
Compute per-run and per-user metrics from run-directory trace files.

Mirrors key logic from scripts/plot_bitrate_playback.py for machine-readable outputs.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from ai_llm.decision_parser import parse_decisions
except ImportError:
    from decision_parser import parse_decisions


def parse_dash_client_rx(path: Path) -> List[Tuple[float, int]]:
    samples: List[Tuple[float, int]] = []
    if not path.is_file():
        return samples
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    samples.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    return samples


def compute_mean_throughput_kbps(rx_data: List[Tuple[float, int]]) -> float:
    if not rx_data or len(rx_data) < 2:
        return 0.0
    total_bytes = sum(size for _, size in rx_data)
    duration = rx_data[-1][0] - rx_data[0][0]
    if duration <= 0:
        return 0.0
    return (total_bytes * 8) / (duration * 1e3)


def find_rx_file(run_dir: Path, node_id: int) -> Optional[Path]:
    for name in (
        f"DashClientRx_Node_{node_id}.txt",
        f"clientQUIC-rx-data{node_id}.txt",
    ):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def parse_playback_events(log_path: Path) -> Dict[int, List]:
    events_by_player: Dict[int, List] = defaultdict(list)
    if not log_path.is_file():
        return events_by_player
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            play = re.search(
                r"(\d+\.?\d*)\s+PLAYING FRAME:.*?PlayerId:\s*(\d+).*?Res:\s*(\d+)",
                line,
            )
            if play:
                events_by_player[int(play.group(2))].append(
                    ("playing", float(play.group(1)), float(play.group(3)))
                )
    return events_by_player


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute run_metrics.json and user_metrics.csv.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    sim_log = run_dir / "sim.log"
    if not sim_log.is_file():
        logs = list(run_dir.glob("*.log"))
        sim_log = logs[0] if logs else sim_log

    decisions = parse_decisions(str(sim_log)) if sim_log.is_file() else []
    nodes = sorted({d["node_id"] for d in decisions})

    user_rows: List[Dict] = []
    for node_id in nodes:
        rx_path = find_rx_file(run_dir, node_id)
        rx = parse_dash_client_rx(rx_path) if rx_path else []
        node_decisions = [d for d in decisions if d["node_id"] == node_id]
        bitrates = [d["new_bitrate_bps"] for d in node_decisions]
        user_rows.append(
            {
                "node_id": node_id,
                "mean_throughput_kbps": compute_mean_throughput_kbps(rx),
                "decision_count": len(node_decisions),
                "avg_chosen_bitrate_bps": sum(bitrates) / len(bitrates) if bitrates else 0,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "decision_count": len(decisions),
        "user_count": len(user_rows),
        "mean_throughput_kbps": (
            sum(r["mean_throughput_kbps"] for r in user_rows) / len(user_rows) if user_rows else 0.0
        ),
    }

    with open(run_dir / "run_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(run_dir / "user_metrics.csv", "w", encoding="utf-8", newline="") as handle:
        if user_rows:
            writer = csv.DictWriter(handle, fieldnames=list(user_rows[0].keys()))
            writer.writeheader()
            writer.writerows(user_rows)

    print(f"Wrote run_metrics.json and user_metrics.csv under {run_dir}")


if __name__ == "__main__":
    main()
