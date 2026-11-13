#!/usr/bin/env python3
"""Build Pensieve training episodes from EXISTING ns-3 artifacts (no new sims, no new data).

Each `throughput_trace.jsonl` under Quic_artifacts/<algo>/<video>/run_<seed>/ gives, per (run,node),
an ordered per-segment effective-bandwidth series. We pair it with that video's canonical per-chunk
size table (the metadata CSV the trace line points at) to form an Episode. Heuristic arms (default
Fdash) are used as the network-condition source so the bandwidth timelines are policy-independent.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

# scratch/ on path so `import ai_llm.*` works whether run as a module or a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_llm.pensieve.env import Episode                       # noqa: E402
from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS  # noqa: E402

# Repo root = .../ns3-ntn-iab (this file is scratch/ai_llm/pensieve/traces.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_ARTIFACTS = os.path.join(_REPO_ROOT, "Quic_artifacts")


def _tables(metadata_csv: str, _cache: Dict[str, tuple]) -> Optional[tuple]:
    """(sizes_bytes, vmafs) each [num_segments][A_DIM] for a video, cached by CSV path."""
    if metadata_csv in _cache:
        return _cache[metadata_csv]
    if not metadata_csv or not os.path.exists(metadata_csv):
        _cache[metadata_csv] = None  # type: ignore
        return None
    idx = load_metadata_index(metadata_csv)
    segs = idx.segment_ids()
    sizes = [[float(idx.get(s).size_bytes[lvl]) for lvl in LADDER_LEVELS] for s in segs]
    vmafs = [[float(idx.get(s).vmaf[lvl]) for lvl in LADDER_LEVELS] for s in segs]
    _cache[metadata_csv] = (sizes, vmafs)
    return sizes, vmafs


def build_episodes(
    artifacts_root: str = DEFAULT_ARTIFACTS,
    algos: Optional[List[str]] = None,
    max_episodes: Optional[int] = None,
) -> List[Episode]:
    """Collect Episodes from throughput traces. `algos` defaults to ["Fdash"] (policy-independent
    network conditions). `max_episodes` caps the pool to keep training minimal."""
    algos = algos or ["Fdash"]
    size_cache: Dict[str, tuple] = {}
    episodes: List[Episode] = []

    trace_files: List[str] = []
    for algo in algos:
        trace_files.extend(sorted(glob.glob(
            os.path.join(artifacts_root, algo, "*", "run_*", "throughput_trace.jsonl"))))

    for path in trace_files:
        # group rows by (run_id, node_id) -> ordered by segment_id
        groups: Dict[tuple, List[dict]] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # pylint: disable=broad-except
                    continue
                key = (row.get("run_id"), row.get("node_id"))
                groups.setdefault(key, []).append(row)

        for rows in groups.values():
            rows.sort(key=lambda r: int(r.get("segment_id", 0)))
            bw = [float(r.get("throughput_bps", 0.0)) for r in rows if float(r.get("throughput_bps", 0.0)) > 0]
            if len(bw) < 2:
                continue
            meta_csv = rows[0].get("metadata_csv", "")
            tbl = _tables(meta_csv, size_cache)
            if not tbl:
                continue
            sizes, vmafs = tbl
            episodes.append(Episode(bw_bps=bw, sizes_bytes=sizes, vmafs=vmafs,
                                    video_name=rows[0].get("video_name", "unknown")))
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


if __name__ == "__main__":
    eps = build_episodes()
    print("episodes: %d" % len(eps))
    if eps:
        lens = [e.length for e in eps]
        print("length min/mean/max: %d / %.1f / %d" % (min(lens), sum(lens) / len(lens), max(lens)))
        print("videos: %d distinct" % len({e.video_name for e in eps}))
