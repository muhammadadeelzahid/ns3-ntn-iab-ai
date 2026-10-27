#!/usr/bin/env python3
"""Convert bridge decisions.jsonl -> throughput_trace.jsonl per AiDash run dir.

The live bridge writes per-run decisions.jsonl but NOT the ns-3 trace files (they clobber the repo
root). To score / DAgger-relabel V2 online runs we reconstruct the throughput_trace.jsonl that
dp_oracle consumes. throughput_bps must be the REALIZED channel rate (so dp_oracle + replay_session
re-simulate downloads faithfully) - computed as the chosen rung's ACTUAL (VBR) segment bytes / the
realized segment_fetch_time_s. (The logged throughput_estimate_bps is the ABR's lagged estimate, ~order
of magnitude below the realized rate, and using it injects huge phantom stalls into replay_session.)
Segment ids are shifted to 0-indexed; looped playback (short videos replaying to fill the sim cap) maps
overflow segments onto their source segment by position.

Usage: decisions_to_throughput.py <AiDash_root> [metadata_dir]
  With metadata_dir: throughput_bps = realized (actual_bytes/fetch_time). Without: falls back to the
  logged estimate (legacy behaviour).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # -> scratch/
from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS

SEG_DUR = 3.3  # player segment duration (s), matches mpeg-header.h (100*33ms); encoded bytes are 2.0s -> 0.606x effective demand
MIN_FETCH_S = 0.05   # guard against divide-by-zero / buffer-full near-instant fetches


def _rung_of(d):
    r = d.get("chosen_rung")
    if r is not None:
        return int(r)
    # fall back to nearest nominal rung of chosen_bitrate_bps
    from ai_llm.paths import NS3_RATE_LADDER_BPS
    bps = float(d.get("chosen_bitrate_bps", 0.0))
    return min(range(len(NS3_RATE_LADDER_BPS)), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "Quic_artifacts/AiDash"
    meta_dir = sys.argv[2] if len(sys.argv) > 2 else None
    meta_cache = {}
    n_files = n_rows = 0
    for dec in glob.glob(os.path.join(root, "**", "run_*", "decisions.jsonl"), recursive=True):
        run_dir = os.path.dirname(dec)
        run_id = os.path.basename(run_dir)                 # run_N
        video = dec.split(os.sep)[-3]
        mi = None
        if meta_dir:
            if video not in meta_cache:
                mp = os.path.join(meta_dir, "%s_segments.csv" % video)
                meta_cache[video] = load_metadata_index(mp, None) if os.path.exists(mp) else None
            mi = meta_cache[video]
        n_src = len(mi._rows) if mi is not None else 0  # pylint: disable=protected-access
        out = os.path.join(run_dir, "throughput_trace.jsonl")
        rows = []
        for line in open(dec, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # pylint: disable=broad-except
                continue
            seg = d.get("segment_id")
            if seg is None:
                continue
            seg0 = int(seg) - 1                            # 1-indexed -> 0-indexed
            fetch = float(d.get("segment_fetch_time_s", 0.0))
            thr = float(d.get("throughput_estimate_bps", 0.0))   # legacy fallback
            seg_bytes = max(1.0, thr * SEG_DUR / 8.0)
            if mi is not None and n_src > 0 and fetch > MIN_FETCH_S:
                meta = mi.get(seg0) or mi.get(seg0 % n_src)
                if meta is not None:
                    rung = _rung_of(d)
                    actual_bytes = float(meta.size_bytes.get(LADDER_LEVELS[rung], 0.0))
                    if actual_bytes > 0:
                        thr = actual_bytes * 8.0 / fetch        # realized channel rate (bps)
                        seg_bytes = actual_bytes
            rows.append({
                "run_id": run_id,
                "node_id": int(d["node_id"]),
                "segment_id": seg0,
                "throughput_bps": thr,
                "segment_fetch_time_s": fetch if fetch > 0 else SEG_DUR,
                "segment_bytes": seg_bytes,
            })
        if rows:
            with open(out, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            n_files += 1
            n_rows += len(rows)
    print("wrote %d throughput_trace.jsonl files, %d rows under %s (realized=%s)" %
          (n_files, n_rows, root, bool(meta_dir)))


if __name__ == "__main__":
    main()
