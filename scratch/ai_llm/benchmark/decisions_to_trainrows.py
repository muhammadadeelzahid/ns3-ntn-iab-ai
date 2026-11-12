#!/usr/bin/env python3
"""Converter B: bridge decisions.jsonl (+ dp_oracle labels + metadata) -> train_jsonl rows.

For DAgger round 2 we reuse V2's OWN online states as training data. The bridge writes per-run
decisions.jsonl (full 6-field network obs after the logging fix) but not trace files, so we rebuild
train_jsonl-format rows here:
  observation = 6 logged net feats + reconstructed last_segment_bitrate_bps (prev seg's chosen rung x
                metadata.actual_bitrate, exactly as inference_server does; 0-indexed; 0 at startup)
  semantic_raw, ladder = from video metadata (0-indexed segment)
  teacher.chosen_bitrate_class = dp_oracle's oracle_bitrate_idx for that (run,node,seg)

PREREQ: run decisions_to_throughput.py + dp_oracle (ALGO_SHORT=AiDash) first so oracle_labels_state.jsonl
exists. Decisions segment_id is 1-indexed; metadata/oracle are 0-indexed -> emit seg-1.

Usage: decisions_to_trainrows.py <AiDash_root> <video_list_file> <metadata_dir> <out_dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # -> scratch/
from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS

NET_FEATS = ["throughput_estimate_bps", "buffer_level_s", "buffer_differential_s",
             "segment_fetch_time_s", "last_segment_bitrate_bps", "smoothed_rtt_ms", "loss_rate_recent"]


def ladder_dict(meta):
    return {"levels": list(LADDER_LEVELS), "size_bytes": meta.size_bytes,
            "actual_bitrate_kbps": meta.actual_bitrate_kbps, "vmaf": meta.vmaf}


def main():
    root, vlist_file, meta_dir, out_dir = sys.argv[1:5]
    keep = set(open(vlist_file).read().split())
    os.makedirs(out_dir, exist_ok=True)
    meta_cache = {}
    n_rows = n_runs = 0
    for dec in glob.glob(os.path.join(root, "*", "run_*", "decisions.jsonl")):
        video = dec.split(os.sep)[-3]   # <root>/<video>/run_N/decisions.jsonl
        if video not in keep:
            continue
        run_dir = os.path.dirname(dec)
        run_id = os.path.basename(run_dir)
        orf = os.path.join(os.path.dirname(run_dir), "oracle_labels_state.jsonl")  # per-VIDEO
        if not os.path.exists(orf):
            continue
        if video not in meta_cache:
            mp = os.path.join(meta_dir, "%s_segments.csv" % video)
            meta_cache[video] = load_metadata_index(mp, None) if os.path.exists(mp) else None
        mi = meta_cache[video]
        if mi is None:
            continue
        # oracle: (run_id, node, seg0) -> full dp_oracle record (video-level file covers all runs).
        # Keep the WHOLE record (not just the rung) so the teacher carries semantic_weight / is_scene_cut /
        # vmaf_chosen - replay_session's quality reward is w_vmaf*semantic_weight*vmaf, so dropping
        # semantic_weight zeroes V2's quality term (the bug that made the %-oracle read 0%).
        oracle = {}
        for l in open(orf):
            try:
                d = json.loads(l)
            except Exception:  # pylint: disable=broad-except
                continue
            oracle[(str(d["run_id"]), int(d["node_id"]), int(d["segment_id"]))] = d
        # decisions grouped by node, ordered by seg (1-indexed)
        by_node = {}
        for l in open(dec):
            try:
                r = json.loads(l)
            except Exception:  # pylint: disable=broad-except
                continue
            by_node.setdefault(int(r["node_id"]), []).append(r)
        out_rows = []
        n_src = len(mi._rows)  # source segments 0..N-1; looped playback overflows past N
        for node, recs in by_node.items():
            recs.sort(key=lambda x: int(x["segment_id"]))
            prev_rung = None
            for r in recs:
                seg1 = int(r["segment_id"])            # 1-indexed
                seg0 = seg1 - 1                          # 0-indexed (metadata/oracle)
                meta = mi.get(seg0)
                if meta is None and n_src > 0:          # looped playback: map overflow onto source segment
                    meta = mi.get(seg0 % n_src)
                if meta is None or (run_id, node, seg0) not in oracle:
                    prev_rung = int(r.get("chosen_rung", 0)); continue
                # reconstruct last_segment_bitrate_bps (prev seg's chosen rung x its actual bitrate)
                prev_meta = mi.get((seg0 - 1) % n_src if n_src > 0 else seg0 - 1)
                last_bps = 0.0
                if prev_meta is not None and prev_rung is not None:
                    last_bps = float(prev_meta.actual_bitrate_kbps.get(LADDER_LEVELS[prev_rung], 0.0)) * 1000.0
                obs = {
                    "throughput_estimate_bps": float(r.get("throughput_estimate_bps", 0.0)),
                    "buffer_level_s": float(r.get("buffer_level_s", 0.0)),
                    "buffer_differential_s": float(r.get("buffer_differential_s", 0.0)),
                    "segment_fetch_time_s": float(r.get("segment_fetch_time_s", 0.0)),
                    "last_segment_bitrate_bps": last_bps,
                    "smoothed_rtt_ms": float(r.get("smoothed_rtt_ms", -1.0)),
                    "loss_rate_recent": float(r.get("loss_rate_recent", -1.0)),
                }
                # realized channel rate = chosen rung's actual bytes / realized fetch time (the basis
                # replay_session/dp_oracle must use; the logged estimate is ~10x too low -> phantom stalls).
                fetch = float(r.get("segment_fetch_time_s", 0.0))
                cur_rung = int(r.get("chosen_rung", 0))
                cur_bytes = float(meta.size_bytes.get(LADDER_LEVELS[cur_rung], 0.0))
                thr_replay = (cur_bytes * 8.0 / fetch) if (fetch > 0.05 and cur_bytes > 0) else \
                    float(r.get("throughput_estimate_bps", 0.0))
                orec = oracle[(run_id, node, seg0)]   # full dp_oracle label record
                out_rows.append({
                    "run_id": run_id, "node_id": node, "segment_id": seg0, "video_name": video,
                    "observation": obs, "semantic_raw": meta.semantic_raw,
                    "ladder": ladder_dict(meta),
                    "teacher": {
                        "name": "dp_oracle",
                        "chosen_bitrate_class": int(orec["oracle_bitrate_idx"]),
                        "throughput_bps_replay": thr_replay,
                        # quality-reward inputs for replay_session (w_vmaf*semantic_weight*vmaf - ...):
                        "semantic_weight": float(orec.get("semantic_weight", 0.0)),
                        "is_scene_cut": int(orec.get("is_scene_cut", 0)),
                        "vmaf_chosen": float(orec.get("vmaf_chosen", 0.0)),
                    },
                })
                prev_rung = int(r.get("chosen_rung", 0))
        if out_rows:
            with open(os.path.join(out_dir, "%s_%s.jsonl" % (video, run_id)), "w") as fh:
                for row in out_rows:
                    fh.write(json.dumps(row) + "\n")
            n_runs += 1; n_rows += len(out_rows)
    print("wrote %d rows across %d runs -> %s" % (n_rows, n_runs, out_dir))


if __name__ == "__main__":
    main()
