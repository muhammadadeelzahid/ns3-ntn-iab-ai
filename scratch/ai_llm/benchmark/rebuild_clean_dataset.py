#!/usr/bin/env python3
"""Rebuild a train_jsonl dataset on CORRECTED metadata (VMAF + semantic) and re-derive greedy labels.

OJ-COMS Phase 2. The served dataset (train_jsonl_bola_down) embeds per-row `ladder.vmaf`, `semantic_raw`
and `model_input.ladder_context` copied from abr `output/metadata/<video>_segments.csv` at build time.
For 32 catalog videos those CSVs carried mis-aligned VMAF/semantic values (abr realign fix, 2026-10-26;
sizes unchanged). The ns-3 traces (observation, segment_id, fetch time, delivered bitrate) are untouched
by that fix, so instead of re-collecting we PATCH the content columns and RE-LABEL:

  per row : ladder.vmaf / semantic_raw / model_input.ladder_context  <- current metadata CSV (by segment_id)
            guard: ladder.size_bytes must equal the CSV sizes byte-for-byte (else abort: wrong CSV/keying)
  per video: semantic_weight = compute_semantic_weights(metadata, mixture)   (same as dataset_builder)
  per (run_id,node_id) session: greedy_rung(...) in segment order with prev_vmaf tracking
            (same function + reward config as greedy_teacher.assign_greedy_labels / dataset_builder)
  teacher.throughput_bps_replay is KEPT from the source row: it is the realized channel rate derived from
            the DELIVERED bitrate at build time (sizes unchanged -> still exact); assign_greedy_labels
            cannot recompute it here because the source teacher block is already the greedy label.

Every video is processed the same way, so videos whose CSV did not change must come out IDENTICAL to the
source (ladder, semantic, labels) - that equality is the regression check that this re-derivation
reproduces the served pipeline exactly; only the realigned videos may differ. The report records both.

Usage: rebuild_clean_dataset.py --src ../../train_jsonl_bola_down --out ../../train_jsonl_clean \
         --metadata-dir /project/6030214/adeel/abr/output/metadata [--splits train,val,test]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # -> scratch/
from ai_llm.greedy_teacher import compute_semantic_weights, greedy_rung, load_reward_config, _ladder  # noqa: E402
from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402


def _video_of(shard_path):
    return os.path.basename(shard_path).split("_run_")[0]


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def patch_shard(rows, meta, weights, W):
    """Patch content columns from `meta` and relabel. Returns per-shard change counters."""
    c = defaultdict(int)
    for r in rows:
        seg_id = int(r["segment_id"])
        m = meta.get(seg_id)
        if m is None:
            raise RuntimeError("segment_id %d not in metadata (%s)" % (seg_id, r.get("video_name")))
        lad = r["ladder"]
        for lv in LADDER_LEVELS:
            if not _close(lad["size_bytes"][lv], m.size_bytes[lv]):
                raise RuntimeError("size mismatch %s seg %d rung %s: row %s csv %s" % (
                    r.get("video_name"), seg_id, lv, lad["size_bytes"][lv], m.size_bytes[lv]))
        new_vmaf = {lv: float(m.vmaf[lv]) for lv in LADDER_LEVELS}
        if any(not _close(lad["vmaf"][lv], new_vmaf[lv]) for lv in LADDER_LEVELS):
            c["vmaf_changed"] += 1
        new_sem = {k: float(v) for k, v in m.semantic_raw.items()}
        old_sem = r.get("semantic_raw") or {}
        if set(old_sem) != set(new_sem) or any(not _close(old_sem[k], new_sem[k]) for k in new_sem):
            c["semantic_changed"] += 1
        lad["vmaf"] = new_vmaf
        r["semantic_raw"] = new_sem
        # raw 6x2 [size_bytes, vmaf] exactly as features._build_ladder_context
        r["model_input"]["ladder_context"] = [[float(m.size_bytes[lv]), float(m.vmaf[lv])] for lv in LADDER_LEVELS]

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("run_id"), r.get("node_id"))].append(r)
    for _, sess in groups.items():
        sess.sort(key=lambda x: int(x.get("segment_id", 0)))
        prev_vmaf = None
        for r in sess:
            obs = r.get("observation", {}) or {}
            thr = float(obs.get("throughput_estimate_bps", 0.0))
            buf = float(obs.get("buffer_level_s", 0.0))
            sizes, vmafs, brs = _ladder(r)
            seg_id = int(r.get("segment_id", 0))
            sem = float(weights.get(seg_id, 0.0))
            cut = 1 if float((r.get("semantic_raw") or {}).get("scene_cut_count", 0.0)) > 0 else 0
            g = greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W)
            old_t = r.get("teacher") or {}
            if old_t.get("name") != "greedy_reward" or "throughput_bps_replay" not in old_t:
                raise RuntimeError("source teacher is not a greedy_reward block with throughput_bps_replay")
            if int(old_t.get("chosen_bitrate_class", -1)) != g:
                c["label_changed"] += 1
            if not _close(old_t.get("semantic_weight", 0.0), sem):
                c["semantic_weight_changed"] += 1
            r["teacher"] = {
                "name": "greedy_reward",
                "chosen_bitrate_class": g,
                "oracle_bitrate_idx": g,
                "chosen_bitrate_bps": int(round(brs[g])),
                "throughput_bps_replay": float(old_t["throughput_bps_replay"]),
                "vmaf_chosen": vmafs[g],
                "semantic_weight": sem,
                "is_scene_cut": cut,
            }
            c["rung_%d" % g] += 1
            prev_vmaf = vmafs[g]
    c["rows"] = len(rows)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--reward-config", default=None, help="default: config/oracle_reward.json")
    args = ap.parse_args()

    cfg = load_reward_config(args.reward_config)
    W = dict(cfg["reward_weights"])
    mixture = cfg["semantic_weight_mixture"]
    meta_cache, weight_cache = {}, {}
    per_video = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for split in args.splits.split(","):
        shards = sorted(glob.glob(os.path.join(args.src, split, "*.jsonl")))
        if not shards:
            raise SystemExit("no shards under %s/%s" % (args.src, split))
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        for fp in shards:
            video = _video_of(fp)
            if video not in meta_cache:
                csv_path = os.path.join(args.metadata_dir, "%s_segments.csv" % video)
                meta_cache[video] = load_metadata_index(csv_path, None)
                weight_cache[video] = compute_semantic_weights(meta_cache[video], mixture)
            rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
            c = patch_shard(rows, meta_cache[video], weight_cache[video], W)
            for k, v in c.items():
                per_video[video][k] += v
                totals[k] += v
            per_video[video]["split"] = split
            with open(os.path.join(args.out, split, os.path.basename(fp)), "w", encoding="utf-8") as fh:
                fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
            totals["files"] += 1

    changed = {v: dict(c) for v, c in per_video.items()
               if c["vmaf_changed"] or c["semantic_changed"] or c["label_changed"] or c["semantic_weight_changed"]}
    unchanged = sorted(v for v in per_video if v not in changed)
    report = {
        "src": os.path.abspath(args.src), "out": os.path.abspath(args.out),
        "metadata_dir": os.path.abspath(args.metadata_dir),
        "reward_weights": W, "semantic_weight_mixture": mixture,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "totals": dict(totals),
        "videos_total": len(per_video), "videos_changed": len(changed), "videos_identical": len(unchanged),
        "changed": changed, "identical": unchanged,
    }
    src_manifest = os.path.join(args.src, "manifest.json")
    manifest = json.load(open(src_manifest)) if os.path.exists(src_manifest) else {"schema_version": 2}
    manifest.update({"rows": totals["rows"], "rebuilt_from": os.path.abspath(args.src),
                     "rebuild_report": "rebuild_report.json", "created_at": report["created_at"]})
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(args.out, "rebuild_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("rebuilt %d rows / %d files -> %s" % (totals["rows"], totals["files"], args.out))
    print("videos: %d total, %d changed, %d byte-identical to source" % (len(per_video), len(changed), len(unchanged)))
    print("rows: vmaf_changed=%d semantic_changed=%d semantic_weight_changed=%d label_changed=%d" % (
        totals["vmaf_changed"], totals["semantic_changed"], totals["semantic_weight_changed"], totals["label_changed"]))
    print("rung-dist:", {k: totals["rung_%d" % k] for k in range(len(LADDER_LEVELS))})
    for v in sorted(changed, key=lambda x: -changed[x].get("label_changed", 0)):
        c = defaultdict(int, changed[v])
        print("  %-22s %-5s rows=%4d vmaf=%4d sem=%4d w=%4d label=%4d (%.0f%%)" % (
            v, c["split"], c["rows"], c["vmaf_changed"], c["semantic_changed"], c["semantic_weight_changed"],
            c["label_changed"], 100.0 * c["label_changed"] / max(1, c["rows"])))


if __name__ == "__main__":
    main()
