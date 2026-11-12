#!/usr/bin/env python3
"""Diagnose the model's low-headroom collapse (benchmark 2026-07-05: pexels_35132716 0%,
pexels_30886101 20% of the greedy ceiling, both via systematic UNDER-asking).

Three read-only stages (see plan "DIAGNOSIS PLAN - low-headroom collapse"):
  1. PARITY   - offline checkpoint argmax vs the ONLINE chosen rung on the exact benchmark states
                (rules a serving bug in/out; online pre-guard rung recovered from the reason string).
  2. FORENSICS- at model=r0 / greedy=r1+ mismatch states: softmax shape (prior vs calibration),
                kNN of (semantic, thr, buffer) against TRAIN rows -> neighbourhood label mix +
                buffer shift; P(label=0 | buffer) on train low-headroom rows; semantic OOD check.
  3. METRIC   - QoE span (oracle-floor) by headroom bin from perceptual_qoe.score_run's _qoe +
                VMAF points/segment actually lost on the failing videos (honest magnitude).

Inputs are all existing artifacts: greedy shards (decisions_to_trainrows->relabel_greedy output),
bench decisions.jsonl, train_jsonl_bola/train, stage_b_subset/manifest.csv, pinned scalers.
Run on a GPU node (CPU inference of the bf16 LLM is ~28s/decision - infeasible).
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO)                                     # -> scripts.dp_oracle (perceptual_qoe)
sys.path.insert(0, os.path.join(_REPO, "scratch"))            # -> ai_llm.*
sys.path.insert(0, os.path.join(_REPO, "scripts"))            # -> dp_oracle (evaluate.py imports it)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_llm.train import AbrJsonlDataset  # noqa: E402
from ai_llm.evaluate import load_policy_model  # noqa: E402
from ai_llm.perceptual_qoe import score_run  # noqa: E402
from ai_llm.trace_schema import LABEL_FIELD_CLASS  # noqa: E402
from ai_llm.greedy_teacher import load_reward_config  # noqa: E402

GUARD_RE = re.compile(r"guard\(buf[\d.]+:r(\d)->r(\d)\)")
# NET_FEATS order in the shards/dataset (thr, buffer are dims 0,1 of the last window step)
IDX_THR, IDX_BUF = 0, 1


def load_online(bench_root, videos):
    """(video, run_id, node, seg0) -> (post_guard_rung, pre_guard_rung, buffer_raw)."""
    out = {}
    for vid in videos:
        for dec in glob.glob(os.path.join(bench_root, vid, "run_*", "decisions.jsonl")):
            run_id = dec.split(os.sep)[-2]
            for line in open(dec):
                d = json.loads(line)
                post = int(d.get("chosen_rung", 0))
                m = GUARD_RE.search(d.get("reason", ""))
                pre = int(m.group(1)) if m else post
                out[(vid, run_id, int(d["node_id"]), int(d["segment_id"]) - 1)] = (
                    post, pre, float(d.get("buffer_level_s", 0.0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(_REPO, "scratch/ai_llm/checkpoints_bola/abr_v2_seed42.pt"))
    # must live on the shared filesystem; /tmp is node-local and empty on the GPU node.
    ap.add_argument("--shards-dir", default=os.path.join(_REPO, "Quic_artifacts/AiDash/shards_greedy"))
    ap.add_argument("--videos", default="pexels_35132716,pexels_30886101")
    ap.add_argument("--bench-root", default=os.path.join(_REPO, "Quic_artifacts/AiDash/bench_scored"))
    ap.add_argument("--train-dir", default=os.path.join(_REPO, "train_jsonl_bola/train"))
    ap.add_argument("--headroom-csv", default=os.path.join(
        _HERE, "catalog_selection", "stage_b_subset", "manifest.csv"))
    ap.add_argument("--reward-config", default=os.path.join(_REPO, "scratch/ai_llm/config/oracle_reward.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--knn-k", type=int, default=50)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    videos = args.videos.split(",")
    headroom = {}
    import csv as _csv
    for r in _csv.DictReader(open(args.headroom_csv)):
        headroom[r["video_id"]] = float(r["headroom"])

    # ---------------- Stage 1: offline forward on the exact benchmark states -----------------
    print("=" * 100)
    print("STAGE 1 - offline/online parity (device=%s)" % device)
    model, window, use_masks, variant, _ = load_policy_model(args.checkpoint, device)
    print("checkpoint: window=%d masks=%s variant=%s" % (window, use_masks, variant))

    shard_paths = []
    for vid in videos:
        shard_paths += sorted(glob.glob(os.path.join(args.shards_dir, "%s_run_*.jsonl" % vid)))
    ds = AbrJsonlDataset([p for p in shard_paths], network_scaler=None, semantic_scaler=None,
                         window=window, use_missing_masks=use_masks, variant=variant)
    online = load_online(args.bench_root, videos)

    rows_out = []          # per-decision dicts for stages 1+2
    batch, meta = [], []

    def flush():
        if not batch:
            return
        net = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
        sem = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=device)
        lad = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits, _ = model(net, sem, lad)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for m, p in zip(meta, probs):
            m["softmax"] = p.tolist()
            m["offline_pred"] = int(np.argmax(p))
            rows_out.append(m)
        batch.clear()
        meta.clear()

    for row in ds._rows:  # pylint: disable=protected-access
        vid = row.get("video_name")
        key = (vid, row.get("run_id"), int(row.get("node_id")), int(row.get("segment_id")))
        net, sem, lad = ds._features(row)  # pylint: disable=protected-access
        onl = online.get(key)
        batch.append((np.asarray(net), np.asarray(sem), np.asarray(lad)))
        meta.append({
            "video": vid, "run": row.get("run_id"), "node": key[2], "seg": key[3],
            "label": int(row["teacher"][LABEL_FIELD_CLASS]),
            "online_post": onl[0] if onl else None,
            "online_pre": onl[1] if onl else None,
            "buffer_raw": float((row.get("observation") or {}).get("buffer_level_s", 0.0)),
            "thr_z": float(np.asarray(net)[-1][IDX_THR]),
            "buf_z": float(np.asarray(net)[-1][IDX_BUF]),
            "sem_z": np.asarray(sem).tolist(),
        })
        if len(batch) >= 64:
            flush()
    flush()

    joined = [r for r in rows_out if r["online_pre"] is not None]
    agree_pre = sum(r["offline_pred"] == r["online_pre"] for r in joined)
    agree_post = sum(r["offline_pred"] == r["online_post"] for r in joined)
    print("states: %d scored, %d joined to online decisions (join rate %.0f%%)" % (
        len(rows_out), len(joined), 100.0 * len(joined) / max(len(rows_out), 1)))
    print("PARITY offline==online(pre-guard) : %d/%d = %.1f%%" % (
        agree_pre, len(joined), 100.0 * agree_pre / max(len(joined), 1)))
    print("       offline==online(post-guard): %d/%d = %.1f%%" % (
        agree_post, len(joined), 100.0 * agree_post / max(len(joined), 1)))
    div = [r for r in joined if r["offline_pred"] != r["online_pre"]][:10]
    for r in div:
        print("  divergent: %s/%s node%d seg%d offline=r%d online_pre=r%d label=r%d" % (
            r["video"], r["run"], r["node"], r["seg"], r["offline_pred"], r["online_pre"], r["label"]))

    # ---------------- Stage 2: why r0? --------------------------------------------------------
    print("=" * 100)
    print("STAGE 2 - forensics at mismatch states (model says r0/low, greedy wants higher)")
    mism = [r for r in rows_out if r["offline_pred"] < r["label"]]
    mism_r0 = [r for r in mism if r["offline_pred"] == 0]
    print("mismatch (pred<label): %d/%d rows; of which pred==r0: %d" % (
        len(mism), len(rows_out), len(mism_r0)))
    if mism:
        p_pred = np.mean([r["softmax"][r["offline_pred"]] for r in mism])
        p_lab = np.mean([r["softmax"][r["label"]] for r in mism])
        second = [sorted(r["softmax"], reverse=True)[1] for r in mism]
        margin = np.mean([max(r["softmax"]) - s for r, s in zip(mism, second)])
        close = sum(1 for r, s in zip(mism, second) if max(r["softmax"]) - s < 0.10)
        # is the LABEL rung the runner-up?
        lab_is_2nd = sum(1 for r in mism
                         if np.argsort(r["softmax"])[-2] == r["label"])
        print("softmax at mismatches: mean p(pred)=%.2f  mean p(label)=%.2f  mean top-2 margin=%.2f" % (
            p_pred, p_lab, margin))
        print("  label is the runner-up class at %d/%d (%.0f%%); margin<0.10 at %d (%.0f%%)" % (
            lab_is_2nd, len(mism), 100.0 * lab_is_2nd / len(mism), close, 100.0 * close / len(mism)))
        print("  -> dominant-prior if p(pred) high & margin large; calibration if label is a close 2nd")

    # kNN against TRAIN in the SAME normalized feature space (sem 5 + thr_z + buf_z)
    print("-" * 100)
    train_paths = sorted(glob.glob(os.path.join(args.train_dir, "*.jsonl")))
    tds = AbrJsonlDataset(train_paths, network_scaler=None, semantic_scaler=None,
                          window=1, use_missing_masks=use_masks, variant="v0")
    feats, labels, bufs_raw, vids_t = [], [], [], []
    for row in tds._rows:  # pylint: disable=protected-access
        net, sem, _ = tds._features(row)  # pylint: disable=protected-access
        n0 = np.asarray(net)[-1]
        feats.append(np.concatenate([np.asarray(sem), [n0[IDX_THR], n0[IDX_BUF]]]))
        labels.append(int(row["teacher"][LABEL_FIELD_CLASS]))
        bufs_raw.append(float((row.get("observation") or {}).get("buffer_level_s", 0.0)))
        vids_t.append(row.get("video_name"))
    F = np.stack(feats)
    L = np.asarray(labels)
    B = np.asarray(bufs_raw)
    print("train rows for kNN: %d" % len(F))

    q = np.stack([np.concatenate([np.asarray(r["sem_z"]), [r["thr_z"], r["buf_z"]]]) for r in mism]) \
        if mism else np.zeros((0, F.shape[1]))
    if len(q):
        nbr_label = collections.Counter()
        nbr_buf, nbr_bufz = [], []
        for x in q:
            d = np.linalg.norm(F - x, axis=1)
            idx = np.argpartition(d, args.knn_k)[:args.knn_k]
            for i in idx:
                nbr_label[int(L[i])] += 1
            nbr_buf.append(float(np.mean(B[idx])))
            nbr_bufz.append(float(np.mean(F[idx, -1])))
        tot = sum(nbr_label.values())
        print("kNN(k=%d) neighbourhood TRAIN-label mix at the mismatch states: %s" % (
            args.knn_k, {k: "%.0f%%" % (100.0 * v / tot) for k, v in sorted(nbr_label.items())}))
        print("  query mean buffer=%.1fs (z=%.2f)  vs neighbour mean buffer=%.1fs (z=%.2f)" % (
            np.mean([r["buffer_raw"] for r in mism]), np.mean([r["buf_z"] for r in mism]),
            np.mean(nbr_buf), np.mean(nbr_bufz)))
        print("  -> if neighbours are r0-labelled LOW-buffer states while queries are HIGH-buffer,")
        print("     the imitation prior transferred across a buffer-state shift (covariate shift).")

    # buffer effect on the TEACHER within low-headroom train videos
    lo_mask = np.asarray([headroom.get(v, 999.0) < 45.0 for v in vids_t])
    if lo_mask.any():
        for thr_b in (4.0, 8.0):
            lo_lab = L[lo_mask]
            lo_buf = B[lo_mask]
            p_lo = float(np.mean(lo_lab[lo_buf < thr_b] == 0)) if (lo_buf < thr_b).any() else float("nan")
            p_hi = float(np.mean(lo_lab[lo_buf >= thr_b] == 0)) if (lo_buf >= thr_b).any() else float("nan")
            print("TRAIN low-headroom videos: P(label=r0 | buffer<%.0fs)=%.0f%%  vs  >=%.0fs: %.0f%%" % (
                thr_b, 100 * p_lo, thr_b, 100 * p_hi))

    # semantic OOD: max |z| of the failing videos vs train base rate
    if mism:
        qz = np.abs(np.stack([np.asarray(r["sem_z"]) for r in rows_out]))
        tz = np.abs(F[:, :5])
        print("semantic OOD: failing-video rows with any |z|>3: %.0f%%  (train base rate: %.0f%%)" % (
            100.0 * float(np.mean((qz > 3).any(axis=1))), 100.0 * float(np.mean((tz > 3).any(axis=1)))))

    # ---------------- Stage 3: metric-cliff spans ---------------------------------------------
    print("=" * 100)
    print("STAGE 3 - QoE span (oracle-floor) by headroom + honest VMAF magnitude")
    cfg = load_reward_config(args.reward_config)
    span_rows = []
    for shard in sorted(glob.glob(os.path.join(args.shards_dir, "*_run_*.jsonl"))):
        base = os.path.basename(shard)[:-6]
        vid, run = base.split("_run_")
        dec = os.path.join(args.bench_root, vid, "run_%s" % run, "decisions.jsonl")
        if not os.path.exists(dec):
            continue
        res = score_run(None, shard, cfg, label=base, decisions_jsonl=dec)
        if res is None:
            continue
        qoe = res["_qoe"]
        span_rows.append((vid, headroom.get(vid, float("nan")), qoe["oracle"] - qoe["floor"],
                          res["pct_of_oracle"], res["segments"]))
    by_vid = collections.defaultdict(list)
    for vid, hr, span, pct, segs in span_rows:
        by_vid[vid].append((hr, span / max(segs, 1), pct))
    print("%-20s %8s | %14s | %10s" % ("video", "headroom", "span/seg (QoE)", "mean pct"))
    for vid in sorted(by_vid, key=lambda v: headroom.get(v, 999)):
        xs = by_vid[vid]
        print("%-20s %8.1f | %14.2f | %9.1f%%" % (
            vid, xs[0][0], np.mean([x[1] for x in xs]), np.mean([x[2] for x in xs])))

    # VMAF actually left on the table for the failing videos (policy vs teacher rung, per segment)
    for vid in videos:
        deltas = []
        for r in rows_out:
            if r["video"] != vid or r["online_post"] is None:
                continue
            row_src = None  # ladder vmafs live in the shard rows; reuse ds lookup
        # simpler: recompute from the shards directly
        for shard in glob.glob(os.path.join(args.shards_dir, "%s_run_*.jsonl" % vid)):
            run = "run_" + os.path.basename(shard)[:-6].split("_run_")[1]
            for line in open(shard):
                d = json.loads(line)
                key = (vid, run, int(d["node_id"]), int(d["segment_id"]))
                onl = online.get(key)
                if onl is None:
                    continue
                vm = d["ladder"]["vmaf"]
                lv = d["ladder"]["levels"]
                lab = int(d["teacher"][LABEL_FIELD_CLASS])
                deltas.append(float(vm[lv[lab]]) - float(vm[lv[onl[0]]]))
        if deltas:
            print("%s: mean VMAF left on the table = %.1f points/segment (teacher rung vs delivered rung; n=%d)"
                  % (vid, float(np.mean(deltas)), len(deltas)))

    print("=" * 100)
    print("done.")


if __name__ == "__main__":
    main()
