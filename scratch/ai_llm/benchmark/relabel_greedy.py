#!/usr/bin/env python3
"""Relabel a train_jsonl dataset with a GREEDY-REWARD CAUSAL teacher.

Per segment, pick the rung that maximises the *immediate* QoE reward, using only present info (causal):

  reward(r) = w_vmaf * semantic_weight * VMAF(r)
            - w_stall * rebuffer_est(r)                 # rebuffer_est = max(0, dl(r) - buffer)
            - w_switch * |VMAF(r) - VMAF_prev| * (1 - scene_cut_discount * is_scene_cut)
  dl(r) = 8 * size_bytes(r) / throughput_estimate_bps   # myopic download time at the CURRENT estimate

This is NOT the DP-oracle: it is single-segment (myopic, no DP) and present-only (causal, no future) — so a
model can reproduce it online. Unlike the throughput-only teacher it depends on content (VMAF, semantic
weight, scene-cut) + buffer, so imitating it forces the model to use content+network.

Only the teacher label changes (chosen_bitrate_class / oracle_bitrate_idx / chosen_bitrate_bps / vmaf_chosen
/ name); observation / semantic_raw / ladder are untouched. Switches need per-session ordering, so rows are
grouped by (run_id, node_id) and sorted by segment_id. Weights default to config/oracle_reward.json but are
overridable (for the Stage-3 constant sweep).

Usage: relabel_greedy.py <in_dir> <out_dir> [--w-vmaf .. --w-stall .. --w-switch .. --scene-cut-discount ..
                                              --w-quality-sensitive ..]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # -> scratch/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")))
from ai_llm.metadata_loader import LADDER_LEVELS
from dp_oracle import load_reward_config  # type: ignore

SEG_DUR = 3.3


def _ladder(row):
    lad = row.get("ladder", {}) or {}
    sb = lad.get("size_bytes", {}) or {}
    vm = lad.get("vmaf", {}) or {}
    abk = lad.get("actual_bitrate_kbps", {}) or {}
    sizes = [float(sb.get(lv, 0.0)) for lv in LADDER_LEVELS]
    vmafs = [float(vm.get(lv, 0.0)) for lv in LADDER_LEVELS]
    brs = [float(abk.get(lv, 0.0)) * 1000.0 for lv in LADDER_LEVELS]
    return sizes, vmafs, brs


def greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W):
    best, bi = -1e18, 0
    for r in range(len(LADDER_LEVELS)):
        dl = 8.0 * sizes[r] / max(thr, 1.0)
        rebuf = max(0.0, dl - buf)
        sw = 0.0 if prev_vmaf is None else W["w_switch"] * abs(vmafs[r] - prev_vmaf) * (1.0 - W["scene_cut_discount"] * cut)
        q = W["w_vmaf"] * sem * vmafs[r]
        # optional extra content term (off by default; w_quality_sensitive=0 in the base config)
        q += W.get("w_quality_sensitive", 0.0) * sem * vmafs[r]
        rew = q - W["w_stall"] * rebuf - sw
        if rew > best:
            best, bi = rew, r
    return bi


def thr_only_rung(thr, brs, sf=0.9):
    r = 0
    for i, b in enumerate(brs):
        if b <= thr * sf:
            r = i
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--reward-config", default=os.path.join(os.path.dirname(__file__), "..", "config", "oracle_reward.json"))
    for k in ("w_vmaf", "w_stall", "w_switch", "scene_cut_discount", "w_quality_sensitive"):
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    args = ap.parse_args()

    cfg = load_reward_config(args.reward_config)
    W = dict(cfg["reward_weights"])
    for k in ("w_vmaf", "w_stall", "w_switch", "scene_cut_discount", "w_quality_sensitive"):
        v = getattr(args, k)
        if v is not None:
            W[k] = v
    os.makedirs(args.out_dir, exist_ok=True)

    n_rows = n_files = n_diff = 0
    dist = {}
    for fp in glob.glob(os.path.join(args.in_dir, "*.jsonl")):
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        # group by (run_id, node_id), order by segment_id (switch term needs prev segment)
        groups = {}
        for r in rows:
            groups.setdefault((r.get("run_id"), r.get("node_id")), []).append(r)
        out = []
        for _, sess in groups.items():
            sess.sort(key=lambda x: int(x.get("segment_id", 0)))
            prev_vmaf = None
            for r in sess:
                obs = r.get("observation", {}) or {}
                thr = float(obs.get("throughput_estimate_bps", 0.0))
                buf = float(obs.get("buffer_level_s", 0.0))
                sizes, vmafs, brs = _ladder(r)
                t = r.get("teacher", {}) or {}
                sem = float(t.get("semantic_weight", 0.0))
                cut = int(t.get("is_scene_cut", 0))
                g = greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W)
                if g != thr_only_rung(thr, brs):
                    n_diff += 1
                t["name"] = "greedy_reward"
                t["chosen_bitrate_class"] = g
                t["oracle_bitrate_idx"] = g
                t["chosen_bitrate_bps"] = int(round(brs[g]))
                t["vmaf_chosen"] = vmafs[g]
                r["teacher"] = t
                dist[g] = dist.get(g, 0) + 1
                prev_vmaf = vmafs[g]
                out.append(r)
        if out:
            with open(os.path.join(args.out_dir, os.path.basename(fp)), "w") as fh:
                fh.write("\n".join(json.dumps(r) for r in out) + "\n")
            n_files += 1; n_rows += len(out)
    print("relabeled %d rows / %d files -> %s" % (n_rows, n_files, args.out_dir))
    print("weights:", {k: round(W[k], 3) for k in ("w_vmaf", "w_stall", "w_switch", "scene_cut_discount", "w_quality_sensitive") if k in W})
    print("greedy rung-dist:", {k: dist[k] for k in sorted(dist)})
    print("OFFLINE GATE: greedy differs from throughput-only on %.0f%% of rows" % (100.0 * n_diff / max(n_rows, 1)))


if __name__ == "__main__":
    main()
