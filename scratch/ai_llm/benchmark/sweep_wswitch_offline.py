#!/usr/bin/env python3
"""Offline w_switch gate (no sims, no GPU) for the low-headroom under-ask fix (plan option (b)).

The greedy teacher locks in (hysteresis): the myopic reward is maximized at V_prev whenever
w_vmaf*sem < w_switch (0.65*sem < 0.8, ALWAYS true), so upward moves never pay and the label sticks
one rung low on low-headroom content -> the model imitates a 34%-r0 plurality -> the benchmark
collapse. This tool previews teacher-label candidates that break the lock-in, WITHOUT retraining:

  sym          : sw = w_switch * |V_r - V_prev| * (1 - scd*cut)          (current; w_switch swept)
  down_only    : sw = w_switch * max(0, V_prev - V_r) * (1 - scd*cut)    (free to climb; penalize drops)

For each candidate it relabels train_jsonl_bola/train (split by VMAF headroom) + the model's own
benchmark states (Quic_artifacts/AiDash/shards_greedy) and reports the gate metrics that decide the fix:
  - rung-dist / mean-rung / r0%  (does the low-headroom r0 prior break?)
  - upgrade-off-cut%             (does hysteresis break? fraction of UPWARD label moves NOT at a cut)
  - predicted-stall%            (do we reintroduce over-ask? label whose download underruns the buffer)
  - switch%                     (oscillation guard: consecutive labels that differ)
  - r0% on the two collapse videos (pexels_35132716, pexels_30886101)

Read-only: writes nothing; labels are recomputed in memory.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))                                  # -> scratch/
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "scripts")))  # -> dp_oracle
from ai_llm.metadata_loader import LADDER_LEVELS                                       # noqa: E402
from dp_oracle import load_reward_config                                              # noqa: E402

NR = len(LADDER_LEVELS)
COLLAPSE = ("pexels_35132716", "pexels_30886101")


def _ladder(row):
    lad = row.get("ladder", {}) or {}
    sb, vm = lad.get("size_bytes", {}) or {}, lad.get("vmaf", {}) or {}
    abk = lad.get("actual_bitrate_kbps", {}) or {}
    sizes = [float(sb.get(lv, 0.0)) for lv in LADDER_LEVELS]
    vmafs = [float(vm.get(lv, 0.0)) for lv in LADDER_LEVELS]
    br_mbps = [float(abk.get(lv, 0.0)) / 1000.0 for lv in LADDER_LEVELS]  # actual VBR Mbps per rung
    return sizes, vmafs, br_mbps


def greedy_rung(thr, buf, sizes, vmafs, br_mbps, prev_vmaf, sem, cut, W, mode):
    best, bi = -1e18, 0
    thr_eff = max(thr * W.get("thr_safety", 1.0), 1.0)
    w_rate = W.get("w_rate", 0.0)
    for r in range(NR):
        rebuf = max(0.0, 8.0 * sizes[r] / thr_eff - buf)
        if prev_vmaf is None:
            sw = 0.0
        else:
            dv = vmafs[r] - prev_vmaf
            mag = abs(dv) if mode == "sym" else max(0.0, -dv)   # down_only: penalize drops only
            sw = W["w_switch"] * mag * (1.0 - W["scene_cut_discount"] * cut)
        q = (W["w_vmaf"] + W.get("w_quality_sensitive", 0.0)) * sem * vmafs[r]
        # rate-efficiency term: high rungs only pay off when the VMAF gain beats the bitrate cost, so on
        # saturated (low-headroom) content the teacher stays low; on steep-VMAF content it climbs.
        rew = q - W["w_stall"] * rebuf - sw - w_rate * br_mbps[r]
        if rew > best:
            best, bi = rew, r
    return bi


def relabel_dir(files, W, mode):
    """Return per-row (video, headroom-agnostic) label stream with session order preserved."""
    recs = []  # (video, label, cut, stalled, is_upgrade, is_switch)
    for fp in files:
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(r.get("run_id"), r.get("node_id"))].append(r)
        for _, sess in groups.items():
            sess.sort(key=lambda x: int(x.get("segment_id", 0)))
            prev_vmaf = prev_lab = None
            thr_safety = W.get("thr_safety", 1.0)
            for r in sess:
                obs = r.get("observation", {}) or {}
                thr = float(obs.get("throughput_estimate_bps", 0.0))
                buf = float(obs.get("buffer_level_s", 0.0))
                sizes, vmafs, br_mbps = _ladder(r)
                t = r.get("teacher", {}) or {}
                sem = float(t.get("semantic_weight", 0.0))
                cut = int(t.get("is_scene_cut", 0))
                g = greedy_rung(thr, buf, sizes, vmafs, br_mbps, prev_vmaf, sem, cut, W, mode)
                stalled = (8.0 * sizes[g] / max(thr * thr_safety, 1.0) - buf) > 0
                is_up = prev_lab is not None and g > prev_lab
                is_sw = prev_lab is not None and g != prev_lab
                recs.append((r.get("video_name"), g, cut, stalled, is_up, is_sw))
                prev_vmaf, prev_lab = vmafs[g], g
    return recs


def summarize(recs, tag):
    n = len(recs)
    if not n:
        return
    dist = collections.Counter(r[1] for r in recs)
    mean_rung = sum(r[1] for r in recs) / n
    r0 = 100.0 * dist[0] / n
    ups = [r for r in recs if r[4]]
    up_off_cut = 100.0 * sum(1 for r in ups if r[2] == 0) / max(len(ups), 1)
    stall = 100.0 * sum(1 for r in recs if r[3]) / n
    sw = 100.0 * sum(1 for r in recs if r[5]) / n
    pct = " ".join("%4.0f" % (100.0 * dist[i] / n) for i in range(NR))
    print("  %-22s n=%-6d r0=%4.1f%% mean=%.2f | up_off_cut=%4.1f%% stall=%4.1f%% switch=%4.1f%% | dist[%s]" % (
        tag, n, r0, mean_rung, up_off_cut, stall, sw, pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="train_jsonl_bola/train")
    ap.add_argument("--bench-dir", default="Quic_artifacts/AiDash/shards_greedy")
    ap.add_argument("--headroom-csv", default=os.path.join(
        _HERE, "catalog_selection", "stage_b_subset", "manifest.csv"))
    ap.add_argument("--reward-config", default=os.path.join(_HERE, "..", "config", "oracle_reward.json"))
    ap.add_argument("--low", type=float, default=45.0, help="headroom < LOW = low-headroom split")
    args = ap.parse_args()

    hr = {r["video_id"]: float(r["headroom"]) for r in csv.DictReader(open(args.headroom_csv))}
    base = dict(load_reward_config(args.reward_config)["reward_weights"])
    print("base weights: w_vmaf=%.2f w_switch=%.2f w_stall=%.0f scene_cut=%.2f" % (
        base["w_vmaf"], base["w_switch"], base["w_stall"], base["scene_cut_discount"]))
    print("lock-in threshold: upward move rational iff w_vmaf*sem > w_switch  =>  sem > w_switch/%.2f" % base["w_vmaf"])

    candidates = [
        ("sym w_switch=0.8 (BASE)",       {"w_switch": 0.8},                 "sym"),
        ("down_only w_switch=0.8",        {"w_switch": 0.8},                 "down_only"),
        ("RATE w_rate=0.5 + down_only",   {"w_switch": 0.8, "w_rate": 0.5},  "down_only"),
        ("RATE w_rate=1.0 + down_only",   {"w_switch": 0.8, "w_rate": 1.0},  "down_only"),
        ("RATE w_rate=1.5 + down_only",   {"w_switch": 0.8, "w_rate": 1.5},  "down_only"),
    ]
    train_files = sorted(f for f in glob.glob(os.path.join(args.train_dir, "*.jsonl"))
                         if os.path.basename(f) != "all_runs.jsonl")
    bench_files = sorted(glob.glob(os.path.join(args.bench_dir, "*.jsonl")))

    def per_video_meanrung(recs):
        d = collections.defaultdict(list)
        for v, lab, *_ in recs:
            d[v].append(lab)
        return {v: sum(x) / len(x) for v, x in d.items()}

    def corr(xs, ys):
        n = len(xs)
        if n < 3:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sx = (sum((a - mx) ** 2 for a in xs)) ** 0.5
        sy = (sum((b - my) ** 2 for b in ys)) ** 0.5
        return cov / (sx * sy) if sx * sy > 0 else 0.0

    for name, over, mode in candidates:
        W = dict(base); W.update(over)
        print("\n" + "=" * 118)
        print("CANDIDATE: %s" % name)
        print("=" * 118)
        for label, files in (("TRAIN", train_files), ("BENCH(model states)", bench_files)):
            recs = relabel_dir(files, W, mode)
            lo = [r for r in recs if hr.get(r[0], 999) < args.low]
            hi = [r for r in recs if hr.get(r[0], 999) >= args.low]
            print(" %s:" % label)
            summarize(recs, "all")
            summarize(lo, "low-headroom(<%.0f)" % args.low)
            summarize(hi, "high-headroom(>=%.0f)" % args.low)
            # CONTENT-DIFFERENTIATION: per-video mean rung vs headroom (want mean rung to RISE w/ headroom)
            pv = per_video_meanrung(recs)
            xs = [hr[v] for v in pv if v in hr]
            ys = [pv[v] for v in pv if v in hr]
            lo_m = sum(pv[v] for v in pv if hr.get(v, 999) < args.low) / max(sum(1 for v in pv if hr.get(v, 999) < args.low), 1)
            hi_m = sum(pv[v] for v in pv if hr.get(v, 999) >= args.low) / max(sum(1 for v in pv if hr.get(v, 999) >= args.low), 1)
            print("    content-diff: low-hr mean rung=%.2f  high-hr=%.2f  gap=%+.2f  corr(headroom,meanrung)=%+.2f" % (
                lo_m, hi_m, hi_m - lo_m, corr(xs, ys)))
        for v in COLLAPSE:
            vr = relabel_dir([f for f in bench_files if os.path.basename(f).startswith(v + "_run_")], W, mode)
            if vr:
                print("  collapse %-18s (hr=%.0f) bench r0=%.0f%% mean_rung=%.2f" % (
                    v, hr.get(v, -1), 100.0 * sum(1 for r in vr if r[1] == 0) / len(vr),
                    sum(r[1] for r in vr) / len(vr)))

    print("\nRead - TWO different goals:")
    print("  * FIX COLLAPSE (headline): want low-hr r0 down + a DECISIVE teacher; stall low; switch bounded.")
    print("  * RESTORE CONTENT-AWARENESS (Fig-3): want a LARGE content-diff gap + POSITIVE corr(headroom,")
    print("    mean rung) - i.e. teacher stays low on saturated content, climbs on high-headroom. Only the")
    print("    RATE term can do this (w_switch alone climbs everywhere). Pick w_rate where the gap is large")
    print("    AND collapse videos still park sensibly low (mean rung ~1-2) without stalls.")


if __name__ == "__main__":
    main()
