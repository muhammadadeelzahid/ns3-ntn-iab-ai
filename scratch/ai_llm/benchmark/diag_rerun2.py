#!/usr/bin/env python3
"""Phase-A diagnosis of the model arm on a scored benchmark run (default: re-run 2, fixed binary).

Read-only. Joins each arm's live decisions (Quic_artifacts/<Stem>/<video>/run_<cond>/decisions.jsonl) with the
greedy-teacher labels of the SAME states (Quic_artifacts/<Stem>_shards/<video>_run_<cond>.jsonl) and reports:
  1. per-cell deltas Ours vs a reference arm (%teacher, true stall) and the worst cells;
  2. over/under-ask vs the teacher by link rate, by segment-index bucket and by buffer bucket;
  3. guard activity (reason "llm_policy_model+guard(buf..:rA->rB)") by rate;
  4. stall attribution: decisions after which the UE's interruption time grew, with rung-vs-teacher and guard.

Usage: python3.11 diag_rerun2.py [--arm AiDashClean] [--ref Qmpc] [--out results/diag_rerun2.txt]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
Q = os.path.join(REPO, "Quic_artifacts")
RES = os.path.join(HERE, "results")
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
GUARD_RE = re.compile(r"guard\(buf([0-9.]+):r(\d)->r(\d)\)")


def load_results(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(\S+)/run_(\d)\s+\d+\s+([0-9.]+)%\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", line)
        if m:
            out[(m.group(1), int(m.group(2)))] = {"pct": float(m.group(3)), "vmaf": float(m.group(5)),
                                                  "switch": int(m.group(8))}
    return out


def load_true_stall(arm_key):
    out = {}
    for r in csv.DictReader(open(os.path.join(RES, "true_stall.csv"))):
        if r["arm"] == arm_key:
            out[(r["video"], int(r["cond"]))] = float(r["stall_s"])
    return out


def joined_rows(stem):
    """yield (video, cond, node, seg, decision_row, teacher_rung) for every decision with a teacher label."""
    for dec in glob.glob(os.path.join(Q, stem, "*", "run_*", "decisions.jsonl")):
        parts = dec.split(os.sep)
        video, cond = parts[-3], int(parts[-2].split("_")[1])
        shard = os.path.join(Q, stem + "_shards", "%s_run_%d.jsonl" % (video, cond))
        if not os.path.exists(shard):
            continue
        teach = {}
        for line in open(shard):
            r = json.loads(line)
            teach[(int(r["node_id"]), int(r["segment_id"]))] = int(r["teacher"]["chosen_bitrate_class"])
        for line in open(dec):
            d = json.loads(line)
            key = (int(d["node_id"]), int(d["segment_id"]))
            if key in teach and d.get("chosen_rung") is not None:
                yield video, cond, key[0], key[1], d, teach[key]


def pct(n, d):
    return 100.0 * n / d if d else float("nan")


def ask_table(rows, keyfn, keys, title, w):
    agg = defaultdict(lambda: [0, 0, 0, 0, 0.0])   # n, over, under, exact, sum(rung-teacher)
    for r in rows:
        a = agg[keyfn(r)]
        diff = r["rung"] - r["teacher"]
        a[0] += 1; a[1] += diff > 0; a[2] += diff < 0; a[3] += diff == 0; a[4] += diff
    w("\n%s\n  %-14s %7s %8s %8s %8s %9s" % (title, "bucket", "n", "over%", "under%", "agree%", "meanΔrung"))
    for k in keys:
        a = agg.get(k)
        if a and a[0]:
            w("  %-14s %7d %7.1f%% %7.1f%% %7.1f%% %+9.2f" % (k, a[0], pct(a[1], a[0]), pct(a[2], a[0]),
                                                               pct(a[3], a[0]), a[4] / a[0]))


def seg_bucket(s):
    return "seg 0-3" if s < 4 else "seg 4-13" if s < 14 else "seg 14-27" if s < 28 else "seg 28+"


def buf_bucket(b):
    return "buf <5s" if b < 5 else "buf 5-15s" if b < 15 else "buf 15-25s" if b < 25 else "buf >=25s"


def collect(stem):
    rows = []
    for video, cond, node, seg, d, t in joined_rows(stem):
        reason = d.get("reason", "")
        g = GUARD_RE.search(reason)
        rows.append({"video": video, "cond": cond, "node": node, "seg": seg, "rung": int(d["chosen_rung"]),
                     "teacher": t, "buf": float(d.get("buffer_level_s", 0.0)),
                     "thr": float(d.get("throughput_estimate_bps", 0.0)),
                     "intr": float(d.get("interruption_time_s", 0.0)), "t": float(d.get("sim_time_s", 0.0)),
                     "guard": bool(g), "pre": int(g.group(2)) if g else None})
    return rows


def stall_attribution(rows, w, label):
    by_ue = defaultdict(list)
    for r in rows:
        by_ue[(r["video"], r["cond"], r["node"])].append(r)
    events = []
    for k, rs in by_ue.items():
        rs.sort(key=lambda r: r["t"])
        for a, b in zip(rs, rs[1:]):
            grow = b["intr"] - a["intr"]
            if grow > 0.05:
                events.append((grow, k, a))
    tot = sum(e[0] for e in events)
    w("\n[%s] stall attribution: %d growth events, %.1f s total interruption growth" % (label, len(events), tot))
    if not events:
        return
    over = sum(e[0] for e in events if e[2]["rung"] > e[2]["teacher"])
    guard = sum(e[0] for e in events if e[2]["guard"])
    lowbuf = sum(e[0] for e in events if e[2]["buf"] < 5)
    w("  share of stall after a decision ABOVE the teacher rung: %.0f%%" % pct(over, tot))
    w("  share after a GUARDED decision (guard already lowered it): %.0f%%" % pct(guard, tot))
    w("  share with buffer < 5 s at decision time: %.0f%%" % pct(lowbuf, tot))
    byc = defaultdict(float)
    for g, k, a in events:
        byc[COND[k[1]]] += g
    w("  by rate: " + "  ".join("%s %.1fs" % (c, byc.get(c, 0.0)) for c in COND.values()))
    w("  largest events (growth s | video cond node seg | rung teacher buf thr_Mbps guard):")
    for g, k, a in sorted(events, key=lambda e: -e[0])[:8]:
        w("    %6.2f | %s %s n%d s%d | r%d t%d buf %.1f thr %.2f %s" % (
            g, k[0], COND[k[1]], k[2], a["seg"], a["rung"], a["teacher"], a["buf"], a["thr"] / 1e6,
            "G(r%d->r%d)" % (a["pre"], a["rung"]) if a["guard"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="AiDashClean")
    ap.add_argument("--arm-stall-key", default="AiDash", help="arm name used in results/true_stall.csv")
    ap.add_argument("--ref", default="Qmpc")
    ap.add_argument("--out", default=os.path.join(RES, "diag_rerun2.txt"))
    args = ap.parse_args()
    lines = []
    w = lambda s="": (print(s), lines.append(s))   # noqa: E731

    # 1. per-cell deltas
    ra, rr = load_results(os.path.join(RES, args.arm + ".txt")), load_results(os.path.join(RES, args.ref + ".txt"))
    sa, sr = load_true_stall(args.arm_stall_key), load_true_stall(args.ref)
    cells = sorted(set(ra) & set(rr))
    w("== 1. per-cell %s vs %s over %d cells" % (args.arm, args.ref, len(cells)))
    wins = sum(ra[c]["pct"] > rr[c]["pct"] + 0.05 for c in cells)
    loss = sum(ra[c]["pct"] < rr[c]["pct"] - 0.05 for c in cells)
    w("  %%teacher: %s better in %d cells, worse in %d, tie %d" % (args.arm, wins, loss, len(cells) - wins - loss))
    for c in COND:
        cc = [x for x in cells if x[1] == c]
        dp = sum(ra[x]["pct"] - rr[x]["pct"] for x in cc) / len(cc)
        dv = sum(ra[x]["vmaf"] - rr[x]["vmaf"] for x in cc) / len(cc)
        ds = sum(sa.get(x, 0) - sr.get(x, 0) for x in cc) / len(cc)
        w("  %-8s Δ%%teacher %+5.1f  ΔVMAF %+5.1f  Δstall/UE %+5.2f s" % (COND[c], dp, dv, ds))
    w("  worst 10 cells for %s (Δ%%teacher | ΔVMAF | stall ours/ref):" % args.arm)
    for x in sorted(cells, key=lambda x: ra[x]["pct"] - rr[x]["pct"])[:10]:
        w("    %-24s %-8s %+6.1f | %+5.1f | %5.2f / %5.2f s" % (x[0], COND[x[1]], ra[x]["pct"] - rr[x]["pct"],
                                                            ra[x]["vmaf"] - rr[x]["vmaf"], sa.get(x, 0), sr.get(x, 0)))
    stall_cells = sorted(cells, key=lambda x: -(sa.get(x, 0) - sr.get(x, 0)))[:8]
    w("  cells where %s stalls most more than %s:" % (args.arm, args.ref))
    for x in stall_cells:
        w("    %-24s %-8s stall %5.2f vs %5.2f s/UE" % (x[0], COND[x[1]], sa.get(x, 0), sr.get(x, 0)))

    # 2-4 per arm
    for stem in (args.arm, args.ref):
        rows = collect(stem)
        w("\n== %s: %d decisions joined with teacher labels" % (stem, len(rows)))
        ask_table(rows, lambda r: COND[r["cond"]], list(COND.values()), "2a. rung vs teacher by link rate", w)
        ask_table(rows, lambda r: seg_bucket(r["seg"]), ["seg 0-3", "seg 4-13", "seg 14-27", "seg 28+"],
                  "2b. by segment index (training sessions covered seg 0-13 only)", w)
        ask_table(rows, lambda r: buf_bucket(r["buf"]), ["buf <5s", "buf 5-15s", "buf 15-25s", "buf >=25s"],
                  "2c. by buffer level at decision time", w)
        if stem == args.arm:
            w("\n3. guard activity by rate (fired %% | mean rungs removed | pre-guard rung above teacher %%)")
            for c in COND:
                rc = [r for r in rows if r["cond"] == c]
                g = [r for r in rc if r["guard"]]
                w("  %-8s %5.1f%% | %.2f | %5.1f%%" % (COND[c], pct(len(g), len(rc)),
                                                      sum(r["pre"] - r["rung"] for r in g) / max(1, len(g)),
                                                      pct(sum(r["pre"] > r["teacher"] for r in g), len(g))))
            post_over = [r for r in rows if r["guard"] and r["rung"] > r["teacher"]]
            w("  guarded decisions still above the teacher after the guard: %d (%.1f%% of guarded)" % (
                len(post_over), pct(len(post_over), sum(r["guard"] for r in rows))))
        stall_attribution(rows, w, stem)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
