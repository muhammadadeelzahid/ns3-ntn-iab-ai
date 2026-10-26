#!/usr/bin/env python3
"""Extract TRUE closed-loop ns-3 rebuffering per (arm, video, condition) for a paper-safe Fig-1.

The perceptual_qoe REPLAY stall is an artifact for the model arm (it re-simulates downloads and ignores
the serving buffer-guard, inflating the model's stall ~10-100x). This reads the ACTUAL ns-3 MPEG-player
interruption instead:

  MODEL (bridge runs, Quic_artifacts/AiDash/bench_down_scored/<video>/run_<cond>/decisions.jsonl):
    each decision logs cumulative `interruption_time_s` per node -> per-node stall = max over its decisions;
    per-run stall = mean over the run's nodes (mean per-UE rebuffering).

  HEURISTICS (ns-3 runs, Quic_artifacts/<ARM>/<video>/run_<seed>/sim.log):
    the MPEG player prints `FinalizeInterruptionTime ... current interruption_time: X s` per UE. X is the
    REAL mid-playback rebuffer; the end-of-run `ue-id: N InterruptionTime: M` is CONTAMINATED by idle time
    after the video finishes (player DONE, idling to sim-stop) so it is NOT used. per-run stall = mean of
    the per-UE current interruption_time. Odd seeds {1,3,5,7,9} map to conditions {1..5} (same as the model).

Writes results/true_stall.csv (arm,video,cond,stall_s) + prints per-arm/per-cond means.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
Q = os.path.join(_REPO, "Quic_artifacts")
VIDS = ["big_buck_bunny", "chef_cooking", "city_night_traffic", "hiking", "peace_autumn_lake",
        "pexels_28887170", "pexels_30886101", "pexels_34254266", "pexels_34797065", "pexels_35132716"]
SEED2COND = {1: 1, 3: 2, 5: 3, 7: 4, 9: 5}
CUR_RE = re.compile(r"current interruption_time:\s*([0-9.]+)\s*s")
UPD_RE = re.compile(r"Updated interruption_time:\s*([0-9.]+)\s*s")
PAUSE_RE = re.compile(r"No frames to play at t=([0-9.]+)s\. Player ID=(\d+)")
RESUME_RE = re.compile(r"Play resumed at t=([0-9.]+)s\. Player ID=(\d+)")
RATE = {1: "100Gbs", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}
LOGS_BENCH = os.path.join(_REPO, "logs_bench")
METADIR = "/project/6030214/adeel/abr/output/metadata"


def _segcount(video):
    """Number of 2 s segments for a video (metadata CSV rows), for the freeze-vs-finished test."""
    p = os.path.join(METADIR, "%s_segments.csv" % video)
    if not os.path.isfile(p):
        return None
    with open(p, errors="ignore") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def true_stall_per_ue(text, segment_count):
    """Per-UE TRUE rebuffer from a run's ns-3 sim.log: resolved mid-playback stalls PLUS a never-resumed
    terminal freeze — the terminal part counted ONLY when that UE did not finish the video
    (played < segment_count), so end-of-content idle (player DONE, idling to sim-stop) is excluded.
    This is the fix for permanent freezes that decisions.jsonl's interruption_time_s (finalized only on
    resume) and the old pre-finalize 'current interruption_time' read both missed. Same rule for every
    arm so the comparison stays fair. Returns a list of per-UE stall seconds."""
    st = defaultdict(lambda: {"pause": [], "resume": [], "played": -1, "last": 0.0})
    for t, p in PAUSE_RE.findall(text):
        st[int(p)]["pause"].append(float(t))
    for t, p in RESUME_RE.findall(text):
        st[int(p)]["resume"].append(float(t))
    for line in text.splitlines():
        if "PLAYING FRAME" not in line:
            continue
        mt = re.search(r"([0-9]+\.[0-9]+) PLAYING FRAME", line)
        mp = re.search(r"PlayerId: (\d+)", line)
        ms = re.search(r"SegId: (\d+)", line)
        if not (mt and mp):
            continue
        p = int(mp.group(1))
        st[p]["last"] = max(st[p]["last"], float(mt.group(1)))
        if ms:
            st[p]["played"] = max(st[p]["played"], int(ms.group(1)))
    if not st:
        return []
    cur = [float(x) for x in CUR_RE.findall(text)]
    upd = [float(x) for x in UPD_RE.findall(text)]
    players = sorted(st)
    # sim end: for a UE that ended paused, last_pause + (Updated-current) (finalize pair, player order);
    # all paused UEs agree. Fall back to last observed activity if finalize lines are absent.
    sim_end = max([d["last"] for d in st.values()] + [max(d["pause"]) for d in st.values() if d["pause"]] + [0.0])
    if len(upd) == len(players) and len(cur) == len(players):
        for i, p in enumerate(players):
            d = st[p]
            if len(d["pause"]) > len(d["resume"]) and d["pause"]:
                sim_end = max(sim_end, max(d["pause"]) + (upd[i] - cur[i]))
    vals = []
    for p in players:
        d = st[p]
        resolved = sum(r - q for q, r in zip(sorted(d["pause"]), sorted(d["resume"])))
        ended_paused = len(d["pause"]) > len(d["resume"])
        finished = segment_count is not None and (d["played"] + 1) >= segment_count
        terminal = (sim_end - max(d["pause"])) if (ended_paused and d["pause"] and not finished) else 0.0
        vals.append(resolved + terminal)
    return vals


def _find_sim_log(arm_subdir, video, cond, seed=None):
    """Locate a run's sim.log: run-dir first, then the per-cell paper logs in logs_bench/ (model arms,
    whose sim.log was captured there, matched by video+rate; pick the most complete run)."""
    run_tag = "run_%d" % (seed if seed is not None else cond)
    rundir = os.path.join(Q, arm_subdir, video, run_tag, "sim.log")
    if os.path.isfile(rundir):
        return rundir
    cands = glob.glob(os.path.join(LOGS_BENCH, "task_*_%s_%s_run*.log" % (video, RATE[cond])))
    if cands:
        return max(cands, key=lambda f: open(f, errors="ignore").read().count("ue-id:"))
    return None


def model_stall(dec_subdir=os.path.join("AiDash", "bench_down_scored")):
    """{(video,cond): (mean-per-UE, sum-over-UEs) TRUE rebuffer} for a bridge arm (AiDash model or
    Pensieve), from the run's ns-3 sim.log via true_stall_per_ue — which COUNTS never-resumed
    permanent freezes (the ones decisions.jsonl's resume-finalized interruption_time_s reported as 0).
    Falls back silently to no entry when the run's sim.log cannot be located."""
    out = {}
    for v in VIDS:
        sc = _segcount(v)
        for cond in range(1, 6):
            log = _find_sim_log(dec_subdir, v, cond)
            if log is None:
                continue
            vals = true_stall_per_ue(open(log, errors="ignore").read(), sc)
            if vals:
                out[(v, cond)] = (mean(vals), sum(vals))
    return out


def heuristic_stall(arm):
    """{(video,cond): (mean-per-UE, sum-over-UEs)} from each run's sim.log via true_stall_per_ue
    (same freeze-aware rule as the model arms, replacing the old pre-finalize 'current
    interruption_time' read which also missed never-resumed freezes)."""
    out = {}
    for v in VIDS:
        sc = _segcount(v)
        for seed, cond in SEED2COND.items():
            log = _find_sim_log(arm, v, cond, seed=seed)
            if log is None:
                continue
            vals = true_stall_per_ue(open(log, errors="ignore").read(), sc)
            if vals:
                out[(v, cond)] = (mean(vals), sum(vals))
    return out


def main():
    arms = {"AiDash": model_stall(),
            "Pensieve": model_stall("Pensieve"),
            "Bola": heuristic_stall("Bola"),
            "Raahs": heuristic_stall("Raahs"),
            "Aaash": heuristic_stall("Aaash")}
    outcsv = os.path.join(_HERE, "results", "true_stall.csv")
    with open(outcsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "video", "cond", "stall_s", "stall_sum_s"])  # stall_s=mean per-UE (Fig 1); stall_sum_s=sum over UEs
        for arm, d in arms.items():
            for (v, c), (m, s) in sorted(d.items()):
                w.writerow([arm, v, c, "%.4f" % m, "%.4f" % s])
    print("wrote", outcsv)
    print("\n%-8s | %6s %6s %6s %6s %6s | %7s | n   (mean per-UE stall, s)" %
          ("arm", "100G", "36M", "18M", "9M", "4.5M", "overall"))
    for arm in ("AiDash", "Pensieve", "Raahs", "Aaash", "Bola"):
        d = arms[arm]
        per_c = {c: [m for (v, cc), (m, s) in d.items() if cc == c] for c in range(1, 6)}
        cells = [mean(per_c[c]) if per_c[c] else float("nan") for c in range(1, 6)]
        allv = [m for (m, s) in d.values()]
        print("%-8s | %6.2f %6.2f %6.2f %6.2f %6.2f | %7.3f | %d" %
              (arm, cells[0], cells[1], cells[2], cells[3], cells[4], mean(allv), len(allv)))


if __name__ == "__main__":
    main()
