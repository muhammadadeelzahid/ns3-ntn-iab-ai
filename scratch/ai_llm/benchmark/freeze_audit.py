#!/usr/bin/env python3
"""Freeze / reconnect audit of ns-3 DASH session logs (bridge-arm cell logs or native sim.log files).

For every log: per-UE (player) pause/resume events, highest segment played, whether the UE ended paused
without finishing the video (= permanent freeze), the client's dead-segment reconnects ('[DashClient] ...
reconnect #k' lines + 'reconnects: N' in the final ue-id stats) and the number of 'giving up' events.

  python3 freeze_audit.py <log file or glob> [...]      e.g. 'logs_bench_qmpc/task_*_100Gbs_run1.log'
"""
import glob
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
METADIR = "/project/6030214/adeel/abr/output/metadata"
PAUSE_RE = re.compile(r"No frames to play at t=([0-9.]+)s\. Player ID=(\d+)")
RESUME_RE = re.compile(r"Play resumed at t=([0-9.]+)s\. Player ID=(\d+)")
PLAY_RE = re.compile(r"PlayerId: (\d+) VidId: \d+ SegId: (\d+)")
RECON_RE = re.compile(r"\[DashClient\] node (\d+) seg (\d+) dead \(bytes=(\d+) frames=(\d+)\) at t=([0-9.]+)s -> reconnect #(\d+)")
GIVEUP_RE = re.compile(r"\[DashClient\] node (\d+) seg (\d+) still dead")
STATS_RE = re.compile(r"ue-id: (\d+) InterruptionTime: ([0-9.]+) .*?(?:reconnects: (\d+))?$", re.M)
VIDEO_RE = re.compile(r"(?:task_\d+_)?(.+?)_(100Gbs|36Mbps|18Mbps|9Mbps|4\.5Mbps)_run\d+\.log$")


def segcount(video):
    p = os.path.join(METADIR, "%s_segments.csv" % video)
    if not os.path.isfile(p):
        return None
    with open(p, errors="ignore") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def video_of(path):
    m = VIDEO_RE.search(os.path.basename(path))
    if m:
        return m.group(1), m.group(2)
    parts = path.split(os.sep)           # Quic_artifacts/<Arm>/<video>/run_<k>/sim.log
    return (parts[-3], parts[-2]) if len(parts) >= 3 else (os.path.basename(path), "?")


def audit(path):
    t = open(path, errors="ignore").read()
    video, tag = video_of(path)
    sc = segcount(video)
    np_ = Counter(p for _, p in PAUSE_RE.findall(t)); nr = Counter(p for _, p in RESUME_RE.findall(t))
    played = {}
    for p, s in PLAY_RE.findall(t):
        played[p] = max(played.get(p, -1), int(s))
    ues = sorted(set(list(np_) + list(played)), key=int)
    frozen = [p for p in ues if np_[p] > nr[p] and sc is not None and played.get(p, -1) + 1 < sc]
    finished = [p for p in ues if sc is not None and played.get(p, -1) + 1 >= sc]
    recon = RECON_RE.findall(t); giveup = GIVEUP_RE.findall(t)
    done = "ue-id:" in t and "InterruptionTime" in t
    return {"video": video, "tag": tag, "nseg": sc, "ues": len(ues), "frozen": len(frozen), "finished": len(finished),
            "played": {p: played.get(p, -1) for p in ues}, "reconnects": len(recon), "giveups": len(giveup),
            "recon_events": [(int(n), int(s), int(b), int(f), float(tt), int(k)) for n, s, b, f, tt, k in recon],
            "done": done, "stalls": {p: (np_[p], nr[p]) for p in ues}}


def main(argv):
    paths = []
    for a in argv:
        paths += sorted(glob.glob(a)) if any(ch in a for ch in "*?[") else [a]
    tot_ue = tot_frz = tot_rec = tot_give = 0
    print("%-28s %-8s %4s | ues frozen finished | reconnects giveups | played/seg per UE" % ("video", "cond", "done"))
    for p in paths:
        r = audit(p)
        tot_ue += r["ues"]; tot_frz += r["frozen"]; tot_rec += r["reconnects"]; tot_give += r["giveups"]
        print("%-28s %-8s %4s | %3d %6d %8d | %10d %7d | %s" % (
            r["video"][:28], r["tag"], "yes" if r["done"] else "no", r["ues"], r["frozen"], r["finished"],
            r["reconnects"], r["giveups"], " ".join("p%s:%d/%s" % (k, v + 1, r["nseg"]) for k, v in r["played"].items())))
        for n, s, b, f, tt, k in r["recon_events"][:6]:
            print("      reconnect: node %d seg %d bytes %d frames %d t=%.1fs #%d" % (n, s, b, f, tt, k))
    print("TOTAL: UEs %d  frozen %d (%.1f%%)  reconnects %d  giveups %d" % (
        tot_ue, tot_frz, 100.0 * tot_frz / max(tot_ue, 1), tot_rec, tot_give))


if __name__ == "__main__":
    main(sys.argv[1:])
