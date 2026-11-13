#!/usr/bin/env python3
"""Export existing ns-3 ABR benchmark runs into DATA_SPEC v1 "run bundles" for the front-end viewer.

This reads the LLM (AiDash) scored runs in Quic_artifacts/AiDash/bench_down_scored/<video>/run_N/
(decisions.jsonl + throughput_trace.jsonl) plus the encoded-video assets under
<abr-root>/{metadata,by_video}/ and writes one self-contained bundle per (video, link-rate).

NO new simulation is run. The script is strictly READ-ONLY on every existing path: it only ever
writes under --out. See README/plan for the full safety contract; the guard in _assert_write_target()
enforces that every write lands inside the resolved output dir.

Bundle format: DATA_SPEC.md v1 (run.json, segments.jsonl, player_events.jsonl, samples.jsonl,
summary.json, logs/policy.log, videos/<video>/{video.json,segments.jsonl,r{rung}/seg_*.mp4}).

Key conventions (all sourced from the repo, not invented):
  * Ladder: scratch/ai_llm/paths.py NS3_RATE_LADDER_BPS (6 rungs 250k..12M).
  * run-index -> feeder rate: COND below, mirrors scratch/ai_llm/benchmark/paper_style.py.
  * decisions.jsonl segment_id is 1-indexed; 0-indexed segment S = segment_id - 1, and chosen_rung
    is the rung DELIVERED for S (see perceptual_qoe._algo_rungs_from_decisions). So the first
    decision (segment_id=1) is segment 0 -- there is no separate bootstrap row.
  * Encoded segments are 2.0 s; the simulated player buffered in 3.3 s units (100 frames x 33 ms).
    Both are reported; sim times are copied verbatim (not rescaled).

Usage:
  python scratch/ai_llm/export_viewer_bundle.py --list          # dry-run, write nothing
  python scratch/ai_llm/export_viewer_bundle.py --videos hiking --runs run_4
  python scratch/ai_llm/export_viewer_bundle.py --with-media    # all defaults, copy media
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from paths import NS3_RATE_LADDER_BPS, NUM_LADDER_RUNGS  # noqa: E402
from metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402

# run-index -> S1-U feeder rate label. Mirrors scratch/ai_llm/benchmark/paper_style.py COND
# (kept inline so we don't import matplotlib via paper_style on headless nodes).
COND = {1: "100Gb/s", 2: "36Mbps", 3: "18Mbps", 4: "9Mbps", 5: "4.5Mbps"}

# Fallback per-rung max-height ladder (abr/config/default.yaml representations), used only if ffprobe
# is unavailable. Widths are then assumed from catalog orientation (16:9 / 9:16).
_MAX_HEIGHT = {"250k": 240, "750k": 360, "1500k": 480, "3000k": 720, "6000k": 1080, "12000k": 2160}

ENCODED_SEGMENT_DURATION_S = 2.0   # DATA_SPEC Q1: encoded media duration
PLAYER_SEGMENT_DURATION_S = 3.3    # DATA_SPEC Q1: simulated player's buffered unit (100 x 33 ms)

# Default 5-video set spanning the headroom bins {0,2,3,5,6} of the 10-video test set.
DEFAULT_VIDEOS = ["big_buck_bunny", "pexels_30886101", "pexels_34254266", "hiking", "peace_autumn_lake"]
DEFAULT_RUNS = ["run_2", "run_3", "run_4"]  # 36 / 18 / 9 Mbps


# --------------------------------------------------------------------------------------------------
# Safety: every write MUST be inside the resolved output root.
# --------------------------------------------------------------------------------------------------
class ExportError(RuntimeError):
    pass


_OUT_ROOT_REAL: Optional[str] = None


def _assert_write_target(path: Path) -> Path:
    """Abort unless `path` resolves inside the output root. Last line of defence against clobbering
    any source tree."""
    if _OUT_ROOT_REAL is None:
        raise ExportError("output root not initialised before a write")
    rp = os.path.realpath(str(path))
    if not (rp == _OUT_ROOT_REAL or rp.startswith(_OUT_ROOT_REAL + os.sep)):
        raise ExportError("refusing to write outside --out: %s" % path)
    return path


def _write_text(path: Path, text: str) -> None:
    _assert_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _sanitize(obj):
    """Recursively replace non-finite floats (NaN/Inf) with None so output is strict, viewer-safe JSON.
    Some decisions.jsonl rows carry NaN throughput estimates; `NaN` is not valid JSON."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _write_jsonl(path: Path, records: List[dict]) -> None:
    _write_text(path, "".join(
        json.dumps(_sanitize(r), separators=(",", ": "), allow_nan=False) + "\n" for r in records))


def _write_json(path: Path, obj: dict) -> None:
    _write_text(path, json.dumps(_sanitize(obj), indent=2, allow_nan=False) + "\n")


# --------------------------------------------------------------------------------------------------
# Small readers / helpers (all read-only on sources)
# --------------------------------------------------------------------------------------------------
def _rate_bps(run: str) -> Optional[int]:
    """run_N label -> feeder link rate in bps, or None for the uncongested 100Gb/s baseline."""
    idx = int(run.split("_")[1])
    label = COND[idx]
    if label.endswith("Gb/s"):
        return None  # uncongested; DATA_SPEC wants null
    num = float(label.replace("Mbps", ""))
    return int(num * 1_000_000)


def _rate_label(run: str) -> str:
    return COND[int(run.split("_")[1])].replace("/", "")  # "9Mbps" / "100Gbs"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(_HERE),
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def _read_decisions(path: Path) -> Dict[int, List[dict]]:
    """node_id -> decision rows sorted by segment_id (1-indexed as logged)."""
    by_node: Dict[int, List[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_node.setdefault(int(d["node_id"]), []).append(d)
    for rows in by_node.values():
        rows.sort(key=lambda r: int(r["segment_id"]))
    return by_node


def _ffprobe_wh(mp4: Path) -> Optional[Tuple[int, int]]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(mp4)],
            capture_output=True, text=True, check=True)
        w, h = out.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return None


def _ffprobe_fps(mp4: Path) -> Optional[float]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True, check=True)
        s = out.stdout.strip()
        if "/" in s:
            n, d = s.split("/")
            return round(float(n) / float(d), 3) if float(d) else None
        return float(s)
    except Exception:
        return None


def _load_catalog(abr_root: Path) -> Dict[str, dict]:
    """video id -> catalog entry (title/source/license). Best-effort.

    Prefers in-process PyYAML; if this interpreter lacks it, shells out to the system `python3`
    (which has yaml in this project) to emit JSON. Read-only; returns {} if nothing works."""
    cat = abr_root.parent / "config" / "video_catalog.yaml"
    if not cat.is_file():
        return {}
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(open(cat, encoding="utf-8"))
        return {v["id"]: v for v in data.get("videos", []) if "id" in v}
    except Exception:
        pass
    code = ("import yaml,json,sys;"
            "d=yaml.safe_load(open(sys.argv[1]));"
            "print(json.dumps({v['id']:v for v in d.get('videos',[]) if 'id' in v}))")
    for interp in ("python3", "/usr/bin/python3"):
        try:
            out = subprocess.run([interp, "-c", code, str(cat)],
                                 capture_output=True, text=True, check=True)
            return json.loads(out.stdout)
        except Exception:
            continue
    return {}


# --------------------------------------------------------------------------------------------------
# Per-video assets (written once under <out>/_videos/<video>, shared by that video's run bundles)
# --------------------------------------------------------------------------------------------------
def build_video_assets(video: str, abr_root: Path, out_root: Path, catalog: Dict[str, dict],
                       with_media: bool, media_mode: str) -> Path:
    """Write <out>/_videos/<video>/{video.json,segments.jsonl,r{rung}/...}. Returns that dir."""
    vdir = out_root / "_videos" / video
    if (vdir / "video.json").is_file():
        return vdir  # already built in this run

    meta_csv = abr_root / "metadata" / f"{video}_segments.csv"
    if not meta_csv.is_file():
        raise ExportError("metadata CSV not found for %s: %s" % (video, meta_csv))
    index = load_metadata_index(str(meta_csv))
    seg_ids = index.segment_ids()
    segment_count = len(seg_ids)
    src_seg_root = abr_root / "by_video" / video / "segments"

    # video-level segment metadata (one record per segment,rung)
    vseg_records: List[dict] = []
    for s in seg_ids:
        m = index.get(s)
        for r, lvl in enumerate(LADDER_LEVELS):
            vseg_records.append({
                "type": "video_segment", "segment": s, "rung": r,
                "bytes": int(round(m.size_bytes[lvl])),
                "actual_bitrate_bps": int(round(m.actual_bitrate_kbps[lvl] * 1000)),
                "vmaf": round(float(m.vmaf[lvl]), 4),
            })
    _write_jsonl(vdir / "segments.jsonl", vseg_records)

    # per-rung resolution via ffprobe (fallback to max-height ladder + orientation aspect)
    entry = catalog.get(video, {})
    orientation = str(entry.get("orientation", "landscape")).lower()
    rungs = []
    fps = None
    for r, lvl in enumerate(LADDER_LEVELS):
        first = src_seg_root / lvl / (index.get(seg_ids[0]).segment_file if seg_ids else "seg_0000.mp4")
        wh = _ffprobe_wh(first) if first.is_file() else None
        if wh is None:
            h = _MAX_HEIGHT[lvl]
            w = int(round(h * (9 / 16))) if orientation == "portrait" else int(round(h * (16 / 9)))
            wh = (w, h)
        if fps is None and first.is_file():
            fps = _ffprobe_fps(first)
        rungs.append({"rung": r, "bitrate_bps": NS3_RATE_LADDER_BPS[r], "width": wh[0], "height": wh[1]})

    video_json = {
        "name": video,
        "title": entry.get("title", video),
        "source": entry.get("source"),
        "license": entry.get("license"),
        "duration_s": round(segment_count * ENCODED_SEGMENT_DURATION_S, 3),
        "segment_duration_s": ENCODED_SEGMENT_DURATION_S,
        "frame_rate": fps,
        "segment_count": segment_count,
        "rungs": rungs,
        "segment_file_pattern": "r{rung}/seg_{segment:04d}.mp4",
        "init_file_pattern": None,
    }
    _write_json(vdir / "video.json", video_json)

    # media: copy/symlink the 6 ABR rungs into r0..r5 (skip the unused 'original' rung)
    if with_media:
        for r, lvl in enumerate(LADDER_LEVELS):
            dst_dir = _assert_write_target(vdir / f"r{r}")
            dst_dir.mkdir(parents=True, exist_ok=True)
            for s in seg_ids:
                src = src_seg_root / lvl / index.get(s).segment_file
                if not src.is_file():
                    raise ExportError("media segment missing: %s" % src)
                dst = _assert_write_target(dst_dir / f"seg_{s:04d}.mp4")
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if media_mode == "symlink":
                    dst.symlink_to(os.path.realpath(str(src)))
                else:
                    shutil.copy2(str(src), str(dst))  # copy OUT; source untouched
    return vdir


def _link_video_into_bundle(bundle: Path, video: str, out_root: Path, media_mode: str) -> None:
    """Point <bundle>/videos/<video> at the shared _videos/<video> tree (symlink by default)."""
    dst = _assert_write_target(bundle / "videos" / video)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shared = out_root / "_videos" / video
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if media_mode == "copy":
        shutil.copytree(str(shared), str(dst))
    else:
        dst.symlink_to(os.path.relpath(os.path.realpath(str(shared)), str(dst.parent)))


# --------------------------------------------------------------------------------------------------
# Per-(video, run) bundle
# --------------------------------------------------------------------------------------------------
def _clean_rows(rows: List[dict], segment_count: int, label: str) -> List[dict]:
    """Dedupe decision rows by segment_id (last wins) and drop any that index past the video's
    segment_count (would be a looped/overflow id with no media). Returns rows sorted by segment_id.
    These are real serving-log artifacts (occasional duplicate / dropped decision lines)."""
    by_seg: Dict[int, dict] = {}
    dropped = 0
    for d in rows:
        s = int(d["segment_id"]) - 1  # 0-indexed
        if s < 0 or s >= segment_count:
            dropped += 1
            continue
        by_seg[s] = d  # last occurrence wins
    if dropped:
        print(f"        [note] {label}: dropped {dropped} decision(s) indexing past segment_count={segment_count}")
    return [by_seg[s] for s in sorted(by_seg)]


def _ue_segments(cleaned: List[dict], index, label: str) -> List[dict]:
    """Cleaned, sorted decision rows -> contiguous 0-indexed DATA_SPEC segment records.

    Missing segments (dropped log lines) are carry-forward filled and flagged gap_filled=true so the
    per-UE sequence is contiguous (DATA_SPEC: segment increases by one) without inventing rung changes."""
    present: Dict[int, dict] = {int(d["segment_id"]) - 1: d for d in cleaned}
    if not present:
        return []
    recs: List[dict] = []
    prev_bps: Optional[int] = None
    prev_prev_bps: Optional[int] = None
    last_rec: Optional[dict] = None
    ngaps = 0
    for s in range(0, max(present) + 1):
        d = present.get(s)
        if d is None:  # gap: carry forward last delivered rung, interpolate time, flag it
            ngaps += 1
            rung = last_rec["rung"] if last_rec else 0
            bitrate = NS3_RATE_LADDER_BPS[rung]
            m = index.get(s)
            nbytes = int(round(m.size_bytes[LADDER_LEVELS[rung]])) if m else None
            t = last_rec["done_s"] if last_rec else 0.0
            rec = {
                "type": "segment", "ue": None, "segment": s,
                "request_s": round(t, 6), "done_s": round(t, 6),
                "rung": rung, "bitrate_bps": bitrate, "bytes": nbytes,
                "decision": {"gap_filled": True, "throughput_estimate_bps": None, "buffer_s": None,
                             "current_bitrate_bps": prev_bps, "prev_bitrate_bps": prev_prev_bps,
                             "rtt_ms": None, "loss_rate": None, "fallback": False,
                             "decision_delay_s": None},
            }
            recs.append(rec)
            last_rec = rec
            prev_prev_bps, prev_bps = prev_bps, bitrate
            continue
        rung = int(d["chosen_rung"])
        bitrate = int(d["chosen_bitrate_bps"])
        if bitrate != NS3_RATE_LADDER_BPS[rung]:
            raise ExportError("bitrate %d != ladder[rung %d]=%d (seg %d)"
                              % (bitrate, rung, NS3_RATE_LADDER_BPS[rung], s))
        m = index.get(s)
        nbytes = int(round(m.size_bytes[LADDER_LEVELS[rung]])) if m else None
        done = float(d["sim_time_s"])
        fetch = float(d.get("segment_fetch_time_s", 0.0) or 0.0)
        rtt = d.get("smoothed_rtt_ms")
        loss = d.get("loss_rate_recent")
        rec = {
            "type": "segment", "ue": None, "segment": s,
            "request_s": round(done - fetch, 6), "done_s": round(done, 6),
            "rung": rung, "bitrate_bps": bitrate, "bytes": nbytes,
            "decision": {
                "throughput_estimate_bps": d.get("throughput_estimate_bps"),
                "buffer_s": d.get("buffer_level_s"),
                "current_bitrate_bps": prev_bps,
                "prev_bitrate_bps": prev_prev_bps,
                "rtt_ms": None if rtt in (None, -1, -1.0) else rtt,
                "loss_rate": None if loss in (None, -1, -1.0) else loss,
                "fallback": d.get("reason", "") != "llm_policy_model",
                "decision_delay_s": round(float(d.get("inference_latency_ms", 0.0) or 0.0) / 1000.0, 6),
            },
        }
        recs.append(rec)
        last_rec = rec
        prev_prev_bps, prev_bps = prev_bps, bitrate
    if ngaps:
        print(f"        [note] {label}: carry-forward filled {ngaps} missing segment(s)")
    return recs


def _ue_events(cleaned: List[dict], segs: List[dict], sim_stop: float) -> List[dict]:
    """stream_start / playback_start / derived stalls / playback_end for one UE.

    Stalls are derived from cumulative interruption_time_s growth between consecutive (cleaned)
    decisions, placed on the timeline via each decision's segment done_s."""
    ev: List[dict] = []
    stream_start = min(s["request_s"] for s in segs) if segs else 0.0
    ev.append({"type": "player_event", "ue": None, "t": round(stream_start, 6), "event": "stream_start"})
    first_done = segs[0]["done_s"] if segs else stream_start
    ev.append({"type": "player_event", "ue": None, "t": round(first_done, 6), "event": "playback_start"})
    seg_done = {s["segment"]: s["done_s"] for s in segs}
    prev_it = 0.0
    prev_t = first_done
    for d in cleaned:
        seg_idx = int(d["segment_id"]) - 1
        it = float(d.get("interruption_time_s", 0.0) or 0.0)
        if it - prev_it > 1e-3:
            start = prev_t
            end = start + (it - prev_it)
            ev.append({"type": "player_event", "ue": None, "t": round(start, 6),
                       "event": "stall_start", "segment": seg_idx, "derived": True})
            ev.append({"type": "player_event", "ue": None, "t": round(end, 6),
                       "event": "stall_end", "segment": seg_idx, "derived": True})
        prev_it = it
        prev_t = seg_done.get(seg_idx, prev_t)
    end_t = max((s["done_s"] for s in segs), default=stream_start)
    ev.append({"type": "player_event", "ue": None, "t": round(min(end_t, sim_stop) if sim_stop else end_t, 6),
               "event": "playback_end"})
    ev.sort(key=lambda e: (e["t"], 0 if e["event"] != "stall_end" else 1))
    return ev


def _ue_samples(cleaned: List[dict]) -> List[dict]:
    out: List[dict] = []
    for d in cleaned:
        rtt = d.get("smoothed_rtt_ms")
        loss = d.get("loss_rate_recent")
        out.append({
            "type": "sample", "ue": None, "t": round(float(d["sim_time_s"]), 6),
            "buffer_s": d.get("buffer_level_s"),
            "throughput_bps": d.get("throughput_estimate_bps"),
            "rtt_ms": None if rtt in (None, -1, -1.0) else rtt,
            "loss_rate": None if loss in (None, -1, -1.0) else loss,
        })
    out.sort(key=lambda r: r["t"])
    return out


def _ue_summary(ue: int, rows: List[dict], segs: List[dict], events: List[dict]) -> dict:
    import math
    n = len(segs)
    avg_bitrate = sum(s["bitrate_bps"] for s in segs) / n if n else 0.0
    thr = [d.get("throughput_estimate_bps") for d in rows
           if isinstance(d.get("throughput_estimate_bps"), (int, float)) and math.isfinite(d["throughput_estimate_bps"])]
    avg_thr = sum(thr) / len(thr) if thr else None
    rebuffer = max((float(d.get("interruption_time_s", 0.0) or 0.0) for d in rows), default=0.0)
    stall_count = max((int(d.get("interruptions", 0) or 0) for d in rows), default=0)
    switches = sum(1 for a, b in zip(segs, segs[1:]) if a["rung"] != b["rung"])
    ss = next((e["t"] for e in events if e["event"] == "stream_start"), 0.0)
    pb = next((e["t"] for e in events if e["event"] == "playback_start"), ss)
    return {
        "ue": ue,
        "avg_bitrate_bps": int(round(avg_bitrate)),
        "avg_throughput_bps": int(round(avg_thr)) if avg_thr is not None else None,
        "rebuffer_time_s": round(rebuffer, 4),
        "stall_count": stall_count,
        "switch_count": switches,
        "startup_delay_s": round(pb - ss, 4),
    }


def _best_task_log(task_dir: Path, video: str, rate_label: str,
                   scored: Dict[Tuple[int, int], int]) -> Optional[Tuple[Path, float, int]]:
    """Find the real per-cell ns-3 stdout log (logs_bench/task_*_<video>_<rate>_run*.log) for this
    (video, rate). The filename identifies the cell; among duplicate/partial executions pick the one
    whose decision stream best agrees with the scored decisions.jsonl. Returns (path, agreement, n)."""
    if not task_dir.is_dir():
        return None
    cands = sorted(task_dir.glob(f"task_*_{video}_{rate_label}_run*.log"))
    best: Optional[Tuple[Path, float, int]] = None
    for f in cands:
        log: Dict[Tuple[int, int], int] = {}
        for line in open(f, errors="ignore", encoding="utf-8"):
            m = re.search(r"ue-id: (\d+) videoId: \d+ segmentId: (\d+) newBitRate: (\d+)", line)
            if m:
                bps = int(m.group(3))
                rung = min(range(NUM_LADDER_RUNGS), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))
                log[(int(m.group(1)), int(m.group(2)))] = rung
        common = set(scored) & set(log)
        if not common:
            continue
        agree = sum(1 for k in common if scored[k] == log[k]) / len(common)
        # rank by agreement, then by coverage (prefer the complete, non-partial execution)
        if best is None or (agree, len(common)) > (best[1], best[2]):
            best = (f, agree, len(common))
    return best


_PAUSE_RE = re.compile(r"No frames to play at t=([0-9.]+)s\. Player ID=(\d+)")
_RESUME_RE = re.compile(r"Play resumed at t=([0-9.]+)s\. Player ID=(\d+)")
_SIMCUR_RE = re.compile(r"current interruption_time:\s*([0-9.]+)\s*s")
_SIMUPD_RE = re.compile(r"Updated interruption_time:\s*([0-9.]+)\s*s")


def _parse_simlog(text: str, nodes: List[int]):
    """Parse the REAL ns-3 stdout for ground-truth playback: per-node pause/resume, first/last frame,
    segments played, and the terminal (never-resumed) freeze. Player ID p maps to sorted nodes[p]
    (creation order). Returns (per_node dict, sim_end)."""
    pid2node = {p: n for p, n in enumerate(nodes)}
    st = {n: {"pause": [], "resume": [], "first": None, "last": 0.0, "played_max": -1} for n in nodes}
    for t, p in _PAUSE_RE.findall(text):
        n = pid2node.get(int(p))
        if n is not None:
            st[n]["pause"].append(float(t))
    for t, p in _RESUME_RE.findall(text):
        n = pid2node.get(int(p))
        if n is not None:
            st[n]["resume"].append(float(t))
    for line in text.splitlines():
        if "PLAYING FRAME" not in line:
            continue
        mt = re.search(r"([0-9]+\.[0-9]+) PLAYING FRAME", line)
        mp = re.search(r"PlayerId: (\d+)", line)
        ms = re.search(r"SegId: (\d+)", line)
        if not (mt and mp):
            continue
        n = pid2node.get(int(mp.group(1)))
        if n is None:
            continue
        tt = float(mt.group(1))
        if st[n]["first"] is None:
            st[n]["first"] = tt
        st[n]["last"] = max(st[n]["last"], tt)
        if ms:
            st[n]["played_max"] = max(st[n]["played_max"], int(ms.group(1)))
    cur = [float(x) for x in _SIMCUR_RE.findall(text)]
    upd = [float(x) for x in _SIMUPD_RE.findall(text)]
    # sim end: for a player that ended paused, sim_end = last_pause + (Updated - current) (finalize pair,
    # in player order). All paused players agree; fall back to last activity if finalize lines absent.
    sim_end = max((d["last"] for d in st.values()), default=0.0)
    if len(upd) == len(nodes) and len(cur) == len(nodes):
        for p, n in pid2node.items():
            d = st[n]
            if len(d["pause"]) > len(d["resume"]) and d["pause"]:
                sim_end = max(sim_end, max(d["pause"]) + (upd[p] - cur[p]))
    for n, d in st.items():
        d["ended_paused"] = len(d["pause"]) > len(d["resume"])
        d["resolved"] = sum(r - p for p, r in zip(sorted(d["pause"]), sorted(d["resume"])))
        d["terminal"] = (sim_end - max(d["pause"])) if (d["ended_paused"] and d["pause"]) else 0.0
    return st, sim_end


def _pick_keeper(by_node: Dict[int, List[dict]], segment_count: int) -> int:
    """The single UE to keep: the one that received the whole video (downloaded the most segments,
    tie-break by least reported rebuffer). A node that downloaded all segment_count segments has the
    full video buffered and can play it start-to-finish."""
    def key(nid: int):
        rows = by_node[nid]
        downloaded = max(int(d["segment_id"]) for d in rows)  # 1-indexed count delivered
        rebuffer = max(float(d.get("interruption_time_s", 0.0) or 0.0) for d in rows)
        return (downloaded >= segment_count, downloaded, -rebuffer)
    return max(by_node.keys(), key=key)


def _keeper_events_summary(node: int, sl, segs: List[dict], segment_count: int):
    """Build player_events + summary extras for the kept UE from the REAL sim.log player record."""
    info = sl.get(node) if sl else None
    stream_start = min((s["request_s"] for s in segs), default=0.0)
    if info and info["first"] is not None:
        playback_start = info["first"]
    else:
        playback_start = segs[0]["done_s"] if segs else stream_start
    ev = [{"type": "player_event", "ue": 0, "t": round(stream_start, 6), "event": "stream_start"},
          {"type": "player_event", "ue": 0, "t": round(playback_start, 6), "event": "playback_start"}]
    resolved = 0.0
    stall_count = 0
    frozen = False
    # what the viewer can replay = segments present in segments.jsonl (downloaded), NOT the number the
    # ns-3 player rendered before sim-stop (big_buck_bunny is 195 s of playout vs a ~180 s sim window).
    downloaded = len({s["segment"] for s in segs})
    completed_dl = downloaded >= segment_count
    if info:
        pairs = list(zip(sorted(info["pause"]), sorted(info["resume"])))
        for ps, rs in pairs:  # resolved mid-playback stalls (real)
            ev.append({"type": "player_event", "ue": 0, "t": round(ps, 6), "event": "stall_start",
                       "source": "sim.log"})
            ev.append({"type": "player_event", "ue": 0, "t": round(rs, 6), "event": "stall_end",
                       "source": "sim.log"})
            resolved += rs - ps
            stall_count += 1
        # genuine terminal freeze = player starved (ended paused) AND never got the whole video
        if info["ended_paused"] and not completed_dl:
            frozen = True
            fp = max(info["pause"])
            ev.append({"type": "player_event", "ue": 0, "t": round(fp, 6), "event": "stall_start",
                       "source": "sim.log", "terminal": True})
            ev.append({"type": "player_event", "ue": 0, "t": round(fp + info["terminal"], 6),
                       "event": "stall_end", "source": "sim.log", "terminal": True})
            stall_count += 1
    # session end: last played frame (finisher) or sim end (frozen)
    if frozen and info:
        playback_end = max(info["pause"]) + info["terminal"]
    elif info and info["last"]:
        playback_end = info["last"]
    else:
        playback_end = max((s["done_s"] for s in segs), default=stream_start)
    ev.append({"type": "player_event", "ue": 0, "t": round(playback_end, 6), "event": "playback_end"})
    ev.sort(key=lambda e: (e["t"], 0 if e["event"] != "stall_end" else 1))
    rebuffer = resolved + (info["terminal"] if frozen else 0.0)
    completed = completed_dl and not frozen
    extra = {"completed": completed, "segments_available": downloaded, "total_segments": segment_count,
             "rebuffer_time_s": round(rebuffer, 4), "stall_count": stall_count,
             "startup_delay_s": round(playback_start - stream_start, 4),
             "stall_source": "sim.log" if info else "none"}
    return ev, extra, round(playback_end, 6)


def build_bundle(video: str, run: str, src_root: Path, abr_root: Path, out_root: Path,
                 catalog: Dict[str, dict], rng_run: int, sim_stop_arg: Optional[float],
                 commit: Optional[str], now_iso: str, task_dir: Optional[Path]) -> Tuple[Path, dict]:
    run_dir = src_root / video / run
    dec_path = run_dir / "decisions.jsonl"
    if not dec_path.is_file():
        raise ExportError("decisions.jsonl not found: %s" % dec_path)
    meta_csv = abr_root / "metadata" / f"{video}_segments.csv"
    index = load_metadata_index(str(meta_csv))

    by_node = _read_decisions(dec_path)
    nodes = sorted(by_node.keys())
    node_to_ue = {nid: i for i, nid in enumerate(nodes)}
    segment_count = len(index)

    # Real per-cell ns-3 stdout for the paper run (logs_bench/task_*). Matched by filename
    # (video+rate) and, among duplicate/partial executions, by best decision agreement.
    rate_label = _rate_label(run)
    scored_rungs = {(nid, int(d["segment_id"]) - 1): int(d["chosen_rung"])
                    for nid in nodes for d in by_node[nid]}
    task_match = _best_task_log(task_dir, video, rate_label, scored_rungs) if task_dir else None

    # Parse the real ns-3 log (if attached) for ground-truth playback/stalls.
    sl, sim_end = (None, 0.0)
    if task_match:
        sl, sim_end = _parse_simlog(open(task_match[0], errors="ignore", encoding="utf-8").read(), nodes)

    # Keep ONE UE: the one that received the whole video (re-indexed to ue 0).
    keeper = _pick_keeper(by_node, segment_count)
    cleaned = _clean_rows(by_node[keeper], segment_count, f"{video}/{run} node {keeper}")
    all_segs = _ue_segments(cleaned, index, f"{video}/{run} node {keeper}")
    all_samples = _ue_samples(cleaned)
    all_events, ue_extra, playback_end = _keeper_events_summary(keeper, sl, all_segs, segment_count)
    for lst in (all_segs, all_samples):
        for r in lst:
            r["ue"] = 0
    # per-UE throughput average for the summary
    thr = [d.get("throughput_estimate_bps") for d in cleaned
           if isinstance(d.get("throughput_estimate_bps"), (int, float))
           and math.isfinite(d["throughput_estimate_bps"])]
    summaries = [{
        "ue": 0,
        "avg_bitrate_bps": int(round(sum(s["bitrate_bps"] for s in all_segs) / len(all_segs))) if all_segs else 0,
        "avg_throughput_bps": int(round(sum(thr) / len(thr))) if thr else None,
        "switch_count": sum(1 for a, b in zip(all_segs, all_segs[1:]) if a["rung"] != b["rung"]),
        **ue_extra,
    }]
    ue_entries = [{
        "ue": 0, "node_id": keeper, "video": video,
        "stream_start_s": next((e["t"] for e in all_events if e["event"] == "stream_start"), 0.0),
    }]
    sim_stop = sim_stop_arg if sim_stop_arg is not None else playback_end
    print(f"        [keeper] node {keeper} ue0: {ue_extra['segments_available']}/{segment_count} segs"
          f" completed={ue_extra['completed']} rebuffer={ue_extra['rebuffer_time_s']}s "
          f"stalls={ue_extra['stall_count']}  (dropped {len(nodes)-1} other UE(s))")

    all_segs.sort(key=lambda r: r["done_s"])
    all_events.sort(key=lambda r: (r["t"], 0 if r["event"] != "stall_end" else 1))
    all_samples.sort(key=lambda r: r["t"])

    rate_bps = _rate_bps(run)
    run_id = f"{video}_llm_{_rate_label(run)}_{run}"
    bundle = out_root / run_id

    run_json = {
        "bundle_version": 1,
        "run_id": run_id,
        "created_utc": now_iso,
        "source_commit": commit,
        "scenario": "ntn-iab-quic-dash-ai",
        "policy": {
            "name": "llm",
            "client": "ns3::AiDashClient",
            "model": "abr-llm-v2",
            "description": ("Content-aware SLM ABR policy (greedy-teacher). NOTE: encoded segments are "
                            "2.0 s but the simulated player buffered in 3.3 s units; sim times are "
                            "copied verbatim (not rescaled)."),
        },
        "network": {
            "num_ues": 1,                       # this bundle includes ONE UE (the whole-video receiver)
            "simulated_num_ues": len(nodes),    # UEs actually contending on the feeder during the run
            "num_relays": 1,
            "feeder_link_rate_bps": rate_bps,
            "congestion_control": "ns3::QuicBbr",
            "rlc_mode": "AM",
            "rng_run": rng_run,
        },
        "player": {
            "segment_duration_s": PLAYER_SEGMENT_DURATION_S,
            "ladder_bps": list(NS3_RATE_LADDER_BPS),
        },
        "sim_stop_time_s": sim_stop,
        "ues": ue_entries,
        "log_time_format": "leading seconds",
        "sim_log_source": (task_match[0].name if task_match else None),
        "sim_log_decision_agreement": (round(task_match[1], 3) if task_match else None),
        "notes": ("SINGLE-UE bundle: only the UE that received the whole video is included (re-indexed "
                  "ue=0); the feeder was shared by simulated_num_ues UEs during the run. STALLS/rebuffer "
                  "in player_events.jsonl + summary.json are computed from the REAL ns-3 sim.log MPEG "
                  "player (stall_start/stall_end with source:'sim.log'; a never-resumed freeze is a "
                  "terminal:true stall to session end), NOT from decisions.jsonl. summary.ue.completed "
                  "says whether the stream played to the end. segments/samples come from decisions.jsonl. "
                  "sim_log_decision_agreement<1.0 means the attached sim.log is a sibling re-execution "
                  "of this cell. Encoded segments are 2.0 s but the player buffered in 3.3 s units; sim "
                  "times verbatim. sim_stop_time_s is this UE's session end."
                  if task_match else
                  "SINGLE-UE bundle (whole-video receiver, ue=0). No ns-3 sim.log attached -> stalls fall "
                  "back to decisions.jsonl deltas. 2.0 s media vs 3.3 s player buffering; times verbatim."),
    }

    # policy.log: human-readable reconstruction of the decision stream (exact to decisions.jsonl)
    log_lines = [f"# reconstructed from {dec_path} (exact decision record)\n"]
    for d in sorted((d for rows in by_node.values() for d in rows), key=lambda r: float(r["sim_time_s"])):
        log_lines.append(
            "%10.4f ue-id: %d seg(1-idx): %d rung: %d bitrate: %d thr_est: %.0f buf: %.3f "
            "infer_ms: %.1f reason: %s\n" % (
                float(d["sim_time_s"]), int(d["node_id"]), int(d["segment_id"]), int(d["chosen_rung"]),
                int(d["chosen_bitrate_bps"]), float(d.get("throughput_estimate_bps", 0.0) or 0.0),
                float(d.get("buffer_level_s", 0.0) or 0.0),
                float(d.get("inference_latency_ms", 0.0) or 0.0), d.get("reason", "")))

    # write bundle files
    _write_json(bundle / "run.json", run_json)
    _write_jsonl(bundle / "segments.jsonl", all_segs)
    _write_jsonl(bundle / "player_events.jsonl", all_events)
    _write_jsonl(bundle / "samples.jsonl", all_samples)
    _write_json(bundle / "summary.json", {"ues": summaries})
    _write_text(bundle / "logs" / "policy.log", "".join(log_lines))
    if task_match:  # copy the REAL ns-3 stdout for this cell as logs/sim.log (read-only on source)
        _write_text(bundle / "logs" / "sim.log",
                    open(task_match[0], errors="ignore", encoding="utf-8").read())
    _link_video_into_bundle(bundle, video, out_root, media_mode=_MEDIA_MODE)

    return bundle, run_json


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------
def validate_bundle(bundle: Path) -> List[str]:
    """Re-read a written bundle and return a list of problems ([] == OK). Structural issues are hard;
    timing monotonicity is reported but non-fatal (best-effort reconstruction)."""
    problems: List[str] = []
    run = json.load(open(bundle / "run.json", encoding="utf-8"))
    ladder = run["player"]["ladder_bps"]
    for f in ("bundle_version", "run_id", "scenario", "policy", "network", "player", "sim_stop_time_s", "ues"):
        if f not in run:
            problems.append(f"run.json missing required field: {f}")

    def _jl(name):
        return [json.loads(l) for l in open(bundle / name, encoding="utf-8") if l.strip()]

    segs = _jl("segments.jsonl")
    by_ue: Dict[int, List[dict]] = {}
    for r in segs:
        by_ue.setdefault(r["ue"], []).append(r)
    for ue, rows in by_ue.items():
        rows.sort(key=lambda r: r["segment"])
        if [r["segment"] for r in rows] != list(range(len(rows))):
            problems.append(f"ue {ue}: segment indices not contiguous 0..N")
        for r in rows:
            if r["bitrate_bps"] != ladder[r["rung"]]:
                problems.append(f"ue {ue} seg {r['segment']}: bitrate != ladder[rung]")
            if r["done_s"] < r["request_s"] - 1e-6:
                problems.append(f"ue {ue} seg {r['segment']}: done_s < request_s")
        reqs = [r["request_s"] for r in rows]
        if any(b < a - 1e-3 for a, b in zip(reqs, reqs[1:])):
            problems.append(f"[warn] ue {ue}: request_s not non-decreasing (best-effort timing)")

    for r in _jl("player_events.jsonl"):
        pass
    for ue, rows in by_ue.items():
        evs = [e for e in _jl("player_events.jsonl") if e["ue"] == ue]
        opens = sum(1 for e in evs if e["event"] == "stall_start")
        closes = sum(1 for e in evs if e["event"] == "stall_end")
        if opens != closes:
            problems.append(f"ue {ue}: {opens} stall_start vs {closes} stall_end")

    # video metadata completeness + media presence
    for ue_entry in run["ues"]:
        vdir = bundle / "videos" / ue_entry["video"]
        vj = json.load(open(vdir / "video.json", encoding="utf-8"))
        vsegs = _jl(str(Path("videos") / ue_entry["video"] / "segments.jsonl"))
        want = {(s, r) for s in range(vj["segment_count"]) for r in range(NUM_LADDER_RUNGS)}
        have = {(v["segment"], v["rung"]) for v in vsegs}
        if want - have:
            problems.append(f"{ue_entry['video']}: missing {len(want - have)} (segment,rung) metadata rows")
        if (vdir / "r0").exists():  # media present -> every referenced file must exist
            for s in range(vj["segment_count"]):
                for r in range(NUM_LADDER_RUNGS):
                    if not (vdir / f"r{r}" / f"seg_{s:04d}.mp4").exists():
                        problems.append(f"{ue_entry['video']}: missing media r{r}/seg_{s:04d}.mp4")
                        break
    return problems


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
_MEDIA_MODE = "symlink"


def main() -> None:
    global _OUT_ROOT_REAL, _MEDIA_MODE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS), help="comma-separated video names")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma-separated run_N (link rates)")
    ap.add_argument("--src", default="Quic_artifacts/AiDash/bench_down_scored",
                    help="root of <video>/run_N/ scored LLM runs")
    ap.add_argument("--abr-root", default="/project/6030214/adeel/abr/output",
                    help="abr output root holding metadata/ and by_video/")
    ap.add_argument("--task-logs", default="logs_bench",
                    help="dir of real per-cell ns-3 stdout (task_<id>_<video>_<rate>_run*.log); "
                         "attached as logs/sim.log. Empty string to skip.")
    ap.add_argument("--out", default="viewer_bundles", help="output dir (the ONLY writable location)")
    ap.add_argument("--with-media", action="store_true", help="copy/symlink the 6 ABR rungs (enables replay)")
    ap.add_argument("--media-mode", choices=["symlink", "copy"], default="symlink",
                    help="symlink (default, space-saving) or copy (portable) for media + per-video tree")
    ap.add_argument("--rng-run", type=int, default=1, help="RNG seed recorded in run.json")
    ap.add_argument("--sim-stop", type=float, default=None, help="override sim_stop_time_s (else observed max)")
    ap.add_argument("--force", action="store_true", help="overwrite existing bundles")
    ap.add_argument("--list", action="store_true", help="dry-run: print planned bundles, write nothing")
    args = ap.parse_args()

    repo = _HERE.parent.parent
    src_root = Path(args.src) if os.path.isabs(args.src) else repo / args.src
    abr_root = Path(args.abr_root)
    task_dir = None
    if args.task_logs:
        task_dir = Path(args.task_logs) if os.path.isabs(args.task_logs) else repo / args.task_logs
    out_root = Path(args.out) if os.path.isabs(args.out) else repo / args.out
    videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    _MEDIA_MODE = args.media_mode

    # safety: out must not live inside a source tree
    out_real = os.path.realpath(str(out_root))
    for guard in (src_root, abr_root, repo / "cache", repo / "train_jsonl"):
        g = os.path.realpath(str(guard))
        if out_real == g or out_real.startswith(g + os.sep):
            ap.error(f"--out ({out_real}) resolves inside a source tree ({g}); refusing")

    plan = [(v, r) for v in videos for r in runs]
    if args.list:
        print(f"[dry-run] out={out_real}  media={'on' if args.with_media else 'off'} ({args.media_mode})")
        total = 0.0
        for v, r in plan:
            rd = src_root / v / r / "decisions.jsonl"
            ok = "OK" if rd.is_file() else "MISSING decisions.jsonl"
            print(f"  {v}_llm_{_rate_label(r)}_{r}  <- {rd}  [{ok}]")
        if args.with_media:
            for v in videos:
                segdir = abr_root / "by_video" / v / "segments"
                mb = sum(f.stat().st_size for lvl in LADDER_LEVELS
                         for f in (segdir / lvl).glob("*.mp4")) / 1e6 if segdir.is_dir() else 0.0
                total += mb
                print(f"  media {v}: {mb:.1f} MB (6 rungs)")
            print(f"  media total (copy mode would use): {total:.1f} MB; symlink mode ~0 extra")
        print(f"[dry-run] {len(plan)} bundles planned. Nothing written.")
        return

    _OUT_ROOT_REAL = out_real
    out_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffprobe") is None:
        print("[warn] ffprobe not on PATH -> per-rung width/height use the max-height ladder fallback "
              "(approximate). Run on a node with ffmpeg/ffprobe for exact dimensions.")
    catalog = _load_catalog(abr_root)
    commit = _git_commit()
    now_iso = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    built: List[Path] = []
    all_problems: Dict[str, List[str]] = {}
    good_videos = []
    for v in videos:
        if args.force:  # refresh the shared per-video tree too (dims/fps/media), not just bundles
            shared = out_root / "_videos" / v
            if shared.exists():
                shutil.rmtree(_assert_write_target(shared))
        try:
            build_video_assets(v, abr_root, out_root, catalog, args.with_media, args.media_mode)
            good_videos.append(v)
        except ExportError as e:
            print(f"[SKIP video] {v}: {e}")
            shared = out_root / "_videos" / v
            if shared.exists():
                shutil.rmtree(_assert_write_target(shared))
    plan = [(v, r) for v, r in plan if v in good_videos]
    for v, r in plan:
        run_id = f"{v}_llm_{_rate_label(r)}_{r}"
        bundle = out_root / run_id
        if bundle.exists() and not args.force:
            print(f"[skip] {run_id} exists (use --force)")
            continue
        if bundle.exists():
            shutil.rmtree(_assert_write_target(bundle))
        try:
            bundle, _ = build_bundle(v, r, src_root, abr_root, out_root, catalog,
                                     args.rng_run, args.sim_stop, commit, now_iso, task_dir)
        except ExportError as e:
            print(f"[FAIL] {run_id}: {e}")
            continue
        probs = validate_bundle(bundle)
        all_problems[run_id] = probs
        tag = "OK" if not any(not p.startswith("[warn]") for p in probs) else "PROBLEMS"
        rj = json.load(open(bundle / "run.json", encoding="utf-8"))
        sl = rj.get("sim_log_source")
        slmsg = (f"  sim.log={sl} (agree {rj['sim_log_decision_agreement']:.0%})" if sl
                 else "  sim.log=NONE (reconstructed policy.log only)")
        print(f"[{tag}] {run_id}{slmsg}")
        for p in probs:
            print(f"        {p}")
        built.append(bundle)

    hard = {k: [p for p in v if not p.startswith("[warn]")] for k, v in all_problems.items()}
    nbad = sum(1 for v in hard.values() if v)
    print(f"\nDone: {len(built)} bundles under {out_root}  ({nbad} with hard validation problems).")


if __name__ == "__main__":
    main()
