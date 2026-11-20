#!/usr/bin/env python3
"""Per-segment, per-rung picture quality for the viewer bundles (export_viewer_bundle.py).

For one video, writes <out>/_quality/<video>.json with one record per (segment, rung):
  vmaf      mean frame VMAF, copied from the corrected metadata CSV (the value the paper uses);
  vmaf_min  lowest frame VMAF, from the content pipeline's libvmaf logs
            (by_video/<video>/vmaf_logs/<rung>/seg_NNNN.json); left out when a log's mean does not
            match the CSV (logs not refreshed by the realign fix);
  ssim      ffmpeg ssim filter, luma (Y), mean over the segment's frames;
  psnr      ffmpeg psnr filter, luma (y), mean over the segment's frames;
  duration_s  the rung file's real duration (ffprobe).
Reference = the source segment by_video/<video>/segments/original/seg_NNNN.mp4. Both inputs are
scaled to the source resolution with bicubic scaling, exactly as the pipeline's VMAF step
(abr/src/ffmpeg_utils.py compute_vmaf). Read-only on every source; writes only under --out.

Needs ffmpeg/ffprobe on PATH (module load ffmpeg/7.0.2). 4K: run as a CPU job, not on the login node.
Usage: python viewer_quality.py --video chef_cooking --out viewer_bundles/phasef_seed203 [--jobs 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402

_SSIM_RE = re.compile(r"SSIM Y:([0-9.]+)")
_PSNR_RE = re.compile(r"PSNR y:([0-9.]+|inf)")


def _probe(mp4: Path, entries: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
                           "-of", "csv=p=0", str(mp4)], capture_output=True, text=True, check=True).stdout.strip()


def _duration(mp4: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(mp4)], capture_output=True, text=True, check=True).stdout.strip()
    return round(float(out), 4)


def _ssim_psnr(dist: Path, ref: Path, w: int, h: int):
    lavfi = (f"[0:v]scale={w}:{h}:flags=bicubic,split[d1][d2];"
             f"[1:v]scale={w}:{h}:flags=bicubic,split[r1][r2];"
             f"[d1][r1]ssim;[d2][r2]psnr")
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(dist), "-i", str(ref),
                          "-lavfi", lavfi, "-f", "null", "-"], capture_output=True, text=True).stderr
    ms, mp = _SSIM_RE.search(err), _PSNR_RE.search(err)
    if not (ms and mp):
        raise RuntimeError("ffmpeg ssim/psnr failed for %s:\n%s" % (dist, err[-800:]))
    psnr = None if mp.group(1) == "inf" else round(float(mp.group(1)), 3)
    return round(float(ms.group(1)), 5), psnr


def _vmaf_min(log: Path, csv_mean: float):
    if not log.is_file():
        return None
    frames = [f["metrics"]["vmaf"] for f in json.load(open(log, encoding="utf-8"))["frames"]]
    if not frames or abs(sum(frames) / len(frames) - csv_mean) > 0.05:
        return None  # stale log: does not describe the media the CSV (and the paper) describe
    return round(min(frames), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--abr-root", default="/project/6030214/adeel/abr/output")
    ap.add_argument("--out", required=True, help="viewer bundle root; writes <out>/_quality/<video>.json")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    abr = Path(args.abr_root)
    vdir = abr / "by_video" / args.video
    index = load_metadata_index(str(abr / "metadata" / f"{args.video}_segments.csv"))
    seg_ids = index.segment_ids()
    ref0 = vdir / "segments" / "original" / index.get(seg_ids[0]).segment_file
    w, h = (int(x) for x in _probe(ref0, "stream=width,height").split(","))

    def one(task):
        s, r, lvl = task
        m = index.get(s)
        dist = vdir / "segments" / lvl / m.segment_file
        ref = vdir / "segments" / "original" / m.segment_file
        ssim, psnr = _ssim_psnr(dist, ref, w, h)
        return {"segment": s, "rung": r, "vmaf": round(float(m.vmaf[lvl]), 4),
                "vmaf_min": _vmaf_min(vdir / "vmaf_logs" / lvl / m.segment_file.replace(".mp4", ".json"),
                                      float(m.vmaf[lvl])),
                "ssim": ssim, "psnr": psnr, "duration_s": _duration(dist)}

    tasks = [(s, r, lvl) for s in seg_ids for r, lvl in enumerate(LADDER_LEVELS)]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        records = sorted(ex.map(one, tasks), key=lambda x: (x["segment"], x["rung"]))

    out = Path(args.out) / "_quality" / f"{args.video}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    notes = (f"Reference: source segments at {w}x{h}; both inputs bicubic-scaled to that resolution. "
             "vmaf: mean frame VMAF from the content pipeline (libvmaf default model vmaf_v0.6.1, not the "
             "4K model), as used in the paper. vmaf_min: lowest frame VMAF from the same libvmaf logs. "
             "ssim, psnr: ffmpeg ssim/psnr filters, luma, mean over the segment's frames. duration_s: "
             "the rung file's duration.")
    json.dump({"video": args.video, "reference_resolution": [w, h], "notes": notes, "records": records},
              open(out, "w", encoding="utf-8"), indent=1)
    nmin = sum(r["vmaf_min"] is not None for r in records)
    print(f"wrote {out}: {len(records)} records, vmaf_min on {nmin}")


if __name__ == "__main__":
    main()
