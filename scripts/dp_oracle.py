#!/usr/bin/env python3
"""
Generate state-keyed oracle labels from ns-3 throughput traces + video metadata.

Input:
  - <artifacts>/<AlgoShort>/<video>/run_*/throughput_trace.jsonl  (per-(run,node) trajectories)
  - <metadata-dir>/<video>_segments.csv
  - scratch/ai_llm/config/oracle_reward.json

Output:
  - <artifacts>/<AlgoShort>/<video>/oracle_labels_state.jsonl
    One row per (run_id, node_id, segment_id) carrying the optimal action under
    a forward DP with backpointers that maximizes a semantic-aware QoE objective
    over the whole session. See docs/dash-trace-pipeline.md for the reward equation.
"""

from __future__ import print_function

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scratch"))

from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402
from ai_llm.paths import NS3_RATE_LADDER_BPS  # noqa: E402
from ai_llm.trace_schema import (  # noqa: E402
    ORACLE_LABELS_STATE_REQUIRED_FIELDS,
    THROUGHPUT_TRACE_REQUIRED_FIELDS,
)

DP_POLICY = "backward_dp_semantic_v1"

DEFAULT_REWARD_CONFIG = str(
    REPO_ROOT / "scratch" / "ai_llm" / "config" / "oracle_reward.json"
)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(str(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_reward_config(path: Optional[str]) -> Dict:
    cfg_path = Path(path) if path else Path(DEFAULT_REWARD_CONFIG)
    if not cfg_path.is_file():
        raise FileNotFoundError("oracle reward config not found: {}".format(cfg_path))
    with open(str(cfg_path), "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    for section in ("reward_weights", "semantic_weight_mixture", "player_model", "solver"):
        if section not in cfg:
            raise ValueError("oracle reward config missing section: {}".format(section))
    cfg["__path__"] = str(cfg_path.resolve())
    return cfg


def _ladder_sizes(meta_row, segment_id: int) -> Optional[List[float]]:
    if meta_row is None:
        return None
    sizes = []
    for level in LADDER_LEVELS:
        size = meta_row.size_bytes.get(level)
        if size is None:
            return None
        sizes.append(float(size))
    return sizes


def _ladder_vmaf(meta_row) -> Optional[List[float]]:
    if meta_row is None:
        return None
    vmaf = []
    for level in LADDER_LEVELS:
        v = meta_row.vmaf.get(level)
        if v is None:
            return None
        vmaf.append(float(v))
    return vmaf


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng < 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / rng for v in values]


def compute_semantic_weights(
    metadata,
    mixture: Dict[str, float],
) -> Dict[int, float]:
    """Min-max normalize each mixture component across this video's segments."""
    seg_ids = sorted(metadata._rows.keys())  # type: ignore[attr-defined]
    components = list(mixture.keys())
    raw_by_component: Dict[str, List[float]] = {c: [] for c in components}

    for seg_id in seg_ids:
        meta = metadata.get(seg_id)
        for c in components:
            raw_by_component[c].append(float(meta.semantic_raw.get(c, 0.0)))

    norm_by_component = {c: _minmax(raw_by_component[c]) for c in components}

    weights: Dict[int, float] = {}
    for i, seg_id in enumerate(seg_ids):
        weight = 0.0
        for c in components:
            weight += mixture[c] * norm_by_component[c][i]
        weights[seg_id] = weight
    return weights


def discover_throughput_files(
    artifacts_root: Path,
    algo_short: str,
    video: str,
    run_glob: str = "run_*",
) -> List[Path]:
    base = artifacts_root / algo_short / video
    if not base.is_dir():
        return []
    return sorted(base.glob("{}/throughput_trace.jsonl".format(run_glob)))


def load_sessions(
    throughput_files: List[Path],
) -> Dict[Tuple[str, int], List[Dict]]:
    """Group throughput_trace rows by (run_id, node_id), sorted by segment_id."""
    sessions: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for path in throughput_files:
        for row in load_jsonl(path):
            missing = [f for f in THROUGHPUT_TRACE_REQUIRED_FIELDS if f not in row]
            if missing:
                continue
            run_id = str(row["run_id"])
            node_id = int(row["node_id"])
            sessions[(run_id, node_id)].append(row)

    for key in sessions:
        sessions[key].sort(key=lambda r: int(r["segment_id"]))
    return sessions


def _safe_throughput(row: Dict, throughput_floor_bps: float) -> float:
    bps = float(row.get("throughput_bps", 0.0))
    if bps < throughput_floor_bps:
        return throughput_floor_bps
    return bps


def _planned_throughput(
    observed_bps: float,
    throughput_floor_bps: float,
    planning_factor: float,
) -> float:
    """Conservative throughput used for download-time / buffer transitions."""
    factor = max(0.1, min(1.0, float(planning_factor)))
    planned = observed_bps * factor
    return max(throughput_floor_bps, planned)


def _buffer_to_bin(buffer_s: float, bin_size: float) -> int:
    if buffer_s < 0.0:
        buffer_s = 0.0
    return int(round(buffer_s / bin_size))


def _bin_to_buffer(bin_idx: int, bin_size: float) -> float:
    return float(bin_idx) * bin_size


def _segment_transition(
    buffer_before_s: float,
    playing_before: int,
    download_time_s: float,
    segment_duration_s: float,
    max_buffer_s: float,
    startup_threshold_s: float,
) -> Tuple[float, float, int]:
    """Return (rebuffer_s, buffer_after_s, playing_after)."""
    if playing_before == 0:
        buffer_after = min(max_buffer_s, buffer_before_s + segment_duration_s)
        playing_after = 1 if buffer_after >= startup_threshold_s else 0
        return 0.0, buffer_after, playing_after
    rebuf = max(0.0, download_time_s - buffer_before_s)
    drained = max(0.0, buffer_before_s - download_time_s)
    buffer_after = min(max_buffer_s, drained + segment_duration_s)
    return rebuf, buffer_after, 1


def _segment_reward(
    weight: float,
    vmaf_chosen: float,
    rebuffer_s: float,
    switch_penalty_value: float,
    weights: Dict[str, float],
    rung_idx: int = 0,
    n_rungs: int = 6,
) -> float:
    reward = (
        weights["w_vmaf"] * weight * vmaf_chosen
        - weights["w_stall"] * rebuffer_s
        - switch_penalty_value
    )
    w_rung = float(weights.get("w_rung", 0.0))
    if w_rung > 0.0 and n_rungs > 1:
        reward -= w_rung * (float(rung_idx) / float(n_rungs - 1))
    return reward


def _switch_penalty(
    vmaf_chosen: float,
    vmaf_prev: float,
    is_cut: int,
    weights: Dict[str, float],
) -> float:
    cut_discount = weights["scene_cut_discount"] * (1 if is_cut else 0)
    # switch_mode=down_only: directional smoothness, penalize only quality drops (up-switches free),
    # which removes the symmetric-|dVMAF| under-ask lock-in. Default "sym" preserves prior behavior.
    if weights.get("switch_mode") == "down_only":
        mag = max(0.0, vmaf_prev - vmaf_chosen)
    else:
        mag = abs(vmaf_chosen - vmaf_prev)
    return weights["w_switch"] * mag * (1.0 - cut_discount)


def solve_session_dp(
    session_rows: List[Dict],
    metadata,
    semantic_weights: Dict[int, float],
    reward_cfg: Dict,
) -> Optional[Dict]:
    """Forward DP with backpointers over one (run_id, node_id) session.

    Equivalent to the backward Bellman V_t(s) = max_a [R_t(s,a) + V_{t+1}(s')]
    under deterministic transitions: forward DP tracks the max accumulated
    reward to reach each state, and traceback recovers the argmax policy.
    """
    weights = reward_cfg["reward_weights"]
    player = reward_cfg["player_model"]
    solver = reward_cfg["solver"]

    segment_duration_s = float(player["segment_duration_s"])
    max_buffer_s = float(player["max_buffer_s"])
    startup_threshold_s = float(player["startup_threshold_s"])
    initial_buffer_s = float(player["initial_buffer_s"])
    bin_size = float(player["buffer_bin_size_s"])
    throughput_floor_bps = float(solver.get("throughput_floor_bps", 1e4))
    throughput_planning_factor = float(solver.get("throughput_planning_factor", 1.0))

    n_src_segments = len(metadata._rows)  # type: ignore[attr-defined]  # source segs are 0..N-1
    traj: List[Dict] = []
    for row in session_rows:
        seg_id = int(row["segment_id"])
        meta = metadata.get(seg_id)
        if meta is None and n_src_segments > 0:
            # Looped playback: short videos replay to fill the sim cap (ns-3 AiDashClient loops natively).
            # Map the overflow segment onto its source segment by position (seg_id % N) so its ladder
            # VMAF/sizes are available; the label keeps the original (overflow) seg_id for the join.
            meta = metadata.get(seg_id % n_src_segments)
        if meta is None:
            continue
        sizes = _ladder_sizes(meta, seg_id)
        vmaf = _ladder_vmaf(meta)
        if sizes is None or vmaf is None:
            continue
        tp_observed = _safe_throughput(row, throughput_floor_bps)
        tp_planned = _planned_throughput(
            tp_observed, throughput_floor_bps, throughput_planning_factor
        )
        is_cut = 1 if float(meta.semantic_raw.get("scene_cut_count", 0.0)) > 0 else 0
        traj.append(
            {
                "segment_id": seg_id,
                "throughput_bps": tp_observed,
                "throughput_bps_planned": tp_planned,
                "sizes": sizes,
                "vmaf": vmaf,
                "weight": semantic_weights.get(seg_id, 0.0),
                "is_cut": is_cut,
            }
        )
    if not traj:
        return None

    R = len(NS3_RATE_LADDER_BPS)
    N = len(traj)

    dp: List[Dict[Tuple[int, int, int],
                  Tuple[float, Optional[Tuple[int, int, int]], Dict]]] = [
        {} for _ in range(N)
    ]

    seg0 = traj[0]
    for r in range(R):
        download_s = 8.0 * seg0["sizes"][r] / seg0["throughput_bps_planned"]
        rebuf, buffer_after, playing_after = _segment_transition(
            buffer_before_s=initial_buffer_s,
            playing_before=0,
            download_time_s=download_s,
            segment_duration_s=segment_duration_s,
            max_buffer_s=max_buffer_s,
            startup_threshold_s=startup_threshold_s,
        )
        immediate = _segment_reward(
            weight=seg0["weight"],
            vmaf_chosen=seg0["vmaf"][r],
            rebuffer_s=rebuf,
            switch_penalty_value=0.0,
            weights=weights,
            rung_idx=r,
            n_rungs=R,
        )
        state = (r, _buffer_to_bin(buffer_after, bin_size), int(playing_after))
        diag = {
            "buffer_before_s": initial_buffer_s,
            "buffer_after_s": buffer_after,
            "predicted_download_time_s": download_s,
            "predicted_rebuffer_s": rebuf,
            "playing_before": 0,
            "playing_after": int(playing_after),
            "switch_penalty": 0.0,
            "vmaf_chosen": seg0["vmaf"][r],
            "vmaf_prev": 0.0,
            "immediate_reward": immediate,
        }
        prev = dp[0].get(state)
        if prev is None or prev[0] < immediate:
            dp[0][state] = (immediate, None, diag)

    for t in range(1, N):
        seg = traj[t]
        for prev_state, (prev_val, _, _) in dp[t - 1].items():
            r_prev, b_prev, p_prev = prev_state
            buffer_before_s = _bin_to_buffer(b_prev, bin_size)
            vmaf_prev = traj[t - 1]["vmaf"][r_prev]
            for r in range(R):
                download_s = 8.0 * seg["sizes"][r] / seg["throughput_bps_planned"]
                rebuf, buffer_after, playing_after = _segment_transition(
                    buffer_before_s=buffer_before_s,
                    playing_before=p_prev,
                    download_time_s=download_s,
                    segment_duration_s=segment_duration_s,
                    max_buffer_s=max_buffer_s,
                    startup_threshold_s=startup_threshold_s,
                )
                switch_pen = _switch_penalty(
                    vmaf_chosen=seg["vmaf"][r],
                    vmaf_prev=vmaf_prev,
                    is_cut=seg["is_cut"],
                    weights=weights,
                )
                immediate = _segment_reward(
                    weight=seg["weight"],
                    vmaf_chosen=seg["vmaf"][r],
                    rebuffer_s=rebuf,
                    switch_penalty_value=switch_pen,
                    weights=weights,
                    rung_idx=r,
                    n_rungs=R,
                )
                val = prev_val + immediate
                new_state = (r, _buffer_to_bin(buffer_after, bin_size), int(playing_after))
                cur = dp[t].get(new_state)
                if cur is None or cur[0] < val:
                    diag = {
                        "buffer_before_s": buffer_before_s,
                        "buffer_after_s": buffer_after,
                        "predicted_download_time_s": download_s,
                        "predicted_rebuffer_s": rebuf,
                        "playing_before": int(p_prev),
                        "playing_after": int(playing_after),
                        "switch_penalty": switch_pen,
                        "vmaf_chosen": seg["vmaf"][r],
                        "vmaf_prev": vmaf_prev,
                        "immediate_reward": immediate,
                    }
                    dp[t][new_state] = (val, prev_state, diag)

    terminal_state, terminal_payload = max(dp[N - 1].items(), key=lambda kv: kv[1][0])
    total_qoe = terminal_payload[0]

    chosen_rungs: List[int] = [-1] * N
    diagnostics: List[Dict] = [None] * N  # type: ignore[list-item]
    cumulative_after: List[float] = [0.0] * N
    s: Optional[Tuple[int, int, int]] = terminal_state
    for t in range(N - 1, -1, -1):
        if s is None:
            break
        val_t, prev_state, diag = dp[t][s]
        cumulative_after[t] = val_t
        chosen_rungs[t] = s[0]
        diagnostics[t] = diag
        s = prev_state

    out_records: List[Dict] = []
    for t in range(N):
        seg = traj[t]
        rung = chosen_rungs[t]
        diag = diagnostics[t]
        immediate = float(diag["immediate_reward"])
        future_value = float(total_qoe - cumulative_after[t])
        total_q = immediate + future_value
        out_records.append(
            {
                "segment_id": int(seg["segment_id"]),
                "oracle_bitrate_idx": int(rung),
                "oracle_bitrate_bps": int(NS3_RATE_LADDER_BPS[rung]),
                "immediate_reward": immediate,
                "future_value": future_value,
                "total_q_value": total_q,
                "buffer_before_s": float(diag["buffer_before_s"]),
                "buffer_after_s": float(diag["buffer_after_s"]),
                "predicted_download_time_s": float(diag["predicted_download_time_s"]),
                "predicted_rebuffer_s": float(diag["predicted_rebuffer_s"]),
                "playing_before": int(diag["playing_before"]),
                "playing_after": int(diag["playing_after"]),
                "prev_rung": int(chosen_rungs[t - 1]) if t > 0 else -1,
                "semantic_weight": float(seg["weight"]),
                "is_scene_cut": int(seg["is_cut"]),
                "scene_cut_count": float(
                    metadata.get(seg["segment_id"]).semantic_raw.get("scene_cut_count", 0.0)
                ),
                "vmaf_chosen": float(diag["vmaf_chosen"]),
                "vmaf_prev": float(diag["vmaf_prev"]),
                "switch_penalty": float(diag["switch_penalty"]),
                "throughput_bps": float(seg["throughput_bps"]),
                "throughput_bps_planned": float(seg["throughput_bps_planned"]),
            }
        )

    return {
        "records": out_records,
        "total_qoe": float(total_qoe),
        "segment_count": int(N),
    }


def build_dp_records_for_video(
    video: str,
    metadata_csv: str,
    throughput_files: List[Path],
    reward_cfg: Dict,
    algo_short: str,
    strict: bool,
) -> Tuple[List[Dict], Dict]:
    metadata = load_metadata_index(metadata_csv, scaler_json_path=None)
    sessions = load_sessions(throughput_files)
    if not sessions:
        msg = "No sessions parsed from {} files for video {}".format(
            len(throughput_files), video
        )
        if strict:
            raise RuntimeError(msg)
        return [], {
            "video": video,
            "rows_in": 0,
            "rows_written": 0,
            "sessions_solved": 0,
            "sessions_skipped": 0,
            "rung_histogram": {},
            "warning": msg,
        }

    mixture = reward_cfg["semantic_weight_mixture"]
    semantic_weights = compute_semantic_weights(metadata, mixture)

    records: List[Dict] = []
    sessions_solved = 0
    sessions_skipped = 0
    rungs_counter: Counter = Counter()
    throughput_source = reward_cfg["solver"].get("throughput_source", "per_run_node")

    for (run_id, node_id), session_rows in sorted(sessions.items()):
        try:
            result = solve_session_dp(
                session_rows=session_rows,
                metadata=metadata,
                semantic_weights=semantic_weights,
                reward_cfg=reward_cfg,
            )
        except Exception as exc:
            if strict:
                raise
            sessions_skipped += 1
            print(
                "WARN DP failed for {} run={} node={}: {}".format(
                    video, run_id, node_id, exc
                ),
                file=sys.stderr,
            )
            continue
        if result is None:
            sessions_skipped += 1
            continue
        sessions_solved += 1
        total_qoe = result["total_qoe"]
        seg_count = result["segment_count"]
        for rec in result["records"]:
            rec["run_id"] = str(run_id)
            rec["node_id"] = int(node_id)
            rec["video_name"] = video
            rec["oracle_policy"] = DP_POLICY
            rec["session_total_qoe"] = float(total_qoe)
            rec["session_segment_count"] = int(seg_count)
            rec["segment_duration_s"] = float(
                reward_cfg["player_model"]["segment_duration_s"]
            )
            rec["metadata_csv"] = metadata_csv
            rec["algo_short"] = algo_short
            rec["throughput_source"] = throughput_source
            rec["reward_config"] = reward_cfg.get("__path__", "")
            rungs_counter[rec["oracle_bitrate_idx"]] += 1
            records.append(rec)

    stats = {
        "video": video,
        "sessions_solved": sessions_solved,
        "sessions_skipped": sessions_skipped,
        "rows_in": sum(len(s) for s in sessions.values()),
        "rows_written": len(records),
        "missing_segments": 0,
        "rung_histogram": dict(rungs_counter),
    }
    return records, stats


def write_state_jsonl(records: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "w", encoding="utf-8") as handle:
        for record in records:
            for field in ORACLE_LABELS_STATE_REQUIRED_FIELDS:
                if field not in record:
                    raise RuntimeError(
                        "Missing oracle-state field {} in record".format(field)
                    )
            handle.write(json.dumps(record))
            handle.write("\n")


def _resolve_metadata_csv(
    rows: List[Dict],
    metadata_csv: Optional[str],
    metadata_dir: Optional[str],
    video: str,
) -> str:
    if metadata_csv:
        return str(Path(metadata_csv).resolve())
    if rows and rows[0].get("metadata_csv"):
        return rows[0]["metadata_csv"]
    if metadata_dir:
        return str(Path(metadata_dir) / "{}_segments.csv".format(video))
    raise RuntimeError("metadata CSV not resolvable for {}".format(video))


def process_video(
    artifacts_root: Path,
    algo_short: str,
    video: str,
    metadata_dir: Optional[str],
    metadata_csv: Optional[str],
    output: Optional[Path],
    reward_cfg: Dict,
    run_glob: str,
    strict: bool,
) -> Dict:
    files = discover_throughput_files(artifacts_root, algo_short, video, run_glob)
    if not files:
        msg = "No throughput_trace.jsonl under {}".format(
            artifacts_root / algo_short / video
        )
        if strict:
            raise RuntimeError(msg)
        return {"video": video, "rows_written": 0, "warning": msg}

    first_rows: List[Dict] = []
    if files:
        first_rows = load_jsonl(files[0])[:1]
    csv_path = _resolve_metadata_csv(first_rows, metadata_csv, metadata_dir, video)

    records, stats = build_dp_records_for_video(
        video=video,
        metadata_csv=csv_path,
        throughput_files=files,
        reward_cfg=reward_cfg,
        algo_short=algo_short,
        strict=strict,
    )

    out_path = (
        output
        if output
        else (artifacts_root / algo_short / video / "oracle_labels_state.jsonl")
    )
    write_state_jsonl(records, out_path)
    stats["output"] = str(out_path)
    return stats


def _video_worker(payload: Tuple) -> Dict:
    (
        artifacts_root_s,
        algo_short,
        video,
        metadata_dir,
        reward_cfg,
        run_glob,
        strict,
    ) = payload
    return process_video(
        artifacts_root=Path(artifacts_root_s),
        algo_short=algo_short,
        video=video,
        metadata_dir=metadata_dir,
        metadata_csv=None,
        output=None,
        reward_cfg=reward_cfg,
        run_glob=run_glob,
        strict=strict,
    )


def discover_videos(artifacts_root: Path, algo_short: str) -> List[str]:
    base = artifacts_root / algo_short
    if not base.is_dir():
        return []
    return [child.name for child in sorted(base.iterdir()) if child.is_dir()]


def _format_stats(stats: Dict) -> str:
    if "output" not in stats:
        return "SKIP {}: {}".format(stats.get("video"), stats.get("warning", ""))
    hist = stats.get("rung_histogram", {})
    hist_str = ",".join("{}:{}".format(k, hist[k]) for k in sorted(hist.keys()))
    return (
        "OK {video}: sessions={sessions_solved} rows={rows_written} "
        "rungs=[{hist}]"
    ).format(
        video=stats["video"],
        sessions_solved=stats.get("sessions_solved", 0),
        rows_written=stats["rows_written"],
        hist=hist_str,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build oracle_labels_state.jsonl via semantic backward-DP."
    )
    parser.add_argument("--artifacts-root", default="Quic_artifacts")
    parser.add_argument("--algo-short", default="Fdash")
    parser.add_argument("--video", help="Only this video (else all under algo).")
    parser.add_argument(
        "--metadata-dir",
        help="Dir of <video>_segments.csv (fallback if trace lacks path).",
    )
    parser.add_argument(
        "--metadata",
        help="Single-video: explicit metadata CSV path (overrides discovery).",
    )
    parser.add_argument(
        "--output",
        help="Single-video: override oracle_labels_state.jsonl output path.",
    )
    parser.add_argument("--run-glob", default="run_*", help="Trace glob under <video>.")
    parser.add_argument(
        "--reward-config",
        default=DEFAULT_REWARD_CONFIG,
        help="Path to oracle_reward.json.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    artifacts_root = (REPO_ROOT / args.artifacts_root).resolve()
    metadata_dir = str(Path(args.metadata_dir).resolve()) if args.metadata_dir else None
    reward_cfg = load_reward_config(args.reward_config)

    if args.video:
        stats = process_video(
            artifacts_root=artifacts_root,
            algo_short=args.algo_short,
            video=args.video,
            metadata_dir=metadata_dir,
            metadata_csv=str(Path(args.metadata).resolve()) if args.metadata else None,
            output=Path(args.output).resolve() if args.output else None,
            reward_cfg=reward_cfg,
            run_glob=args.run_glob,
            strict=args.strict,
        )
        print(_format_stats(stats))
        return

    videos = discover_videos(artifacts_root, args.algo_short)
    if not videos:
        raise SystemExit("No videos under {}".format(artifacts_root / args.algo_short))

    payloads = [
        (
            str(artifacts_root),
            args.algo_short,
            video,
            metadata_dir,
            reward_cfg,
            str(args.run_glob),
            bool(args.strict),
        )
        for video in videos
    ]

    total_rows = 0
    failures = 0

    def _run_sequential() -> Iterable[Dict]:
        for payload in payloads:
            video = payload[2]
            try:
                yield _video_worker(payload)
            except Exception as exc:
                print("FAIL {}: {}".format(video, exc), file=sys.stderr)
                yield {"video": video, "failed": True, "error": str(exc)}

    def _run_parallel() -> Iterable[Dict]:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(_video_worker, p): p[2] for p in payloads}
            for fut in as_completed(futures):
                video = futures[fut]
                try:
                    yield fut.result()
                except Exception as exc:
                    print("FAIL {}: {}".format(video, exc), file=sys.stderr)
                    yield {"video": video, "failed": True, "error": str(exc)}

    iterator = _run_sequential() if args.workers <= 1 else _run_parallel()
    for stats in iterator:
        if stats.get("failed"):
            failures += 1
            continue
        total_rows += int(stats.get("rows_written", 0))
        print(_format_stats(stats))

    print(
        "Batch done: {n} videos, {rows} rows, {fail} failures".format(
            n=len(videos), rows=total_rows, fail=failures
        )
    )
    if failures and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
