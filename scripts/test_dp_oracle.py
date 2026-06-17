#!/usr/bin/env python3
"""
Smoke + regression checks for scripts/dp_oracle.py backward-DP oracle.

What this exercises (no ns-3 / no real artifacts required):

  1) DP optimality vs brute-force enumeration on a tiny synthetic session.
  2) Buffer / stall transition invariants over a longer synthetic trajectory.
  3) Reward equation surface equality (R = w*q - alpha*stall - switch_penalty).
  4) Rung histogram sanity: DP exploits middle rungs under variable throughput.
  5) Schema contract: every produced record carries all required state-key fields.

Usage:
  python3 scripts/test_dp_oracle.py
Exits non-zero on any failure with a descriptive error.
"""

from __future__ import print_function

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scratch"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ai_llm.paths import NS3_RATE_LADDER_BPS  # noqa: E402
from ai_llm.trace_schema import (  # noqa: E402
    ORACLE_LABELS_STATE_REQUIRED_FIELDS,
)

import dp_oracle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeMetaRow:
    def __init__(
        self,
        segment_index: int,
        size_bytes: Dict[str, float],
        vmaf: Dict[str, float],
        semantic_raw: Dict[str, float],
    ) -> None:
        self.segment_index = segment_index
        self.segment_file = "fake_{}.m4s".format(segment_index)
        self.size_bytes = size_bytes
        self.actual_bitrate_kbps = {lvl: 0.0 for lvl in size_bytes}
        self.vmaf = vmaf
        self.semantic_raw = semantic_raw
        self.semantic_norm = dict(semantic_raw)


class FakeMetaIndex:
    def __init__(self, rows: Dict[int, FakeMetaRow]) -> None:
        self._rows = rows

    def get(self, seg_id: int):
        return self._rows.get(seg_id)


# ladder kbps -> bytes for a 3.3s segment at the given rate
def _size_from_bps(bps: float, duration_s: float = 3.3) -> float:
    return bps * duration_s / 8.0


def _make_metadata(n_segments: int, scene_cut_at: List[int]) -> FakeMetaIndex:
    rows: Dict[int, FakeMetaRow] = {}
    for seg_id in range(n_segments):
        size_bytes = {}
        vmaf = {}
        for lvl, bps in zip(dp_oracle.LADDER_LEVELS, NS3_RATE_LADDER_BPS):
            size_bytes[lvl] = _size_from_bps(bps)
            # Saturating VMAF that grows sub-linearly with bitrate.
            kbps = bps / 1000.0
            vmaf[lvl] = 100.0 * (kbps / (kbps + 1500.0))
        semantic_raw = {
            "motion_score": 0.5 + 0.05 * (seg_id % 3),
            "saliency_score": 0.4 + 0.1 * ((seg_id + 1) % 4) / 4.0,
            "scene_cut_count": 1.0 if seg_id in scene_cut_at else 0.0,
            "object_score": 0.3 + 0.07 * (seg_id % 5),
            "quality_sensitivity": 0.5 + 0.1 * (seg_id % 6),
        }
        rows[seg_id] = FakeMetaRow(seg_id, size_bytes, vmaf, semantic_raw)
    return FakeMetaIndex(rows)


def _make_session(throughputs: List[float]) -> List[Dict]:
    return [
        {
            "run_id": "run_smoke",
            "node_id": 0,
            "segment_id": idx,
            "throughput_bps": float(bps),
            "segment_fetch_time_s": 0.0,
            "segment_bytes": 0,
        }
        for idx, bps in enumerate(throughputs)
    ]


def _default_cfg() -> Dict:
    return {
        "reward_weights": {
            "w_vmaf": 1.0,
            "w_stall": 10.0,
            "w_switch": 0.3,
            "scene_cut_discount": 0.6,
            "w_quality_sensitive": 0.0,
            "w_rung": 0.0,
        },
        "semantic_weight_mixture": {
            "quality_sensitivity": 0.5,
            "saliency_score": 0.3,
            "object_score": 0.2,
        },
        "player_model": {
            "segment_duration_s": 3.3,
            "max_buffer_s": 30.0,
            "startup_threshold_s": 10.0,
            "initial_buffer_s": 0.0,
            "buffer_bin_size_s": 0.5,
        },
        "solver": {
            "gamma": 1.0,
            "throughput_floor_bps": 10000.0,
            "throughput_planning_factor": 1.0,
            "throughput_source": "per_run_node",
        },
        "__path__": "<test-default>",
    }


# ---------------------------------------------------------------------------
# Brute-force reference DP (no buffer quantization) over policies a_0..a_{N-1}
# ---------------------------------------------------------------------------


def brute_force_optimum(
    session_rows: List[Dict],
    metadata: FakeMetaIndex,
    reward_cfg: Dict,
) -> Tuple[float, List[int]]:
    weights = reward_cfg["reward_weights"]
    player = reward_cfg["player_model"]
    seg_duration = player["segment_duration_s"]
    max_buf = player["max_buffer_s"]
    startup = player["startup_threshold_s"]
    init_buf = player["initial_buffer_s"]

    semantic_weights = dp_oracle.compute_semantic_weights(
        metadata, reward_cfg["semantic_weight_mixture"]
    )

    solver = reward_cfg["solver"]
    tp_floor = float(solver.get("throughput_floor_bps", 1e4))
    tp_factor = float(solver.get("throughput_planning_factor", 1.0))

    R = len(NS3_RATE_LADDER_BPS)
    N = len(session_rows)
    best_val = -1e30
    best_path: List[int] = []

    for path in itertools.product(range(R), repeat=N):
        buf = init_buf
        playing = 0
        total = 0.0
        for t in range(N):
            seg_id = int(session_rows[t]["segment_id"])
            tp_obs = dp_oracle._safe_throughput(session_rows[t], tp_floor)
            tp = dp_oracle._planned_throughput(tp_obs, tp_floor, tp_factor)
            meta = metadata.get(seg_id)
            sizes = [meta.size_bytes[lvl] for lvl in dp_oracle.LADDER_LEVELS]
            vmaf = [meta.vmaf[lvl] for lvl in dp_oracle.LADDER_LEVELS]
            a = path[t]
            dt = 8.0 * sizes[a] / tp
            rebuf, buf_after, playing_after = dp_oracle._segment_transition(
                buffer_before_s=buf,
                playing_before=playing,
                download_time_s=dt,
                segment_duration_s=seg_duration,
                max_buffer_s=max_buf,
                startup_threshold_s=startup,
            )
            if t == 0:
                switch_pen = 0.0
            else:
                vmaf_prev = [
                    metadata.get(int(session_rows[t - 1]["segment_id"])).vmaf[lvl]
                    for lvl in dp_oracle.LADDER_LEVELS
                ][path[t - 1]]
                is_cut = 1 if metadata.get(seg_id).semantic_raw["scene_cut_count"] > 0 else 0
                switch_pen = dp_oracle._switch_penalty(
                    vmaf_chosen=vmaf[a],
                    vmaf_prev=vmaf_prev,
                    is_cut=is_cut,
                    weights=weights,
                )
            total += dp_oracle._segment_reward(
                weight=semantic_weights[seg_id],
                vmaf_chosen=vmaf[a],
                rebuffer_s=rebuf,
                switch_penalty_value=switch_pen,
                weights=weights,
                rung_idx=a,
                n_rungs=R,
            )
            buf = buf_after
            playing = playing_after
        if total > best_val:
            best_val = total
            best_path = list(path)
    return best_val, best_path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_dp_matches_brute_force() -> None:
    metadata = _make_metadata(n_segments=4, scene_cut_at=[2])
    throughputs = [4_000_000.0, 1_200_000.0, 600_000.0, 3_000_000.0]
    session = _make_session(throughputs)
    # Use a very fine buffer bin so quantization-induced sub-optimality is
    # negligible vs. brute force. Production uses 0.5s as documented.
    cfg = _default_cfg()
    cfg["player_model"]["buffer_bin_size_s"] = 0.01

    dp_result = dp_oracle.solve_session_dp(
        session_rows=session,
        metadata=metadata,
        semantic_weights=dp_oracle.compute_semantic_weights(
            metadata, cfg["semantic_weight_mixture"]
        ),
        reward_cfg=cfg,
    )
    assert dp_result is not None
    dp_total = dp_result["total_qoe"]
    dp_path = [r["oracle_bitrate_idx"] for r in dp_result["records"]]

    bf_total, bf_path = brute_force_optimum(session, metadata, cfg)

    if abs(dp_total - bf_total) > 1e-4:
        raise RuntimeError(
            "DP total {:.6f} != brute force {:.6f} (DP path={}, BF path={})".format(
                dp_total, bf_total, dp_path, bf_path
            )
        )
    print(
        "OK dp-vs-brute (fine bin): path={} total={:.4f} (matches reference)".format(
            dp_path, dp_total
        )
    )

    # Also exercise the production-grade coarse bin and ensure it is at most a
    # bounded fraction below the brute-force optimum.
    cfg_coarse = _default_cfg()
    cfg_coarse["player_model"]["buffer_bin_size_s"] = 0.5
    coarse_result = dp_oracle.solve_session_dp(
        session_rows=session,
        metadata=metadata,
        semantic_weights=dp_oracle.compute_semantic_weights(
            metadata, cfg_coarse["semantic_weight_mixture"]
        ),
        reward_cfg=cfg_coarse,
    )
    assert coarse_result is not None
    coarse_total = coarse_result["total_qoe"]
    # Quantization with 0.5s bins should not lose more than 10% of optimum here.
    if coarse_total < 0.9 * bf_total - 1e-6:
        raise RuntimeError(
            "Coarse-bin DP lost >10% vs brute force: coarse={:.4f}, bf={:.4f}".format(
                coarse_total, bf_total
            )
        )
    print(
        "OK dp-vs-brute (coarse bin=0.5s): {:.4f} within 10% of {:.4f}".format(
            coarse_total, bf_total
        )
    )


def check_buffer_invariants() -> None:
    metadata = _make_metadata(n_segments=20, scene_cut_at=[3, 11])
    throughputs = [
        2_500_000.0,
        300_000.0,
        4_000_000.0,
        2_000_000.0,
        800_000.0,
        6_000_000.0,
        1_500_000.0,
        400_000.0,
        5_000_000.0,
        2_500_000.0,
    ] * 2
    session = _make_session(throughputs)
    cfg = _default_cfg()

    result = dp_oracle.solve_session_dp(
        session_rows=session,
        metadata=metadata,
        semantic_weights=dp_oracle.compute_semantic_weights(
            metadata, cfg["semantic_weight_mixture"]
        ),
        reward_cfg=cfg,
    )
    assert result is not None
    max_buf = cfg["player_model"]["max_buffer_s"]
    for rec in result["records"]:
        if not (0.0 <= rec["buffer_before_s"] <= max_buf + 1e-9):
            raise RuntimeError("buffer_before out of bounds: {}".format(rec))
        if not (0.0 <= rec["buffer_after_s"] <= max_buf + 1e-9):
            raise RuntimeError("buffer_after out of bounds: {}".format(rec))
        if rec["predicted_rebuffer_s"] < -1e-9:
            raise RuntimeError("negative rebuffer: {}".format(rec))
        if rec["predicted_download_time_s"] <= 0.0:
            raise RuntimeError("non-positive download time: {}".format(rec))
    print(
        "OK invariants: {} records, max buffer_after={:.2f}, max rebuffer={:.2f}".format(
            len(result["records"]),
            max(r["buffer_after_s"] for r in result["records"]),
            max(r["predicted_rebuffer_s"] for r in result["records"]),
        )
    )


def check_reward_surface_equation() -> None:
    cfg = _default_cfg()
    w = cfg["reward_weights"]
    expected = w["w_vmaf"] * 0.8 * 75.0 - w["w_stall"] * 0.5 - 1.234
    actual = dp_oracle._segment_reward(
        weight=0.8,
        vmaf_chosen=75.0,
        rebuffer_s=0.5,
        switch_penalty_value=1.234,
        weights=w,
    )
    if abs(actual - expected) > 1e-9:
        raise RuntimeError(
            "Reward equation mismatch: expected {:.6f}, got {:.6f}".format(
                expected, actual
            )
        )

    # Scene-cut switch penalty discount.
    full = dp_oracle._switch_penalty(
        vmaf_chosen=90.0,
        vmaf_prev=70.0,
        is_cut=0,
        weights=w,
    )
    discounted = dp_oracle._switch_penalty(
        vmaf_chosen=90.0,
        vmaf_prev=70.0,
        is_cut=1,
        weights=w,
    )
    if discounted >= full or discounted < 0:
        raise RuntimeError(
            "Scene-cut discount failed: full={:.6f} discounted={:.6f}".format(
                full, discounted
            )
        )
    print("OK reward equation + scene-cut discount")


def check_rung_distribution_not_bimodal() -> None:
    """DP should distribute over middle rungs under fluctuating throughput."""
    metadata = _make_metadata(n_segments=30, scene_cut_at=[7, 14, 22])
    # Cycle of comfortable / tight / tiny throughputs.
    base = [3_500_000.0, 1_500_000.0, 800_000.0, 5_500_000.0, 2_500_000.0]
    throughputs = [base[i % len(base)] for i in range(30)]
    session = _make_session(throughputs)
    cfg = _default_cfg()

    result = dp_oracle.solve_session_dp(
        session_rows=session,
        metadata=metadata,
        semantic_weights=dp_oracle.compute_semantic_weights(
            metadata, cfg["semantic_weight_mixture"]
        ),
        reward_cfg=cfg,
    )
    assert result is not None
    counts = {}
    for rec in result["records"]:
        counts[rec["oracle_bitrate_idx"]] = counts.get(rec["oracle_bitrate_idx"], 0) + 1
    total = sum(counts.values())
    rung_0_5_share = (counts.get(0, 0) + counts.get(5, 0)) / float(total)
    used = sorted(counts.keys())
    # Sanity: with this fluctuating throughput we expect at least one middle rung used.
    middle_used = any(idx in counts for idx in (1, 2, 3, 4))
    if not middle_used:
        raise RuntimeError(
            "DP collapsed to extremes (rungs={}, counts={})".format(used, counts)
        )
    print(
        "OK rung-distribution: counts={} extremes-share={:.2f}".format(
            counts, rung_0_5_share
        )
    )


def check_record_schema() -> None:
    metadata = _make_metadata(n_segments=5, scene_cut_at=[])
    session = _make_session([2_000_000.0] * 5)
    cfg = _default_cfg()

    result = dp_oracle.solve_session_dp(
        session_rows=session,
        metadata=metadata,
        semantic_weights=dp_oracle.compute_semantic_weights(
            metadata, cfg["semantic_weight_mixture"]
        ),
        reward_cfg=cfg,
    )
    assert result is not None
    for rec in result["records"]:
        # Stamp the session-level fields the per-video orchestrator normally adds.
        rec.setdefault("run_id", session[0]["run_id"])
        rec.setdefault("node_id", session[0]["node_id"])
        rec.setdefault("video_name", "smoke")
        rec.setdefault("oracle_policy", dp_oracle.DP_POLICY)
        for field in ORACLE_LABELS_STATE_REQUIRED_FIELDS:
            if field not in rec:
                raise RuntimeError("record missing required field {}: {}".format(field, rec))
    print(
        "OK schema: {} required fields present on {} records".format(
            len(ORACLE_LABELS_STATE_REQUIRED_FIELDS), len(result["records"])
        )
    )


def check_reward_config_loads() -> None:
    cfg = dp_oracle.load_reward_config(dp_oracle.DEFAULT_REWARD_CONFIG)
    for section in ("reward_weights", "semantic_weight_mixture", "player_model", "solver"):
        if section not in cfg:
            raise RuntimeError("default reward config missing {}".format(section))
    mix_sum = sum(cfg["semantic_weight_mixture"].values())
    if abs(mix_sum - 1.0) > 1e-3:
        # Not a hard error but worth flagging.
        print(
            "WARN semantic_weight_mixture sums to {:.3f}, not 1.0".format(mix_sum)
        )
    print(
        "OK reward config: from {}".format(cfg.get("__path__"))
    )


def main() -> None:
    check_reward_config_loads()
    check_reward_surface_equation()
    check_dp_matches_brute_force()
    check_buffer_invariants()
    check_rung_distribution_not_bimodal()
    check_record_schema()
    print("dp_oracle smoke checks passed")


if __name__ == "__main__":
    main()
