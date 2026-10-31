#!/usr/bin/env python3
"""RobustMPC ABR baseline (Yin, Jindal, Sekar, Sinopoli - "A Control-Theoretic Approach for Dynamic
Adaptive Video Streaming over HTTP", SIGCOMM 2015), served through the same ns-3 bridge as the LLM.

Controller = the reference implementation everyone benchmarks against (Pensieve repo, test/mpc.py):
  * throughput prediction = harmonic mean of the last HIST_LEN realized per-chunk throughputs,
    discounted by the worst recent relative prediction error:  bw = hm / (1 + max_err)   ("robust");
  * exhaustive enumeration of all n_rungs^K rung sequences over a K=HORIZON chunk lookahead (K shrinks
    to the chunks remaining), rollout with bw held constant over the horizon:
        dl = 8*size/bw ; rebuf = max(0, dl - buf) ; buf = min(max_buf, max(buf - dl, 0) + seg_dur)
  * action = first rung of the best sequence; ties -> last sequence in lexicographic order.
Torch-free, numpy only; runs on CPU in the 3.11 inference server.

Two objectives (the controller is as published, the objective is the evaluating paper's metric - the
same convention Comyco [MM'19] and Puffer [NSDI'20] used when they re-pointed RobustMPC at their QoE):
  "linear" : QoE_lin of Yin'15/Pensieve - sum R(Mbps) - REBUF_PENALTY*rebuf - SMOOTH_PENALTY*|dR|,
             REBUF_PENALTY = R_max (12 Mbps on our ladder; 4.3 on Pensieve's) = pensieve/env.py's reward,
             i.e. the exact objective our Pensieve arm was trained on.             -> arm "Mpc"
  "qoe"    : the benchmark scoring QoE (config/oracle_reward.json), per segment
             w_vmaf*sem*VMAF - w_stall*rebuf - w_switch*drop(VMAF)*(1-scene_cut_discount*cut) [- w_rung..]
             mirrored term-for-term from scripts/dp_oracle._segment_reward/_switch_penalty (tests assert
             parity), semantic weights from dp_oracle.compute_semantic_weights exactly as the server's
             --policy greedy path. With K=1 and no robust discount this IS greedy_teacher.greedy_rung.
                                                                                    -> arm "Qmpc"
Per-UE state is keyed by node_id (one server per cell, like PensievePolicy). The ns-3 request carries
no delivered-bytes field, so the realized throughput sample for the just-downloaded segment is
rebuilt as 8*size_bytes[seg0-1][last chosen rung]/segment_fetch_time_s (same reconstruction the model
path uses for last_segment_bitrate_bps). ns-3's throughput_estimate_bps is a 10 s plain average (NOT a
harmonic mean) and is only used as the seed sample on a UE's first decision.
"""
from __future__ import annotations

import itertools
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "scripts")))  # dp_oracle
try:
    from ai_llm.metadata_loader import LADDER_LEVELS
    from ai_llm.paths import NS3_RATE_LADDER_BPS
except ImportError:  # run from scratch/ai_llm directly
    from metadata_loader import LADDER_LEVELS
    from paths import NS3_RATE_LADDER_BPS
from dp_oracle import compute_semantic_weights, load_reward_config  # type: ignore

CITATION = ("Yin et al., SIGCOMM'15 (RobustMPC); reference impl. hongzimao/pensieve test/mpc.py; "
            "quality objective per Comyco MM'19 / Puffer NSDI'20 convention (evaluating paper's QoE)")

HORIZON = 5                # MPC_FUTURE_CHUNK_COUNT
HIST_LEN = 5               # harmonic-mean window AND max-error window (both 5 in the reference)
DEFAULT_QUALITY = 1        # first chunk when no throughput sample exists (reference DEFAULT_QUALITY)
MIN_FETCH_S = 0.05         # guard buffer-full near-instant fetches (decisions_to_throughput.py)
SEG_DUR_S = 3.3            # ns-3 player segment duration (trace_schema.MPEG_SEGMENT_DURATION_S)
MAX_BUFFER_S = 30.0        # oracle_reward.json player_model.max_buffer_s
LINEAR_REBUF_PENALTY = max(NS3_RATE_LADDER_BPS) / 1.0e6   # 12.0 = R_max convention (pensieve/env.py)
LINEAR_SMOOTH_PENALTY = 1.0
OBJECTIVES = ("linear", "qoe")


class RobustThroughputEstimator:
    """Per-UE RobustMPC throughput predictor (shared by RobustMpcPolicy and the model arm's serving guard).

    Keeps the last `hist_len` realized per-chunk throughput samples and the relative errors of the
    predictions that preceded them; prediction = harmonic mean / (1 + max recent error), exactly the
    reference (hongzimao/pensieve test/mpc.py). The UNdiscounted harmonic mean is stored as the step's
    prediction, as the reference does."""

    def __init__(self, hist_len: int = HIST_LEN) -> None:
        self.hist_len = int(hist_len)
        self._samples: Dict[int, deque] = {}
        self._preds: Dict[int, deque] = {}
        self._errors: Dict[int, deque] = {}

    def reset(self) -> None:
        self._samples.clear(); self._preds.clear(); self._errors.clear()

    def has_samples(self, node: int) -> bool:
        return bool(self._samples.get(node))

    def ingest(self, node: int, sample_bps: float) -> None:
        """Record a realized per-chunk throughput and the error of the prediction that preceded it."""
        if sample_bps <= 0.0:
            return
        preds = self._preds.setdefault(node, deque(maxlen=self.hist_len))
        errs = self._errors.setdefault(node, deque(maxlen=self.hist_len))
        if preds:  # curr_error = |past_bandwidth_ests[-1] - actual| / actual
            errs.append(abs(preds[-1] - sample_bps) / sample_bps)
        self._samples.setdefault(node, deque(maxlen=self.hist_len)).append(sample_bps)

    def predict(self, node: int) -> Tuple[Optional[float], float, float]:
        """(future_bw_bps or None if no samples, harmonic_mean, max_err)."""
        samples = self._samples.get(node)
        if not samples:
            return None, 0.0, 0.0
        hm = len(samples) / sum(1.0 / s for s in samples)
        errs = self._errors.get(node)
        max_err = max(errs) if errs else 0.0
        self._preds.setdefault(node, deque(maxlen=self.hist_len)).append(hm)
        return hm / (1.0 + max_err), hm, max_err


class RobustMpcPolicy:
    def __init__(
        self,
        metadata_index,
        objective: str = "qoe",
        reward_cfg: Optional[Dict] = None,
        reward_config_path: Optional[str] = None,
        horizon: int = HORIZON,
        hist_len: int = HIST_LEN,
        seg_dur_s: float = SEG_DUR_S,
        max_buffer_s: float = MAX_BUFFER_S,
        ladder_bps: Optional[Sequence[int]] = None,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError("objective must be one of %s, got %r" % (OBJECTIVES, objective))
        self.objective = objective
        self.horizon = int(horizon)
        self.hist_len = int(hist_len)
        self.ladder_bps = [int(b) for b in (ladder_bps or NS3_RATE_LADDER_BPS)]
        self.ladder_mbps = np.asarray(self.ladder_bps, dtype=np.float64) / 1.0e6
        self.n_rungs = len(self.ladder_bps)

        self.seg_ids: List[int] = sorted(metadata_index.segment_ids())
        self._pos = {sid: i for i, sid in enumerate(self.seg_ids)}
        self.n_segments = len(self.seg_ids)
        self.sizes = np.asarray([[float(metadata_index.get(s).size_bytes.get(lv, 0.0)) for lv in LADDER_LEVELS]
                                 for s in self.seg_ids], dtype=np.float64)           # [N, R] bytes
        self.vmafs = np.asarray([[float(metadata_index.get(s).vmaf.get(lv, 0.0)) for lv in LADDER_LEVELS]
                                 for s in self.seg_ids], dtype=np.float64)           # [N, R]

        self.seg_dur = float(seg_dur_s)
        self.max_buffer = float(max_buffer_s)
        self.W: Dict[str, float] = {}
        self.sem = np.ones(self.n_segments)
        self.cut = np.zeros(self.n_segments)
        if objective == "qoe":
            cfg = reward_cfg or load_reward_config(reward_config_path)
            self.W = dict(cfg["reward_weights"])
            player = cfg.get("player_model", {}) or {}
            # Player constants come from the SAME config the scorer replays with (parity, not a choice).
            self.seg_dur = float(player.get("segment_duration_s", self.seg_dur))
            self.max_buffer = float(player.get("max_buffer_s", self.max_buffer))
            sw = compute_semantic_weights(metadata_index, cfg["semantic_weight_mixture"])
            self.sem = np.asarray([float(sw.get(s, 0.0)) for s in self.seg_ids], dtype=np.float64)
            self.cut = np.asarray([1.0 if float((metadata_index.get(s).semantic_raw or {}).get("scene_cut_count", 0.0)) > 0
                                   else 0.0 for s in self.seg_ids], dtype=np.float64)

        self._combos: Dict[int, np.ndarray] = {}
        # per-UE rolling state (mirrors the reference's past_bandwidths / past_bandwidth_ests / past_errors)
        self._est = RobustThroughputEstimator(self.hist_len)
        self._last_rung: Dict[int, int] = {}
        self._seen: Dict[int, bool] = {}

    # ------------------------------------------------------------------ state
    def reset_history(self) -> None:
        self._est.reset()
        self._last_rung.clear(); self._seen.clear()

    # read-only views of the estimator state (tests and diagnostics inspect them)
    @property
    def _samples(self) -> Dict[int, deque]:
        return self._est._samples

    @property
    def _preds(self) -> Dict[int, deque]:
        return self._est._preds

    @property
    def _errors(self) -> Dict[int, deque]:
        return self._est._errors

    def _ingest_sample(self, node: int, sample_bps: float) -> None:
        """Record a realized per-chunk throughput and the error of the prediction that preceded it."""
        self._est.ingest(node, sample_bps)

    def predict_throughput(self, node: int) -> Tuple[Optional[float], float, float]:
        """(future_bw_bps or None if no samples, harmonic_mean, max_err) - see RobustThroughputEstimator."""
        return self._est.predict(node)

    # ------------------------------------------------------------------ planning
    def _combos_for(self, k: int) -> np.ndarray:
        if k not in self._combos:
            self._combos[k] = np.asarray(list(itertools.product(range(self.n_rungs), repeat=k)), dtype=np.int64)
        return self._combos[k]

    def horizon_rewards(self, pos: int, buf_s: float, future_bw_bps: float, last_rung: Optional[int],
                        k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised rollout. Returns (combos [M,K], total reward [M]) for segment position `pos`."""
        if k is None:
            k = min(self.horizon, self.n_segments - pos)
        if k <= 0:
            raise ValueError("no segments left to plan at pos=%d" % pos)
        combos = self._combos_for(k)
        m = combos.shape[0]
        bw = max(float(future_bw_bps), 1.0)
        buf = np.full(m, float(buf_s))
        total = np.zeros(m)
        if self.objective == "linear":
            prev = None if last_rung is None else np.full(m, self.ladder_mbps[last_rung])
        else:
            prev_pos = pos - 1
            prev = None if (last_rung is None or prev_pos < 0) else np.full(m, self.vmafs[prev_pos, last_rung])
            w_vmaf, w_stall, w_switch = self.W["w_vmaf"], self.W["w_stall"], self.W["w_switch"]
            scd = float(self.W.get("scene_cut_discount", 0.0))
            w_rung = float(self.W.get("w_rung", 0.0))
            down_only = self.W.get("switch_mode") == "down_only"
        for step in range(k):
            r = combos[:, step]
            dl = 8.0 * self.sizes[pos + step][r] / bw
            rebuf = np.maximum(0.0, dl - buf)
            buf = np.minimum(self.max_buffer, np.maximum(buf - dl, 0.0) + self.seg_dur)
            if self.objective == "linear":
                rate = self.ladder_mbps[r]
                total += rate - LINEAR_REBUF_PENALTY * rebuf
                if prev is not None:
                    total -= LINEAR_SMOOTH_PENALTY * np.abs(rate - prev)
                prev = rate
            else:
                v = self.vmafs[pos + step][r]
                total += w_vmaf * self.sem[pos + step] * v - w_stall * rebuf
                if prev is not None:
                    mag = np.maximum(0.0, prev - v) if down_only else np.abs(v - prev)
                    total -= w_switch * mag * (1.0 - scd * self.cut[pos + step])
                if w_rung > 0.0 and self.n_rungs > 1:
                    total -= w_rung * (r.astype(np.float64) / float(self.n_rungs - 1))
                prev = v
        return combos, total

    def plan(self, seg0: int, buf_s: float, future_bw_bps: float, last_rung: Optional[int]) -> Tuple[int, int]:
        """Best first action for segment id `seg0`. Returns (rung, horizon_used)."""
        pos = self._pos[seg0]
        combos, total = self.horizon_rewards(pos, buf_s, future_bw_bps, last_rung)
        best = int(np.flatnonzero(total >= total.max())[-1])  # ties -> last combo (reference `>=`)
        return int(combos[best, 0]), combos.shape[1]

    # ------------------------------------------------------------------ serving entry point
    def predict(self, req: Dict, seg0: int) -> Tuple[int, str]:
        """One bridge decision. `seg0` = 0-indexed id of the segment to fetch next."""
        node = int(req.get("node_id", -1))
        if seg0 not in self._pos:
            raise KeyError("segment %d not in metadata" % seg0)
        fetch = float(req.get("segment_fetch_time_s", 0.0))
        buf = float(req.get("buffer_level_s", 0.0))
        last_rung = self._last_rung.get(node)
        prev_pos = self._pos.get(seg0 - 1)
        if last_rung is not None and prev_pos is not None and fetch > 0.0:
            # realized channel rate of the segment we just downloaded at the rung we chose for it
            self._ingest_sample(node, 8.0 * self.sizes[prev_pos, last_rung] / max(fetch, MIN_FETCH_S))
        elif not self._seen.get(node):
            est = float(req.get("throughput_estimate_bps", 0.0))
            if est > 0.0:
                self._ingest_sample(node, est)   # seed sample (ns-3's 10 s average), first decision only
        self._seen[node] = True

        future_bw, hm, max_err = self.predict_throughput(node)
        tag = "qmpc" if self.objective == "qoe" else "robustmpc"
        if future_bw is None:
            rung = min(DEFAULT_QUALITY, self.n_rungs - 1)
            reason = "%s(cold-start:r%d)" % (tag, rung)
        else:
            rung, k = self.plan(seg0, buf, future_bw, last_rung)
            reason = "%s(bw=%.2fMbps,hm=%.2f,err=%.2f,K=%d,buf=%.1f)" % (
                tag, future_bw / 1.0e6, hm / 1.0e6, max_err, k, buf)
        self._last_rung[node] = rung
        return rung, reason
