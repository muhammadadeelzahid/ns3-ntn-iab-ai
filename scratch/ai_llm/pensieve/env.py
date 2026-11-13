#!/usr/bin/env python3
"""Chunk-level offline ABR simulator for training Pensieve on OUR ladder + traces.

One step = downloading one video chunk at the chosen rung against a per-slot bandwidth sample taken
from a real ns-3 throughput trace (see traces.py). The reward is the CANONICAL Pensieve linear QoE
(bitrate - rebuffer penalty - smoothness penalty), network-only - no VMAF/semantic terms, matching
the "as-published" baseline framing. Deliberately simple (per-slot bandwidth, not a continuous
integrator) because the trace granularity is per-segment; this keeps the baseline honest and the
training minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

try:
    from ai_llm.pensieve.model import (
        A_DIM, PENSIEVE_LADDER_MBPS, SEGMENT_DURATION_S, build_state,
    )
except ImportError:  # imported from within the pensieve/ dir
    from model import A_DIM, PENSIEVE_LADDER_MBPS, SEGMENT_DURATION_S, build_state

# Canonical linear-QoE weights (Mbps units): rebuffer penalty = ladder max Mbps, smoothness = 1.
REBUF_PENALTY = max(PENSIEVE_LADDER_MBPS)   # 12.0
SMOOTH_PENALTY = 1.0
BUFFER_CAP_S = 60.0


@dataclass
class Episode:
    """One training episode: a bandwidth timeline paired with a video's per-chunk size table.

    `vmafs` (per-chunk VMAF per rung) is optional and unused by training (linear-QoE on bitrate); it
    is carried only for the offline VMAF-%oracle checkpoint-selection eval (offline_eval.py)."""
    bw_bps: List[float]                 # per-slot effective bandwidth (one sample per chunk)
    sizes_bytes: List[List[float]]      # [num_segments][A_DIM] chunk sizes (bytes) for this video
    video_name: str
    vmafs: Optional[List[List[float]]] = None  # [num_segments][A_DIM] VMAF per rung

    @property
    def length(self) -> int:
        return min(len(self.bw_bps), len(self.sizes_bytes))


# Median-bandwidth bands (Mbps) used to balance training exposure across congestion levels, so a
# congestion-heavy trace pool doesn't collapse the policy to a single safe rung. Mirrors the
# benchmark's uniform-over-5-conditions design (100Gb/s / 36 / 18 / 9 / 4.5 Mbps).
_BAND_EDGES_MBPS = [0.0, 3.0, 9.0, 18.0, 40.0, float("inf")]


class ABREnv:
    """Replays Episodes chunk by chunk. reset() -> state; step(action) -> (state, reward, done)."""

    def __init__(self, episodes: Sequence[Episode], seed: int = 42, stratify: bool = True) -> None:
        self.episodes = [e for e in episodes if e.length >= 2]
        if not self.episodes:
            raise ValueError("ABREnv: no usable episodes (need length >= 2).")
        self._rng = np.random.RandomState(seed)
        self._stratify = stratify
        # Group episode indices by median-bandwidth band for balanced sampling.
        self._bands: List[List[int]] = [[] for _ in range(len(_BAND_EDGES_MBPS) - 1)]
        for i, e in enumerate(self.episodes):
            med = float(np.median(e.bw_bps)) / 1.0e6
            b = next(k for k in range(len(_BAND_EDGES_MBPS) - 1)
                     if _BAND_EDGES_MBPS[k] <= med < _BAND_EDGES_MBPS[k + 1])
            self._bands[b].append(i)
        self._nonempty_bands = [b for b in self._bands if b]
        self._ep: Optional[Episode] = None
        self.reset()

    def _next_sizes(self, chunk: int) -> List[float]:
        seg = min(chunk, len(self._ep.sizes_bytes) - 1)
        return list(self._ep.sizes_bytes[seg])

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        if episode_idx is not None:
            idx = episode_idx
        elif self._stratify:
            band = self._nonempty_bands[self._rng.randint(len(self._nonempty_bands))]
            idx = band[self._rng.randint(len(band))]
        else:
            idx = self._rng.randint(len(self.episodes))
        self._ep = self.episodes[idx]
        self._chunk = 0
        self._buffer = 0.0
        self._last_bitrate_mbps = 0.0
        self._thr_hist: List[float] = []
        self._dly_hist: List[float] = []
        return self._state()

    def _state(self) -> np.ndarray:
        return build_state(
            last_bitrate_mbps=self._last_bitrate_mbps,
            buffer_s=self._buffer,
            throughput_hist_mbps=self._thr_hist,
            delay_hist_s=self._dly_hist,
            next_sizes_bytes=self._next_sizes(self._chunk),
            chunks_remaining=self._ep.length - self._chunk,
            total_chunks=self._ep.length,
        )

    def step(self, action: int):
        action = int(np.clip(action, 0, A_DIM - 1))
        seg = min(self._chunk, len(self._ep.sizes_bytes) - 1)
        size_bytes = float(self._ep.sizes_bytes[seg][action])
        bw = max(float(self._ep.bw_bps[min(self._chunk, len(self._ep.bw_bps) - 1)]), 1.0)

        download_time = size_bytes * 8.0 / bw
        rebuf = max(0.0, download_time - self._buffer)
        self._buffer = min(max(0.0, self._buffer - download_time) + SEGMENT_DURATION_S, BUFFER_CAP_S)

        bitrate_mbps = PENSIEVE_LADDER_MBPS[action]
        reward = (bitrate_mbps
                  - REBUF_PENALTY * rebuf
                  - SMOOTH_PENALTY * abs(bitrate_mbps - self._last_bitrate_mbps))

        # History carries the realized download outcome (measured throughput == per-slot bw).
        self._thr_hist.append(bw / 1.0e6)
        self._dly_hist.append(download_time)
        self._last_bitrate_mbps = bitrate_mbps
        self._chunk += 1

        done = self._chunk >= self._ep.length
        state = self._state() if not done else self._state()
        return state, float(reward), bool(done)
