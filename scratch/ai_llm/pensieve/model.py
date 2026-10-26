#!/usr/bin/env python3
"""Pensieve actor-critic network + the shared state featurizer.

`build_state()` is the SINGLE source of truth for how raw player/network observations become the
[S_INFO, S_LEN] matrix the net consumes. Both the training env (env.py) and the serving wrapper
(ai_llm/inference.py::PensievePolicy) call it, so train-time and serve-time features are identical
by construction - the usual silent-OOD trap for a learned ABR policy.

Architecture mirrors the original Pensieve A3C actor (1-D conv over the throughput/download-time
history and over the next-chunk size vector; dense over the scalar state), emitting a softmax over
the 6 ladder rungs plus a scalar value head.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # torch is only needed where the net actually runs (train / serve)
    torch = None
    nn = object  # type: ignore

# Our 6-rung ladder in Mbps (matches ai_llm.paths.NS3_RATE_LADDER_BPS). Kept local so this module
# has no hard dependency on the package layout when imported by the trainer.
PENSIEVE_LADDER_MBPS: List[float] = [0.25, 0.75, 1.5, 3.0, 6.0, 12.0]
A_DIM = len(PENSIEVE_LADDER_MBPS)            # 6 actions, 1:1 with the ladder rungs
S_INFO = 6                                   # feature rows (see build_state)
S_LEN = 8                                    # history window (Pensieve default k=8)

# Normalization constants - shared by train + serve (do NOT retune one side only).
BITRATE_NORM = 12.0      # Mbps (ladder max)
BUFFER_NORM = 10.0       # seconds
THROUGHPUT_NORM = 12.0   # Mbps
DELAY_NORM = 10.0        # seconds (per-chunk download time)
SIZE_NORM = 1.0e6        # bytes -> MB
SEGMENT_DURATION_S = 2.0 # 100 frames @ 20 ms, matches the 2.0s encoded segment duration (abr pipeline)

# Feature clips: idle-link throughput measurements can hit hundreds of Mbps (p99 ~335), which
# saturates the normalized throughput channel and makes the net blind to throughput. Clip the
# feature (NOT the download-time physics) to a sane band so the channel stays informative. Applied
# identically at train + serve via build_state, so parity is preserved.
THROUGHPUT_CLIP_MBPS = 2.0 * BITRATE_NORM  # 24 Mbps - well above the ladder, below the outliers
DELAY_CLIP_S = 30.0


def build_state(
    last_bitrate_mbps: float,
    buffer_s: float,
    throughput_hist_mbps: Sequence[float],
    delay_hist_s: Sequence[float],
    next_sizes_bytes: Sequence[float],
    chunks_remaining: int,
    total_chunks: int,
) -> np.ndarray:
    """Assemble the normalized [S_INFO, S_LEN] Pensieve state matrix.

    Rows: 0 last bitrate, 1 buffer, 2 throughput history (k steps), 3 download-time history,
    4 next-chunk sizes for the A_DIM rungs, 5 chunks remaining. History sequences are right-aligned
    (most recent in the last column); scalars live in the last column of their row.
    """
    state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    state[0, -1] = float(last_bitrate_mbps) / BITRATE_NORM
    state[1, -1] = float(buffer_s) / BUFFER_NORM

    thr = list(throughput_hist_mbps)[-S_LEN:]
    dly = list(delay_hist_s)[-S_LEN:]
    if thr:
        clipped = np.clip(np.asarray(thr, dtype=np.float32), 0.0, THROUGHPUT_CLIP_MBPS)
        state[2, -len(thr):] = clipped / THROUGHPUT_NORM
    if dly:
        clipped = np.clip(np.asarray(dly, dtype=np.float32), 0.0, DELAY_CLIP_S)
        state[3, -len(dly):] = clipped / DELAY_NORM

    sizes = list(next_sizes_bytes)[:A_DIM]
    state[4, :len(sizes)] = np.asarray(sizes, dtype=np.float32) / SIZE_NORM

    denom = float(total_chunks) if total_chunks > 0 else 1.0
    state[5, -1] = float(chunks_remaining) / denom
    return state


if torch is not None:

    class ActorCritic(nn.Module):
        """Pensieve actor-critic. Input [B, S_INFO, S_LEN] -> (action_logits [B, A_DIM], value [B,1])."""

        def __init__(self, hidden: int = 128) -> None:
            super().__init__()
            self.conv_thr = nn.Conv1d(1, 128, kernel_size=4)   # over throughput history
            self.conv_dly = nn.Conv1d(1, 128, kernel_size=4)   # over download-time history
            self.conv_size = nn.Conv1d(1, 128, kernel_size=4)  # over the A_DIM next-chunk sizes
            self.fc_bitrate = nn.Linear(1, 128)
            self.fc_buffer = nn.Linear(1, 128)
            self.fc_remain = nn.Linear(1, 128)

            conv_thr_out = 128 * (S_LEN - 4 + 1)   # 128 * 5
            conv_dly_out = 128 * (S_LEN - 4 + 1)
            conv_size_out = 128 * (A_DIM - 4 + 1)  # 128 * 3
            merged = 128 * 3 + conv_thr_out + conv_dly_out + conv_size_out
            self.fc_merge = nn.Linear(merged, hidden)
            self.actor = nn.Linear(hidden, A_DIM)
            self.critic = nn.Linear(hidden, 1)

        def forward(self, x):  # x: [B, S_INFO, S_LEN]
            b = x.shape[0]
            f_bitrate = F.relu(self.fc_bitrate(x[:, 0:1, -1]))
            f_buffer = F.relu(self.fc_buffer(x[:, 1:2, -1]))
            f_thr = F.relu(self.conv_thr(x[:, 2:3, :])).reshape(b, -1)
            f_dly = F.relu(self.conv_dly(x[:, 3:4, :])).reshape(b, -1)
            f_size = F.relu(self.conv_size(x[:, 4:5, :A_DIM])).reshape(b, -1)
            f_remain = F.relu(self.fc_remain(x[:, 5:6, -1]))
            merged = torch.cat([f_bitrate, f_buffer, f_thr, f_dly, f_size, f_remain], dim=1)
            h = F.relu(self.fc_merge(merged))
            return self.actor(h), self.critic(h)
