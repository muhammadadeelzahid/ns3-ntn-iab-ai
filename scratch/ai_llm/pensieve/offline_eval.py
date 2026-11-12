#!/usr/bin/env python3
"""Offline VMAF-%oracle proxy for Pensieve checkpoint SELECTION.

Pensieve trains on canonical linear QoE (bitrate), but the benchmark scores the content-aware
VMAF-greedy %-of-oracle. Those disagree: more RL training lowers the linear-QoE loss yet collapses
the policy to a conservative near-constant rung that scores WORSE on VMAF-%oracle. So we pick the
training iterate that is best on THIS proxy (early-stopping on the target metric), not the final one.

Per eval episode we replay the policy through the same per-slot-bandwidth chunk model as env.py,
score it with the benchmark reward (oracle_reward.json: w_vmaf*VMAF - w_stall*rebuf - w_switch*max(0,
VMAF drop); down-only smoothness), and floor-normalize against a causal-greedy-VMAF oracle:
    pct = (policy_reward - floor_reward) / (oracle_reward - floor_reward)
floor = always-lowest-rung. Mean pct over a fixed held-out episode set. Absolute values won't equal
the ns-3 %oracle (per-slot bw != full download dynamics), but the RANKING tracks adaptivity/benchmark
alignment, which is all we need to select a checkpoint.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

try:
    from ai_llm.pensieve.model import (
        A_DIM, PENSIEVE_LADDER_MBPS, SEGMENT_DURATION_S, build_state,
    )
except ImportError:
    from model import A_DIM, PENSIEVE_LADDER_MBPS, SEGMENT_DURATION_S, build_state

# Benchmark reward weights (mirror config/oracle_reward.json reward_weights; semantic-free proxy).
W_VMAF = 0.65
W_STALL = 65.0
W_SWITCH = 0.8          # down-only
MAX_BUFFER_S = 30.0


def _seg(ep, chunk, attr):
    tbl = getattr(ep, attr)
    return tbl[min(chunk, len(tbl) - 1)]


def _rebuf_buffer(buffer, size_bytes, bw_bps):
    dl = size_bytes * 8.0 / max(bw_bps, 1.0)
    rebuf = max(0.0, dl - buffer)
    buffer = min(max(0.0, buffer - dl) + SEGMENT_DURATION_S, MAX_BUFFER_S)
    return rebuf, buffer, dl


def _reward_of_rungs(ep, rungs: Sequence[int]) -> float:
    buffer, prev_v, total = 0.0, None, 0.0
    for t, r in enumerate(rungs):
        bw = ep.bw_bps[min(t, len(ep.bw_bps) - 1)]
        rebuf, buffer, _ = _rebuf_buffer(buffer, _seg(ep, t, "sizes_bytes")[r], bw)
        v = _seg(ep, t, "vmafs")[r]
        sw = 0.0 if prev_v is None else W_SWITCH * max(0.0, prev_v - v)
        total += W_VMAF * v - W_STALL * rebuf - sw
        prev_v = v
    return total


def _greedy_oracle_rungs(ep) -> List[int]:
    """Causal greedy: at each chunk pick the rung maximizing immediate benchmark reward on realized bw."""
    buffer, prev_v, rungs = 0.0, None, []
    L = ep.length
    for t in range(L):
        bw = ep.bw_bps[min(t, len(ep.bw_bps) - 1)]
        sizes, vmafs = _seg(ep, t, "sizes_bytes"), _seg(ep, t, "vmafs")
        best_r, best_rew = 0, -1e18
        for r in range(A_DIM):
            rebuf, _, _ = _rebuf_buffer(buffer, sizes[r], bw)
            sw = 0.0 if prev_v is None else W_SWITCH * max(0.0, prev_v - vmafs[r])
            rew = W_VMAF * vmafs[r] - W_STALL * rebuf - sw
            if rew > best_rew:
                best_rew, best_r = rew, r
        _, buffer, _ = _rebuf_buffer(buffer, sizes[best_r], bw)
        prev_v = vmafs[best_r]
        rungs.append(best_r)
    return rungs


def _policy_rungs(model, ep, device, window: int) -> List[int]:
    from collections import deque
    hist = deque(maxlen=window)
    last_bitrate, rungs = 0.0, []
    L = ep.length
    for t in range(L):
        thr = [h[0] for h in hist]
        dly = [h[1] for h in hist]
        s = build_state(last_bitrate, _buffer_at(ep, rungs), thr, dly,
                        _seg(ep, t, "sizes_bytes"), L - t, L)
        with torch.inference_mode():
            logits, _ = model(torch.from_numpy(s).unsqueeze(0).to(device))
        r = int(torch.argmax(logits[0]).item())
        rungs.append(r)
        bw = ep.bw_bps[min(t, len(ep.bw_bps) - 1)]
        dl = _seg(ep, t, "sizes_bytes")[r] * 8.0 / max(bw, 1.0)
        hist.append((bw / 1.0e6, dl))
        last_bitrate = PENSIEVE_LADDER_MBPS[r]
    return rungs


def _buffer_at(ep, rungs_so_far: List[int]) -> float:
    """Replay buffer level after the rungs chosen so far (so the state's buffer matches the replay)."""
    buffer = 0.0
    for t, r in enumerate(rungs_so_far):
        bw = ep.bw_bps[min(t, len(ep.bw_bps) - 1)]
        _, buffer, _ = _rebuf_buffer(buffer, _seg(ep, t, "sizes_bytes")[r], bw)
    return buffer


def eval_policy(model, episodes, device="cpu", window: int = 8) -> float:
    """Mean floor-normalized VMAF-%oracle (0..100) of `model` over `episodes`."""
    model.eval()
    pcts = []
    for ep in episodes:
        if ep.vmafs is None or ep.length < 2:
            continue
        oracle = _reward_of_rungs(ep, _greedy_oracle_rungs(ep))
        floor = _reward_of_rungs(ep, [0] * ep.length)
        pol = _reward_of_rungs(ep, _policy_rungs(model, ep, device, window))
        denom = oracle - floor
        if abs(denom) < 1e-6:
            continue
        pcts.append(max(0.0, min(1.0, (pol - floor) / denom)))
    model.train()
    return 100.0 * float(np.mean(pcts)) if pcts else float("nan")
