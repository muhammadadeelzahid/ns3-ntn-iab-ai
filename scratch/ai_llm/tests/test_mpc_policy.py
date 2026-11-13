#!/usr/bin/env python3
"""Tests for mpc_policy.RobustMpcPolicy (RobustMPC baselines). Runs under pytest or directly:
    PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/tests/test_mpc_policy.py
Uses a real metadata CSV (big_buck_bunny, 59 segments) so the ladder/VMAF/semantic columns are genuine.
"""
import itertools
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))            # scratch/ -> ai_llm.*
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "scripts")))  # dp_oracle
from ai_llm.greedy_teacher import greedy_rung  # noqa: E402
from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402
from ai_llm.mpc_policy import (DEFAULT_QUALITY, LINEAR_REBUF_PENALTY, LINEAR_SMOOTH_PENALTY,  # noqa: E402
                               RobustMpcPolicy)
from ai_llm.paths import NS3_RATE_LADDER_BPS  # noqa: E402
from dp_oracle import _segment_reward, _segment_transition, _switch_penalty, load_reward_config  # noqa: E402

META = os.environ.get("MPC_TEST_META", "/project/6030214/adeel/abr/output/metadata/big_buck_bunny_segments.csv")
CFG = load_reward_config(None)


def _idx():
    return load_metadata_index(META, None)


def _ladders(idx, seg):
    m = idx.get(seg)
    sizes = [float(m.size_bytes[lv]) for lv in LADDER_LEVELS]
    vmafs = [float(m.vmaf[lv]) for lv in LADDER_LEVELS]
    return sizes, vmafs


# ---------------------------------------------------------------- (a) extremes
def test_extremes_top_and_bottom():
    idx = _idx()
    for obj in ("linear", "qoe"):
        p = RobustMpcPolicy(idx, objective=obj)
        rung, k = p.plan(5, 10.0, 1e12, None)
        assert rung == len(NS3_RATE_LADDER_BPS) - 1, (obj, rung)      # infinite bandwidth -> top rung
        rung, k = p.plan(5, 0.5, 1.0, None)
        assert rung == 0, (obj, rung)                                 # ~zero bandwidth -> rung 0
        assert k == 5


# ---------------------------------------------------------------- (b) horizon shrinks at the end
def test_horizon_shrinks_near_end():
    idx = _idx()
    p = RobustMpcPolicy(idx, objective="qoe")
    n = p.n_segments
    combos, total = p.horizon_rewards(n - 2, 8.0, 5e6, 3)
    assert combos.shape == (len(NS3_RATE_LADDER_BPS) ** 2, 2) and total.shape == (36,)
    rung, k = p.plan(n - 1, 8.0, 5e6, 3)
    assert k == 1
    rung, k = p.plan(n - 7, 8.0, 5e6, 3)
    assert k == 5


# ---------------------------------------------------------------- (c) robust prediction
def test_robust_discount_and_monotone_in_bandwidth():
    idx = _idx()
    p = RobustMpcPolicy(idx, objective="linear")
    # hand sequence: samples 4,8,2 Mbps -> hm = 3/(1/4+1/8+1/2) Mbps; preds after each step; errors vs last pred
    for s in (4e6, 8e6, 2e6):
        p._ingest_sample(7, s)
        p.predict_throughput(7)
    samples = list(p._samples[7]); preds = list(p._preds[7]); errs = list(p._errors[7])
    hm = len(samples) / sum(1.0 / s for s in samples)
    assert abs(preds[-1] - hm) < 1e-6
    # errors were computed against the prediction that preceded each sample
    assert len(errs) == 2 and abs(errs[0] - abs(4e6 - 8e6) / 8e6) < 1e-12
    bw, hm2, max_err = p.predict_throughput(7)
    assert abs(hm2 - hm) < 1e-6 and abs(bw - hm / (1.0 + max_err)) < 1e-6 and max_err == max(errs)
    # Monotonicity in predicted bandwidth. Every combo's rebuffer falls as bw rises, so the best horizon
    # reward is non-decreasing for both objectives, and at K=1 the chosen rung is too (greedy rule). With
    # K=5 and the DOWN-ONLY switch penalty the first action may legitimately dip (a lower start can avoid a
    # paid drop later), so the K=5 rung test is only asserted for the symmetric linear objective.
    grid = np.logspace(5, 8, 25)
    for obj in ("linear", "qoe"):
        q = RobustMpcPolicy(idx, objective=obj)
        best = [q.horizon_rewards(10, 6.0, bw, 2)[1].max() for bw in grid]
        assert all(a <= b + 1e-9 for a, b in zip(best, best[1:])), (obj, best)
        q1 = RobustMpcPolicy(idx, objective=obj, horizon=1)
        rungs1 = [q1.plan(10, 6.0, bw, 2)[0] for bw in grid]
        assert all(a <= b for a, b in zip(rungs1, rungs1[1:])), (obj, rungs1)
    q = RobustMpcPolicy(idx, objective="linear")
    rungs = [q.plan(10, 6.0, bw, 2)[0] for bw in grid]
    assert all(a <= b for a, b in zip(rungs, rungs[1:])), rungs


# ---------------------------------------------------------------- (d) linear objective = Pensieve mpc.py formula
def _linear_reference(p, pos, buf, bw, last_rung, k):
    best, best_combo, out = -1e18, None, []
    for combo in itertools.product(range(p.n_rungs), repeat=k):
        b, rebuf, total, prev = buf, 0.0, 0.0, (None if last_rung is None else p.ladder_mbps[last_rung])
        for step, r in enumerate(combo):
            dl = 8.0 * p.sizes[pos + step, r] / bw
            if b < dl:
                rebuf += dl - b
                b = 0.0
            else:
                b -= dl
            b = min(p.max_buffer, b + p.seg_dur)
            rate = p.ladder_mbps[r]
            total += rate
            if prev is not None:
                total -= LINEAR_SMOOTH_PENALTY * abs(rate - prev)
            prev = rate
        total -= LINEAR_REBUF_PENALTY * rebuf
        out.append(total)
        if total >= best:                      # reference tie rule: later combo wins
            best, best_combo = total, combo
    return np.asarray(out), best_combo


def test_linear_matches_reference_enumeration():
    idx = _idx()
    p = RobustMpcPolicy(idx, objective="linear", horizon=3)
    rng = random.Random(1)
    for _ in range(20):
        pos = rng.randrange(0, p.n_segments - 3)
        buf, bw, last = rng.uniform(0, 25), 10 ** rng.uniform(5.5, 7.5), rng.choice([None, 0, 2, 5])
        combos, total = p.horizon_rewards(pos, buf, bw, last)
        ref, ref_best = _linear_reference(p, pos, buf, bw, last, 3)
        assert np.allclose(total, ref, atol=1e-9)
        assert p.plan(p.seg_ids[pos], buf, bw, last)[0] == ref_best[0]


# ---------------------------------------------------------------- (e) qoe objective parity with dp_oracle + greedy_rung
def test_qoe_parity_with_dp_oracle_and_greedy_rung():
    idx = _idx()
    W = dict(CFG["reward_weights"])
    player = CFG["player_model"]
    p = RobustMpcPolicy(idx, objective="qoe", reward_cfg=CFG, horizon=3)
    rng = random.Random(2)
    for _ in range(200):
        pos = rng.randrange(1, p.n_segments - 3)
        buf, bw, last = rng.uniform(0, 25), 10 ** rng.uniform(5.5, 7.5), rng.choice([None, 0, 1, 3, 5])
        combos, total = p.horizon_rewards(pos, buf, bw, last)
        i = rng.randrange(combos.shape[0])
        combo = combos[i]
        # scalar reference: dp_oracle's transition (steady state, playing=1) + reward + switch penalty
        b, ref = buf, 0.0
        prev_v = None if last is None else p.vmafs[pos - 1, last]
        for step, r in enumerate(combo):
            sizes, vmafs = _ladders(idx, p.seg_ids[pos + step])
            dl = 8.0 * sizes[r] / bw
            rebuf, b, _ = _segment_transition(b, 1, dl, float(player["segment_duration_s"]),
                                              float(player["max_buffer_s"]), float(player["startup_threshold_s"]))
            cut = int(p.cut[pos + step])
            sw = 0.0 if prev_v is None else _switch_penalty(vmafs[r], prev_v, cut, W)
            ref += _segment_reward(float(p.sem[pos + step]), vmafs[r], rebuf, sw, W, r, p.n_rungs)
            prev_v = vmafs[r]
        assert abs(total[i] - ref) < 1e-9, (total[i], ref)
    # K=1, no robust discount == the greedy teacher rule (ties excluded)
    p1 = RobustMpcPolicy(idx, objective="qoe", reward_cfg=CFG, horizon=1)
    W1 = dict(W); W1.pop("thr_safety", None)
    n_cmp = 0
    for _ in range(200):
        pos = rng.randrange(1, p1.n_segments)
        buf, bw, last = rng.uniform(0, 25), 10 ** rng.uniform(5.5, 7.5), rng.choice([None, 0, 2, 4])
        combos, total = p1.horizon_rewards(pos, buf, bw, last)
        top2 = np.sort(total)[-2:]
        if top2[1] - top2[0] < 1e-6:
            continue
        sizes, vmafs = _ladders(idx, p1.seg_ids[pos])
        prev_v = None if last is None else p1.vmafs[pos - 1, last]
        g = greedy_rung(bw, buf, sizes, vmafs, prev_v, float(p1.sem[pos]), int(p1.cut[pos]), W1)
        assert p1.plan(p1.seg_ids[pos], buf, bw, last)[0] == g
        n_cmp += 1
    assert n_cmp > 150


# ---------------------------------------------------------------- (f) tie-break = last combo
def test_tie_break_last_combo():
    idx = _idx()
    p = RobustMpcPolicy(idx, objective="linear", horizon=2)
    p.horizon_rewards = lambda pos, buf, bw, last, k=None: (p._combos_for(2), np.zeros(36))  # all tie
    assert p.plan(3, 5.0, 1e6, None) == (p.n_rungs - 1, 2)


# ---------------------------------------------------------------- (g) serving flow + determinism
def _drive(p, node, n_steps, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n_steps):
        req = {"node_id": node, "segment_fetch_time_s": rng.uniform(0.3, 3.0) if i else 0.0,
               "buffer_level_s": rng.uniform(0, 20), "throughput_estimate_bps": 0.0 if i else 3e6}
        out.append(p.predict(req, i))
    return out


def test_serving_flow_and_determinism():
    idx = _idx()
    for obj in ("linear", "qoe"):
        p = RobustMpcPolicy(idx, objective=obj)
        a = _drive(p, 5, 12, 3)
        p.reset_history()
        b = _drive(p, 5, 12, 3)
        assert a == b
        assert all(0 <= r < p.n_rungs for r, _ in a)
        tag = "qmpc" if obj == "qoe" else "robustmpc"
        assert all(reason.startswith(tag) for _, reason in a)
        assert "K=5" in a[1][1] and "bw=" in a[1][1]
        # two UEs keep independent state
        p.reset_history()
        c = _drive(p, 5, 6, 3); d = _drive(p, 6, 6, 4)
        assert len(p._samples) == 2 and p._last_rung[5] == c[-1][0] and p._last_rung[6] == d[-1][0]
    # cold start without any estimate -> DEFAULT_QUALITY
    p = RobustMpcPolicy(idx, objective="qoe")
    rung, reason = p.predict({"node_id": 1, "segment_fetch_time_s": 0.0, "buffer_level_s": 0.0,
                              "throughput_estimate_bps": 0.0}, 0)
    assert rung == DEFAULT_QUALITY and "cold-start" in reason


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print("%d tests passed" % len(tests))
