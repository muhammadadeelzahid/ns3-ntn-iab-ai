# How the oracle picks a rung

A summary of what the per-segment reward means and how the DP oracle turns it into a
chosen bitrate rung (the supervised label `oracle_bitrate_idx`). Covers both the
**greedy** rule and the **dynamic-programming (DP)** rule the oracle actually uses.

Default weights (`scratch/ai_llm/config/oracle_reward.json`): `w_vmaf = 1.0`,
`w_stall = 10.0`, `w_switch = 0.3`, `scene_cut_discount = 0.6`.

---

## 1. The reward is a score, not an index

The reward

```
R_t(a) = w_vmaf · semantic_weight_t · vmaf_t(a)   # quality earned
       − w_stall · rebuffer_seconds_t              # freezing penalty
       − switch_penalty_t                          # instability penalty
```

is a **real number** (in VMAF-equivalent points) that scores *one* candidate rung `a`
on segment `t`. It is **not** the rung index. `a ∈ {0,1,2,3,4,5}` is the action; `R_t(a)`
is how good that action looks.

The three terms pull against each other:

- **Quality** rises with the rung (higher VMAF), amplified on semantically important
  segments (`semantic_weight_t` near 1).
- **Stall** punishes a rung whose segment is too big to download before the buffer runs
  out (`download_time = 8 · size_bytes / throughput`). At `w_stall = 10`, one second of
  freeze costs 10 quality points.
- **Switch** punishes a large VMAF jump from the previous segment (softened at scene cuts).

## 2. Selection = argmax over rungs

To pick, you **score all six rungs and take the index of the best score** — an `argmax`.
The score tells you *how good*; the *position* of the best score is the *which*:

```
rung a:   0     1     2     3     4     5
score:   10    30    45    40    20    -5
                      ↑ best  →  pick rung 2   (label = 2, not 45)
```

The number `45` is stored only as a diagnostic (`total_q_value`); the **index 2** is the
label the student learns to predict.

The two approaches below differ only in **what score** the argmax ranks: the immediate
reward (greedy) or the immediate reward plus the best achievable future (DP).

---

## 3. Greedy rule (myopic)

At each segment, pick the rung with the highest **immediate** reward given the current
player state, then advance the state and move on:

```
state = initial_player_state
for t in 0 .. N-1:
    for a in 0..5:
        R[a] = reward(state, a)        # quality − stall − switch, this segment only
    a_star   = argmax_a R[a]           # chosen rung
    state    = transition(state, a_star)   # update buffer / playing / prev_rung
```

Greedy never looks past the current segment. It *feels* a stall when it happens, but it
cannot foresee that a choice now sets up a stall later.

(A full 5-segment, 3-rung greedy walkthrough is in
[`reduced_dp_oracle_eq.md`](reduced_dp_oracle_eq.md) → "Worked example".)

## 4. DP rule (lookahead) — what the oracle uses

The oracle maximizes the **sum of rewards over the whole video**, not each segment in
isolation. For state `s` at segment `t` it ranks each rung by immediate reward **plus the
best reward reachable afterwards**:

```
Q_t(s, a) = R_t(s, a) + γ · V_{t+1}( transition(s, a) )
V_t(s)    = max_a Q_t(s, a)            # value of being in state s at t
a*_t(s)   = argmax_a Q_t(s, a)         # chosen rung  →  oracle_bitrate_idx
```

`V` is the best total reward from here to the end; `γ` is a discount (≈1 for this finite
horizon). Compute `V` from the last segment backwards (`V_N = 0`), then replay forward
from the start state to read off `a*` per segment. Because the buffer transition is
deterministic, the implementation runs this as a forward pass with backpointers —
mathematically identical, just cheaper.

The argmax now ranks `Q` (reward **+** future), so a rung with a great *immediate* reward
can still lose if it drains the buffer and forces a costly stall on a later segment.

---

## 5. Worked example where greedy and DP differ

3 segments, 2 rungs (low VMAF 40, high VMAF 80). `segment_duration = 2 s`,
`w_vmaf = 1`, `w_stall = 10`, switch penalty omitted (`w_switch = 0`) to isolate the
buffer effect. Constant link where **high** takes 7 s to download, **low** takes 1 s.
Player starts in playback with **buffer = 5 s**.

Player model per segment: `rebuffer = max(0, download − buffer_before)`,
`buffer_after = max(0, buffer_before − download) + 2`.

**Greedy** (picks the higher immediate reward each step):

| seg | buffer in | low R | high R | greedy pick | stall | buffer out |
|-----|-----------|-------|--------|-------------|-------|------------|
| 0 | 5 | 40 | 80 − 10·2 = **60** | **high** | 2 s | 2 |
| 1 | 2 | **40** | 80 − 10·5 = 30 | **low** | 0 | 3 |
| 2 | 3 | **40** | 80 − 10·4 = 40 | low (tie) | 0 | 4 |

Greedy total = 60 + 40 + 40 = **140** (quality 160, minus a 2 s stall at seg 0).

**DP** (maximizes the 3-segment sum). The best path is **low, low, high**:

| seg | buffer in | pick | reward | stall | buffer out |
|-----|-----------|------|--------|-------|------------|
| 0 | 5 | low | 40 | 0 | 6 |
| 1 | 6 | low | 40 | 0 | 7 |
| 2 | 7 | high | 80 − 10·0 = 80 | 0 | 2 |

DP total = 40 + 40 + 80 = **160**, with **zero stalls**.

Same total quality (160 VMAF) as greedy, but DP **delays the high rung until the buffer
can afford it stall-free**, while greedy spent its high rung early and paid a 2 s freeze.
Greedy's myopic `argmax` at segment 0 (immediate 60 > 40) is the mistake DP avoids by
banking buffer first.

---

## 6. Greedy vs DP, and the output

- **DP ≥ greedy always.** Greedy is a special case that ignores future value
  (`γ = 0`). DP can only match or beat it.
- **They agree** when the immediately-best rung does not create a future cost — e.g.
  steady throughput with comfortable buffer, where grabbing quality now never sets up a
  later stall or a worse switch trajectory. (The 5-segment example in
  `reduced_dp_oracle_eq.md` is such a case.)
- **They diverge** exactly when an early high-reward rung drains the buffer (future
  stall) or forces a sharper later switch — the trap in §5.
- The oracle ships the **DP** label. `scripts/test_dp_oracle.py` checks DP optimality
  against brute force, so the labels are the true sum-optimal actions, not greedy ones.

What each segment's label row (`oracle_labels_state.jsonl`) records:

| field | meaning |
|-------|---------|
| `oracle_bitrate_idx` | `a*` — the chosen rung index (**the training label**) |
| `total_q_value` | `Q_t(s, a*)` — winning score (reward + future); diagnostic |
| `immediate_reward` | `R_t(s, a*)` — just the reward term |
| `future_value` | `γ · V_{t+1}` — the rest |

The student is trained (cross-entropy) to predict `oracle_bitrate_idx` from
`model_input`; the reward machinery above is how that label was produced.
