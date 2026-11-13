# Diagnosis: why the model "collapses" on low-headroom content

**Date:** 2026-07-05 · **Benchmark:** fixed-binary, 10 headroom-stratified test videos × 5 rates,
greedy causal ceiling (`config/oracle_reward.json`, w_stall=65) · **Tool:**
`scratch/ai_llm/benchmark/diag_low_headroom.py` (GPU job 7208104; full log
`slurm-diag-lowheadroom-7208104.out`).

**Symptom.** Table 1 by-content cut: the model scores 0% of the greedy ceiling on pexels_35132716
(headroom 25) and 20% on pexels_30886101 (headroom 38) while heuristics score 61–100% there — the
opposite of the content-aware hypothesis. Mode of failure: systematic UNDER-asking (model parks 1–2
rungs below the teacher; zero stalls; guard interventions rare and −1 rung only).

## Root cause (three stacked layers, each evidenced)

### 1. Policy layer — a confident learned under-ask prior (NOT a serving bug, NOT OOD, NOT calibration)
- **Parity:** offline checkpoint argmax reproduces the online decision at **92.5%** of the 199
  benchmark states (divergences cluster at early segments = window warm-up, ±1 rung). The low rungs
  come from the policy itself.
- **Confidence:** at the 119 under-ask states, mean softmax on the chosen (low) rung = **0.56** vs
  **0.08** on the teacher's rung; the teacher's rung is even the runner-up class only 19% of the
  time; top-2 margin 0.35. This is a dominant prior, not a near-miss calibration issue.
- **In-distribution:** kNN (k=50) of the failing states against the 32,466 train rows in the pinned
  normalized feature space: neighbour buffer 10.9 s ≈ query buffer 10.8 s (NO buffer covariate
  shift), semantic |z|>3 rate 0% (vs 7% train base rate — if anything the failing videos are
  *more* typical than average).
- **Where the prior comes from:** those train neighbourhoods carry a **34% r0 plurality label**
  (rest spread r1–r5). Cross-entropy imitation of an ambiguous/plurality label sharpens into a hard
  argmax — the model votes r0 with 0.56 confidence where the teacher itself only chose r0 a third
  of the time.

### 2. Teacher layer — the greedy teacher's r0 labels are switch-cost hysteresis, and the label is trajectory-dependent
- On low-headroom train videos, P(label=r0 | buffer ≥ 4s) = **33%** vs 18% at low buffer —
  *inverted* from a stall-avoidance story. The r0s are not "protect the buffer"; they are
  **lock-in**: on low-headroom content the per-rung quality gain (w_vmaf·sem·ΔVMAF) is smaller than
  the switch penalty (w_switch·|ΔVMAF|), so the greedy teacher stays wherever its own chain started.
- The relabeled ceiling on the model's own benchmark trajectories tracks the TEACHER's counterfactual
  prev-VMAF chain, which locks at r1 on pexels_35132716 — while the model's chain locked at r0.
  Two hysteretic policies one rung apart; the label says the model is "wrong" at almost every step.

### 3. Metric layer — floor-normalization turns a 6–8 VMAF-point gap into 0%
- Honest magnitude: the model's under-ask costs **5.8 VMAF pts/segment** (pexels_35132716) and
  **8.1 pts/segment** (pexels_30886101), with **zero stalls**. That is a modest quality loss, not a
  QoE catastrophe.
- The bounded %-of-greedy = (policy−floor)/(ceiling−floor) clipped to [0,100]. On pexels_35132716
  the span is ≈0 (−1.34 QoE/seg) because ceiling(r1-lock) ≈ floor(r0) — so the metric is effectively
  **binary**: match the teacher → 100%, sit one rung under → 0%.
- **Wider finding:** the ceiling−floor span is **negative on 7 of 10 test videos** (teacher replay
  under realized throughput stalls below the all-r0 floor; e.g. chef_cooking −2096 QoE/seg). On
  those videos the headline pct degenerates to per-run binary "beat the teacher replay or not" for
  EVERY arm. Cross-arm ranking within a video stays consistent (all arms scored identically), but
  per-video pct values are not comparable in magnitude, and `regret_per_seg` / the QoE pillars
  (VMAF, stall, switches) are the trustworthy quantities.

## One-paragraph root cause
The model does not "collapse" on low-headroom content — it **confidently imitates the greedy
teacher's own r0 hysteresis** (a 34%-plurality label in exactly these feature neighbourhoods,
sharpened to a 0.56-confidence argmax by cross-entropy), landing one rung below the teacher's
counterfactual r1 chain. The real cost is 6–8 VMAF points/segment with zero stalls; the
floor-normalized %-of-greedy then reports this as 0–20% because on these videos the ceiling sits
almost on the floor (span ≈ 0 or negative), making the metric near-binary. The content-aware
hypothesis fails not because the model ignores content, but because the teacher's own
content-conditioned behaviour is ambiguous precisely where headroom is low.

## Options (user decision — none executed)
**(a) Paper-only (no retrain).** Report Table 1 with the regret/pillar framing; footnote the
binary-span caveat; state the cause in one line ("conservative imitation prior; one-rung under-ask
maximally penalized by floor-normalization"). Cheapest; honest; keeps the "competitive, lowest
regret, no hand-tuning" narrative.

**(b) Reduce teacher hysteresis + retrain.** The r0 lock-in is w_switch-driven on low-headroom
content: relabel with a smaller w_switch (or a switch cost floored by the actual VMAF gap) →
teacher prefers the quality-optimal rung more often → less ambiguous labels → retrain
(train_jsonl_bola rebuild is a one-flag relabel + 1 GPU retrain). Attacks the cause at the source;
risks re-tuning ripples (offline gate first via `sweep_wstall_offline`-style comparison).

**(c) DAgger round on the model's own benchmark states.** The 50 scored cells are already
greedy-relabelled (`Quic_artifacts/AiDash/shards_greedy/`) = ready aggregation data; mix into train
and retrain. Fixes the trajectory mismatch (teacher chain vs model chain) directly, but inherits
the teacher's hysteresis labels unchanged.

**(d) Metric hardening for the paper.** Report per-video regret (QoE units) or VMAF/stall pillars
as the by-content figure instead of pct; or floor-normalize against a stronger floor. Orthogonal to
(a)–(c); recommended for whichever narrative is chosen.
