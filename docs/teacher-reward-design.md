# Teacher reward design — terminology & rationale (paper reference)

Reference for the paper's methodology/teacher section. Defines every term in the greedy-reward causal
teacher, the two candidate fixes explored for the low-headroom under-ask ("down-only" smoothness and
the rate-efficiency term), and the offline evidence for each. Config: `config/oracle_reward.json`.
Sweep tool: `scratch/ai_llm/benchmark/sweep_wswitch_offline.py`. Diagnosis that motivated the fix:
`docs/low-headroom-collapse-diagnosis.md`.

## 1. The greedy-reward causal teacher (current baseline)

Per segment `n`, the teacher picks the rung `r` that maximizes the **immediate** QoE reward using only
present information (causal, myopic — so a model can reproduce it online):

```
reward_n(r) =  w_vmaf · sem_n · VMAF_n(r)                          (quality, content-weighted)
             − w_stall · max(0, 8·size_n(r)/(thr_safety·thr_n) − buf_n)   (rebuffering)
             − SWITCH_n(r)                                          (smoothness — see §2)
             [ − w_rate · bitrate_n(r) ]                            (rate efficiency — OPTIONAL, see §3)
```

Base weights: `w_vmaf=0.65`, `w_stall=65`, `w_switch=0.8`, `scene_cut_discount=0.65`, `thr_safety=1.0`,
`w_rate=0` (off). `sem_n ∈ [0,1]` is the per-segment content weight (min-max mixture of
motion/saliency/object/quality-sensitivity). This is the **VMAF-quality family** (Comyco/GreenABR-like)
— the `sem` multiplier + scene-cut discount are our additions; nothing else is novel structurally.

## 2. Smoothness term: symmetric vs. "down-only" (directional)

The smoothness term penalizes changing the delivered quality between consecutive segments.

- **Symmetric (current):**
  ```
  SWITCH_n(r) = w_switch · |VMAF_n(r) − VMAF_{n-1}(chosen)| · (1 − scene_cut_discount · is_cut_n)
  ```
  Up-switches and down-switches cost the same.

- **"down-only" (directional smoothness):**
  ```
  SWITCH_n(r) = w_switch · max(0, VMAF_{n-1}(chosen) − VMAF_n(r)) · (1 − scene_cut_discount · is_cut_n)
  ```
  Only **quality drops** are penalized; **climbing to a higher rung is free.**

**Why "down-only" exists (the lock-in problem it fixes).** With the symmetric term, an upward move of
`ΔV` gains `w_vmaf·sem·ΔV` but costs `w_switch·ΔV`. Since `w_vmaf·sem ≤ 0.65 < 0.8 = w_switch` for all
`sem`, an upward move is **never** myopically rational (except at scene cuts). The reward is therefore
maximized at `VMAF_prev` → the teacher **locks in** wherever its chain started, sitting one rung low on
low-headroom content. The model imitates this as a confident under-ask prior → the benchmark
"collapse" (0–20% of the ceiling on low-headroom videos; see diagnosis doc). Making up-switches free
removes the trap: the teacher climbs to the quality-optimal rung, while still discouraged from dropping
quality.

**Why it is principled (paper framing).** In perceived QoE a quality **drop** is far more noticeable
and annoying than a quality **increase**; asymmetric/directional smoothness (penalize down-switches
more than up-switches) is well-motivated by subjective-quality studies and used in several QoE models.
So "down-only" is not an ad-hoc hack — it is *directional smoothness*.

## 3. Rate-efficiency term (`w_rate`) — what actually restores content-awareness

"down-only" (or any `w_switch` reduction) fixes the collapse but makes the teacher climb to the highest
**sustainable** rung **everywhere** — because the greedy reward has no cost for spending bandwidth, so
on saturated (low-headroom) content where all rungs give ~the same VMAF, max-VMAF = max-rung. That is
NOT content-aware (it wastes bandwidth on saturated content and gives no low-vs-high differentiation).

The **rate-efficiency term** `− w_rate · bitrate_n(r)` (bitrate in Mbps) fixes this: a higher rung now
only pays off when its VMAF gain beats its bitrate cost. On low-headroom content (flat VMAF) the rate
cost dominates → the teacher stays modest; on high-headroom content (steep VMAF) the quality gain wins
→ it climbs. This is exactly the "content-aware ABR" behavior the paper's Fig-3 hypothesis needs.

## 4. Offline evidence (gate on the model's own benchmark states + train set)

Metric = `corr(video headroom, teacher mean rung)` — a content-aware teacher climbs with headroom.
(Full table + rung distributions: run `sweep_wswitch_offline.py`; stall stays ≈1–2% for all rows.)

| Teacher | corr(headroom, mean-rung) | low-hr → high-hr mean rung | reads as |
|---|--:|--:|---|
| symmetric `w_switch=0.8` (base) | +0.13 | 2.2 → 2.5 | flat; lock-in |
| down-only `w_switch=0.8` | +0.39–0.49 | 3.6 → 4.3 | climbs **everywhere** (not content-aware) |
| **down-only + `w_rate=1.0`** | **+0.85** | **1.8 → 3.9** | **content-aware** (modest on saturated, climbs on high-headroom) |
| down-only + `w_rate=5.0` | +0.86 | 0.7 → 2.4 | over-frugal (parks r0 even at headroom 38) |

Collapse videos under **down-only + `w_rate=1.0`**: pexels_35132716 (headroom 25) → mean rung 1.8;
pexels_30886101 (headroom 38) → 3.2 — sensible, no r0 collapse. This is the current sweet-spot
calibration (pending final selection + retrain).

## 5. Status / how it maps to the paper
- **Baseline result (already in floats):** symmetric `w_switch=0.8`; model 61.3% of greedy (2nd,
  RAAHS 64.3); content-aware hypothesis NOT supported (Fig-3 NO-GO). Floats in
  `scratch/ai_llm/benchmark/paper/` (Table 1, Fig 1 Pareto, Fig 2 rate-curve).
- **Proposed fix (option b):** retrain on **down-only + `w_rate=1.0`** teacher → re-benchmark →
  re-score all arms → regenerate Table 1 + Fig 1/2 and **re-open Fig-3** (by-content) on the new
  numbers. If the retrained model tracks the content-aware teacher, Fig-3 becomes the "we win on
  content" figure; if not, we fall back to the competitive/lowest-regret narrative.
- Changing `w_switch`/`w_rate` in `oracle_reward.json` changes BOTH the teacher labels AND the scoring
  QoE (one source of truth), so every headline number refreshes — this must be owned in the paper's
  QoE-definition paragraph (same posture as the earlier `w_stall` literature calibration).
