# LLM-ABR benchmark plan — model, evaluation, and paper figures

This document specifies the **single content-aware LLM-ABR policy** we train and benchmark for
NTN: its architecture, the evaluation protocol, and the paper figure/table selection.

> Earlier drafts described a V0–V3 ablation ladder (Tier-0 MLP, random-LLM, pretrained-LLM, and a
> language-channel variant). **That is abandoned.** There is now **one model** — the combined
> network+semantic LLM policy trained on the **greedy-reward causal teacher**. The language/text
> channel (V3) was tested and dropped (no robust gain; see the `v3-language-channel-result`
> finding). See git history for the ablation rationale.

**Frozen contract:** `model_input = network_norm (7) + semantic_norm (5) + ladder_context (6×2
[size_bytes, vmaf])`, optional time-series **window** `W` on the network branch (current `W=4`);
label = the teacher rung `∈ {0..5}` (field `oracle_bitrate_idx`, now holding the **greedy** label);
**teacher = the greedy-reward causal policy** (§4.0). Goal: imitation (SFT) now, QoE evaluation
in/out of distribution, RL later.

---

## 1. The model

### 1.1 Inputs (per decision)
- **network_norm**: `[W, 7]` — time-series window of `W` past segments (same `(run_id, node_id)`),
  z-scored per timestep (`feature_scaler_network.json`; throughput & last-segment-bitrate are
  `log1p`-scaled). `W=1` is the snapshot case.
- **semantic_norm**: `[5]` — current segment's **content** features (motion, saliency, object,
  quality-sensitivity, scene-cut-derived), z-scored (`feature_scaler_semantic.json`). This is the
  *numeric* content signal — there is **no language/text channel** (that variant was abandoned).
- **ladder_context**: `[6, 2]` — current segment `[size_bytes, vmaf]` per rung (raw).

### 1.2 Multimodal encoder
Produces a fixed embedding `e_fused` (all branches trainable).

| Branch | Input | Layers | Output |
|--------|-------|--------|--------|
| Network (temporal) | `[B, W, 7]` | `Conv1d(7→128,k3,pad1)→SiLU→Dropout→Conv1d(128→128,k3,pad1)→SiLU → AdaptiveAvgPool1d(1) → LayerNorm` | `net_emb [B,128]` |
| Semantic | `[B, 5]` | `Linear(5→64)→SiLU→LayerNorm→Dropout→Linear(64→64)→SiLU→LayerNorm` | `sem_emb [B,64]` |
| Ladder | `[B, 6, 2]` | per-rung `Linear(2→32)→SiLU`, then attention-pool over the 6 rungs | `lad_emb [B,32]` |
| Fusion | concat `[B,224]` | `Linear(224 → d_fuse)` | `e_fused [B, d_fuse]` |

The 1D-CNN network branch is length-agnostic (handles any window `W`).

### 1.3 Soft-token injection → LLM → policy head
The encoder output is projected into the LLM embedding space and injected as **soft tokens**
(bypassing the tokenizer):
```
e_fused → Linear(d_fuse → k·d_model) → soft_tokens [B, k, 1536]
        → Qwen2.5-1.5B (PRETRAINED, FROZEN + LoRA)  via inputs_embeds=
        → pool last hidden [B,1536] → policy head → 6 bitrate logits
```
- **Backbone:** Qwen2.5-1.5B-Instruct (hidden `d_model=1536`), **frozen**; **LoRA** (r=16, α=32,
  dropout 0.05) on attention proj + MLP. `k` soft tokens (default `k=4`).
- **Policy head:** `Linear(1536→256)→SiLU→Dropout(0.1)→Linear(256→128)→SiLU→Linear(128→6)`.
- **Action:** `argmax(logits)` → rung → bps via `NS3_RATE_LADDER_BPS`. (A `Linear(128→1)` value
  head is added only for the RL phase, not for these SFT benchmarks.)
- **Trainable:** encoder + soft-token projection + LoRA adapters + head. **Frozen:** base LLM.
  Checkpoint stores encoder + projection + LoRA + head + `window` + scaler hashes + `model_id`
  (frozen base loaded from HF, not saved).

### 1.4 Teacher, label, loss
- **Teacher = the greedy-reward causal policy** (§4.0) — NOT the DP-oracle. Labels are produced by
  `relabel_greedy.py` from the QoE reward in `config/oracle_reward.json` (`w_stall=65`).
- **Label** = teacher rung (`oracle_bitrate_idx`). **Loss** = class-weighted cross-entropy
  (inverse-frequency on the **train split only**; the rung distribution is skewed).
- Normalization applied **at load** via the two pinned scalers.

### 1.5 Training recipe
- QLoRA 4-bit base for training; bf16 elsewhere; `torch.inference_mode` at eval.
- AdamW, lr 1e-3 (encoder/head) / 1e-4 (LoRA), cosine, ~5–10 epochs, early-stop on val.
- Group-aware splits (§2.2); multiple seeds; report mean ± std. Current window `W=4`.

---

## 2. Evaluation protocol

### 2.1 Metrics
**A. Imitation quality (sanity — does it match the teacher?)**
- **macro-F1** and **per-class recall** (NOT top-1 accuracy — rung imbalance makes it a trap);
  6×6 **confusion matrix**. (Offline F1 ≠ online QoE — diagnostic only.)

**B. QoE (the headline)** — replay the policy's chosen rungs through the player model (offline
buffer model first; live ns-3 for the finalists). Report:
- **avg VMAF** ↑, **rebuffering time** ↓, **switch magnitude/frequency** ↓, and
- **combined QoE** `= w_vmaf·Σ(ω·VMAF) − w_stall·Σstall − w_switch·Σ|ΔVMAF|` (weights from
  `oracle_reward.json`), reported as **%-of-greedy** (§4.0).

**C. Systems (deployment story)** — decision **latency p50/p95**, model **size / GPU memory**.

### 2.2 Splits — report in-distribution AND out-of-distribution separately
1. **In-distribution:** held-out runs of seen videos (group-aware by `run_id`).
2. **Unseen content:** held-out **videos** (group-aware by `video_name`) — where content-awareness
   should help most (the Fig-3 story).
3. **Unseen network/NTN regime:** test traces from network conditions not in training.

No `(run_id, node_id)` session may cross a split (also required by windowing). The current split is
the **headroom-stratified, group-aware** split pinned in
`scratch/ai_llm/benchmark/catalog_selection/stage_b_subset/manifest.json` (so the low-headroom
bucket is represented in test).

### 2.3 Baselines and reference
- **Heuristics:** FDASH, BOLA, SVAA, SFTM, RAAHS, AAASH, OSMP.
- **Reference ceiling:** the **greedy-reward causal policy** ("greedy oracle"; §4.0) — the metric
  denominator. (The DP-oracle may be computed for an appendix sanity check only.)

### 2.4 Result matrix
For every (ours/baseline × test-split × link-rate): QoE components + combined **%-of-greedy**, and
latency. Paired comparisons across the same seeds; report mean ± std with a significance check.

---

## 3. Reproducibility & artifacts
- Pin per run: dataset `schema_version` + `manifest.json` (scaler SHAs), `window`, backbone
  `model_id`, LoRA config, seed, class-weight scheme, split definition, and the **reward config**
  (`oracle_reward.json`, `w_stall=65`).
- Each checkpoint records `window`, scaler hashes, `model_id` (so eval/inference reproduce
  training exactly).
- Log: training curves (loss, macro-F1), result rows, confusion matrices, QoE breakdowns, latency
  profiles. One results table per test-split.

---

## 4. Paper presentation — figure & table selection (5-page budget)

Target venue is a **5-page double-column** paper (≈4 float objects fit comfortably before the
eval section gets crowded). This section records the **full menu** of candidate floats, the
**selected set**, and the **conditional decision** on the content-novelty figure. Mockups with
illustrative placeholder data live in `scratch/ai_llm/benchmark/paper/mockups/` (gitignored,
each stamped `[DUMMY]`); the real versions regenerate from scored CSVs once Stage C + the
fixed-binary benchmark land.

### 4.0 Benchmarking reference — the GREEDY equation, NOT the DP-oracle  [DECISION 2026-06-30]

All relative metrics ("%-of-oracle") are scored against the **greedy-reward causal policy**, not
the acausal DP-oracle. The DP-oracle is **dropped** as the headline reference.

- **Reference = the greedy equation** (per-segment argmax of the QoE reward, present-only/causal):
  `a_n = argmax_r [ ω_n·VMAF_n(r) − μ·rebuffer_n(r) − λ·|ΔVMAF| ]`
  (same reward in `config/oracle_reward.json`; implemented in `relabel_greedy.py` /
  `inference_server.py --policy greedy`).
- **Why:** the DP-oracle is future-aware (acausal) and over-asks — not a target a causal online
  policy can reach. The greedy equation is the **realizable causal ceiling** the teacher targets,
  so "% of greedy" measures how close a policy gets to the best *causal* decision. (See
  `[[acausal-oracle-misspecified-expert]]` / `[[causal-expert-fixes-online-collapse]]`.)
- **Naming:** for brevity, figures/tables call this reference the **"oracle"** — but it is the
  greedy equation. Define once in the paper ("we report % of a causal greedy oracle"), use
  "oracle" thereafter.
- **Metric:** `%-of-greedy = (policy − floor) / (greedy − floor)`, floor = lowest-rung policy.
- The DP solver may still be computed for an appendix sanity check, but it is not the headline
  standard. The mockup PNGs still say "DP-oracle" in titles — relabel to "greedy oracle" on
  regeneration.

### 4.1 The full menu of candidates (options)

**Figures**
| ID | Figure | Axes | What it proves | Mockup |
|----|--------|------|----------------|--------|
| Arch | System/architecture diagram | — | frozen Qwen + LoRA + multimodal encoder → policy head; ns-3 closed loop | (have it) |
| F1 | Quality–Rebuffering Pareto | x = mean VMAF ↑, y = rebuffer s ↓ | we're Pareto-efficient (high quality *and* low stall) | `fig1_pareto_DUMMY.png` |
| F2 | %-of-greedy vs link rate | x = 5 feeder rates (congestion →), y = %-of-greedy | robustness under congestion (the NTN angle) | `fig2_curve_DUMMY.png` |
| F3 | Content-awareness gain | x = content type (easy→demanding), y = delivered VMAF; bars net-only vs +semantic | the **novelty**: semantic helps, content-specifically | `fig3_bycontent_DUMMY.png` |
| F4 | Per-session QoE CDF | x = normalized QoE, y = CDF | we lift worst-case sessions, not just the mean | `fig4_cdf_DUMMY.png` |

**Tables**
| ID | Table | Contents | What it proves |
|----|-------|----------|----------------|
| T1 | Main benchmark | ours + heuristics × 5 rates + Overall, %-of-greedy | complete, credible quantitative comparison |
| T2 | QoE pillars | mean VMAF / rebuffer s / #switches / regret | decomposes *where* the score comes from |
| T3 | Ablation / standard-QoE | ±semantic, ±VMAF; scored under our metric AND a semantic-free QoE | gain isn't a self-grading artifact |

### 4.2 Selected set for 5 pages

**Keep as floats (4):**
1. **Arch diagram** — system anchor (mandatory).
2. **T1 — main benchmark** — the rigor anchor; one compact table carries every baseline ×
   every congestion level. Also tells the NTN congestion story via its 9 / 4.5 columns.
3. **F3 — content-awareness gain** — the novelty justification (see §4.3 for the conditional).
4. **F1 — Quality–Rebuffering Pareto** — efficiency at a glance; complements T1 (tradeoff
   geometry vs per-rate numbers, no overlap).

**Demote to prose / appendix (avoid redundant floats):**
- **F2** is redundant with T1's per-rate columns → cut, *unless* the Fig 3 fallback triggers
  (§4.3), in which case F2 is promoted and T1 trims to its Overall column.
- **T2** is redundant with F1's axes → one sentence in prose ("Ours matches the throughput
  baseline's VMAF at a fraction of the rebuffering").
- **T3** → two sentences with inline numbers, **%-of-greedy** (kept *in text* because reviewers
  will ask: "removing the semantic head drops overall %-of-greedy by N pts; under a canonical
  semantic-free QoE we still lead the throughput baseline"). Numbers computed at Stage C.
- **F4 (CDF)** → appendix / extended version.

### 4.3 Conditional decision on F3 (the content-novelty figure) — FINAL CALL

F3 is the only float whose inclusion is **gated on the result supporting our theory**, because our
prior offline language probe found *no* robust semantic advantage in pure imitation (see the
`v3-language-channel-result` finding). Decision rule, applied once the real Stage-C benchmark
decision logs exist:

- **Go/no-go criterion (theory supported):** the gain (ours − network-only delivered VMAF)
  **trends up with content demand** (monotonic or clearly increasing across the content-type
  ordering) **and** the high-demand gain is materially > noise (rule of thumb: ≥ ~5 VMAF on the
  most demanding bucket, and the trend holds across videos, not one outlier).
- **DEFAULT (criterion met):** **F3 = `fig3_bycontent` (delivered-VMAF grouped bar by content
  type)** is the locked Fig 3 design. Paper headline = *content-aware ABR for NTN*.
- **FALLBACK (criterion NOT met — flat/within-noise gap):** **drop F3 entirely**, promote
  **F2 (%-of-greedy vs feeder rate)** into slot 4, trim T1 to its Overall column, and **re-pitch
  the paper around congestion robustness** (our other genuine strength, still on-theme for NTN)
  rather than content-awareness. Report the content-null honestly as a secondary finding — do
  not manufacture the staircase.

Rationale for the by-content bar over the earlier "VMAF-headroom tercile" version: identical
message but the axes are immediately legible (recognizable content categories + delivered VMAF),
and the near-tie on easy content is an honest built-in control (the gain is content-specific, not
a global quality inflation).

### 4.4 Auto-regeneration (deferred)
When real scored CSVs land, wire the four selected floats into `make_paper_figs.py` /
`make_paper_table.py` so they regenerate deterministically (F3 from the per-content slice of the
benchmark decision logs; T1 from the scored `results/*.txt`). No new sims needed — all are
re-slices of the benchmark output, scored against the greedy reference (§4.0).
