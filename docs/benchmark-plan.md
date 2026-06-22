# LLM-ABR benchmark plan — variants, architectures, and evaluation

This document specifies the ablation suite for the semantic LLM-ABR policy: the exact
architecture of each variant, what each comparison proves, and the evaluation protocol.
It is grounded in the frozen contract: `model_input` = `network_norm` (7) +
`semantic_norm` (5) + `ladder_context` (6×2 `[size_bytes, vmaf]`), optional time-series
**window** `W` on the network branch; label = `oracle_bitrate_idx ∈ {0..5}`; teacher = the
semantic DP oracle. Goal: imitation now (SFT), QoE evaluation in/out of distribution,
RL later.

---

## 1. The ablation ladder

Each variant changes **exactly one factor** from the one below it, so each *comparison*
answers one question.

| ID | Variant | One-line | Factor introduced |
|----|---------|----------|-------------------|
| **V0** | Tier-0 (no LLM) | encoder → policy head | — (baseline) |
| **V1** | Random LLM | encoder → **randomly-initialized** LLM (full fine-tune) → head | transformer **capacity** |
| **V2** | Pretrained LLM | encoder → **pretrained** LLM, frozen + LoRA → head; **numeric soft-tokens only** | **pretrained knowledge** |
| **V3** | Text + soft (ours) | V2 **+ content-derived language** (genre / scene description / NTN regime) as real text tokens | the **language channel** — *new information*, not the numbers re-spelled |

**Claim per comparison**

| Comparison | Question | Expected result |
|-----------|----------|-----------------|
| V0 vs V2 | Does an LLM beat a strong small encoder? | V2 ≥ V0, mainly **out-of-distribution** |
| V1 vs V2 | Pretrained knowledge vs raw capacity? | V2 ≫ V1 (replicates NetLLM Fig 13) |
| V2 vs V3 | Does *language carrying new information* (genre/scene/regime) help over numbers alone? | V3 > V2, **strongest on unseen content** (**our contribution**) |
| (opt) V2 vs V2-noSem | Does the semantic modality help at all? | V2 > V2-noSem |

---

## 2. Shared components (identical across variants)

These are held constant so any difference is attributable to the factor under test.

### 2.1 Inputs (per decision)
- **network_norm**: `[W, 7]` — time-series window of `W` past segments (same `(run_id, node_id)`), z-scored per timestep (`feature_scaler_network.json`; throughput & last-segment-bitrate are `log1p`-scaled). `W=1` is the snapshot case.
- **semantic_norm**: `[5]` — current segment, z-scored (`feature_scaler_semantic.json`).
- **ladder_context**: `[6, 2]` — current segment `[size_bytes, vmaf]` per rung (raw).
- **(V3 only) content/context text**: a short **content-derived** descriptor that carries information *not present in the numeric vectors* (genre, scene/object description, NTN regime) — **not** a restatement of the numeric semantics. Generated offline from the source video (see §4.5).

### 2.2 Multimodal encoder (the shared body front-end)
Trainable in every variant. Produces a fixed embedding `e_fused`.

| Branch | Input | Layers | Output |
|--------|-------|--------|--------|
| Network (temporal) | `[B, W, 7]` | `Conv1d(7→128,k3,pad1)→SiLU→Dropout→Conv1d(128→128,k3,pad1)→SiLU → AdaptiveAvgPool1d(1) → LayerNorm` | `net_emb [B,128]` |
| Semantic | `[B, 5]` | `Linear(5→64)→SiLU→LayerNorm→Dropout→Linear(64→64)→SiLU→LayerNorm` | `sem_emb [B,64]` |
| Ladder | `[B, 6, 2]` | per-rung `Linear(2→32)→SiLU`, then **attention-pool over 6 rungs** (upgrade from mean-pool) | `lad_emb [B,32]` |
| Fusion | concat `[B,224]` | `Linear(224 → d_fuse)` | `e_fused [B, d_fuse]` |

`net_emb` already exists in `model.py` (1D-CNN, length-agnostic). The ladder
attention-pool is a small upgrade from the current mean-pool. `d_fuse` depends on the
variant (see §4).

### 2.3 Policy head (the shared body back-end)
Trainable in every variant.
```
Linear(d_in → 256) → SiLU → Dropout(0.1) → Linear(256 → 128) → SiLU → Linear(128 → 6)  → bitrate_logits
```
Action = `argmax(bitrate_logits)` → rung → bps via `NS3_RATE_LADDER_BPS`. (A `Linear(128→1)`
**value head** is added only for the RL phase, not for these SFT benchmarks.)

### 2.4 Normalization, label, loss
- Normalization applied **at load** via the two pinned scalers (already implemented).
- Label = `teacher.chosen_bitrate_class`. Loss = **class-weighted cross-entropy** (rung 5 ≈ 51%, rung 1 ≈ 2%); weights = inverse-frequency, computed on the **train split only**.

### 2.5 The embedding-injection mechanism (V1–V3)
The transformer consumes `d_model`-dim vectors, not text. Variants with an LLM project the
encoder output into the **LLM's embedding space** and inject it as **token-like embeddings**
(soft tokens), bypassing the tokenizer:
```
e_fused [B, d_fuse] → Linear(d_fuse → k · d_model) → reshape → soft_tokens [B, k, d_model]
```
- `d_model` = the LLM hidden size (e.g., **1536** for Qwen2.5-1.5B, 4096 for 7B).
- `k` = number of soft tokens (default **k=4**).
- Injected via HF `inputs_embeds=` (prefix). The pooled output is the **last hidden state**
  over the injected sequence (or attention-pool over the `k` positions) → fed to the head.

---

## 3. Backbone, training, and compute (shared)

- **Backbone (V1–V3):** **Qwen2.5-1.5B-Instruct** (hidden 1536). ≥1B is required — NetLLM's
  OPT-0.35B lost to every ABR baseline; 1.3–1.5B is the tractable sweet spot for a 4-way
  sweep. Scale-up runs (3B/7B) optional after the 1.5B sweep.
- **Adaptation:** V2/V3 = base **frozen**, **LoRA** (r=16, α=32, dropout 0.05) on attention
  proj + MLP; encoder + projection + head fully trained. V1 = **all weights trainable**
  (random init can't be frozen — see §4.2 caveat).
- **Precision:** QLoRA 4-bit base for V2/V3 training; bf16 elsewhere. `torch.inference_mode`
  at eval.
- **Optimizer/schedule:** AdamW, lr 1e-3 (encoder/head) / 1e-4 (LoRA), cosine, ~5–10 epochs,
  early-stop on val macro-F1.
- **Splits/seeds:** group-aware splits (§5.2); **3 seeds** per variant; report mean ± std.
- **Held constant across variants:** dataset, splits, scalers, window `W`, oracle labels,
  class weights, training budget, eval harness.

---

## 4. Complete architecture per variant

### 4.0 V0 — Tier-0 (no LLM)
**Purpose:** strong small-model control / efficiency floor. This is essentially today's
`LlmAbrModel` with the placeholder projector.
```
[network W×7] ─┐
[semantic 5] ──┤→ MultimodalEncoder → e_fused [B,256] → backbone_projector
[ladder 6×2] ──┘                                         (Linear256→SiLU→LayerNorm)
                                                              → policy head → 6 logits
```
- `d_fuse = 256`. **No LLM.** Everything trainable.
- **Params:** ~0.3–1 M trainable. **Latency:** <1 ms. **No GPU/transformers dependency.**

### 4.1 V1 — Random-initialized LLM (full fine-tune)
**Purpose:** isolate *pretrained knowledge* — same architecture as V2 but the LLM starts
from random weights.
```
encoder → e_fused → Linear(→ k·1536) → soft_tokens [B,4,1536]
        → LLM(Qwen2.5-1.5B config, RANDOM init, ALL weights trainable)
        → pool last hidden [B,1536] → policy head → 6 logits
```
- **Trainable:** encoder + projection + **entire LLM** + head. **Frozen:** nothing.
- **Params trainable:** ~1.5 B. **Caveat (inherent):** you cannot freeze random weights, so
  V1 is *full* fine-tune while V2 is frozen+LoRA — this bundles "random-vs-pretrained" with
  "fullFT-vs-LoRA," exactly as NetLLM did. Interpret V1 as **"no pretrained knowledge"**, and
  **expect it to over/underfit on ~12k rows** — that underperformance *is* the result.
- **Latency/memory:** like V2 at inference; training is the heaviest (full 1.5B grads).

### 4.2 V2 — Pretrained LLM (frozen + LoRA), numeric soft-tokens
**Purpose:** the NetLLM-equivalent Tier-1. Numeric modalities only (network + semantic +
ladder), no language.
```
encoder → e_fused → Linear(→ k·1536) → soft_tokens [B,4,1536]
        → LLM(Qwen2.5-1.5B, PRETRAINED, FROZEN + LoRA)
        → pool last hidden [B,1536] → policy head → 6 logits
```
- **Trainable:** encoder + projection + **LoRA adapters** + head. **Frozen:** base LLM.
- **Params trainable:** a few M (encoder/head) + LoRA (~0.5–2% of 1.5B ≈ 8–30 M).
- **Checkpoint stores:** encoder + projection + LoRA + head + `window` + scaler hashes +
  `model_id` (frozen base loaded separately from HF — not saved).

### 4.3 V3 — Text + soft embeddings (ours)
**Purpose:** use the LLM as a *reasoner over content-as-language* — the channel NetLLM never
uses. Identical to V2 plus a content-derived text sequence.
```
source video ── offline pipeline (§4.5) ─→ "Content: live sports broadcast — fast camera pan
                                   over a crowd, motion-masked. Network: handover imminent,
                                   beam congested."
                                   → tokenizer → embed lookup → text_emb [B, T, 1536]
encoder → e_fused → Linear(→ k·1536) → soft_tokens [B,4,1536]
sequence = concat(soft_tokens, text_emb)  [B, 4+T, 1536]
        → LLM(Qwen2.5-1.5B, PRETRAINED, FROZEN + LoRA)
        → pool last hidden [B,1536] → policy head → 6 logits
```
- Same trainable/frozen split as V2; only the **input sequence** gains real text tokens.
  **V3 requires its own training run** — a *fresh* LoRA fit with the text present. You cannot
  reuse V2's adapters: they were fit on a soft-tokens-only sequence and never learned to attend
  to text. The LLM's *language understanding* is free (pretrained); learning to *use* the text
  for the bitrate decision is not. Same recipe/budget/seeds as V2 — only the text channel differs.

- **The redundancy rule (why the old template was dropped):** templated text derived from the
  same 5 semantic scalars (`"motion: high; scene-cut: yes; ..."`) carries **identical information**
  to `semantic_norm` — it is the numbers re-spelled, and gives the LLM nothing the encoder lacks.
  V3's text must pass **both** tests:
  1. **Non-redundant** — information *not in* `network_norm` / `semantic_norm` / `ladder_context`.
  2. **Open-vocabulary / world-knowledge-dependent** — the text→action mapping benefits from
     pretrained priors and the vocabulary is too large/compositional to enumerate as features.
     (A single quantity, e.g. a handover countdown, fails this — feed it as a scalar instead.)

- **What the text contains (passes both tests):**
  - **Content identity / genre** — `"live sports broadcast"`, `"talking-head news"`, `"2D animation"`.
    Not derivable from the 5 scalars; the LLM knows a fast pan over a crowd *masks* artifacts
    (drop the rung) while a static face is *unforgiving of blur* (hold quality) — even at identical
    motion/saliency values.
  - **Scene / object description** — `"soccer pitch, crowd, fast pan"` vs `"news anchor, studio desk"`.
    Open-vocabulary, compositional.
  - *(optional, the NTN edge)* **compositional regime phrase** — `"rain-fade during a handover with a
    congested beam"` — a conjunction you'd never enumerate as features. Push the *scalar* parts
    (handover countdown, elevation, UE count) into the numeric branch, not the text.

- Keep `T` small (~20–40 tokens). Both `soft_tokens` and `text_emb` are `d_model=1536` vectors in the
  **same space**; the text ones come from the normal embedding lookup, the numeric ones from the encoder.
- **Provable-win shape:** because the text adds information the encoder cannot recover, V3 > V2 is
  predicted to be **strongest on the unseen-content split** (§5.2) — a genre held out of training where
  the LLM's priors transfer but the encoder has no anchor. That is the falsifiable claim, not "numbers
  spelled as words help."

### 4.4 Optional variants
- **V2-no-semantics:** V2 with the semantic branch removed (network + ladder only) → proves
  the **semantic modality** helps independent of the LLM. One extra run.
- **Distilled student:** distill the best variant (likely V3) into V0's small encoder+head via
  KD (soft-label CE on the teacher logits) → demonstrate **LLM-level QoE at MLP-level
  latency**, closing the deployment gap NetLLM leaves open. The capstone for the ns-3 loop.

### 4.5 Generating the V3 language features (offline pipeline)
The V3 text is produced **once, offline, over the source videos** and cached — **nothing runs in
the per-decision loop** (this also keeps V3 compatible with the NTN latency budget: the text is a
precomputed dataset column, not a per-segment forward pass). It is joined into the dataset by
`video_name` + segment index, exactly like the scalers, and its hash is pinned in the manifest.

**Algorithm families (by output type):**

| Family | Output | Representative models | Granularity | Use for |
|--------|--------|----------------------|-------------|---------|
| VLM caption/QA (promptable, open-vocab) | free-text description | **Qwen2.5-VL** (3B/7B), InternVL3, **Florence-2** (light, MIT) | shot keyframe | scene description (rich channel) |
| Open-vocab tagger | tag list | **RAM++** (6.4k+ tags; beats CLIP/BLIP), Tag2Text | keyframe | compact object/scene tags |
| Zero-shot classifier | label from a fixed candidate list | CLIP / OpenCLIP / **SigLIP** | keyframe | **genre** (closed set, deterministic) |
| Action/video classifier | action label | SlowFast, Video-Swin (Kinetics) | clip | motion/activity type |
| Audio / ASR | transcript, audio class | Whisper, PANNs/AST | clip | live/commentary cue (optional) |
| Shot segmentation | shot boundaries + keyframes | **PySceneDetect** (classical) | video | picks keyframes for the above |

**Two design axes:** (1) *open vs closed vocab* — genre is a small stable set → **SigLIP zero-shot
against a fixed candidate list** (reproducible); scene description is open-ended → a **VLM**.
(2) *granularity* — genre is ~static → **once per video**; scene/object content changes → **per shot**
(PySceneDetect keyframe), each segment inherits its containing shot's description.

**Recommended pipeline (cheap, reproducible):**
1. **PySceneDetect** → shot boundaries + one center keyframe per shot.
2. **SigLIP zero-shot** vs a fixed genre list (`["live sports","news studio","animation","nature doc",…]`)
   → per-video `genre`.
3. **Florence-2** (`<DETAILED_CAPTION>` + dense-region caption) **or Qwen2.5-VL** on each shot keyframe
   → per-shot scene text. Qwen2.5-VL shares the text-backbone family; Florence-2 is lighter/MIT for bulk runs.
4. *(optional)* **RAM++** tags as a structured fallback; **Whisper** on the audio for a live/commentary cue.
5. Cache to `video_semantics.json` keyed by `(video_name, shot_id)`; join at dataset-build time; freeze
   its SHA in `manifest.json` alongside the scalers (§6).

**Reproducibility:** the V3 text is a frozen artifact — re-running the pipeline must reproduce it, so pin
the extractor `model_id`s, prompts, candidate genre list, and PySceneDetect threshold in the manifest.

---

## 5. Evaluation protocol

The variants are only as meaningful as the measurement. Fix all of this before training.

### 5.1 Metrics
**A. Imitation quality (does it match the oracle?)**
- **macro-F1** and **per-class recall** (NOT top-1 — rung 5 ≈ 51% makes accuracy a trap).
- 6×6 **confusion matrix**.

**B. QoE (the metric that matters)** — replay each policy's chosen rungs through the player
model (offline via the oracle's buffer model first; live ns-3 for the finalists). Report the
components and a combined score consistent with the oracle reward:
- **avg quality** (VMAF / bitrate) ↑
- **rebuffering time** ↓
- **switch magnitude / frequency** (smoothness) ↓
- **combined QoE** `= w_vmaf·Σ(sem·VMAF) − w_stall·Σstall − w_switch·Σ|ΔVMAF|` (same weights as
  `oracle_reward.json`), reported as **% of the DP-oracle ceiling**.

**C. Systems (deployment story)**
- decision **latency p50/p95**, model **size / GPU memory**, params trainable.

### 5.2 Splits — report in-distribution AND out-of-distribution separately
The LLM's edge (V2/V3) shows up OOD, per NetLLM. Three test sets:
1. **In-distribution:** held-out runs of seen videos (group-aware by `run_id`).
2. **Unseen content:** held-out **videos** (group-aware by `video_name`) — tests content generalization (where semantics/V3 should help most).
3. **Unseen network/NTN regime:** test traces from network conditions not in training (different elevation/handover/throughput regimes) — tests the pretrained-knowledge generalization NetLLM highlights.

No `(run_id, node_id)` session may cross a split (also required by the windowing).

### 5.3 Anchors / baselines
- **FDASH** (your heuristic) — lower reference.
- **DP oracle** — upper bound / ceiling (QoE % is measured against this).
- (optional) a classic ABR (BOLA/MPC-style) if available, à la NetLLM's BBA/MPC/GENET.

### 5.4 Result matrix
For every (variant × test-split): macro-F1, per-class recall, QoE components + combined
(% of oracle), latency p50/p95. Plus the **claim table** from §1 filled with numbers and
seed variance. Use paired comparisons across the same seeds; report mean ± std and a
significance check (e.g., paired test over seeds/videos).

### 5.5 Success criteria (what each claim needs)
- **V2 > V0** on QoE in **unseen content / network** splits (in-dist may tie) → LLM earns its keep.
- **V2 ≫ V1** → pretrained knowledge matters (sanity check vs NetLLM).
- **V3 > V2** on QoE, strongest on **unseen content** → the language/semantic channel is the contribution.
- **V2 > V2-noSem** → the semantic modality is justified.
- **Distilled student** retains ≥ most of V3's QoE at V0's latency → deployable.

---

## 6. Reproducibility & artifacts
- Pin per run: dataset `schema_version` + `manifest.json` (scaler SHAs), `window`, backbone
  `model_id`, LoRA config, seed, class-weight scheme, split definition.
- **(V3)** also pin the language-feature artifact: `video_semantics.json` SHA + the extractor
  `model_id`s, prompts, genre candidate list, and PySceneDetect threshold used to build it (§4.5).
- Each checkpoint records `window`, scaler hashes, `model_id` (so eval/inference reproduce
  training exactly).
- Log: training curves (loss, macro-F1), per-variant result rows, confusion matrices, QoE
  breakdowns, latency profiles. One results table per test-split.

## 7. Recommended order of execution
1. **V0** (no GPU needed beyond torch) — establishes the floor + validates the eval harness.
2. **V2** (the workhorse Tier-1) — the main result.
3. **V1** (sanity ablation vs NetLLM) and **V3** (the contribution).
4. **V2-no-semantics** and the **distilled student** if budget allows.

Run all on the regenerated raw dataset with refit scalers (see `cluster-rerun-handoff.md`);
the windowed loader, scalers, head, and FDASH fallback are already implemented — the
remaining build is the LLM injection (§2.5) for V1–V3.