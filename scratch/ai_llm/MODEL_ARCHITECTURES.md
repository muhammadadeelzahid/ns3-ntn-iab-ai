# ABR Policy Model Architectures: V0 and V2

## Overview

All variants share the same **MultimodalEncoder** front-end. They differ in how the
fused embedding is processed before the policy head.

| Variant | Backbone | Pretrained knowledge | Trainable weights |
|---|---|---|---|
| V0 | Placeholder projector (Linear+SiLU+LN) | No | All (small network) |
| V1 | Qwen2.5-1.5B, random-init, full fine-tune | No | All transformer weights |
| V2 | Qwen2.5-1.5B, frozen + LoRA | Yes | Encoder + SoftProj + LoRA + Head |
| V3 | Same as V2 + language text prefix | Yes | Same as V2 |

---

## Input Features

All variants consume the same three input streams.

### Network features — `network_norm` — shape `[B, 7]`

| # | Feature | Notes |
|---|---|---|
| 0 | `throughput_estimate_bps` | Harmonic-mean estimate |
| 1 | `buffer_level_s` | Playback buffer depth |
| 2 | `buffer_differential_s` | Buffer change since last segment |
| 3 | `segment_fetch_time_s` | Wall-clock download time |
| 4 | `last_segment_bitrate_bps` | Actual encoded bitrate (VBR; not the nominal rung) |
| 5 | `smoothed_rtt_ms` | Imputed to scaler mean if missing (sentinel = −1) |
| 6 | `loss_rate_recent` | Imputed to scaler mean if missing (sentinel = −1) |

All 7 features are z-scored. `smoothed_rtt_ms` and `loss_rate_recent` also receive a
log transform before z-scoring (matching the scaler fit), and two 0/1 missing-indicator
channels are optionally appended (making `network_dim` 9 in that mode).

### Semantic features — `semantic_norm` — shape `[B, 5]`

| # | Feature |
|---|---|
| 0 | `motion_score` |
| 1 | `saliency_score` |
| 2 | `scene_cut_count` |
| 3 | `object_score` |
| 4 | `quality_sensitivity` |

Z-scored via `config/feature_scaler_semantic.json`.

### Ladder context — `ladder_context` — shape `[B, 6, 2]`

One row per bitrate rung (250k → 12000k), two columns: `[size_bytes, vmaf]`.
`actual_bitrate_kbps` is dropped (perfectly collinear with `size_bytes` at a fixed
2 s segment duration).

---

## Shared: MultimodalEncoder

Defined in `model.py:MultimodalEncoder`. Identical across all variants.

```
network_norm [B, 7]
  Conv1d(7 → 128, kernel=3, pad=1) + SiLU + Dropout(0.1)
  Conv1d(128 → 128, kernel=3, pad=1) + SiLU
  AdaptiveAvgPool1d(1)  →  [B, 128]
  LayerNorm(128)                                    ► net_emb  [B, 128]

semantic_norm [B, 5]
  Linear(5 → 64) + SiLU + LayerNorm(64) + Dropout(0.1)
  Linear(64 → 64) + SiLU + LayerNorm(64)           ► sem_emb  [B, 64]

ladder_ctx [B, 6, 2]
  Linear(2 → 32) + SiLU  (applied per rung)
  mean over 6 rungs                                 ► ldr_emb  [B, 32]

concat([net_emb, sem_emb, ldr_emb])  =  [B, 224]
  Linear(224 → 256)                                 ► e_fused  [B, 256]
```

| Sub-module | Layers | Output dim |
|---|---|---|
| Network CNN | 2× Conv1d (7→128→128) + pool | 128 |
| Semantic MLP | 2× Linear (5→64→64) | 64 |
| Ladder proj | Linear (2→32) + mean-pool | 32 |
| Fusion | Linear (224→256) | 256 |

---

## V0 — Baseline (No LLM)

**Role:** Control condition. No transformer; purely the encoder + a small placeholder
projection + policy head.

### Block Diagram

```
network_norm  [B, 7]  ─┐
semantic_norm [B, 5]  ──┤  MultimodalEncoder  ──► e_fused [B, 256]
ladder_ctx    [B, 6,2]─┘

e_fused [B, 256]
  Linear(256 → 256) + SiLU + LayerNorm(256)        ► state_hidden [B, 256]

state_hidden [B, 256]
  Linear(256 → 256) + SiLU + Dropout(0.1)
  Linear(256 → 128) + SiLU
  Linear(128 → 6)                                   ► logits [B, 6]
```

### Layer-by-Layer Details

| Block | Module | In → Out | Notes |
|---|---|---|---|
| Encoder | MultimodalEncoder | (7, 5, 6×2) → 256 | Shared with all variants |
| Backbone projector | Linear + SiLU + LayerNorm | 256 → 256 | V0-only placeholder; keeps checkpoint keys stable |
| Policy head — trunk | Linear + SiLU + Dropout(0.1) | 256 → 256 | |
| Policy head — trunk | Linear + SiLU | 256 → 128 | |
| Policy head — output | Linear | 128 → **6** | 6-rung bitrate ladder |

### V0 Summary

- **LLM:** None
- **Total unique blocks:** Encoder → Projector → Head
- **Output classes:** 6 (bitrate rungs: 250k, 750k, 1.5M, 3M, 6M, 12M bps)
- **Training:** All weights trainable; CPU-friendly inference

---

## V2 — Pretrained LLM with LoRA (Soft-Token Injection)

**Role:** Main LLM-augmented variant. The encoder's 256-dim fused vector is projected
into 4 "soft tokens" (learned numeric prefix embeddings in the LLM's hidden space).
These are fed as `inputs_embeds` to a frozen pretrained Qwen2.5-1.5B transformer.
Only the LoRA adapters and the non-LLM modules are trained.

### Block Diagram

```
network_norm  [B, 7]  ─┐
semantic_norm [B, 5]  ──┤  MultimodalEncoder  ──► e_fused [B, 256]
ladder_ctx    [B, 6,2]─┘

e_fused [B, 256]
  Linear(256 → 4 × 1536)
  reshape                                           ► soft_tokens [B, 4, 1536]

soft_tokens [B, 4, 1536]   (bypasses tokenizer — no text/input_ids)
  attention_mask = ones [B, 4]
  ┌─────────────────────────────────────────────┐
  │  Qwen2.5-1.5B-Instruct (frozen)             │
  │  28 × TransformerBlock:                     │
  │    LayerNorm → GQA(12 heads / 2 KV heads)   │
  │    + LoRA on q/k/v/o/gate/up/down _proj     │
  │    → SwiGLU FFN (hidden=8960)               │
  └─────────────────────────────────────────────┘
  hidden_states[-1]  [B, 4, 1536]
  mean over 4 token positions                       ► pooled [B, 1536]

pooled [B, 1536]
  Linear(1536 → 256) + SiLU + Dropout(0.1)
  Linear(256 → 128) + SiLU
  Linear(128 → 6)                                   ► logits [B, 6]
```

### Layer-by-Layer Details

| Block | Module | In → Out | Notes |
|---|---|---|---|
| Encoder | MultimodalEncoder | (7, 5, 6×2) → 256 | Shared with all variants |
| Soft token projector | Linear | 256 → 4×1536, reshaped to [B, 4, 1536] | Projects numeric state into LLM embedding space |
| LLM backbone | Qwen2.5-1.5B (28-layer transformer) | [B, 4, 1536] → [B, 4, 1536] | Frozen; LoRA adapters only |
| Pooling | Mean over token dim | [B, 4, 1536] → [B, 1536] | Mean of last-layer hidden states |
| Policy head — trunk | Linear + SiLU + Dropout(0.1) | 1536 → 256 | |
| Policy head — trunk | Linear + SiLU | 256 → 128 | |
| Policy head — output | Linear | 128 → **6** | 6-rung bitrate ladder |

### LLM Backbone: Qwen2.5-1.5B-Instruct

| Property | Value |
|---|---|
| Model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Parameters | ~1.54B |
| Transformer layers | **28** |
| Hidden size (H) | **1536** |
| Attention heads | **12** |
| KV heads (GQA) | **2** |
| FFN intermediate size | **8960** |
| Activation | SwiGLU |
| Quantization | QLoRA 4-bit NF4, double-quant, bfloat16 compute |
| Base weights | **Frozen** |

### LoRA Adapter Configuration

| Property | Value |
|---|---|
| PEFT type | LoRA |
| Rank (r) | **16** |
| Alpha (α) | **32** (effective scale α/r = 2.0) |
| Dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Bias | none |
| Trainable per layer | 2 × r × (in_dim + out_dim) per target module |

### V2 Summary

- **LLM:** Qwen2.5-1.5B-Instruct, frozen, 4-bit QLoRA
- **Soft tokens injected:** 4 (numeric prefix; no text)
- **Output classes:** 6 (same ladder as V0)
- **Training:** Encoder + SoftTokenProjection + LoRA adapters + PolicyHead
- **Inference:** GPU required (4-bit quantized transformer)

---

## V0 vs V2: Side-by-Side

| | V0 | V2 |
|---|---|---|
| Encoder | MultimodalEncoder (shared) | MultimodalEncoder (shared) |
| Fused dim | 256 | 256 |
| Backbone | Linear+SiLU+LN (256→256) | Qwen2.5-1.5B, 28 layers |
| State representation going into head | 256-dim | 1536-dim (mean-pooled LLM output) |
| LLM pretrained knowledge | No | Yes (frozen weights) |
| Soft tokens | — | 4 numeric prefix tokens |
| Policy head input dim | 256 | 1536 |
| Trainable params | All (encoder + projector + head) | Encoder + SoftProj + LoRA + Head |
| Hardware requirement | CPU-capable | GPU (QLoRA 4-bit) |
| Output | 6-class bitrate logit | 6-class bitrate logit |
