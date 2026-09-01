# LLM-ABR v2 architecture (compact figure) - description

Companion to `model_v2_compact.drawio`. The diagram shows only stage labels; the input/output
shapes and sub-layers for each box are listed here. Depicts variant **v2** (the pretrained,
frozen backbone + LoRA workhorse), faithful to `scratch/ai_llm/model.py`.

**SLM = Small Language Model** (Qwen2.5-1.5B); it is small enough that "SLM" is the accurate term.

Data flows left-to-right along row 1, wraps down on the right, then right-to-left along row 2:

```
Network State
Content Features --> Multi-Modal Encoder --> Feature Fusion --> Soft-Token Projection --> Frozen SLM + LoRA
Bitrate Ladder                                    |                                              |
                                                  | residual (dashed)                            v
Bitrate Logits <-- ABR Policy Head <-- Residual Fusion <-- Token Mean-Pool
```

## Color legend
| Color | Meaning |
|---|---|
| Gray | Inputs |
| Green | Trainable modules (encoder, fusion, soft-token projection, residual fusion, policy head) |
| Blue | Frozen backbone (SLM weights frozen; only LoRA adapters train) |
| Yellow | Output |

## Per-box detail
| Box | Sub-layers | Input | Output |
|---|---|---|---|
| Network State | (raw features) | `[B, W, 7]` | `[B, W, 7]` |
| Content Features | (raw features) | `[B, 5]` | `[B, 5]` |
| Bitrate Ladder | (raw features) | `[B, 6, 2]` | `[B, 6, 2]` |
| Multi-Modal Encoder | 1D-CNN (network) + MLP (content) + Linear proj (ladder) | 3 inputs above | branch embeds `[B,128]`,`[B,64]`,`[B,32]` |
| Feature Fusion | Concat + Linear(224 -> 256) | concat `[B, 224]` | `e_fused [B, 256]` |
| Soft-Token Projection | Linear(256 -> k*H) + reshape + per-token LayerNorm | `[B, 256]` | `[B, k=4, H=1536]` |
| Frozen SLM + LoRA | Qwen2.5-1.5B, base frozen + LoRA (QLoRA 4-bit), via `inputs_embeds` | `[B, 4, 1536]` | last hidden `[B, 4, 1536]` |
| Token Mean-Pool | mean over the k soft tokens | `[B, 4, 1536]` | `pooled [B, 1536]` |
| Residual Fusion | concat pooled with `e_fused` (encoder_residual skip) | `[B,1536]` + `[B,256]` | `[B, 1792]` |
| ABR Policy Head | Linear(1792 -> 256) + SiLU + Linear(256 -> 128) + SiLU + Linear(128 -> 6) | `[B, 1792]` | `logits [B, 6]` |
| Bitrate Logits | argmax over the 6-rung ladder | `[B, 6]` | chosen rung |

**Residual skip (dashed):** `e_fused` from **Feature Fusion** is routed directly into **Residual Fusion**,
giving the policy head an un-throttled path to the encoder signal (prevents collapse to the
majority-rung prior). Disable via `encoder_residual=False` to ablate.
