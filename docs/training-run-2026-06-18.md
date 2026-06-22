# Training Run — 2026-06-18

## Environment

- Cluster: Narval (Compute Canada), account def-ekram
- GPU nodes: L40S (48 GB VRAM)
- Dataset: 25 videos × 5 network runs = 125 shards, ~24,228 decision rows
- Split: group-aware (by run_id), 21,698 train / 2,530 val
- Epochs: 10, early-stop patience 3, batch 32, seed 42

---

## Results — Single Seed (seed 42, full data, in-distribution)

| Variant | val macro-F1 | Wall time | Notes |
|---------|-------------|-----------|-------|
| V1 — random LLM, full fine-tune | 0.048 | 8m47s | Collapses to single rung — the **negative control** |
| V0 — no-LLM encoder (1D-CNN) | 0.118 | 5m (CPU) | Class-balanced; beats majority-class (0.508 accuracy) |
| V2 — pretrained LLM, frozen + LoRA | **0.256** | 11m20s | Best; recall spreads across all 6 rungs |

V2 ≈ 2× V0 — the predicted "pretrained LLM earns its keep" direction. V1's collapse replicates NetLLM Fig 13's prediction that a randomly initialised 1.5B model cannot learn from ~12k rows.

### V2 Per-Rung Recall (epoch 8, best checkpoint)

| Rung 0 | Rung 1 | Rung 2 | Rung 3 | Rung 4 | Rung 5 |
|--------|--------|--------|--------|--------|--------|
| 0.32   | —      | 0.04   | 0.22   | 0.14   | 0.82   |

Rung 1 (second-lowest) rarely appears in training (~1.9% of labels); the class-weighted loss is helping but rung 1 recall is still near zero.

---

## Checkpoints

| File | Size | Contents |
|------|------|----------|
| `checkpoints/abr_v0_seed42.pt` | ~few MB | V0 encoder weights |
| `checkpoints/abr_v2_seed42.pt` | 8.5 MB | V2 head + LoRA delta |
| `checkpoints/abr_v2_seed42.pt.lora/adapter_model.safetensors` | — | LoRA adapter (mergeable) |
| `checkpoints/abr_v1_seed42.pt` | 3.0 GB | V1 full 1.5B weights |

---

## Jobs Submitted (as of 2026-06-18)

| Job ID | Variant | Status |
|--------|---------|--------|
| 7086135 | V2 seed 42 | Completed (best, 11m) |
| 7086136 | V0 seed 42 | Completed (5m, CPU) |
| 7086139 | V1 seed 42 | Completed (8m47s, early-stop epoch 4) |
| sweep.sh × V0/V1/V2 seeds 41,43 | All three variants | Queued (6 jobs) |
| eval_gpu.sbatch V2 seed 42 | GPU QoE eval | Queued |
| eval_gpu.sbatch V1 seed 42 | GPU QoE eval | Queued |

---

## Infrastructure Fixes Applied This Session

- `PYTHONUNBUFFERED=1` added to `train_llm.sbatch` and `train_v0.sbatch` — live log flushing for future runs
- `slurm/eval_gpu.sbatch` created — auto-detects variant from checkpoint; `--include-video` flag for unseen-content split
- Group-aware train/val split, class-weighted CrossEntropyLoss, macro-F1 + per-class recall all live in `train.py`

---

## Next Milestones

1. **3-seed mean±std table** — wait for seeds 41/43 to finish, then run `evaluate.py` for all six checkpoints
2. **QoE-vs-oracle %** — GPU eval jobs for V1/V2; V0 CPU eval still running
3. **Unseen-content split** — re-evaluate all variants on held-out videos (not in training)
4. **Dataset expansion** — see `docs/dataset-expansion-plan.md`
