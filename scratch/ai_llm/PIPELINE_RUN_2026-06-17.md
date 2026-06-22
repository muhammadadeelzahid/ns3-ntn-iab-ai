# Pipeline Run — 2026-06-17

## Environment

- Python: 3.11.8 (via `module load python/3.11.8`)
- Venv: `.venv` at project root
- PyTorch: 2.12.0+cpu / NumPy fixed
- Root cause of prior silent failure: `module load … | head` runs in a subshell, `PATH` change is discarded

## Cleanup

- 258 stale SLURM `.err`/`.out` logs removed (~220 MB)
- `__pycache__`, stale `train_jsonl/`, `checkpoints/` cleared
- Left untouched: `Quic_artifacts/`, `src/`, `docs/`, `../abr` metadata

---

## Step 1 — Dataset Build

| Metric | Value |
|---|---|
| Rows | 12,114 |
| Files processed | 125 |
| Missing | 0 |
| Failures | 0 |

Output: `train_jsonl/all_runs.jsonl` + `manifest.json`

---

## Step 2 — Scaler Fit

Both scalers refit from the full corpus.

**Key correction — semantic scaler `motion_score` std:**

| Before (single-video) | After (corpus-wide) |
|---|---|
| 1.64 | 5.76 |

The prior value was fit on a single video's motion scores, producing a miscalibrated Z-score. The corpus-wide std is now correct.

---

## Step 3 — Tier-0 Model Training

- Duration: ~1m49s (CPU)
- Checkpoint confirmed: `network_dim=9, window=8, use_missing_masks=True`
- Mask wiring is live in training

### Result

| Metric | Value |
|---|---|
| Val accuracy | 0.508 |
| Majority-class baseline | 0.511 (rung 5) |

---

## ⚠️ Accuracy Not Yet Meaningful — Majority-Class Trap

Val accuracy of 0.508 equals the majority-class baseline. The model is predicting rung 5 for every sample.

**Label distribution:**

| Rung | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| % | 11.0 | 1.9 | 4.9 | 10.1 | 21.0 | **51.1** |

The plumbing works. Two known fixes are needed before metrics are meaningful:

### Fix 1 — Class-Weighted Loss

Apply inverse-frequency weights to `CrossEntropyLoss`. Rung 1 is 1.9% of data, rung 5 is 51% — without weighting the loss surface overwhelmingly rewards majority-class prediction.

```python
# train.py — replace loss instantiation
weights = compute_class_weights(train_labels)  # inverse freq
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### Fix 2 — Group-Aware Train/Val Split

Current `split_dataset` shuffles row indices randomly. Same-session segments (same UE, same run) leak across train/val, inflating val performance artificially once the model learns anything useful.

Fix: split by run/session ID so all rows from a given session are in either train or val, never both.

### Fix 3 — Better Metrics

Report macro-F1 + per-class recall instead of (or alongside) top-1 accuracy. Top-1 is dominated by majority class and will not surface per-rung learning.

---

## Next Steps

1. Implement class weighting + group-aware split in `train.py`
2. Add macro-F1 / per-class recall to validation loop
3. Re-run Tier-0 training → get trustworthy baseline number
4. Run mask on/off ablation (swap `use_missing_masks=True/False`, compare macro-F1) to confirm mask adds signal
