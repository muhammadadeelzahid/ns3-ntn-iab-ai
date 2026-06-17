# Next steps — AI-DASH feature changes & cluster re-run

*(Written as an email/handoff note. Plain-text version below the rule is copy-paste ready.)*

---

**Subject:** AI-DASH training data — schema changes landed, here's how to regenerate on the cluster

Hi all,

I've made three changes to the ABR training contract in the repo. None of them touch the
ns-3 simulations or the DP oracle, so **we do not need to re-run the sims, throughput
extraction, or oracle labeling** — only the feature extraction + dataset build. That keeps
the turnaround short.

**What changed**

1. **Network feature index 4** is now `last_segment_bitrate_bps` (the *actual* VBR-encoded
   bitrate of the last downloaded segment) instead of `current_bitrate_bps`. The old value
   was the *nominal* ladder rate — a 1:1 restatement of the rung index, so it carried no
   information. The new one is the real network-load signal (88.5% of rows differ from
   nominal, ~26% on average).
2. **`ladder_context` is now 6×2** `[size_bytes, vmaf]` (was 6×3). We dropped
   `actual_bitrate_kbps` because it equals `size_bytes × 8 / 1000 / 2s` exactly — perfectly
   collinear with `size_bytes`, verified across all 72,684 ladder entries.
3. **The network scaler is now fit and actually applied.** It used to be an identity
   (mean 0 / std 1), *and* `dataset_builder.py` was passing `None` so it wouldn't have been
   applied anyway. Both are fixed: `feature_scaler_network.json` is fit (throughput and
   last-segment bitrate are `log1p`-scaled), and `dataset_builder.py` now takes
   `--network-scaler` and bakes the normalization into `network_norm`.

Net effect on the model contract: **7 network + 5 semantic + 6×2 ladder**, label unchanged
(`oracle_bitrate_idx`, 6 classes). Background: `docs/reduced_dp_oracle_eq.md` and
`docs/oracle-rung-selection.md`.

**How to regenerate on the HPC cluster**

No sims, no oracle re-run. `Quic_artifacts/<video>/run_*/` already has `sim.log`,
`throughput_trace.jsonl`, and `oracle_labels_state.jsonl` from the last batch — we reuse
them. Everything below is CPU-only post-processing; run it in an interactive job
(`salloc`) or a small `sbatch` wrapper, not on the login node.

> **Builder reworked (Option 2 — normalize at load).** The dataset is now **raw**:
> `dataset_builder` no longer bakes `network_norm`/`semantic_norm`, so there is **no
> scaler dependency at build time and no "build twice."** You build the raw shards once,
> fit the scalers from them, and `train.py` applies the scalers at load (and pins them in
> the checkpoint for inference). Changing a scaler later = re-fit + retrain, never a
> rebuild.

```bash
# 0. Get the code on the cluster and normalize line endings for any .slurm you touch
cd /path/to/ns3-ntn-iab
git pull                                   # or rsync the scratch/ai_llm + docs changes
sed -i 's/\r$//' *.slurm                   # only if sbatch complains about DOS line breaks

export METADATA_DIR=/project/6030214/adeel/abr/output/metadata

# 1. Build the RAW training JSONL (re-extracts features from existing sim.log, joins the
#    existing oracle labels). No scaler needed at build; writes manifest.json. ONE pass.
python3 scratch/ai_llm/dataset_builder.py \
  --artifacts-root Quic_artifacts --algo-short Fdash \
  --metadata-dir "$METADATA_DIR" --output-dir train_jsonl \
  --teacher-source oracle --strict \
  --merge-output train_jsonl/all_runs.jsonl

# 2. Fit BOTH scalers from the raw shards (network is log-scaled; semantic is plain
#    z-score). Note: refitting the semantic scaler here also corrects it — the committed
#    one looks single-video (motion std ~1.6 vs ~5.8 corpus-wide).
python3 scratch/ai_llm/scripts/fit_network_scaler.py  --data-dir train_jsonl --glob '*_run_*.jsonl'
python3 scratch/ai_llm/scripts/fit_semantic_scaler.py --data-dir train_jsonl --glob '*_run_*.jsonl'

# 3. Train — applies the scalers at load and pins them in the checkpoint. No rebuild.
#    (--network-scaler/--semantic-scaler default to the config/ files just fitted.)
python3 scratch/ai_llm/train.py \
  --data-dir train_jsonl --glob 'all_runs.jsonl' \
  --epochs 5 --batch-size 64 --checkpoint-out checkpoints/abr_policy.pt
```

**Validate before training**

```bash
cd /path/to/ns3-ntn-iab/scratch/ai_llm
python3 - <<'PY'
import json, sys
sys.path.insert(0, '.')
import features                       # shared normalize path (no torch needed)
rows=[json.loads(l) for l in open('../../train_jsonl/all_runs.jsonl') if l.strip()]
print('rows:', len(rows))                                   # expect 12,114
r0=rows[0]
print('schema_version:', r0.get('schema_version'))          # 2
print('observation keys:', len(r0['observation']))          # 7 (raw, incl last_segment_bitrate_bps)
print('baked network_norm present?', 'network_norm' in r0.get('model_input', {}))   # False (raw dataset)
# Normalize through the SAME path train.py / inference use:
import statistics as st
cols=[[],[],[],[],[],[],[]]
for r in rows:
    mi=features.build_model_input_from_row(r)
    for i,v in enumerate(mi.network_norm): cols[i].append(v)
print('network_norm width:', len(mi.network_norm), 'ladder:', len(mi.ladder_context),'x',len(mi.ladder_context[0]))  # 7 ; 6x2
print('col means:', [round(st.fmean(c),3) for c in cols])   # ~0
print('col stds :', [round(st.pstdev(c),3) for c in cols])  # ~1 (RTT lower: ~half imputed)
import collections
print('label classes:', sorted(collections.Counter(r['teacher']['chosen_bitrate_class'] for r in rows)))  # 0..5
PY
```

Expect 12,114 rows, `schema_version` 2, raw `observation` of 7 keys, **no** baked
`network_norm` in the shards, and — after normalizing through `build_model_input_from_row`
— width 7 / ladder 6×2 with per-column mean ≈ 0 / std ≈ 1 (RTT std lower because ~half its
values are imputed sentinels). The `dataset_builder` run also prints `oracle ... missing=0`
per video and writes `train_jsonl/manifest.json` pinning the scalers — confirm both.

**After regeneration (the actual next steps)**

- **First SFT run.** `train.py` is ready; run it on `train_jsonl/all_runs.jsonl` to get a
  baseline. Note: there's still class imbalance (rung 5 ≈ 51%) and a random-row train/val
  split — both are addressed in the `abr_training_optimization` plan (class weighting +
  group-aware split, code-only, no re-sim). Worth landing those before reading too much
  into accuracy.
- **C++ live-inference parity (deferred).** The online IPC
  (`AbrDashCpp2PyMsg.current_bitrate_bps`) still emits the nominal rate. When we wire the
  online policy, that field must provide the actual last-segment bytes so live
  `network_norm` matches the trained contract. Until then the live path feeds 0.0 there.

Shout if anything in step 1–3 errors — most likely cause is a metadata CSV path or a stale
`Quic_artifacts` run dir.

Thanks,
Adeel

---

## Plain-text quick reference

```text
NO re-sim, NO re-oracle. Reuse Quic_artifacts/*/run_*/ {sim.log, oracle_labels_state.jsonl}.
Dataset is RAW (schema v2); normalization is applied at load by train.py/inference.
ONE build, then fit scalers, then train. No second build.

cd /path/to/ns3-ntn-iab && git pull
export METADATA_DIR=/project/6030214/adeel/abr/output/metadata

# 1) build raw dataset (no scaler needed at build) + manifest
python3 scratch/ai_llm/dataset_builder.py --artifacts-root Quic_artifacts --algo-short Fdash \
  --metadata-dir "$METADATA_DIR" --output-dir train_jsonl --teacher-source oracle --strict \
  --merge-output train_jsonl/all_runs.jsonl

# 2) fit both scalers from the raw shards
python3 scratch/ai_llm/scripts/fit_network_scaler.py  --data-dir train_jsonl --glob '*_run_*.jsonl'
python3 scratch/ai_llm/scripts/fit_semantic_scaler.py --data-dir train_jsonl --glob '*_run_*.jsonl'

# 3) train (applies + pins scalers; no rebuild on scaler change)
python3 scratch/ai_llm/train.py --data-dir train_jsonl --glob 'all_runs.jsonl' \
  --epochs 5 --batch-size 64 --checkpoint-out checkpoints/abr_policy.pt

# validate: 12,114 rows / schema_version 2 / raw observation 7 / ladder 6x2 / missing=0 / manifest.json
```
