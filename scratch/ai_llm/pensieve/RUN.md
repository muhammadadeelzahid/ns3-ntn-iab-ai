# Pensieve benchmark — run guide

Pensieve (Mao et al., SIGCOMM 2017) as a learning-based ABR baseline, retrained from scratch on our
ns-3 traces + 6-rung ladder, canonical linear QoE, network-only. Served through the existing ns-3
inference bridge as `--policy pensieve`. Training is kept MINIMAL and ≤ our LLM's budget by design
(single synchronous A2C agent, 2000 updates ≈ 64k decisions) — see the project memory
`pensieve-benchmark-fairness`. Do not raise `--updates` without that fairness ceiling in mind.

## 1. Train (already done → `checkpoints_pensieve/pensieve_actor.pt`)
```
~/abr-llm-venv/bin/python scratch/ai_llm/pensieve/train.py \
  --updates 2000 --n-steps 32 --out checkpoints_pensieve/pensieve_actor.pt
```
Defaults (entropy-coef 0.1, stratified bandwidth sampling, clipped throughput/delay features) are
what produce a state-responsive policy; a congestion-heavy pool + low entropy collapses it to a
single safe rung. Validated behavior: rung rises monotonically with throughput and buffer; a short
uncongested ns-3 cell gave 59 decisions, 0 fallbacks, no stalls.

## 2. Run the benchmark arm (same 10 videos × 5 rates as the model arm)
Reuses the live-cell worker with additive env overrides (model arm behavior unchanged). Pensieve is
CPU-cheap, so `DEVICE=cpu` and a high `KPAR` are fine:
```
MATRIX=Quic_artifacts/model_matrix_shrunk.csv \
POLICY=pensieve DEVICE=cpu \
CKPT=checkpoints_pensieve/pensieve_actor.pt \
OUT_ROOT=Quic_artifacts/Pensieve \
KPAR=16 NUMUES=3 SIMSTOP=120 \
  sbatch scratch/ai_llm/slurm/bench_model_packed.sbatch
```
Writes `Quic_artifacts/Pensieve/<video>/run_<seed>_<rate>/decisions.jsonl` — a separate tree from
the existing arms. (A single cell runs the full simStop ~3-4h; non-empty decisions.jsonl ≠ done.)

## 3. Add to tables/figures (FINAL step, after the runs exist)
Uncomment the `("Pensieve", ...)` line in `scratch/ai_llm/benchmark/paper_style.py` (ALGOS), then
regenerate: `benchmark/make_paper_table.py` + `benchmark/aggregate_table.py`. Scored on the same
VMAF greedy-reward %oracle metric as every other arm.

## Note on version control
`scratch/ai_llm/` is gitignored (`.gitignore:78`), so this package and the `--policy pensieve`
additions to `inference.py` / `inference_server.py` live only in the working tree, not on the
`feat/pensieve-benchmark` branch. Only `scratch/run_ai_dash_llm.py` (the `--policy pensieve` choice)
is tracked on the branch. The ai_llm edits are additive and preserve the existing model/greedy
paths exactly.
```
