# Pipeline Progress — 2026-06-19

Snapshot of the LLM-ABR data → training → evaluation pipeline, the running Slurm jobs, and the
Phase 4/5 (ns-3-in-the-loop) plan state.

## Running jobs (what they are)

There are exactly **two** distinct things in the queue:

| Job | Count | What it does |
|---|---|---|
| `pipeline_chain` (7087244) | 1 (CPU) | **Orchestrator.** Runs the whole pipeline in sequence: wait for ABR → ns-3 trace campaign → throughput extraction → DP oracle → dataset build → **train/val/test split** → scaler fit (on `train/`) → training sweep (V0/V1/V2 × seeds) → auto-eval on held-out `test/` → `benchmark_report`. Does ~no compute itself; gates/launches each stage. Currently parked in **Stage 1**, polling the worker array. |
| `dash_trace_worker` (7087435_1…525) | 525 (array) | **ns-3 simulation army — the training-data generators.** Each task `_N` = one `(video, run-seed)`: a full ns-3 DASH-over-NTN simulation streaming one specific video with the **FDASH** client over the satellite link, recording per-segment network observations + bitrate decisions (`trace_decisions.jsonl`, `throughput_trace.jsonl`). 105 videos × 5 runs = 525. Each: 4 CPU / 16 GB / ≤24 h walltime. |

A trace row = *(network state from ns-3) + (video semantics & VMAF ladder joined from metadata) +
(FDASH/oracle label)* — i.e. the `(network × content) → best rung` tuple the policy learns from.
ns-3 only consumes **segment byte sizes** from the metadata; VMAF/semantics are joined later.

## Campaign status (as of 2026-06-19 ~15:25 CDT)

- **ns-3 trace campaign (7087435):** ~298 running, ~140 pending, ~125 completed of **525**.
- **Throttle removed** (`ArrayTaskThrottle=0`) — see below. Running jumped **40 → 298**.
- **ETA:** ~1 day (was ~2+ days at the 40-throttle).
- Chain stays in Stage 1 until the array drains, then auto-continues to throughput → oracle →
  build → split → scalers → sweep → eval → report.

## Throttle removal (today) — non-destructive speedup

The campaign was self-throttled to 40 concurrent (`MAX_PARALLEL=40`). All pending tasks were
waiting on `JobArrayTaskLimit` (our cap), **not** cluster resources — so widening was free.

Done **in place**, without editing the script or resubmitting (which would rebuild the job matrix,
restart from index 1, and overwrite the 125 completed traces):

```
scontrol update JobId=7087435 ArrayTaskThrottle=0      # 0 = unlimited; scheduler/fair-share governs
```

Result: 40 → 298 running instantly, 125 completed traces preserved, nothing re-ran. Reversible
(`ArrayTaskThrottle=40`).

## Completed prior work (in code + git)

- **Richer QoE metrics** in `scratch/ai_llm/evaluate.py` (rebuffer ratio, mean delivered bitrate,
  switch count) on top of the existing % -of-oracle headline.
- **Auto-eval wiring** in `pipeline_chain.slurm` (`sbatch --dependency=afterok` per checkpoint →
  aggregated `benchmark_report_<date>.md`).
- **Leakage-free, genre-stratified, whole-video train/val/test split** (the methodology fix):
  - `scratch/ai_llm/split_dataset_videos.py` (new) — 70/15/15, stratified by genre (100% coverage
    over all 105 videos), materializes `train_jsonl/{train,val,test}/` symlink dirs +
    `split_manifest.json`.
  - `train.py --val-data-dir` — uses the persisted `val/` for early stopping, bypasses the random
    internal split.
  - Chain reordered: build → split → **fit scalers on `train/` only** → train on `train/`+`val/` →
    eval on held-out `test/`. Scalers isolated before the test set (no leakage).

## Phase 4/5 — ns-3-in-the-loop (the roadmap's real evaluation)

The offline `evaluate.py` replay is a proxy (throughput treated as exogenous). The headline test
is the model running **live inside ns-3** over the ns3-ai IPC.

**Full C++ IPC audit (read end-to-end): functionally complete, not stubbed.**
- `abr-dash-ai-msg.h` v2 structs, `AiDashClient::CalcNextSegment` round-trip, pybind, and the
  `run_ai_dash_llm.py` loop are all complete; `inference.py` maintains a **per-UE window ring
  buffer** and reads window/masks from the checkpoint (train/serve parity).
- **What the LLM *has* done:** trained (V1/V2 checkpoints exist), unit-tested (`test_llm_smoke.py`
  runs a real LLM forward+backward), and **offline-evaluated** (`evaluate.py`; V2 macro-F1 ≈ 0.256).
- **The unproven part — the LIVE closed loop.** No evidence in the repo that
  `run_ai_dash_llm.py` → `AiDashClient` → IPC was ever run to completion **with the LLM or even the
  heuristic**: there is **no `Quic_artifacts/AiDash/` artifact dir** and no run logs (only `.pyc`).
  The IPC code is complete and its pieces are unit-tested, but a full live run is **not demonstrated
  here** (artifacts may have been cleaned — stated as "no evidence found," not certain fact). This
  is exactly what Stage A exists to settle.
- **Other gaps:** **latency-naive** (blocking single-struct handshake — a V2 forward per decision
  stalls the single-threaded sim, though it does *not* corrupt QoE since the sim pauses);
  **multi-UE is homogeneous** (one metadata CSV for all UEs); minor train/serve nits.
- **Latency clarification:** the GPU is only for the **LLM policy forward** (V1/V2). Video
  metadata + semantic scores are precomputed by the ABR pipeline and merely looked up at runtime —
  no GPU for features. **V0 runs live sub-ms on CPU**, so choosing V0 for the live loop removes the
  latency problem entirely.
- **FDASH is not the headline** — the DP oracle is acausal (can't run live), so FDASH is only the
  runnable causal sanity baseline. The claim stays the **V0/V1/V2 ablation + % of oracle**.

**Code-prep done today (safe, no ns-3 execution):**
- `run_ai_dash_llm.py`: refreshed the stale "policy stub" docstring; `--model-id` default
  7B → 1.5B (a checkpoint's own `model_id` overrides it anyway).

**Deferred / blocked:**
- **1-UE live ns-3 smoke (Stage A):** deferred until the trace campaign finishes, so `build/` is
  free (a live `./waf` could recompile in place and SIGBUS the running workers).
- **`eval_qoe.py` + per-UE metadata routing (Phase 5 harness):** blocked on three open decisions.

## Open decisions (gate the Phase 5 harness)

1. **Live-policy scope:** V0 only, or V0 + V2?
2. **Latency strategy:** distill V2 → V0 for live, run V2 slow (cache/cadence), or vector-mode batch?
3. **Deployment realism:** model inference delay as `delay_s`, or keep `0` for the pure comparison?

## Next

- Let the campaign drain (~1 day) → chain auto-runs build → split → scalers → sweep → eval →
  `benchmark_report`.
- Once `build/` is free, run the deferred 1-UE live smoke on `checkpoints/abr_v0_seed42.pt`.
- Confirm the three decisions → build `eval_qoe.py` → 1-UE then 10-UE live QoE benchmark.
