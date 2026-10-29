# Running and scoring the benchmark arms (4-arm, 26-video, capped-session benchmark)

Arms (paper_style.ALGOS): **Ours** (`AiDash`, LLM policy), **Quality-MPC** (`Qmpc`), **RobustMPC** (`Mpc`),
**BOLA** (`BolaFull`). Every arm streams the same session: whole video capped at `SIM_STOP_CAP=180` s
(`NSEG*4+30`, floor 90), 3 UEs, test split = `benchmark/test_videos_26.txt`, 5 S1-U rates
(100Gb/s, 36, 18, 9, 4.5 Mbps). The %-of-teacher reference is the greedy down-only causal teacher
(`config/oracle_reward.json`) for every arm - never the DP oracle.

## Simulation environment (fixed binary, 2026-10-28) - read before submitting anything
Every arm must run on a binary built from `fix/client-freeze` (merged into `feat/mpc-baselines`) or later:
- **Two ns-3 QUIC bugs** made 12-16 % of UEs freeze permanently in EVERY arm at EVERY rate on the old
  binary (commits 13b68f5: ACK frame appended in place to the retransmittable packet; 8babc18: post-RTO
  "retransmit everything as one packet" loop, >65,507 B silently dropped by UDP). Symptoms: `interruptions`
  with the video unfinished, sim clock crawling (~1 sim-s per wall-hour), 3.9 MB cwnd. The DASH client
  also gained a dead-segment reconnect (`[DashClient] node N seg S dead ... -> reconnect #k`, max 8) and a
  30 s `MaxBufferLevel` (= the scoring player model) as defence in depth (a59f4cd, eaed4b3).
- Results produced on the old binary (`Quic_artifacts/*_oldenv`, `logs_bench_*_oldenv`,
  `results/_oldenv_2026-10-27/`: Mpc 130/130 = 89.4 % of teacher, Qmpc 129/130 = 90.9 %, BolaFull 130/130,
  Ours 41 cells) are kept ONLY as the before/after disclosure. Never mix old-env and new-env cells.
- **Beamforming cache (`NS3_ATTRS`)**: all three launchers (`bench_cpu_packed.sbatch`,
  `bench_model_packed.sbatch`, `dash_trace_worker.slurm`) default to
  `NS3_ATTRS=ns3::MmWaveFFTCodebookBeamforming::CacheExpiry=1000s`, forwarded as `--ns3-attr` to
  `run_ai_dash_llm.py` (bridge) or `--ns3::...` to the binary (native). The scenario's UEs/IAB nodes are
  static and the 28 GHz channel has `UpdatePeriod` 0, so the FFT-codebook beam design is identical every
  slot; the A/B on a full 90 s cell was byte-identical (decisions, frame trace, session stats) and 8.5x
  faster (~12 s instead of ~100 s wall per simulated second). Disclose it as a simulation setting.
  `NS3_ATTRS=""` restores the per-slot design; with mobility or a channel update period the expiry must not
  exceed that period (20 ms SSB beam-management period). The worker banner prints `ns3_attrs=`.
- **Freeze gate after every arm**: `python3.11 scratch/ai_llm/benchmark/freeze_audit.py 'logs_bench_<stem>/task_*.log'`
  (native BOLA: `'Quic_artifacts/BolaFull/*/run_*/sim.log'`) must report 0 permanently frozen UEs; a handful
  of reconnects per cell is normal. The validation of the fixed binary on the six worst old-freeze cells is
  `Quic_artifacts/freeze_smoke.csv` -> `Qmpc_fixsmoke` (job 8134633; `Qmpc_fixsmoke_v2` = first QUIC fix only).

## Bridge arms (Python inference server): Qmpc, Mpc (CPU) and AiDash (GPU)
```
# RobustMPC / Quality-MPC: torch-free, skylake, KPAR=20. With the beam cache a 180 s cell is ~35-40 min wall,
# so 2 row slices of 65 (~3-4 h each) replace the old 4 slices; existing non-empty cells are skipped on resubmit.
for pol in qmpc mpc; do for s in "1 65" "66 130"; do set -- $s
  MATRIX=Quic_artifacts/model_matrix_1run.csv POLICY=$pol KPAR=20 ROW_START=$1 ROW_END=$2 \
    sbatch --job-name=bench-$pol-$1 scratch/ai_llm/slurm/bench_cpu_packed.sbatch; done; done
# Ours (Phase 4): lgpu packed worker, clean checkpoint + ITS scaler, SIMSTOP=auto (default) = same session
# rule. The old binary needed ~4-5 h wall per 180 s cell (ns-3 CPU-bound, not the GPU); with the beam cache
# ~40 min, so 3 slices of ~44 rows at KPAR=6 (the OOM-safe cap for 6 servers on one L40s) fit 12 h jobs.
# Existing non-empty cells are skipped on resubmit.
for s in "1 44" "45 88" "89 130"; do set -- $s
  MATRIX=Quic_artifacts/model_matrix_1run.csv CKPT=scratch/ai_llm/checkpoints_clean/abr_v2_seed42.pt \
    SEMANTIC_SCALER=scratch/ai_llm/config/feature_scaler_semantic_clean.json \
    OUT_ROOT=Quic_artifacts/AiDashClean LOGDIR=logs_bench_aidashclean KPAR=6 ROW_START=$1 ROW_END=$2 \
    sbatch --job-name=bench-ours-$1 scratch/ai_llm/slurm/bench_model_packed.sbatch; done
# Score Ours exactly like the other bridge arms (fallback gate, greedy relabel, AiDashClean_shards):
#   bash scratch/ai_llm/benchmark/score_bridge_arm.sh AiDashClean
```
Outputs: `Quic_artifacts/<Stem>/<video>/run_<seed>_<rtag>/decisions.jsonl`, logs `logs_bench_<stem>/`.
Cells with an existing non-empty decisions.jsonl are skipped (safe to resubmit slices).

**Before any `run_ai_dash_llm.py` job: `python3 waf build` on the LOGIN node.** The client launches ns-3
through waf; if any source/header is newer than the binary (even a reverted edit), waf tries to compile
on the compute node (no headers there) and concurrent cells race on `build/.wafpickle-*`.

Score (fallback gate -> cond from the rate tag -> realized throughput -> dp_oracle join -> greedy
relabel -> score_v2_perceptual):
```
bash scratch/ai_llm/benchmark/score_bridge_arm.sh Qmpc     # -> results/Qmpc.txt, Quic_artifacts/Qmpc_shards
bash scratch/ai_llm/benchmark/score_bridge_arm.sh Mpc
```
The gate refuses cells with >1 % `reason=fallback_*` rows (the client decided with FDASH); `FORCE=1`
scores anyway. Check the printed reason histogram.

## Native heuristic (BOLA)
```
ALGORITHM=ns3::BolaClient NUM_UES=3 LINK_RATES="100Gb/s,36Mbps,18Mbps,9Mbps,4.5Mbps" RUNS_PER_METADATA=1 \
  VIDEO_LIST=$PWD/scratch/ai_llm/benchmark/test_videos_26.txt ALGO_SHORT_OVERRIDE=BolaFull \
  SKIP_BUILD=1 SIM_STOP_CAP=180 MAX_PARALLEL=40 sbatch dash_trace_launch.slurm
# seeds 1..5 == the five rates in LINK_RATES order; run_manifest.json carries link_rate
python3.11 scratch/ai_llm/benchmark/build_dscore_links.py --arms BolaFull     # BolaFull_dscore/<v>/run_<cond>
bash scratch/ai_llm/benchmark/score_report.sh BolaFull_dscore                  # greedy teacher; persists BolaFull_teacher_down
```

## Tables / figures / stall
```
PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/benchmark/extract_true_stall.py   # per-arm sim.log, own log dir only
PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/benchmark/absolute_qoe_table.py
python3 scratch/ai_llm/benchmark/make_paper_table.py ; python3 scratch/ai_llm/benchmark/aggregate_table.py
python3 scratch/ai_llm/benchmark/make_paper_figs.py
```
`BENCH_VIDS=<file>` switches the video list (default `test_videos_26.txt`).
`MODEL_STEM=AiDashClean` (see `benchmark/model_arm.py`) points the model arm "Ours" of every table/figure
script at the Phase-4 re-run (`Quic_artifacts/AiDashClean`, `AiDashClean_shards`, `logs_bench_aidashclean`,
`results/AiDashClean.txt`); unset = the legacy 2026-07 layout (`AiDash/bench_down_scored`). Run the four
commands above with `MODEL_STEM=AiDashClean` exported once Phase 4 is scored.

## Tests
`PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/tests/test_mpc_policy.py` (7 tests: objective parity with
`dp_oracle` to 1e-9, K=1 == `greedy_rung`, reference enumeration equality, tie rule, serving flow).
`PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/benchmark/mpc_offline_smoke.py` (open-loop replay sanity).
