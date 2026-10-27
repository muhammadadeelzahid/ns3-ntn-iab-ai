# Running and scoring the benchmark arms (4-arm, 26-video, capped-session benchmark)

Arms (paper_style.ALGOS): **Ours** (`AiDash`, LLM policy), **Quality-MPC** (`Qmpc`), **RobustMPC** (`Mpc`),
**BOLA** (`BolaFull`). Every arm streams the same session: whole video capped at `SIM_STOP_CAP=180` s
(`NSEG*4+30`, floor 90), 3 UEs, test split = `benchmark/test_videos_26.txt`, 5 S1-U rates
(100Gb/s, 36, 18, 9, 4.5 Mbps). The %-of-teacher reference is the greedy down-only causal teacher
(`config/oracle_reward.json`) for every arm - never the DP oracle.

## Bridge arms (Python inference server): Qmpc, Mpc (CPU) and AiDash (GPU)
```
# RobustMPC / Quality-MPC: torch-free, skylake, 130 cells in 4 row slices x KPAR=20 (fits 12 h)
for pol in qmpc mpc; do for s in "1 33" "34 66" "67 99" "100 130"; do set -- $s
  MATRIX=Quic_artifacts/model_matrix_1run.csv POLICY=$pol KPAR=20 ROW_START=$1 ROW_END=$2 \
    sbatch --job-name=bench-$pol-$1 scratch/ai_llm/slurm/bench_cpu_packed.sbatch; done; done
# Ours (Phase 4): lgpu packed worker, clean checkpoint + ITS scaler, same session rule
MATRIX=Quic_artifacts/model_matrix_1run.csv CKPT=scratch/ai_llm/checkpoints_clean/abr_v2_seed42.pt \
  SEMANTIC_SCALER=scratch/ai_llm/config/feature_scaler_semantic_clean.json SIMSTOP=<per-row auto - port
  sim_stop_for() from bench_cpu_packed> OUT_ROOT=Quic_artifacts/AiDashClean KPAR=6 \
  sbatch scratch/ai_llm/slurm/bench_model_packed.sbatch
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

## Tests
`PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/tests/test_mpc_policy.py` (7 tests: objective parity with
`dp_oracle` to 1e-9, K=1 == `greedy_rung`, reference enumeration equality, tie rule, serving flow).
`PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/benchmark/mpc_offline_smoke.py` (open-loop replay sanity).
