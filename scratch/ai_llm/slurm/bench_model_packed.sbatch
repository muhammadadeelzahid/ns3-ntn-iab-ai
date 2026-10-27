#!/bin/bash
# Packed live-model benchmark worker: runs KPAR live AiDash cells CONCURRENTLY on ONE GPU.
# Each cell = one run_ai_dash_llm.py (ns-3 closed loop) + its own inference server, sharing this
# GPU (which is ~97% idle per cell). Concurrency is safe via a per-cell unique --shm-id (the
# ns3-ai shared-memory prefix), so the cells' IPC segments never collide.
#
# Submit one worker per matrix slice, e.g.:
#   MATRIX=Quic_artifacts/model_matrix.csv ROW_START=1 ROW_END=65 KPAR=12 \
#     sbatch scratch/ai_llm/slurm/bench_model_packed.sbatch
# Matrix CSV columns: task_id,video_name,metadata_csv,link_rate,run_seed
#
# SIMSTOP=auto (default) = the 4-arm benchmark session rule shared with bench_cpu_packed.sbatch and the
# native heuristic worker: NSEG*4+30 s, floor 90, cap SIM_STOP_CAP (default 180 s). The pre-Phase-4 model
# runs (bench/, bench_down/) used a fixed SIMSTOP=120/180 - pass SIMSTOP=<seconds> to reproduce those.
# Phase 4 (clean checkpoint, 26x5): see benchmark/RUN_BRIDGE_ARMS.md (OUT_ROOT=Quic_artifacts/AiDashClean,
# LOGDIR=logs_bench_aidashclean, SEMANTIC_SCALER=.../feature_scaler_semantic_clean.json).
#
#SBATCH --job-name=bench-model-pack
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

MATRIX="${MATRIX:?set MATRIX=<matrix csv>}"
ROW_START="${ROW_START:-1}"
ROW_END="${ROW_END:-99999999}"
KPAR="${KPAR:-6}"                                         # concurrent cells sharing this GPU
# NOTE: ~6 Qwen2.5-1.5B inference servers (~4-5 GB each incl CUDA/cuBLAS ctx) fit a 48 GB L40s.
# KPAR=12 OOMs (servers fall back to FDASH -> contaminated cells). 6 is the validated safe cap.
CKPT="${CKPT:-scratch/ai_llm/checkpoints_bola/abr_v2_seed42.pt}"
# Scalers must match the checkpoint's training set. checkpoints_clean (Phase 2, corrected VMAF) was
# trained with SEMANTIC_SCALER=scratch/ai_llm/config/feature_scaler_semantic_clean.json - serving it
# with the default (old) semantic scaler is a silent train/serve mismatch. Unset = server defaults.
NETWORK_SCALER="${NETWORK_SCALER:-}"
SEMANTIC_SCALER="${SEMANTIC_SCALER:-}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
NUMUES="${NUMUES:-3}"
SIMSTOP="${SIMSTOP:-auto}"                                # auto = NSEG*4+30 clamped [90, SIM_STOP_CAP=180]
OUT_ROOT="${OUT_ROOT:-Quic_artifacts/AiDash/bench}"
# Additive knobs (defaults preserve the LLM model arm exactly). The Pensieve baseline overrides
# POLICY=pensieve, DEVICE=cpu (tiny net), CKPT=checkpoints_pensieve/..., OUT_ROOT=Quic_artifacts/Pensieve.
POLICY="${POLICY:-}"
DEVICE="${DEVICE:-cuda}"
LOGDIR="${LOGDIR:-logs_bench}"
mkdir -p "$LOGDIR"

echo "=== packed worker: MATRIX=$MATRIX rows [$ROW_START,$ROW_END] KPAR=$KPAR ckpt=$CKPT numUes=$NUMUES simStop=$SIMSTOP cap=${SIM_STOP_CAP:-180} out=$OUT_ROOT logs=$LOGDIR sem_scaler=${SEMANTIC_SCALER:-default} ==="

sim_stop_for() {  # <metadata csv> -> seconds (identical to bench_cpu_packed.sbatch / dash_trace_worker.slurm)
  if [ "$SIMSTOP" = "auto" ]; then
    local nseg; nseg=$(($(wc -l < "$1") - 1))
    local s=$((nseg * 4 + 30)); [ "$s" -lt 90 ] && s=90; [ "$s" -gt "${SIM_STOP_CAP:-180}" ] && s="${SIM_STOP_CAP:-180}"
    echo "$s"
  else
    echo "$SIMSTOP"
  fi
}

run_cell() {
  local tid="$1" video="$2" meta="$3" rate="$4" seed="$5"
  local rtag="${rate//\//}"                              # 100Gb/s -> 100Gbs (dir-safe)
  local out="${OUT_ROOT}/${video}/run_${seed}_${rtag}/decisions.jsonl"
  [ -s "$out" ] && { echo "SKIP task $tid ($video $rate run$seed): decisions.jsonl exists"; return 0; }
  mkdir -p "$(dirname "$out")"
  local link=(); [ "$rate" != "100Gb/s" ] && link=(--link-rate "$rate")
  local stop; stop=$(sim_stop_for "$meta")
  /usr/bin/python3 scratch/run_ai_dash_llm.py \
    --checkpoint "$CKPT" --metadata "$meta" \
    --numUes "$NUMUES" --run "$seed" --sim-stop-time "$stop" \
    --server-python "$SERVER_PY" --device "$DEVICE" \
    ${POLICY:+--policy "$POLICY"} \
    ${NETWORK_SCALER:+--network-scaler "$NETWORK_SCALER"} \
    ${SEMANTIC_SCALER:+--semantic-scaler "$SEMANTIC_SCALER"} \
    --shm-id "pack_${SLURM_JOB_ID}_${tid}" \
    "${link[@]}" --decisions-out "$out" \
    > "${LOGDIR}/task_${tid}_${video}_${rtag}_run${seed}.log" 2>&1
  local rc=$?
  local nfb; nfb=$(grep -c '"reason": *"fallback' "$out" 2>/dev/null || echo 0)
  echo "done task $tid ($video $rate run$seed simStop=$stop) rc=$rc fallback_rows=$nfb -> $out"
}

n=0
while IFS=, read -r tid video meta rate seed; do
  [ "$tid" = "task_id" ] && continue                     # skip header
  [ -z "${tid:-}" ] && continue
  n=$((n+1))
  [ "$n" -lt "$ROW_START" ] && continue
  [ "$n" -gt "$ROW_END" ] && break
  run_cell "$tid" "$video" "$meta" "$rate" "$seed" &
  while [ "$(jobs -rp | wc -l)" -ge "$KPAR" ]; do sleep 5; done   # throttle to KPAR concurrent
done < "$MATRIX"
wait
echo "=== packed worker done (slice [$ROW_START,$ROW_END]) ==="
