#!/bin/bash
# Packed Pensieve benchmark worker (CPU-only). Mirrors bench_model_packed.sbatch but runs the
# Pensieve A3C baseline, which needs no GPU (tiny net, ~35-100 ms/decision on CPU). Each cell = one
# run_ai_dash_llm.py (ns-3 closed loop) + its own Pensieve inference server, with a per-cell unique
# --shm-id so the ns3-ai IPC segments never collide. Writes to a SEPARATE artifacts tree
# (Quic_artifacts/Pensieve/) so the existing LLM/heuristic arms are untouched.
#
# Submit:
#   MATRIX=Quic_artifacts/model_matrix_shrunk.csv KPAR=12 \
#     sbatch scratch/ai_llm/slurm/bench_pensieve_packed.sbatch
# Matrix CSV columns: task_id,video_name,metadata_csv,link_rate,run_seed
#
#SBATCH --job-name=bench-pensieve
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3800M
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

MATRIX="${MATRIX:?set MATRIX=<matrix csv>}"
ROW_START="${ROW_START:-1}"
ROW_END="${ROW_END:-99999999}"
KPAR="${KPAR:-12}"                                        # concurrent CPU cells
CKPT="${CKPT:-checkpoints_pensieve/pensieve_actor.pt}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
NUMUES="${NUMUES:-3}"                                     # match the model arm
SIMSTOP="${SIMSTOP:-120}"                                 # match the model arm
OUT_ROOT="${OUT_ROOT:-Quic_artifacts/Pensieve}"
LOGDIR="${LOGDIR:-logs_bench_pensieve}"
mkdir -p "$LOGDIR"

echo "=== pensieve worker: MATRIX=$MATRIX rows [$ROW_START,$ROW_END] KPAR=$KPAR ckpt=$CKPT numUes=$NUMUES simStop=$SIMSTOP ==="

run_cell() {
  local tid="$1" video="$2" meta="$3" rate="$4" seed="$5"
  local rtag="${rate//\//}"                               # 100Gb/s -> 100Gbs (dir-safe)
  local out="${OUT_ROOT}/${video}/run_${seed}_${rtag}/decisions.jsonl"
  [ -s "$out" ] && { echo "SKIP task $tid ($video $rate run$seed): decisions.jsonl exists"; return 0; }
  mkdir -p "$(dirname "$out")"
  local link=(); [ "$rate" != "100Gb/s" ] && link=(--link-rate "$rate")
  /usr/bin/python3 scratch/run_ai_dash_llm.py \
    --policy pensieve --checkpoint "$CKPT" --metadata "$meta" \
    --numUes "$NUMUES" --run "$seed" --sim-stop-time "$SIMSTOP" \
    --server-python "$SERVER_PY" --device cpu \
    --shm-id "pens_${SLURM_JOB_ID}_${tid}" \
    "${link[@]}" --decisions-out "$out" \
    > "${LOGDIR}/task_${tid}_${video}_${rtag}_run${seed}.log" 2>&1
  echo "done task $tid ($video $rate run$seed) rc=$? -> $out"
}

n=0
while IFS=, read -r tid video meta rate seed; do
  [ "$tid" = "task_id" ] && continue                      # skip header
  [ -z "${tid:-}" ] && continue
  n=$((n+1))
  [ "$n" -lt "$ROW_START" ] && continue
  [ "$n" -gt "$ROW_END" ] && break
  run_cell "$tid" "$video" "$meta" "$rate" "$seed" &
  while [ "$(jobs -rp | wc -l)" -ge "$KPAR" ]; do sleep 5; done   # throttle to KPAR concurrent
done < "$MATRIX"
wait
echo "=== pensieve worker done (slice [$ROW_START,$ROW_END]) ==="
