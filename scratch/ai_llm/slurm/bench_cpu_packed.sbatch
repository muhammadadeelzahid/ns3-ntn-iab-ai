#!/bin/bash
# Packed CPU benchmark worker for the torch-free / CPU-cheap bridge arms: POLICY=pensieve|mpc|qmpc|greedy.
# Generalises bench_pensieve_packed.sbatch: each cell = one run_ai_dash_llm.py (ns-3 closed loop) + its own
# CPU inference server, per-cell unique --shm-id, KPAR concurrent cells on one skylake node. Each arm writes
# to its OWN artifacts tree Quic_artifacts/<Stem>/<video>/run_<seed>_<rtag>/decisions.jsonl and its OWN
# log dir logs_bench_<stem>/ (extract_true_stall.py resolves sim.log per arm from there - never mix arms).
#
# Submit (from the repo root):
#   MATRIX=Quic_artifacts/model_matrix_1run.csv POLICY=qmpc KPAR=12 \
#     sbatch scratch/ai_llm/slurm/bench_cpu_packed.sbatch
# Matrix CSV columns: task_id,video_name,metadata_csv,link_rate,run_seed
#
# SIMSTOP=auto (default) streams the WHOLE video: NSEG*3.3 s + 30 s, the same rule the native heuristic
# collection used (dash_trace_worker.slurm), so every arm plays the same number of segments. The older
# bridge runs used a fixed 120/180 s cap - pass SIMSTOP=<seconds> to reproduce those.
#
#SBATCH --job-name=bench-cpu
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=3800M
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

MATRIX="${MATRIX:?set MATRIX=<matrix csv>}"
POLICY="${POLICY:?set POLICY=pensieve|mpc|qmpc|greedy}"
case "$POLICY" in
  pensieve) STEM=Pensieve ;;
  mpc)      STEM=Mpc ;;
  qmpc)     STEM=Qmpc ;;
  greedy)   STEM=Greedy ;;
  *) echo "unknown POLICY=$POLICY" >&2; exit 2 ;;
esac
ROW_START="${ROW_START:-1}"
ROW_END="${ROW_END:-99999999}"
KPAR="${KPAR:-12}"                                        # concurrent CPU cells
CKPT="${CKPT:-checkpoints_pensieve/pensieve_actor.pt}"    # pensieve only
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
NUMUES="${NUMUES:-3}"                                     # match the model arm
SIMSTOP="${SIMSTOP:-auto}"                                # auto = NSEG*3.3+30 per video (whole video)
OUT_ROOT="${OUT_ROOT:-Quic_artifacts/${STEM}}"
LOGDIR="${LOGDIR:-logs_bench_$(echo "$STEM" | tr '[:upper:]' '[:lower:]')}"
mkdir -p "$LOGDIR"

echo "=== cpu worker: POLICY=$POLICY STEM=$STEM MATRIX=$MATRIX rows [$ROW_START,$ROW_END] KPAR=$KPAR numUes=$NUMUES simStop=$SIMSTOP out=$OUT_ROOT logs=$LOGDIR ==="

sim_stop_for() {  # <metadata csv> -> seconds, or the fixed SIMSTOP
  # auto = EXACTLY the native heuristic worker's rule (dash_trace_worker.slurm): NSEG*4 + 30, floor 90,
  # cap 600 - so bridge arms and natively-run heuristics stream the same session length per video.
  if [ "$SIMSTOP" = "auto" ]; then
    local nseg; nseg=$(($(wc -l < "$1") - 1))
    local s=$((nseg * 4 + 30)); [ "$s" -lt 90 ] && s=90; [ "$s" -gt 600 ] && s=600
    echo "$s"
  else
    echo "$SIMSTOP"
  fi
}

run_cell() {
  local tid="$1" video="$2" meta="$3" rate="$4" seed="$5"
  local rtag="${rate//\//}"                               # 100Gb/s -> 100Gbs (dir-safe)
  local out="${OUT_ROOT}/${video}/run_${seed}_${rtag}/decisions.jsonl"
  [ -s "$out" ] && { echo "SKIP task $tid ($video $rate run$seed): decisions.jsonl exists"; return 0; }
  mkdir -p "$(dirname "$out")"
  local link=(); [ "$rate" != "100Gb/s" ] && link=(--link-rate "$rate")
  local ckpt=(); [ "$POLICY" = "pensieve" ] && ckpt=(--checkpoint "$CKPT")
  local stop; stop=$(sim_stop_for "$meta")
  /usr/bin/python3 scratch/run_ai_dash_llm.py \
    --policy "$POLICY" "${ckpt[@]}" --metadata "$meta" \
    --numUes "$NUMUES" --run "$seed" --sim-stop-time "$stop" \
    --server-python "$SERVER_PY" --device cpu \
    --shm-id "${POLICY:0:4}_${SLURM_JOB_ID}_${tid}" \
    "${link[@]}" --decisions-out "$out" \
    > "${LOGDIR}/task_${tid}_${video}_${rtag}_run${seed}.log" 2>&1
  local rc=$?
  local nfb; nfb=$(grep -c '"reason": *"fallback' "$out" 2>/dev/null || echo 0)
  echo "done task $tid ($video $rate run$seed simStop=$stop) rc=$rc fallback_rows=$nfb -> $out"
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
echo "=== cpu worker done: POLICY=$POLICY slice [$ROW_START,$ROW_END] ==="
