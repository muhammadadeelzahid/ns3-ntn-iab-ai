#!/bin/bash
# Closed-loop smoke for checkpoints_causal (the throughput-sustainable causal-expert retrain).
# Same bridge as bench_v2_run.sbatch but writes to a SEPARATE artifacts root (AiDash_causal/) so it
# never clobbers the checkpoints_dagger rescore baseline. Validates that the offline over-ask collapse
# (19.5% -> 8.4%) survives closed-loop (no buffer drain -> no collapse). Metric = over-ask / switch rate
# on the causal model's OWN online states.
#
# Submit:  VIDEO=<name> LINKRATE=9Mbps RUN=4 sbatch bench_v2_causal_smoke.sbatch
#SBATCH --job-name=bench-causal
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export WAF_CMD="python3 ./waf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

VIDEO="${VIDEO:?set VIDEO=<name>}"
LINKRATE="${LINKRATE:?set LINKRATE=<rate>}"
RUN="${RUN:-1}"
NUMUES="${NUMUES:-3}"
SIMSTOP="${SIMSTOP:-60}"
CKPT="${CKPT:-scratch/ai_llm/checkpoints_causal_snap.pt}"
METADIR="${METADIR:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
OUT="${OUT:-Quic_artifacts/AiDash_causal/${VIDEO}/run_${RUN}/decisions.jsonl}"

echo "===== causal smoke: video=$VIDEO linkRate=$LINKRATE run=$RUN numUes=$NUMUES simStop=$SIMSTOP ckpt=$CKPT policy=${POLICY:-model} ====="
/usr/bin/python3 scratch/run_ai_dash_llm.py \
  --checkpoint "$CKPT" \
  --metadata "${METADIR}/${VIDEO}_segments.csv" \
  --numUes "$NUMUES" --run "$RUN" --sim-stop-time "$SIMSTOP" \
  --link-rate "$LINKRATE" \
  --server-python "$SERVER_PY" --device cpu \
  ${POLICY:+--policy "$POLICY"} ${W_STALL:+--w-stall "$W_STALL"} ${THR_SAFETY:+--thr-safety "$THR_SAFETY"} \
  --video "$VIDEO" --decisions-out "$OUT"
echo "===== done video=$VIDEO run=$RUN linkRate=$LINKRATE -> $OUT ====="
