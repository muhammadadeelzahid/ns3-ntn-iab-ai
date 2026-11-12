#!/bin/bash
# Per-video CONGESTED live ABR run (V0, CPU) — one job per video so SLURM scatters them across
# nodes (the ns3-ai shared-memory segment uses fixed names, so one run PER NODE; separate nodes =
# true parallelism, no collision). 1 UE on a deliberately narrow S1-U feeder forces the policy to
# back off from the top rung — a clean adaptation test, and fast (no multi-UE superlinear sim cost).
#
# Submit:  VIDEO=hiking sbatch live_congested_video.sbatch
#
#SBATCH --job-name=abr-cong
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-05:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export WAF_CMD="python3 ./waf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

VIDEO="${VIDEO:?set VIDEO=<name>}"
VARIANT="${VARIANT:-v0}"
DEVICE="${DEVICE:-cpu}"           # cpu for V0; cuda for V1/V2 (submit on lgpu)
LINKRATE="${LINKRATE:-5Mbps}"
NUMUES="${NUMUES:-1}"
SIMSTOP="${SIMSTOP:-90}"
SEED="${SEED:-42}"
CKPT="${CKPT:-scratch/ai_llm/checkpoints/abr_${VARIANT}_seed${SEED}.pt}"
METADIR="${METADIR:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
OUT="Quic_artifacts/AiDash/cong_${LINKRATE}_${NUMUES}ue/${VARIANT}_${VIDEO}/decisions.jsonl"

echo "===== congested live: variant=$VARIANT video=$VIDEO linkRate=$LINKRATE numUes=$NUMUES simStop=$SIMSTOP device=$DEVICE ====="
/usr/bin/python3 scratch/run_ai_dash_llm.py \
  --checkpoint "$CKPT" \
  --metadata "${METADIR}/${VIDEO}_segments.csv" \
  --numUes "$NUMUES" --run 1 --sim-stop-time "$SIMSTOP" \
  --link-rate "$LINKRATE" \
  --server-python "$SERVER_PY" --device "$DEVICE" \
  --video "$VIDEO" --decisions-out "$OUT"
echo "===== done variant=$VARIANT video=$VIDEO -> $OUT ====="
