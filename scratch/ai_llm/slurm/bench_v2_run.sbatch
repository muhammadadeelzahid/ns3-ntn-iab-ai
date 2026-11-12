#!/bin/bash
# One V2 (LLM) benchmark run: AiDashClient via the run_ai_dash_llm.py bridge, CPU inference.
# Writes the STANDARD Quic_artifacts/AiDash/<video>/run_<RUN>/ artifacts (sim.log + raw traces +
# decisions.jsonl) so the arm is scorable with ALGO_SHORT=AiDash (extract → dp_oracle → perceptual_qoe).
#
# One job per (video, linkRate); RUN encodes the linkRate (run=link_idx+1) so the 5 rates land in
# distinct run dirs and never collide. Per-job-unique ns3-ai shm names (SLURM_JOB_ID) make concurrent
# runs on one node safe. CPU inference (the ns-3 PHY dominates wall time) lets the arm parallelize
# across skylake instead of bottlenecking on the 2 lgpu GPUs.
#
# Submit:  VIDEO=<name> LINKRATE=18Mbps RUN=3 sbatch bench_v2_run.sbatch
#
#SBATCH --job-name=bench-v2
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
# CPU LLM inference is ~6-7s/decision, so a numUes=3, simStop=120 run needs ~7h wall (the ns-3 PHY
# alone is ~2.5h; inference adds the rest). Use the skylake max (24h) so the sim always reaches
# simStopTime instead of being killed mid-playback.
#SBATCH --time=1-00:00:00
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
SIMSTOP="${SIMSTOP:-120}"
CKPT="${CKPT:-scratch/ai_llm/checkpoints_resid_w1/abr_v2_seed42.pt}"
METADIR="${METADIR:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
OUT="Quic_artifacts/AiDash/${VIDEO}/run_${RUN}/decisions.jsonl"

echo "===== bench V2: video=$VIDEO linkRate=$LINKRATE run=$RUN numUes=$NUMUES simStop=$SIMSTOP ====="
/usr/bin/python3 scratch/run_ai_dash_llm.py \
  --checkpoint "$CKPT" \
  --metadata "${METADIR}/${VIDEO}_segments.csv" \
  --numUes "$NUMUES" --run "$RUN" --sim-stop-time "$SIMSTOP" \
  --link-rate "$LINKRATE" \
  --server-python "$SERVER_PY" --device cpu \
  --video "$VIDEO" --decisions-out "$OUT"
echo "===== done video=$VIDEO run=$RUN linkRate=$LINKRATE -> $OUT ====="
