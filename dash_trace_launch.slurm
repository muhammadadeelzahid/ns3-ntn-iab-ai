#!/bin/bash
#SBATCH --job-name=dash_trace_launch
#SBATCH --account=def-ekram
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=dash_trace_launch_%j.out
#SBATCH --error=dash_trace_launch_%j.err

# Discover *_segments.csv metadata files and submit one worker per (video, run_seed).
# Set ALGORITHM to any ns-3 DASH client, e.g. ns3::FdashClient, ns3::AiDashClient.

set -euo pipefail

NS3_REPO="${NS3_REPO:-/home/adeel/projects/def-ekram/adeel/ns3-ntn-iab}"
METADATA_DIR="${METADATA_DIR:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata}"
ALGORITHM="${ALGORITHM:-ns3::FdashClient}"
PROTOCOL="${PROTOCOL:-QUIC}"
RUNS_PER_METADATA="${RUNS_PER_METADATA:-5}"
RUN_OFFSET="${RUN_OFFSET:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
VIDEO_GLOB="${VIDEO_GLOB:-*_segments.csv}"
# 0 = submit full array with no Slurm % cap (all tasks may run at once, cluster QOS permitting).
# N>0 = at most N concurrent array tasks (e.g. MAX_PARALLEL=5 sbatch ...).
MAX_PARALLEL="${MAX_PARALLEL:-0}"
NUM_RELAY="${NUM_RELAY:-1}"
NUM_UES="${NUM_UES:-10}"
AI_ENABLE="${AI_ENABLE:-false}"
# S1-U feeder rates to sweep (comma list). Default = uncongested only (back-compat).
# Congestion-diverse campaign: LINK_RATES="100Gb/s,36Mbps,18Mbps,9Mbps,4.5Mbps" NUM_UES=3
LINK_RATES="${LINK_RATES:-100Gb/s}"

# shellcheck source=/dev/null
source "${NS3_REPO}/scratch/ai_llm/algo_paths.sh"
ARTIFACTS="$(artifacts_root "${PROTOCOL}")"
MATRIX_DIR="${NS3_REPO}/${ARTIFACTS}/job_matrix"
mkdir -p "${MATRIX_DIR}"
MATRIX_CSV="${MATRIX_DIR}/job_matrix_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}.csv"

python3 "${NS3_REPO}/scratch/ai_llm/build_job_matrix.py" \
  --metadata-dir "${METADATA_DIR}" \
  --output "${MATRIX_CSV}" \
  --runs-per-metadata "${RUNS_PER_METADATA}" \
  --run-offset "${RUN_OFFSET}" \
  --num-relay "${NUM_RELAY}" \
  --num-ues "${NUM_UES}" \
  --algorithm "${ALGORITHM}" \
  --video-glob "${VIDEO_GLOB}" \
  --max-videos "${MAX_VIDEOS}" \
  --link-rates "${LINK_RATES}" \
  ${VIDEO_LIST:+--video-list "${VIDEO_LIST}"}

N=$(($(wc -l < "${MATRIX_CSV}") - 1))
echo "Algorithm: ${ALGORITHM}  numUes=${NUM_UES}  linkRates=[${LINK_RATES}]"
echo "Matrix: ${MATRIX_CSV} (${N} worker tasks)"

# Build once before the array so parallel workers do not race on waf.
# SKIP_BUILD=1 skips it (use when several launchers run concurrently and the binary was already
# built once on the login node — avoids racing the waf lock).
export_ns3_runtime "${NS3_REPO}"
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "SKIP_BUILD=1 — using pre-built binary (no waf build)."
else
  echo "Building ntn-iab-quic-dash-ai ..."
  python3 "${NS3_REPO}/waf" build
fi

if [[ "${MAX_PARALLEL}" -gt 0 ]]; then
  ARRAY_SPEC="1-${N}%${MAX_PARALLEL}"
  echo "Submitting array ${ARRAY_SPEC} (max ${MAX_PARALLEL} concurrent)"
else
  ARRAY_SPEC="1-${N}"
  echo "Submitting array ${ARRAY_SPEC} (all parallel, no Slurm throttle)"
fi

# Worker #SBATCH --time=24:00:00 — sized for max ~60 segments, simStopTime<=600s, 10 UEs, heavy traces.
sbatch --array="${ARRAY_SPEC}" \
  --export=ALL,NS3_REPO="${NS3_REPO}",JOB_MATRIX="${MATRIX_CSV}",PROTOCOL="${PROTOCOL}",AI_ENABLE="${AI_ENABLE}",MAX_VIDEO_DURATION="${MAX_VIDEO_DURATION:-0}",ALGO_SHORT_OVERRIDE="${ALGO_SHORT_OVERRIDE:-}",SKIP_BUILD="${SKIP_BUILD:-0}" \
  "${NS3_REPO}/dash_trace_worker.slurm"

echo "Launcher done. Monitor: squeue -u \$USER -n dash_trace_worker"
