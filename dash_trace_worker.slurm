#!/bin/bash
#SBATCH --job-name=dash_trace_worker
#SBATCH --account=def-ekram
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=dash_trace_worker_%A_%a.out
#SBATCH --error=dash_trace_worker_%A_%a.err

set -euo pipefail

NS3_REPO="${NS3_REPO:-/home/adeel/projects/def-ekram/adeel/ns3-ntn-iab}"
JOB_MATRIX="${JOB_MATRIX:?JOB_MATRIX must be set by launcher}"
PROTOCOL="${PROTOCOL:-QUIC}"
AI_ENABLE="${AI_ENABLE:-false}"

# shellcheck source=/dev/null
source "${NS3_REPO}/scratch/ai_llm/algo_paths.sh"
export_ns3_runtime "${NS3_REPO}"
export PYTHONPATH="${NS3_REPO}/scratch:${PYTHONPATH:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
# Line 1 = header; array task 1 -> CSV line 2, etc.
ROW="$((TASK_ID + 1))"
LINE="$(sed -n "${ROW}p" "${JOB_MATRIX}" | tr -d '\r')"
IFS=',' read -r _tid METADATA_CSV VIDEO_NAME RUN_SEED NUM_RELAY NUM_UES ALGORITHM LINK_RATE <<< "${LINE}"

METADATA_CSV="$(strip_cr "${METADATA_CSV//\"/}")"
VIDEO_NAME="$(strip_cr "${VIDEO_NAME//\"/}")"
ALGORITHM="$(strip_cr "${ALGORITHM//\"/}")"
RUN_SEED="$(strip_cr "${RUN_SEED}")"
NUM_RELAY="$(strip_cr "${NUM_RELAY}")"
NUM_UES="$(strip_cr "${NUM_UES}")"
# link_rate column is optional (older matrices lack it); empty -> scenario default (100Gb/s, uncongested)
LINK_RATE="$(strip_cr "${LINK_RATE//\"/}")"

# ALGO_SHORT_OVERRIDE=<Name> writes to <artifacts>/<Name>/ instead of the class-derived stem (e.g. a
# full-length re-run of an algorithm whose truncated collection runs must stay untouched).
ALGO_SHORT="${ALGO_SHORT_OVERRIDE:-$(algo_short_from_ns3 "${ALGORITHM}")}"
ARTIFACTS="$(artifacts_root "${PROTOCOL}")"
RUN_DIR="$(run_artifacts_dir "${NS3_REPO}" "${ARTIFACTS}" "${ALGO_SHORT}" "${VIDEO_NAME}" "${RUN_SEED}")"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

echo "Worker task ${TASK_ID}: algo=${ALGORITHM} video=${VIDEO_NAME} run_${RUN_SEED}"
echo "RUN_DIR=${RUN_DIR}"

cat > run_manifest.json <<EOF
{
  "task_id": ${TASK_ID},
  "metadata_csv": "${METADATA_CSV}",
  "video_name": "${VIDEO_NAME}",
  "run_seed": ${RUN_SEED},
  "num_relay": ${NUM_RELAY},
  "num_ues": ${NUM_UES},
  "algorithm": "${ALGORITHM}",
  "algo_short": "${ALGO_SHORT}",
  "protocol": "${PROTOCOL}",
  "link_rate": "${LINK_RATE}"
}
EOF

if [[ "${TASK_ID}" -eq 1 && "${SKIP_BUILD:-0}" != "1" ]]; then   # SKIP_BUILD=1: binary already built on the login node
  python3 "${NS3_REPO}/waf" build
fi

SIM_BIN="${NS3_REPO}/build/scratch/ntn-iab-quic-dash-ai"
if [[ ! -x "${SIM_BIN}" ]]; then
  echo "ERROR: simulator binary not found: ${SIM_BIN}" >&2
  exit 1
fi

# Simulated seconds: ns-3 uses ~3.3 s per segment (100 MPEG frames @ 33 ms).
# ABR metadata uses 2 s/segment; longest catalog clips are ~120 s (60 segments).
NSEG=$(($(wc -l < "${METADATA_CSV}") - 1))
NS3_SEG_S=4  # ceil(3.3) + small margin per segment
# Upper clamp on sim duration. Wall time is ~linear in simStopTime (the real speed lever, since
# numUes does NOT drive cost), so the congestion campaign caps it low (e.g. SIM_STOP_CAP=120):
# ~120 s still delivers ~25-30 segments/UE → fills the 8-step window + gives congested decisions.
SIM_STOP_CAP="${SIM_STOP_CAP:-600}"
SIM_STOP_TIME="${SIM_STOP_TIME:-0}"
if [[ "${SIM_STOP_TIME}" == "0" || "${SIM_STOP_TIME}" == "0.0" ]]; then
  SIM_STOP_TIME=$((NSEG * NS3_SEG_S + 30))
  if [[ "${SIM_STOP_TIME}" -lt 90 ]]; then
    SIM_STOP_TIME=90
  fi
  if [[ "${SIM_STOP_TIME}" -gt "${SIM_STOP_CAP}" ]]; then
    SIM_STOP_TIME="${SIM_STOP_CAP}"
  fi
fi
echo "segments=${NSEG} simStopTime=${SIM_STOP_TIME}s numUes=${NUM_UES} linkRate=${LINK_RATE:-<default>}"

SIM_ARGS=(
  --numRelay="${NUM_RELAY}"
  --numUes="${NUM_UES}"
  --algorithm="${ALGORITHM}"
  --dashMetadataCsv="${METADATA_CSV}"
  --dashMetadataTrace=dash_metadata_segments.csv
  --run="${RUN_SEED}"
  --aiEnable="${AI_ENABLE}"
  --simStopTime="${SIM_STOP_TIME}"
)
# S1-U feeder rate (congestion knob). Only pass when set, so older matrices keep the scenario default.
if [[ -n "${LINK_RATE}" ]]; then
  SIM_ARGS+=(--linkRate="${LINK_RATE}")
fi
if [[ "${MAX_VIDEO_DURATION:-0}" != "0" ]]; then
  SIM_ARGS+=(--maxVideoDuration="${MAX_VIDEO_DURATION}")
fi
# ns-3 attribute defaults (space-separated ns3::Type::Attr=value, handled by the scenario CommandLine).
# Default = hold the FFT-codebook beamforming cache for the whole session, the same setting the bridge
# launchers (scratch/ai_llm/slurm/bench_*_packed.sbatch) use and the rationale documented there: static
# UEs + channel matrices generated once (UpdatePeriod 0; the moving satellite only changes path loss)
# -> identical beam design every slot; byte-identical output, ~8.5x faster.
# NS3_ATTRS="" disables (export it before sbatch; the launcher passes --export=ALL).
NS3_ATTRS="${NS3_ATTRS-ns3::MmWaveFFTCodebookBeamforming::CacheExpiry=1000s}"
for _attr in ${NS3_ATTRS}; do SIM_ARGS+=("--${_attr}"); done
echo "ns3 attribute defaults: ${NS3_ATTRS:-none}"

"${SIM_BIN}" "${SIM_ARGS[@]}" 2>&1 | tee sim.log

python3 "${NS3_REPO}/scratch/ai_llm/extract_trace.py" \
  --run-dir "${RUN_DIR}" \
  --metadata "${METADATA_CSV}" \
  --output "${RUN_DIR}/trace_decisions.jsonl" \
  --run-id "run_${RUN_SEED}"

python3 "${NS3_REPO}/scripts/validate_extracted_trace.py" \
  --jsonl "${RUN_DIR}/trace_decisions.jsonl" \
  --metadata "${METADATA_CSV}" || true

python3 "${NS3_REPO}/scratch/ai_llm/compute_run_metrics.py" \
  --run-dir "${RUN_DIR}"

echo "Worker ${TASK_ID} complete: ${RUN_DIR}"
