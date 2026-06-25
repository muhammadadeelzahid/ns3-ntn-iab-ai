#!/bin/bash
# Score one benchmark arm's completed ns-3 runs into a perceptual %-of-oracle table.
#
# PREREQ (run first, per arm, via the existing slurm scripts — they produce throughput_trace.jsonl
# + oracle_labels_state.jsonl that this script consumes):
#   ALGO_SHORT=<X> sbatch extract_throughput_traces.slurm
#   ALGO_SHORT=<X> sbatch --dependency=afterok:<extract> dp_oracle.slurm     # uses w_rung=0 config
#
# THEN:  bash scratch/ai_llm/benchmark/score_report.sh <ALGO_SHORT>
# e.g.   bash scratch/ai_llm/benchmark/score_report.sh Fdash
#        bash scratch/ai_llm/benchmark/score_report.sh AiDash      # the V2 (LLM) arm
#
# Output: scratch/ai_llm/benchmark/results/<ALGO_SHORT>.txt  (per-run + pooled %oracle/regret/VMAF/...)
set -euo pipefail

ALGO="${1:?usage: score_report.sh <ALGO_SHORT>  (e.g. Fdash, Osmp, ..., AiDash)}"
REPO="${REPO_ROOT:-/project/6030214/adeel/ns3-ntn-iab}"
META="${METADIR:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata}"
TEACH="/tmp/${ALGO}_teacher"
RESDIR="${REPO}/scratch/ai_llm/benchmark/results"
mkdir -p "${RESDIR}"

cd "${REPO}"
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/scratch:${PYTHONPATH:-}"

echo "[score] building teacher jsonl for ${ALGO} ..."
rm -rf "${TEACH}"
python3 scratch/ai_llm/dataset_builder.py \
  --artifacts-root Quic_artifacts --algo-short "${ALGO}" \
  --metadata-dir "${META}" --output-dir "${TEACH}" \
  --teacher-source oracle --merge-output "${TEACH}/all.jsonl"

# Quic_artifacts/<ALGO> also holds leftover OLD training-campaign videos (~91 non-test). The benchmark
# is the held-out 14-video test split ONLY (and the V2/AiDash arm ran exactly those), so prune the
# teacher-dir to the test videos — perceptual_qoe only scores runs with a matching teacher shard, so
# this isolates the benchmark and keeps every arm comparable (same 14 videos x 5 conditions).
TEST_LIST="${TEST_LIST:-${REPO}/scratch/ai_llm/benchmark/test_videos.txt}"
echo "[score] pruning teacher-dir to $(wc -l < "${TEST_LIST}") test videos ..."
rm -f "${TEACH}/all.jsonl"
for shard in "${TEACH}"/*_run_*.jsonl; do
  [ -e "${shard}" ] || continue
  vid="$(basename "${shard}")"; vid="${vid%_run_*}"
  grep -qxF "${vid}" "${TEST_LIST}" || rm -f "${shard}"
done
echo "[score] teacher shards kept: $(ls "${TEACH}"/*_run_*.jsonl 2>/dev/null | wc -l)"

echo "[score] scoring ${ALGO} runs (test split only) ..."
python3 scratch/ai_llm/perceptual_qoe.py \
  --run-dir "Quic_artifacts/${ALGO}" \
  --teacher-dir "${TEACH}" \
  --reward-config scratch/ai_llm/config/oracle_reward.json \
  | tee "${RESDIR}/${ALGO}.txt"

echo "[score] wrote ${RESDIR}/${ALGO}.txt"
