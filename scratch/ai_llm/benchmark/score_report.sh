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

# TEACHER=greedy (default): the greedy down-only causal teacher from config/oracle_reward.json - the
# %-of-oracle reference every arm must share (benchmark-plan.md 4.0; = <ARM>_teacher_down /
# AiDash/shards_down / score_bridge_arm.sh). Needs no dp_oracle labels. TEACHER=oracle = the old acausal
# DP-ceiling reference (needs oracle_labels_state.jsonl per video from dp_oracle.slurm).
TEACHER="${TEACHER:-greedy}"
echo "[score] building teacher jsonl for ${ALGO} (teacher=${TEACHER}) ..."
rm -rf "${TEACH}"
python3 scratch/ai_llm/dataset_builder.py \
  --artifacts-root Quic_artifacts --algo-short "${ALGO}" \
  --metadata-dir "${META}" --output-dir "${TEACH}" \
  --teacher-source "${TEACHER}" --merge-output "${TEACH}/all.jsonl"

# Quic_artifacts/<ALGO> also holds the training-campaign videos (non-test). The benchmark is the held-out
# test split ONLY, so prune the teacher-dir to the test videos — perceptual_qoe only scores runs with a
# matching teacher shard, so this isolates the benchmark and keeps every arm comparable. Default = the
# full 26-video stage-B test split (test_videos_26.txt); the older 14/10-video lists stay selectable.
TEST_LIST="${TEST_LIST:-${REPO}/scratch/ai_llm/benchmark/test_videos_26.txt}"
echo "[score] pruning teacher-dir to $(wc -l < "${TEST_LIST}") test videos ..."
rm -f "${TEACH}/all.jsonl"
for shard in "${TEACH}"/*_run_*.jsonl; do
  [ -e "${shard}" ] || continue
  vid="$(basename "${shard}")"; vid="${vid%_run_*}"
  grep -qxF "${vid}" "${TEST_LIST}" || rm -f "${shard}"
done
echo "[score] teacher shards kept: $(ls "${TEACH}"/*_run_*.jsonl 2>/dev/null | wc -l)"
if [ "${TEACHER}" = "greedy" ]; then
  # Persist the scoring shards where absolute_qoe_table / per_video_qoe_ci read them.
  KEEP="Quic_artifacts/${ALGO%_dscore}_teacher_down"
  rm -rf "${KEEP}"; mkdir -p "${KEEP}"; cp "${TEACH}"/*_run_*.jsonl "${KEEP}"/
  echo "[score] teacher shards persisted -> ${KEEP}"
fi

echo "[score] scoring ${ALGO} runs (test split only) ..."
python3 scratch/ai_llm/perceptual_qoe.py \
  --run-dir "Quic_artifacts/${ALGO}" \
  --teacher-dir "${TEACH}" \
  --reward-config scratch/ai_llm/config/oracle_reward.json \
  | tee "${RESDIR}/${ALGO}.txt"

echo "[score] wrote ${RESDIR}/${ALGO}.txt"
