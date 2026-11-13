#!/bin/bash
# Offline evaluation of the model-v3 training grid (checkpoints_v3_<label>/abr_v2_seed{41,42,43}.pt) on the
# v3 VAL split. Every checkpoint is scored on merged_greedy/val (greedy-label val = the common reference for
# replay QoE); K5-label checkpoints are additionally scored on their own label set for imitation metrics.
# Submit from the repo root:   sbatch scratch/ai_llm/slurm/eval_v3_ckpts.slurm
#   LABELS="greedy k5estimate" restricts the label sets; missing checkpoints are skipped.
#
#SBATCH --job-name=eval-v3
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail

source "${LLM_VENV:-$HOME/abr-llm-venv}/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scratch/ai_llm"

DATA="${DATA:-../../train_jsonl_v3}"
OUT="${OUT:-benchmark/eval_v3_ckpts.txt}"
SEM=config/feature_scaler_semantic_clean.json
: > "$OUT"

run_eval() {  # <ckpt> <split>
  local ck="$1" split="$2" net=config/feature_scaler_network.json
  case "$ck" in *hist*) net=config/feature_scaler_network_v3h.json;; esac   # 8-feature checkpoints
  echo "##### ckpt=$ck split=$split net_scaler=$net" | tee -a "$OUT"
  python3 evaluate.py --checkpoint "$ck" --data-dir "$DATA/$split" --glob '*.jsonl' \
    --semantic-scaler "$SEM" --network-scaler "$net" \
    --tag "$(basename "$(dirname "$ck")")/$(basename "$ck" .pt)/$split" 2>&1 \
    | grep -vE 'UserWarning|warnings.warn|Loading weights|deprecated' | tee -a "$OUT"
}

for lab in ${LABELS:-greedy k5estimate k5robust greedy_dag k5estimate_dag k5est_pf085_dag k5est_pf07_dag k5estimate_dag_hist}; do
  for s in 41 42 43; do
    ck="checkpoints_v3_$lab/abr_v2_seed$s.pt"; [ -f "$ck" ] || continue
    run_eval "$ck" merged_greedy/val
    case "$lab" in k5estimate_dag_hist) run_eval "$ck" "merged_k5estimate/val";; k5*) run_eval "$ck" "merged_${lab%_dag}/val";; esac
  done
done
echo "wrote $OUT"
