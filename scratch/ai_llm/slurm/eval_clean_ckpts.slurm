#!/bin/bash
# Offline evaluation of the Phase-2 clean retrain vs the served model, on the CLEAN held-out splits.
#
# Each checkpoint is evaluated with the scaler it was TRAINED with (served = old semantic scaler,
# clean = feature_scaler_semantic_clean.json); labels/QoE are the clean greedy teacher. Reports
# evaluate.py's imitation metrics (acc / macro-F1 / per-rung recall) and the replayed QoE vs the
# teacher replay, per split. Submit from the repo root:
#   sbatch scratch/ai_llm/slurm/eval_clean_ckpts.slurm
#
#SBATCH --job-name=eval-clean
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm-%x-%j.out
set -uo pipefail

source "${LLM_VENV:-$HOME/abr-llm-venv}/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scratch/ai_llm"

DATA="${DATA:-../../train_jsonl_clean}"
OUT="${OUT:-benchmark/eval_clean_ckpts.txt}"
: > "$OUT"

run_eval() {  # <ckpt> <semantic scaler> <split>
  local ck="$1" sem="$2" split="$3"
  echo "##### ckpt=$ck semantic_scaler=$sem split=$split" | tee -a "$OUT"
  python3 evaluate.py --checkpoint "$ck" --data-dir "$DATA/$split" --glob '*.jsonl' \
    --semantic-scaler "$sem" --network-scaler config/feature_scaler_network.json \
    --tag "$(basename "$(dirname "$ck")")/$(basename "$ck" .pt)/$split" 2>&1 \
    | grep -vE 'UserWarning|warnings.warn|Loading weights|deprecated' | tee -a "$OUT"
}

for split in test val; do
  for s in 41 42 43; do
    [ -f "checkpoints_bola_down/abr_v2_seed$s.pt" ] && run_eval "checkpoints_bola_down/abr_v2_seed$s.pt" config/feature_scaler_semantic.json "$split"
    [ -f "checkpoints_clean/abr_v2_seed$s.pt" ] && run_eval "checkpoints_clean/abr_v2_seed$s.pt" config/feature_scaler_semantic_clean.json "$split"
  done
done
echo "wrote $OUT"
