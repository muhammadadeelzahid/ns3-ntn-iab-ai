#!/bin/bash
# QoE + imitation evaluation of a trained checkpoint on an L40S GPU (needed for V1/V2,
# which load the LLM). evaluate.py auto-detects the variant from the checkpoint.
#
# Usage:  sbatch eval_gpu.sbatch checkpoints/abr_v2_seed42.pt "V2 in-dist all-data"
#         sbatch eval_gpu.sbatch checkpoints/abr_v2_seed42.pt "V2 unseen-content" --include-video <name> ...
#
# Route output to a predictable file (overrides the #SBATCH --output directive):
#         sbatch --output=eval_v2_seed42.out eval_gpu.sbatch checkpoints/abr_v2_seed42.pt v2_seed42
# Restrict to a held-out video set without editing args (space- or comma-separated):
#         EVAL_INCLUDE_VIDEO="sintel hiking" sbatch eval_gpu.sbatch <ckpt> <tag>
#         EVAL_EXCLUDE_VIDEO="big_buck_bunny" sbatch eval_gpu.sbatch <ckpt> <tag>
# Eval on a different split (default is the held-out test set):
#         EVAL_DATA_DIR=../../train_jsonl/val sbatch eval_gpu.sbatch <ckpt> <tag>
#
#SBATCH --job-name=abr-eval
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "${LLM_VENV:-$HOME/abr-llm-venv}/bin/activate"
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scratch/ai_llm"

CKPT="${1:?pass a checkpoint path relative to scratch/ai_llm}"
TAG="${2:-eval}"
shift 2 || true   # remaining args (e.g. --include-video X) pass through

# Default to the held-out test split (the leakage-free headline). Override with EVAL_DATA_DIR.
EVAL_DATA_DIR="${EVAL_DATA_DIR:-../../train_jsonl/test}"

# Optional held-out video filter from the environment (space- or comma-separated -> repeated
# --include-video / --exclude-video flags). evaluate.py treats both as repeatable.
# Default to empty first (so `set -u` is satisfied when unset), then split on comma/space.
EVAL_INCLUDE_VIDEO="${EVAL_INCLUDE_VIDEO:-}"
EVAL_EXCLUDE_VIDEO="${EVAL_EXCLUDE_VIDEO:-}"
EVAL_ARGS=()
for vid in ${EVAL_INCLUDE_VIDEO//,/ }; do EVAL_ARGS+=(--include-video "$vid"); done
for vid in ${EVAL_EXCLUDE_VIDEO//,/ }; do EVAL_ARGS+=(--exclude-video "$vid"); done

# Scalers must match what the checkpoint trained with. Default to config/ (E1); the held-out-genre
# experiment (E2) overrides these to config_holdout/ so normalization is identical to E2 training.
NETWORK_SCALER="${NETWORK_SCALER:-config/feature_scaler_network.json}"
SEMANTIC_SCALER="${SEMANTIC_SCALER:-config/feature_scaler_semantic.json}"

python3 evaluate.py \
  --data-dir "$EVAL_DATA_DIR" \
  --glob '*.jsonl' \
  --policy model \
  --checkpoint "$CKPT" \
  --network-scaler "$NETWORK_SCALER" \
  --semantic-scaler "$SEMANTIC_SCALER" \
  --tag "$TAG" \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
  "$@"
