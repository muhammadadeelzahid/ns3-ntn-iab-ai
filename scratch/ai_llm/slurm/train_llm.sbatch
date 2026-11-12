#!/bin/bash
# Train an LLM-backbone ABR variant (V1/V2) on a Grex GPU node.
#
# Grex GPU partitions (sinfo): agro/agro-b (A30 x2), mcordgpu (A30 x4), lgpu (L40s x2),
# gpu/stamps (V100 x4), livi (V100 16GB). We target **L40s (lgpu)** (Ada, native bf16, accessible to def-ekram).
# which Volta V100 does NOT support — so the V100 partitions are unusable here. A30 (Ampere)
# and L40s (Ada) both do bf16 natively. Qwen2.5-1.5B + bf16 LoRA fits easily in A30's 248 GB.
#
# Env: a self-contained venv built off system /usr/bin/python3.11 with a cu124 torch wheel
# (bundles its own CUDA) — NO modules needed (the Alliance CCEnv/StdEnv stack errors on this
# Grex). Default adaptation is **bf16 LoRA** (no bitsandbytes); pass QLORA=1 to opt into 4-bit.
#
# Submit:  sbatch train_llm.sbatch v2     # or: sbatch train_llm.sbatch v1
#
#SBATCH --job-name=abr-llm
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

VARIANT="${1:-v2}"

# Modules: none by default (self-contained venv). Override LLM_MODULES only if your venv
# depends on a Grex-native module on the compute node.
if [ -n "${LLM_MODULES-}" ]; then module load ${LLM_MODULES-}; fi
# Activate the CUDA-enabled LLM venv (cu124 torch + transformers/peft). NOT the CPU .venv.
source "${LLM_VENV:-$HOME/abr-llm-venv}/bin/activate"

# Grex compute nodes have no internet, so use the model cache pre-downloaded on the login
# node (shared /home). The model must already be in ~/.cache/huggingface.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Slurm runs a spooled copy so $0 is unreliable; use the submission dir (submit from the
# repo root). Override REPO_ROOT if you submit from elsewhere.
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scratch/ai_llm"

# bf16 LoRA by default; QLORA=1 opts into 4-bit (needs bitsandbytes in the venv).
QLORA_FLAG=""
if [ "${QLORA:-0}" = "1" ] && { [ "$VARIANT" = "v2" ] || [ "$VARIANT" = "v3" ]; }; then
  QLORA_FLAG="--qlora-4bit"
fi

# Data dirs + checkpoint dir are overridable so the held-out-genre experiment (E2) can target a
# separate split (e.g. DATA_DIR=../../train_jsonl_holdout/train CKPT_DIR=checkpoints_holdout)
# without touching the in-distribution split or its checkpoints. V3 reads its language strings
# from config/video_semantics.json (the train.py default), shared across both experiments.
python3 train.py \
  --data-dir "${DATA_DIR:-../../train_jsonl/train}" \
  --val-data-dir "${VAL_DATA_DIR:-../../train_jsonl/val}" \
  --network-scaler "${NETWORK_SCALER:-config/feature_scaler_network.json}" \
  --semantic-scaler "${SEMANTIC_SCALER:-config/feature_scaler_semantic.json}" \
  --glob '*.jsonl' \
  --variant "$VARIANT" \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --soft-tokens "${SOFT_TOKENS:-4}" \
  --window "${WINDOW:-1}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size 32 \
  --lr 1e-3 \
  --lr-encoder 3e-2 \
  --lr-softproj 1e-2 \
  --lr-llm 3e-4 \
  --class-weight-power "${CWP:-0.0}" \
  ${BALANCED:+--balanced-sampling} \
  --scheduler cosine \
  --warmup-frac 0.05 \
  --early-stop-patience "${PATIENCE:-25}" \
  $QLORA_FLAG \
  ${EXTRA_ARGS:-} \
  --checkpoint-out "${CKPT_DIR:-checkpoints}/abr_${VARIANT}_seed${SEED:-42}.pt" \
  --seed "${SEED:-42}"
