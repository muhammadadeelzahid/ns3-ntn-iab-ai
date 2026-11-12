#!/bin/bash
# Train the V0 baseline (no-LLM encoder -> head) on a Grex CPU node.
#
# V0 is ~0.1-1 M params and sub-millisecond per decision: it has NO GPU/transformers
# dependency, so it runs on a regular CPU partition with the project's CPU torch venv.
# (V1/V2 instead use train_llm.sbatch on a GPU partition.)
#
# Submit:  SEED=42 sbatch train_v0.sbatch
#
#SBATCH --job-name=abr-v0
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

# Cluster modules. The Alliance/CC stack (CCEnv + StdEnv) currently errors on this Grex, and
# the CPU .venv ran fine on the login node with no modules, so we load NOTHING by default.
# Set V0_MODULES if your venv needs a Grex-native python module on the compute node.
if [ -n "${V0_MODULES-}" ]; then module load ${V0_MODULES-}; fi

# Slurm runs a spooled copy so $0 is unreliable; use the submission dir (submit from the
# repo root). Override REPO_ROOT if you submit from elsewhere.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# CPU torch venv (the project .venv at the repo root); override with V0_VENV.
source "${V0_VENV:-$REPO/.venv}/bin/activate"
cd "$REPO/scratch/ai_llm"

# Data dirs + checkpoint dir are overridable so the held-out-genre experiment (E2) can target a
# separate split (e.g. DATA_DIR=../../train_jsonl_holdout/train CKPT_DIR=checkpoints_holdout)
# without touching the in-distribution split or its checkpoints.
python3 train.py \
  --data-dir "${DATA_DIR:-../../train_jsonl/train}" \
  --val-data-dir "${VAL_DATA_DIR:-../../train_jsonl/val}" \
  --network-scaler "${NETWORK_SCALER:-config/feature_scaler_network.json}" \
  --semantic-scaler "${SEMANTIC_SCALER:-config/feature_scaler_semantic.json}" \
  --glob '*.jsonl' \
  --variant v0 \
  --window 8 \
  --epochs 25 \
  --batch-size 64 \
  --lr 1e-3 \
  --lr-encoder 3e-2 \
  --class-weight-power 0.0 \
  --scheduler cosine \
  --warmup-frac 0.05 \
  --early-stop-patience 6 \
  --checkpoint-out "${CKPT_DIR:-checkpoints}/abr_v0_seed${SEED:-42}.pt" \
  --seed "${SEED:-42}"
