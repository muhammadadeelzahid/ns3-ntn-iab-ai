#!/bin/bash
# Fast GPU validation on an L40s (lgpu) node before launching real training:
#   1) GPU visible + bf16 supported (L40s = Ada)
#   2) the real Qwen2.5-1.5B loads from the offline HF cache
#   3) one V2 (bf16 LoRA) training epoch on a single shard runs forward+backward
# Should finish in a few minutes. Submit:  sbatch smoke_gpu.sbatch
#
#SBATCH --job-name=abr-smoke
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

source "${LLM_VENV:-$HOME/abr-llm-venv}/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scratch/ai_llm"

echo "=== GPU / bf16 check ==="
python3 -c "
import torch
print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('bf16 supported:', torch.cuda.is_bf16_supported())
"

echo "=== V2 one-epoch single-shard training (bf16 LoRA) ==="
python3 train.py \
  --data-dir ../../train_jsonl \
  --glob 'aerial_ocean_waves_run_1.jsonl' \
  --variant v2 \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --soft-tokens 4 \
  --window 8 \
  --epochs 1 \
  --batch-size 8 \
  --scheduler cosine \
  --checkpoint-out /tmp/abr_v2_smoke.pt

echo "=== SMOKE_GPU_DONE ==="
