#!/bin/bash
# Per-segment SSIM/PSNR (+ vmaf_min) for the viewer bundles: scratch/ai_llm/viewer_quality.py.
# Submit from the repo root: VIDEO=chef_cooking OUT=viewer_bundles/phasef_seed203 sbatch scratch/ai_llm/slurm/viewer_quality.sbatch
#SBATCH --job-name=viewer_quality
#SBATCH --account=def-ekram
#SBATCH --partition=skylake
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-viewer-quality-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load python/3.11.8 ffmpeg/7.0.2
~/abr-llm-venv/bin/python scratch/ai_llm/viewer_quality.py --video "${VIDEO:?}" --out "${OUT:?}" --jobs 4
