#!/bin/bash
#SBATCH --job-name=quic_dash
#SBATCH --account=def-ekram
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=quic_%j.out
#SBATCH --error=quic_%j.err

cd "${SLURM_SUBMIT_DIR:-/home/adeel/projects/def-ekram/adeel/ns3-ntn-iab}"

export LD_LIBRARY_PATH="${PWD}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python3 scratch/run_quic_bbr_ai_recv.py --show-waf-output