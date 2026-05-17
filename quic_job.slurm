#!/bin/bash
# Slurm job: DASH application-layer ABR via ns3-ai (run_ai_dash_heuristic.py).
#
# Author: Muhammad Adeel Zahid <adeel.m.zahid@gmail.com>
#         University of Manitoba
# Added: Slurm wrapper for run_ai_dash_heuristic.py (DASH ABR over ns3-ai).
#
#SBATCH --job-name=quic_dash_ai
#SBATCH --account=def-ekram
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=quic_%j.out
#SBATCH --error=quic_%j.err

cd "${SLURM_SUBMIT_DIR:-/home/adeel/projects/def-ekram/adeel/ns3-ntn-iab}"

export LD_LIBRARY_PATH="${PWD}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python3 scratch/run_ai_dash_heuristic.py
