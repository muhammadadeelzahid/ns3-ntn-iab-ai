#!/bin/bash
# Submit the full benchmark sweep: V0 (CPU) + V2/V3/V1 (GPU) across seeds.
#
# V0 -> train_v0.sbatch on a CPU partition; V1/V2/V3 -> train_llm.sbatch on a GPU partition.
# Each variant is trained at seeds 41/42/43 so results can be reported as mean +/- std
# (docs/benchmark-plan.md §3). Checkpoints land in scratch/ai_llm/${CKPT_SUBDIR}/.
#
# Each submission is recorded in a manifest CSV so a downstream stage (pipeline_chain.slurm)
# can submit a dependency-gated evaluation per checkpoint. The manifest path is echoed last
# (and exposed as SWEEP_MANIFEST in the environment of a sourcing caller).
#
# Usage:   bash sweep.sh               # all variants, seeds 41 42 43
#          bash sweep.sh v2 v3         # only v2+v3, all seeds
#          SEEDS="42" bash sweep.sh    # single seed, all variants
#
# Held-out-genre experiment (E2): point training at the holdout split + a separate checkpoint
# subdir so the in-distribution split/checkpoints are untouched (DATA_DIR/VAL_DATA_DIR/CKPT_DIR
# are honored by the sbatch scripts):
#   DATA_DIR=../../train_jsonl_holdout/train VAL_DATA_DIR=../../train_jsonl_holdout/val \
#   CKPT_SUBDIR=checkpoints_holdout SEEDS="41 42 43 44 45 46 47 48" bash sweep.sh v0 v2 v3

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

read -r -a VARIANTS <<< "${*:-v0 v2 v1}"
read -r -a SEEDS <<< "${SEEDS:-41 42 43}"

# Checkpoint subdir relative to scratch/ai_llm (the dir train_*.sbatch cd into). E2 overrides to
# checkpoints_holdout; passed to the sbatch scripts as CKPT_DIR. DATA_DIR/VAL_DATA_DIR (if set)
# are inherited by the sbatch scripts from this shell's environment.
CKPT_SUBDIR="${CKPT_SUBDIR:-checkpoints}"

# Manifest: one row per submitted training job. Columns:
#   variant,seed,train_jobid,checkpoint_path
# checkpoint_path is relative to scratch/ai_llm (the dir eval_gpu.sbatch / train_*.sbatch cd into).
CKPT_DIR_ABS="$(cd "$HERE/.." && pwd)/${CKPT_SUBDIR}"
mkdir -p "$CKPT_DIR_ABS"
SWEEP_MANIFEST="${SWEEP_MANIFEST:-$CKPT_DIR_ABS/sweep_manifest_$(date +%Y%m%d_%H%M%S).csv}"
echo "variant,seed,train_jobid,checkpoint_path" > "$SWEEP_MANIFEST"

for seed in "${SEEDS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    ckpt="${CKPT_SUBDIR}/abr_${v}_seed${seed}.pt"
    case "$v" in
      v0)
        echo "submit v0 seed=$seed (CPU) -> $ckpt"
        jobid="$(SEED="$seed" CKPT_DIR="$CKPT_SUBDIR" sbatch --parsable "$HERE/train_v0.sbatch")"
        ;;
      v1|v2|v3)
        echo "submit $v seed=$seed (GPU) -> $ckpt"
        jobid="$(SEED="$seed" CKPT_DIR="$CKPT_SUBDIR" sbatch --parsable "$HERE/train_llm.sbatch" "$v")"
        ;;
      *)
        echo "unknown variant: $v (expected v0|v1|v2|v3)" >&2
        exit 1
        ;;
    esac
    echo "${v},${seed},${jobid},${ckpt}" >> "$SWEEP_MANIFEST"
    echo "  -> jobid=${jobid} ckpt=${ckpt}"
  done
done

echo "SWEEP_MANIFEST=${SWEEP_MANIFEST}"
export SWEEP_MANIFEST
