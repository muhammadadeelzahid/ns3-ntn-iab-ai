#!/bin/bash
# Stage C — live ns-3 closed-loop QoE ablation for V0/V1/V2 (same video, all UEs).
#
# Runs each variant as its OWN complete ns-3 session, SEQUENTIALLY (the ns3-ai shared-memory
# segment uses fixed names, so concurrent runs on one node would collide), then aggregates QoE.
# V1/V2 need a GPU (the 3.11 inference server loads Qwen); V0 runs on the same node harmlessly.
#
# The 3.6 client (ns-3 + ns3-ai binding) spawns the 3.11 torch inference server per run — the
# Python-version bridge. Identical numUes / seed / sim-stop / video across variants -> fair compare.
#
# Submit:  sbatch live_qoe_ablation.sbatch          # v0 v1 v2, 10 UEs, hiking
#          VARIANTS="v0 v2" NUMUES=5 sbatch live_qoe_ablation.sbatch
#
#SBATCH --job-name=abr-liveqoe
#SBATCH --partition=lgpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6000M
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
export WAF_CMD="python3 ./waf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

VARIANTS="${VARIANTS:-v0 v1 v2}"
SEED="${SEED:-42}"
NUMUES="${NUMUES:-10}"
RUN="${RUN:-1}"
SIMSTOP="${SIMSTOP:-75}"
META="${META:-/home/adeel/projects/def-ekram/adeel/abr/output/metadata/hiking_segments.csv}"
SERVER_PY="${SERVER_PY:-$HOME/abr-llm-venv/bin/python}"
CKPT_DIR="${CKPT_DIR:-scratch/ai_llm/checkpoints}"
# LINKRATE narrows the backhaul bottleneck (e.g. 20Mbps) for a congested stress test; unset = the
# scenario default (100Gb/s, uncongested). OUT_ROOT defaults include the link rate so congested and
# baseline runs land in separate dirs.
LINKRATE="${LINKRATE:-}"
LINK_TAG="$([ -n "$LINKRATE" ] && echo "${LINKRATE}" || echo "uncongested")"
OUT_ROOT="${OUT_ROOT:-Quic_artifacts/AiDash/live_qoe_${LINK_TAG}}"
LINK_ARGS=()
[ -n "$LINKRATE" ] && LINK_ARGS+=(--link-rate "$LINKRATE")

DECISION_ARGS=()
for v in $VARIANTS; do
  ckpt="${CKPT_DIR}/abr_${v}_seed${SEED}.pt"
  out="${OUT_ROOT}/${v}_seed${SEED}_run${RUN}/decisions.jsonl"
  echo "===== live run: variant=$v ckpt=$ckpt numUes=$NUMUES simStop=$SIMSTOP linkRate=${LINKRATE:-default} ====="
  /usr/bin/python3 scratch/run_ai_dash_llm.py \
    --checkpoint "$ckpt" \
    --metadata "$META" \
    --numUes "$NUMUES" --run "$RUN" --sim-stop-time "$SIMSTOP" \
    --server-python "$SERVER_PY" --device cuda \
    "${LINK_ARGS[@]}" \
    --decisions-out "$out"
  echo "===== done variant=$v -> $out ====="
  DECISION_ARGS+=("${v}:${out}")
done

echo ""
echo "########## CLOSED-LOOP QoE COMPARISON ##########"
/usr/bin/python3 scratch/ai_llm/eval_qoe.py "${DECISION_ARGS[@]}" --per-ue
