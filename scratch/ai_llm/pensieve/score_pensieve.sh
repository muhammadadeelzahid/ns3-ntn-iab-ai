#!/bin/bash
# Score the Pensieve benchmark arm into results/Pensieve.txt, directly comparable to the
# AiDash/RAAHS/AAASH/BOLA arms, then (optionally) regenerate the paper table.
#
# Chain (same as the AiDash arm, docs/content-network-roadmap.md):
#   rename run dirs -> run_<condIdx>  (so make_paper_table maps conditions like the other arms)
#   decisions_to_throughput  ->  dp_oracle(ALGO_SHORT=Pensieve)  ->  decisions_to_trainrows
#   ->  score_v2_perceptual (writes results/Pensieve.txt)
#
# Usage: bash scratch/ai_llm/pensieve/score_pensieve.sh
set -euo pipefail

REPO="${REPO_ROOT:-/project/6030214/adeel/ns3-ntn-iab}"
META="${METADIR:-/project/6030214/adeel/abr/output/metadata}"
cd "$REPO"

PENS_ROOT="Quic_artifacts/Pensieve"
VLIST="/tmp/pensieve_videos.txt"
TEACH="/tmp/Pensieve_teacher"
RESULTS="scratch/ai_llm/benchmark/results/Pensieve.txt"
REWARD="scratch/ai_llm/config/oracle_reward.json"

# 0. Normalize run-dir names run_<seed>_<rtag> -> run_<condIdx> (COND: 100Gbs=1,36Mbps=2,18Mbps=3,
#    9Mbps=4,4.5Mbps=5), matching the other arms' run_1..run_5 convention. Idempotent.
python3 - <<'PY'
import os, glob, re
cond = {"100Gbs": 1, "36Mbps": 2, "18Mbps": 3, "9Mbps": 4, "4.5Mbps": 5}
for d in glob.glob("Quic_artifacts/Pensieve/*/run_*"):
    if not os.path.isdir(d):
        continue
    mt = re.match(r"run_\d+_(.+)$", os.path.basename(d))
    if not mt:
        continue  # already run_<N>
    rtag = mt.group(1)
    if rtag not in cond:
        raise SystemExit("unknown rtag: %s (%s)" % (rtag, d))
    new = os.path.join(os.path.dirname(d), "run_%d" % cond[rtag])
    if d != new:
        if os.path.exists(new):
            raise SystemExit("rename collision: %s -> %s" % (d, new))
        os.rename(d, new)
        print("renamed", d, "->", new)
PY

# Video list = the 10 benchmark videos actually present under the Pensieve arm.
ls "$PENS_ROOT" > "$VLIST"
echo "[score] videos: $(wc -l < "$VLIST")  reward: $REWARD"

# The repo scoring venv (py3.11) + scratch on PYTHONPATH (enables `import ai_llm.*` / `scripts.*`).
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="$REPO/scratch:${PYTHONPATH:-}"

echo "[score] 1/4 decisions -> throughput_trace.jsonl (realized rate) ..."
python3 scratch/ai_llm/benchmark/decisions_to_throughput.py "$PENS_ROOT" "$META"

echo "[score] 2/4 dp_oracle -> oracle_labels_state.jsonl ..."
python3 scripts/dp_oracle.py --artifacts-root Quic_artifacts --algo-short Pensieve \
  --metadata-dir "$META" --reward-config "$REWARD" --run-glob 'run_*' --workers 8

echo "[score] 3/4 decisions -> teacher shards ..."
rm -rf "$TEACH"; mkdir -p "$TEACH"
python3 scratch/ai_llm/benchmark/decisions_to_trainrows.py "$PENS_ROOT" "$VLIST" "$META" "$TEACH"

echo "[score] 4/4 score_v2_perceptual -> $RESULTS ..."
python3 scratch/ai_llm/benchmark/score_v2_perceptual.py "$PENS_ROOT" "$TEACH" "$VLIST" "$REWARD" "$RESULTS"

echo "[score] DONE -> $RESULTS"
echo "=== Pensieve per-condition %oracle (quick peek) ==="
cat "$RESULTS"
