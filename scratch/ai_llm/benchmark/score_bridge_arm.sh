#!/bin/bash
# Score ANY bridge arm (decisions.jsonl runs produced through the Python inference server: Pensieve, Mpc,
# Qmpc, Greedy, ...) into results/<Stem>.txt, directly comparable to the heuristic arms. Generalises
# pensieve/score_pensieve.sh (hard-coded to Pensieve) - same chain, parameterised by the arm stem:
#
#   fallback gate (refuse cells the server silently handed to FDASH)
#   rename run dirs run_<seed>_<rtag> -> run_<condIdx>   (cond from the RATE tag, never the seed)
#   decisions_to_throughput -> dp_oracle(--algo-short <Stem>) -> decisions_to_trainrows
#   -> score_v2_perceptual (results/<Stem>.txt); teacher shards persisted to Quic_artifacts/<Stem>_shards
#
# Usage: bash scratch/ai_llm/benchmark/score_bridge_arm.sh <Stem> [video-list]
#   video-list defaults to benchmark/test_videos_26.txt filtered to the videos present under the arm.
#   FALLBACK_MAX_FRAC=0.01 (default) - cells with a larger share of fallback rows abort the run (FORCE=1 to
#   score anyway; they are still listed).
set -euo pipefail

STEM="${1:?usage: score_bridge_arm.sh <Stem> [video-list]}"
REPO="${REPO_ROOT:-/project/6030214/adeel/ns3-ntn-iab}"
META="${METADIR:-/project/6030214/adeel/abr/output/metadata}"
cd "$REPO"

ROOT="Quic_artifacts/${STEM}"
[ -d "$ROOT" ] || { echo "no such arm tree: $ROOT" >&2; exit 2; }
TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/score_${STEM}_$$"
mkdir -p "$TMP"
VLIST="$TMP/videos.txt"
TEACH="$TMP/teacher"
SHARDS_KEEP="Quic_artifacts/${STEM}_shards"
RESULTS="scratch/ai_llm/benchmark/results/${STEM}.txt"
REWARD="scratch/ai_llm/config/oracle_reward.json"
LIST_IN="${2:-scratch/ai_llm/benchmark/test_videos_26.txt}"

# Video list = requested list ∩ videos present under the arm (so partial runs still score).
: > "$VLIST"
while read -r v; do [ -n "$v" ] && [ -d "$ROOT/$v" ] && echo "$v" >> "$VLIST"; done < "$LIST_IN"
echo "[score:$STEM] videos: $(wc -l < "$VLIST") of $(grep -c . "$LIST_IN") requested  reward: $REWARD"

# 0a. Fallback gate: a server fallback makes the client decide with FDASH and the row still looks normal
#     (only `reason` says "fallback_..."), so an arm can silently become FDASH. Refuse such cells.
python3 - "$ROOT" "$VLIST" "${FALLBACK_MAX_FRAC:-0.01}" "${FORCE:-0}" <<'PY'
import collections, glob, json, os, sys
root, vlist, max_frac, force = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4] == "1"
keep = set(open(vlist).read().split())
hist = collections.Counter(); bad = []
for dec in sorted(glob.glob(os.path.join(root, "*", "run_*", "decisions.jsonl"))):
    video = dec.split(os.sep)[-3]
    if video not in keep:
        continue
    n = nfb = 0
    for line in open(dec):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line); n += 1
        r = str(d.get("reason", ""))
        hist[r.split("(")[0].split(":")[0]] += 1
        if r.startswith("fallback"):
            nfb += 1
    if n and nfb / n > max_frac:
        bad.append((dec, nfb, n))
print("[score] reason histogram:", dict(hist.most_common(8)))
if bad:
    print("[score] FALLBACK GATE: %d cell(s) exceed %.1f%% fallback rows:" % (len(bad), 100 * max_frac))
    for dec, nfb, n in bad:
        print("   %s  %d/%d" % (dec, nfb, n))
    if not force:
        sys.exit("refusing to score (set FORCE=1 to override)")
PY

# 0b. Normalise run-dir names run_<seed>_<rtag> -> run_<condIdx> from the RATE tag (100Gbs=1, 36Mbps=2,
#     18Mbps=3, 9Mbps=4, 4.5Mbps=5). Idempotent; collisions (two seeds, one rate) abort.
python3 - "$ROOT" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
cond = {"100Gbs": 1, "36Mbps": 2, "18Mbps": 3, "9Mbps": 4, "4.5Mbps": 5}
for d in glob.glob(os.path.join(root, "*", "run_*")):
    if not os.path.isdir(d):
        continue
    mt = re.match(r"run_\d+_(.+)$", os.path.basename(d))
    if not mt:
        continue  # already run_<N>
    rtag = mt.group(1)
    if rtag not in cond:
        raise SystemExit("unknown rtag: %s (%s)" % (rtag, d))
    new = os.path.join(os.path.dirname(d), "run_%d" % cond[rtag])
    if os.path.exists(new):
        raise SystemExit("rename collision: %s -> %s" % (d, new))
    os.rename(d, new)
    print("renamed", d, "->", new)
PY

# Scoring venv (py3.11) + scratch on PYTHONPATH (enables `import ai_llm.*` / `scripts.*`).
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="$REPO/scratch:${PYTHONPATH:-}"

echo "[score:$STEM] 1/4 decisions -> throughput_trace.jsonl (realized rate) ..."
python3 scratch/ai_llm/benchmark/decisions_to_throughput.py "$ROOT" "$META"

echo "[score:$STEM] 2/4 dp_oracle -> oracle_labels_state.jsonl ..."
python3 scripts/dp_oracle.py --artifacts-root Quic_artifacts --algo-short "$STEM" \
  --metadata-dir "$META" --reward-config "$REWARD" --run-glob 'run_*' --workers "${WORKERS:-8}"

echo "[score:$STEM] 3/4 decisions -> teacher shards ..."
rm -rf "$TEACH"; mkdir -p "$TEACH"
python3 scratch/ai_llm/benchmark/decisions_to_trainrows.py "$ROOT" "$VLIST" "$META" "$TEACH"
rm -rf "$SHARDS_KEEP"; mkdir -p "$SHARDS_KEEP"; cp "$TEACH"/*.jsonl "$SHARDS_KEEP"/   # absolute_qoe_table reads <Stem>_shards

echo "[score:$STEM] 4/4 score_v2_perceptual -> $RESULTS ..."
python3 scratch/ai_llm/benchmark/score_v2_perceptual.py "$ROOT" "$TEACH" "$VLIST" "$REWARD" "$RESULTS"

rm -rf "$TMP"
echo "[score:$STEM] DONE -> $RESULTS  (shards: $SHARDS_KEEP)"
