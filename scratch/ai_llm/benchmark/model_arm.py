#!/usr/bin/env python3
"""Where the MODEL arm ("AiDash" = Ours in paper_style.ALGOS) keeps its data - one switch for every
table/figure script, so a re-run of Ours (Phase 4: clean checkpoint, Quic_artifacts/AiDashClean) is
selected with ONE env var and never by editing five scripts:

  MODEL_STEM unset / "AiDash"  -> legacy layout of the 2026-07 fixed-binary benchmark:
      Quic_artifacts/AiDash/bench_down_scored/<v>/run_<cond>/decisions.jsonl,
      Quic_artifacts/AiDash/shards_down/<v>_run_<cond>.jsonl, worker logs in logs_bench/,
      results/AiDash.txt
  MODEL_STEM=<Stem> (e.g. AiDashClean) -> the standard bridge-arm layout written by
      slurm/bench_model_packed.sbatch (OUT_ROOT=Quic_artifacts/<Stem>, LOGDIR=logs_bench_<stem>) and scored by
      benchmark/score_bridge_arm.sh <Stem>:
      Quic_artifacts/<Stem>/<v>/run_<cond>/decisions.jsonl, Quic_artifacts/<Stem>_shards/<v>_run_<cond>.jsonl,
      logs_bench_<stem>/, results/<Stem>.txt

The registry key stays "AiDash" everywhere (true_stall.csv arm column, tables, legend "Ours"); only the
file locations move. No matplotlib import here (the scoring venv has none)."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
Q = os.path.join(_REPO, "Quic_artifacts")

MODEL_STEM = os.environ.get("MODEL_STEM", "AiDash") or "AiDash"
LEGACY = MODEL_STEM == "AiDash"


def dec_subdir():
    """Path under Quic_artifacts holding <v>/run_<cond>/decisions.jsonl for the model arm."""
    return os.path.join("AiDash", "bench_down_scored") if LEGACY else MODEL_STEM


def decisions(video, cond):
    return os.path.join(Q, dec_subdir(), video, "run_%d" % cond, "decisions.jsonl")


def shard(video, cond):
    """Greedy-teacher shard (dp_oracle join + relabel) for one model cell."""
    base = os.path.join(Q, "AiDash", "shards_down") if LEGACY else os.path.join(Q, "%s_shards" % MODEL_STEM)
    return os.path.join(base, "%s_run_%d.jsonl" % (video, cond))


def logdir():
    """The model arm's OWN worker log dir (per-cell ns-3 sim.log capture) - never another arm's."""
    return os.path.join(_REPO, "logs_bench") if LEGACY else os.path.join(_REPO, "logs_bench_%s" % MODEL_STEM.lower())


def results_txt():
    return os.path.join(_HERE, "results", "%s.txt" % MODEL_STEM)


def describe():
    return "model arm 'AiDash' <- %s (decisions=%s, shards=%s, logs=%s, results=%s)" % (
        "legacy layout" if LEGACY else "MODEL_STEM=%s" % MODEL_STEM, dec_subdir(),
        os.path.relpath(os.path.dirname(shard("v", 1)), Q), os.path.basename(logdir()),
        os.path.relpath(results_txt(), _HERE))


if __name__ == "__main__":
    print(describe())
