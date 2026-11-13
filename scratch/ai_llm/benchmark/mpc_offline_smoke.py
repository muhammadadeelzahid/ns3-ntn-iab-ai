#!/usr/bin/env python3
"""Offline sanity for the RobustMPC baselines: drive Mpc / Qmpc over recorded sessions and replay the
resulting rung sequences through the scorer's player model (evaluate.replay_session), next to the greedy
teacher labels stored in the same rows. OPEN-LOOP (the logged buffer/fetch observations come from the
BOLA trace, not from the policy's own downloads) - so this is a plausibility check of the controller and
its objective wiring, not a benchmark number. Expectation: Qmpc (optimises the scoring QoE) lands in the
teacher's range; Mpc (linear objective) below it.

Usage: PYTHONPATH=scratch .venv/bin/python scratch/ai_llm/benchmark/mpc_offline_smoke.py \
         [--data-dir train_jsonl_clean/test] [--max-sessions 60] [--metadata-dir ...]
"""
import argparse
import collections
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "scratch"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from ai_llm.evaluate import replay_session  # noqa: E402
from ai_llm.features import ladder_context_from_row  # noqa: E402
from ai_llm.metadata_loader import load_metadata_index  # noqa: E402
from ai_llm.mpc_policy import RobustMpcPolicy  # noqa: E402
from dp_oracle import load_reward_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "train_jsonl_clean", "test"))
    ap.add_argument("--metadata-dir", default="/project/6030214/adeel/abr/output/metadata")
    ap.add_argument("--max-sessions", type=int, default=60)
    args = ap.parse_args()
    cfg = load_reward_config(None)

    sessions = []  # (video, run, node, rows)
    for fp in sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl"))):
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        by_node = collections.defaultdict(list)
        for r in rows:
            by_node[(r["video_name"], r["run_id"], r["node_id"])].append(r)
        for key, sess in by_node.items():
            sess.sort(key=lambda x: int(x["segment_id"]))
            sessions.append((key, sess))
    sessions = sessions[: args.max_sessions]
    print("sessions:", len(sessions))

    meta_cache, pol_cache = {}, {}
    agg = collections.defaultdict(list)
    per_video = collections.defaultdict(lambda: collections.defaultdict(list))
    for (video, run, node), sess in sessions:
        if video not in meta_cache:
            meta_cache[video] = load_metadata_index(os.path.join(args.metadata_dir, "%s_segments.csv" % video), None)
            pol_cache[video] = {obj: RobustMpcPolicy(meta_cache[video], objective=obj, reward_cfg=cfg)
                                for obj in ("linear", "qoe")}
        ladders = [ladder_context_from_row(r) for r in sess]
        teacher = [int(r["teacher"]["chosen_bitrate_class"]) for r in sess]
        res = {"teacher": replay_session(teacher, sess, ladders, cfg)}
        for obj, tag in (("linear", "Mpc"), ("qoe", "Qmpc")):
            pol = pol_cache[video][obj]
            pol.reset_history()
            rungs = []
            for r in sess:
                obs = r.get("observation", {}) or {}
                req = {"node_id": node, "segment_fetch_time_s": float(obs.get("segment_fetch_time_s", 0.0)),
                       "buffer_level_s": float(obs.get("buffer_level_s", 0.0)),
                       "throughput_estimate_bps": float(obs.get("throughput_estimate_bps", 0.0))}
                rung, _ = pol.predict(req, int(r["segment_id"]))
                rungs.append(rung)
            res[tag] = replay_session(rungs, sess, ladders, cfg)
            res[tag]["mean_rung"] = sum(rungs) / len(rungs)
        res["teacher"]["mean_rung"] = sum(teacher) / len(teacher)
        for tag, rr in res.items():
            agg[tag].append(rr)
            per_video[video][tag].append(rr["combined_qoe"])

    def m(tag, k):
        xs = [rr[k] for rr in agg[tag]]
        return sum(xs) / len(xs)

    print("%-8s %10s %8s %8s %8s %8s" % ("policy", "QoE/sess", "%teach", "stall_s", "avgVMAF", "rung"))
    tq = m("teacher", "combined_qoe")
    for tag in ("teacher", "Qmpc", "Mpc"):
        print("%-8s %10.1f %7.1f%% %8.2f %8.1f %8.2f" % (
            tag, m(tag, "combined_qoe"), 100.0 * m(tag, "combined_qoe") / tq, m(tag, "stall_s"),
            m(tag, "avg_quality"), m(tag, "mean_rung")))
    print("\nper-video %teacher (Qmpc | Mpc), n sessions:")
    for v in sorted(per_video):
        d = per_video[v]
        t = sum(d["teacher"]) / len(d["teacher"])
        print("  %-22s %6.1f%% | %6.1f%%  n=%d" % (v, 100 * sum(d["Qmpc"]) / len(d["Qmpc"]) / t,
                                                   100 * sum(d["Mpc"]) / len(d["Mpc"]) / t, len(d["teacher"])))


if __name__ == "__main__":
    main()
