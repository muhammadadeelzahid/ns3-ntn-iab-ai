#!/usr/bin/env python3
"""Lookahead (K-step RobustMPC-on-our-QoE) teacher for training labels - model v3, Phase D.

The current labels come from greedy_teacher.greedy_rung: a 1-step causal teacher that plans with the
observed 10 s throughput average. This relabeller plans K segments ahead with the SAME QoE
(config/oracle_reward.json, via mpc_policy.RobustMpcPolicy(objective="qoe") which is parity-tested against
dp_oracle; K=1 with the observed estimate == greedy_rung) and a causal throughput prediction:

  bw="robust"   : RobustMPC predictor over the session's realized per-segment throughput of EARLIER rows
                  (teacher.throughput_bps_replay = what the replay scorer uses as realized throughput);
                  first row of a session falls back to the observed estimate.
  bw="estimate" : the row's observed throughput_estimate_bps (what greedy uses).

Conventions copied from the greedy relabel (benchmark/rebuild_clean_dataset.py): rows grouped by
(run_id, node_id), ordered by segment_id; buffer = the row's observed buffer; the switch penalty is taken
w.r.t. the TEACHER's own previous label.

Modes
  --compare DIR   offline replay QoE (evaluate.replay_session) of greedy labels vs lookahead variants over
                  every session in DIR (use a VAL split for selection)
  --relabel IN OUT  write IN's rows with teacher = lookahead labels (name "lookahead_k<K>_<bw>"), keeping
                  the greedy label as teacher.greedy_chosen_bitrate_class for audit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "scripts")))
from ai_llm.evaluate import replay_session  # noqa: E402
from ai_llm.metadata_loader import LADDER_LEVELS, load_metadata_index  # noqa: E402
from ai_llm.mpc_policy import RobustMpcPolicy, RobustThroughputEstimator  # noqa: E402
from dp_oracle import load_reward_config  # noqa: E402

META = os.environ.get("ABR_METADATA_DIR", "/project/6030214/adeel/abr/output/metadata")
_POLICY_CACHE: dict = {}


def policy_for(video: str, cfg, horizon: int) -> RobustMpcPolicy:
    key = (video, horizon)
    if key not in _POLICY_CACHE:
        idx = load_metadata_index(os.path.join(META, "%s_segments.csv" % video), None)
        _POLICY_CACHE[key] = RobustMpcPolicy(idx, objective="qoe", reward_cfg=cfg, horizon=horizon)
    return _POLICY_CACHE[key]


def _pos(pol: RobustMpcPolicy, row) -> int:
    """Metadata position of the segment this row decides; verified against the row's own ladder sizes."""
    sid = int(row["segment_id"])
    sizes = [float(row["ladder"]["size_bytes"][lv]) for lv in LADDER_LEVELS]
    for cand in (sid, sid + 1, sid - 1):
        p = pol._pos.get(cand)
        if p is not None and all(abs(pol.sizes[p, i] - sizes[i]) <= 1.0 for i in range(len(sizes))):
            return p
    raise KeyError("row %s seg %d does not match metadata ladder" % (row.get("video_name"), sid))


def lookahead_labels(sess, cfg, horizon: int, bw: str):
    """Labels for one ordered session (list of rows)."""
    pol = policy_for(sess[0]["video_name"], cfg, horizon)
    est = RobustThroughputEstimator()
    labels, prev = [], None
    for t, r in enumerate(sess):
        obs = r.get("observation", {}) or {}
        observed = float(obs.get("throughput_estimate_bps", 0.0))
        if bw == "robust" and t > 0:
            est.ingest(0, float(sess[t - 1]["teacher"].get("throughput_bps_replay", 0.0)))
            pred, _, _ = est.predict(0)
            future = pred if pred else observed
        else:
            future = observed
        future = max(future, 1e4)
        combos, total = pol.horizon_rewards(_pos(pol, r), float(obs.get("buffer_level_s", 0.0)), future, prev)
        best = int((total >= total.max()).nonzero()[0][-1])
        rung = int(combos[best, 0])
        labels.append(rung)
        prev = rung
    return labels


def sessions(path_glob):
    groups = defaultdict(list)
    for f in sorted(glob.glob(path_glob)):
        for line in open(f):
            r = json.loads(line)
            groups[(os.path.basename(f), r.get("run_id"), r.get("node_id"))].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: int(r.get("segment_id", 0)))
    return groups


def _replay(rungs, sess, cfg):
    ladders = [[[float(r["ladder"]["size_bytes"][lv]), float(r["ladder"]["vmaf"][lv])] for lv in LADDER_LEVELS]
               for r in sess]
    return replay_session(rungs, sess, ladders, cfg)


def compare(data_dir, cfg, variants):
    groups = sessions(os.path.join(data_dir, "*.jsonl"))
    res = defaultdict(lambda: defaultdict(list))
    agree = defaultdict(list)
    for k, sess in groups.items():
        g = [int(r["teacher"]["chosen_bitrate_class"]) for r in sess]
        out = {"greedy": g}
        for (K, bw) in variants:
            out["K%d_%s" % (K, bw)] = lookahead_labels(sess, cfg, K, bw)
        for name, rungs in out.items():
            rep = _replay(rungs, sess, cfg)
            for m in ("combined_qoe", "stall_s", "avg_quality", "switch_count"):
                if m in rep:
                    res[name][m].append(float(rep[m]))
            res[name]["mean_rung"].append(mean(rungs))
            if name != "greedy":
                agree[name].append(mean(int(a == b) for a, b in zip(rungs, g)))
    print("sessions: %d  (dir %s)" % (len(groups), data_dir))
    keys = sorted({m for v in res.values() for m in v})
    print("  %-16s " % "teacher" + " ".join("%12s" % m for m in keys) + " %10s" % "agree_gr")
    for name in ["greedy"] + ["K%d_%s" % v for v in variants]:
        print("  %-16s " % name + " ".join("%12.3f" % mean(res[name][m]) for m in keys) +
              (" %9.1f%%" % (100 * mean(agree[name])) if name in agree else " %10s" % "-"))


def relabel(in_dir, out_dir, cfg, K, bw):
    os.makedirs(out_dir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(in_dir, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        by = defaultdict(list)
        for r in rows:
            by[(r.get("run_id"), r.get("node_id"))].append(r)
        for sess in by.values():
            sess.sort(key=lambda r: int(r.get("segment_id", 0)))
            for r, lab in zip(sess, lookahead_labels(sess, cfg, K, bw)):
                t = r["teacher"]
                t["greedy_chosen_bitrate_class"] = int(t["chosen_bitrate_class"])
                t["name"] = "lookahead_k%d_%s" % (K, bw)
                t["chosen_bitrate_class"] = lab
                t["oracle_bitrate_idx"] = lab
                lv = LADDER_LEVELS[lab]
                t["vmaf_chosen"] = float(r["ladder"]["vmaf"][lv])
                t["chosen_bitrate_bps"] = int(round(float(r["ladder"].get("actual_bitrate_kbps", {}).get(lv, 0.0)) * 1000))
        with open(os.path.join(out_dir, os.path.basename(f)), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("relabelled", in_dir, "->", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", metavar="DIR")
    ap.add_argument("--relabel", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--bw", choices=("robust", "estimate"), default="robust")
    ap.add_argument("--reward-config", default=None)
    args = ap.parse_args()
    cfg = load_reward_config(args.reward_config)
    if args.compare:
        compare(args.compare, cfg, [(1, "estimate"), (1, "robust"), (3, "robust"), (5, "robust"), (5, "estimate")])
    if args.relabel:
        relabel(args.relabel[0], args.relabel[1], cfg, args.horizon, args.bw)


if __name__ == "__main__":
    main()
