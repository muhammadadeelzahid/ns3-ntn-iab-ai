#!/usr/bin/env python3
"""Decision latency of the served policy in the Phase F benchmark (paper: Evaluation, last paragraph).

Reads inference_latency_ms from Quic_artifacts/OursV3_s{1,2,3}/<video>/run_*/decisions.jsonl (26 test videos x
5 rates x 3 seed sets). The first decision of each user (lowest segment_id per node in a cell) is excluded, since
it includes model warm-up. Latency runs from feature assembly to rung selection on the inference server and
excludes client-to-server transport.
Usage: python3 latency_stats.py
"""
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(HERE, "..", "..", "..", "Quic_artifacts")
TEST = set(open(os.path.join(HERE, "test_videos_26.txt")).read().split())


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


total, lat = 0, []
for stem in ("OursV3_s1", "OursV3_s2", "OursV3_s3"):
    for dec in glob.glob(os.path.join(Q, stem, "*", "run_*", "decisions.jsonl")):
        if dec.split(os.sep)[-3] not in TEST:
            continue
        per_node = defaultdict(list)
        for line in open(dec):
            r = json.loads(line)
            per_node[r["node_id"]].append((r["segment_id"], r.get("inference_latency_ms")))
        for rows in per_node.values():
            rows.sort()
            total += len(rows)
            lat += [ms for _, ms in rows[1:] if ms is not None]

print("decisions total %d, after first request per user %d" % (total, len(lat)))
print("latency ms: median %.1f  p95 %.1f  p99 %.1f  max %.0f"
      % (pct(lat, 0.5), pct(lat, 0.95), pct(lat, 0.99), max(lat)))
