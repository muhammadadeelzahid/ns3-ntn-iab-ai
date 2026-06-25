#!/usr/bin/env python3
"""Decisive test: is the online r5-collapse caused by last_segment_bitrate_bps being 0?

Run the V2 model offline on the held-out test split TWICE — once with the real
last_segment_bitrate_bps, once forced to 0 (simulating the online serving bug) — and compare
the predicted-rung distributions. If zeroing it reproduces the online collapse (~72% r5 vs the
real ~34%), the missing feature is the cause and the inference_server fix resolves it.
"""
import collections
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))   # -> scratch/ (ai_llm pkg root)
import torch
from ai_llm.evaluate import load_policy_model
from ai_llm.train import AbrJsonlDataset
from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NUM_LADDER_RUNGS
import glob


def predict_dist(model, dataset, device, zero_feature=None):
    hist = collections.Counter()
    for r in dataset._rows:  # pylint: disable=protected-access
        if zero_feature is not None:
            r = dict(r)
            obs = dict(r.get("observation", {}))
            obs[zero_feature] = 0.0
            r["observation"] = obs
        net, sem, lad = dataset._features(r)  # pylint: disable=protected-access
        net_t = torch.tensor([net], dtype=torch.float32, device=device)
        sem_t = torch.tensor([sem], dtype=torch.float32, device=device)
        lad_t = torch.tensor([lad], dtype=torch.float32, device=device)
        with torch.inference_mode():
            logits, _ = model(net_t, sem_t, lad_t)
        hist[int(torch.argmax(logits[0]).item())] += 1
    return hist


def main():
    ckpt = "checkpoints_resid_w1/abr_v2_seed42.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, *_ = (lambda x: x if isinstance(x, tuple) else (x,))(load_policy_model(ckpt, device))
    model.eval()
    paths = sorted(glob.glob("../../train_jsonl/test/*.jsonl"))
    ds = AbrJsonlDataset(paths, network_scaler=DEFAULT_NETWORK_SCALER,
                         semantic_scaler=DEFAULT_SEMANTIC_SCALER, window=1,
                         use_missing_masks=True, variant="v2")
    n = len(ds._rows)  # pylint: disable=protected-access
    FEATS = ["throughput_estimate_bps", "buffer_level_s", "buffer_differential_s",
             "segment_fetch_time_s", "last_segment_bitrate_bps", "smoothed_rtt_ms", "loss_rate_recent"]
    rows = [("(none) baseline", None)] + [("zero %s" % f, f) for f in FEATS]
    for label, zf in rows:
        h = predict_dist(model, ds, device, zf)
        pct = " ".join("r%d=%2d%%" % (k, round(100 * h[k] / n)) for k in range(NUM_LADDER_RUNGS))
        print("%-34s | %s" % (label, pct))


if __name__ == "__main__":
    main()
