#!/usr/bin/env python3
"""Per-video OFFLINE model rung distribution - the apples-to-apples reference for the congested
live sweep. Runs a trained V0/V2 checkpoint on the uncongested train_jsonl shards (same model,
same feature path as evaluate.py) and reports, per video, the predicted-rung histogram + mean.
This is the correct baseline for "does the model pick LOWER rungs under live congestion?" -
NOT the DP-oracle teacher label (acausal, only loosely imitated)."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from evaluate import load_policy_model, AbrJsonlDataset, DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-dir", default="train_jsonl")
    ap.add_argument("--include-video", action="append", required=True)
    ap.add_argument("--network-scaler", default=DEFAULT_NETWORK_SCALER)
    ap.add_argument("--semantic-scaler", default=DEFAULT_SEMANTIC_SCALER)
    args = ap.parse_args()

    device = args.device
    model, variant, window, use_masks, language_cache = load_policy_model(args.checkpoint, device)
    model.eval()

    include = set(args.include_video)
    # Only load shards for the requested videos (cheap).
    paths = []
    for v in include:
        paths += sorted(Path(args.data_dir).glob("%s_run_*.jsonl" % v))
    dataset = AbrJsonlDataset(
        sorted(paths), network_scaler=args.network_scaler or None,
        semantic_scaler=args.semantic_scaler or None, window=window,
        use_missing_masks=use_masks, variant=variant, language_cache=language_cache,
    )

    by_video = defaultdict(list)
    for r in dataset._rows:  # noqa
        vid = r.get("video_name")
        if vid not in include:
            continue
        net, sem, lad = dataset._features(r)  # noqa
        net_t = torch.tensor([net], dtype=torch.float32, device=device)
        sem_t = torch.tensor([sem], dtype=torch.float32, device=device)
        lad_t = torch.tensor([lad], dtype=torch.float32, device=device)
        lang_args = ()
        if variant == "v3":
            li, lm = dataset._language(r)  # noqa
            lang_args = (li.unsqueeze(0).to(device), lm.unsqueeze(0).to(device))
        with torch.inference_mode():
            logits, _ = model(net_t, sem_t, lad_t, *lang_args)
        by_video[vid].append(int(torch.argmax(logits[0]).item()))

    out = {}
    for v in sorted(include):
        rungs = by_video.get(v, [])
        if not rungs:
            out[v] = {"n": 0, "mean": None, "hist": {}}
            continue
        out[v] = {"n": len(rungs), "mean": sum(rungs) / len(rungs),
                  "hist": dict(sorted(Counter(rungs).items()))}
        print("%-16s n=%-5d mean=%.3f hist=%s" % (v, out[v]["n"], out[v]["mean"], out[v]["hist"]))
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
