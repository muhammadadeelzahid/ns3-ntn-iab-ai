#!/usr/bin/env python3
"""
Fit the network-feature scaler for the LLM ABR policy.

Reads the raw ``observation`` block from every training JSONL row, computes per-feature
mean/std for z-scoring, and writes ``config/feature_scaler_network.json`` in the schema
``features.py`` expects (``feature_names``, ``mean``, ``std``) plus a ``log_features``
list and provenance fields.

Why this script exists
----------------------
The shipped ``feature_scaler_network.json`` was an identity scaler (mean=0, std=1), so
``network_norm`` was the raw observation: throughput (~1e6) and bitrate (~1e5) dwarfed
buffer (~3), fetch time, RTT, and loss. This fits real statistics so the 7-D network
vector is actually normalized.

Design choices (must match ``features._normalize_network``):
  * ``log_features`` (default throughput + current bitrate) are ``log1p``-transformed
    *before* computing mean/std, because they are positive and span orders of magnitude.
  * Sentinel features (``smoothed_rtt_ms``, ``loss_rate_recent``) use ``-1`` when
    unavailable; those ``-1`` values are EXCLUDED from the mean/std fit so the sentinel
    does not bias the statistics. (Runtime imputes missing values to the fitted mean and
    sets a mask flag.)

The ``observation`` block is raw and scaler-independent, so this can be run against the
current ``train_jsonl/`` shards even though those were written with the identity scaler.
After running this, regenerate the dataset (``dataset_builder.py``) so ``network_norm``
reflects the new scaler.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import statistics
from datetime import datetime, timezone
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import DERIVED_NETWORK_FEATURES, derive_network_feature  # noqa: E402

# Fixed feature order - must equal features.NETWORK_FEATURE_NAMES (+ --extra-features, appended).
NETWORK_FEATURE_NAMES: List[str] = [
    "throughput_estimate_bps",
    "buffer_level_s",
    "buffer_differential_s",
    "segment_fetch_time_s",
    "last_segment_bitrate_bps",
    "smoothed_rtt_ms",
    "loss_rate_recent",
]

# Features whose raw values are imputed (not fit) when negative (-1 sentinel).
SENTINEL_FEATURES = {"smoothed_rtt_ms", "loss_rate_recent"}

# Positive, order-of-magnitude features that are log1p-transformed before z-scoring.
# last_segment_bitrate_bps spans ~24 kbps to ~125 Mbps (VBR), so log-scaling is essential.
DEFAULT_LOG_FEATURES = ["throughput_estimate_bps", "last_segment_bitrate_bps"]


def _iter_rows(data_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    # Never double-count the merged file alongside per-run shards.
    paths = [p for p in paths if os.path.basename(p) != "all_runs.jsonl"]
    if not paths:
        raise FileNotFoundError(
            "No JSONL shards matching {} under {} (excluding all_runs.jsonl)".format(
                pattern, data_dir
            )
        )
    n_files = 0
    for path in paths:
        n_files += 1
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                obs = record.get("observation")
                if isinstance(obs, dict):
                    yield obs
    _iter_rows.n_files = n_files  # type: ignore[attr-defined]


def fit(data_dir: str, pattern: str, log_features: List[str]) -> Dict[str, object]:
    log_set = set(log_features)
    # Collect transformed values per feature (sentinels: valid values only).
    values: Dict[str, List[float]] = {name: [] for name in NETWORK_FEATURE_NAMES}
    n_rows = 0
    for obs in _iter_rows(data_dir, pattern):
        n_rows += 1
        for name in NETWORK_FEATURE_NAMES:
            try:
                if name in DERIVED_NETWORK_FEATURES and name not in obs:
                    raw = derive_network_feature(name, obs)   # same derivation as features.py (train/serve parity)
                else:
                    raw = float(obs.get(name, 0.0))
            except (TypeError, ValueError):
                raw = 0.0
            if not math.isfinite(raw):
                raw = 0.0
            if name in SENTINEL_FEATURES and raw < 0.0:
                # -1 sentinel: do not let it bias the fit.
                continue
            if name in log_set:
                raw = math.log1p(max(raw, 0.0))
            values[name].append(raw)

    if n_rows == 0:
        raise RuntimeError("No observation rows found; nothing to fit.")

    means: List[float] = []
    stds: List[float] = []
    per_feature_count: Dict[str, int] = {}
    for name in NETWORK_FEATURE_NAMES:
        vals = values[name]
        per_feature_count[name] = len(vals)
        if not vals:
            # All sentinel / empty - neutral scaler for this column.
            means.append(0.0)
            stds.append(1.0)
            continue
        mu = statistics.fmean(vals)
        sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if not math.isfinite(sigma) or sigma < 1e-12:
            sigma = 1.0  # guard near-constant columns (e.g. buffer_differential_s)
        means.append(mu)
        stds.append(sigma)

    return {
        "feature_names": list(NETWORK_FEATURE_NAMES),
        "log_features": list(log_features),
        "mean": means,
        "std": stds,
        "source": os.path.join(data_dir, pattern),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": n_rows,
        "n_files": getattr(_iter_rows, "n_files", None),
        "per_feature_fit_count": per_feature_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="train_jsonl",
        help="Directory with per-run JSONL shards (default: train_jsonl).",
    )
    parser.add_argument(
        "--glob",
        default="*_run_*.jsonl",
        dest="glob_pattern",
        help="Shard glob (default: *_run_*.jsonl; all_runs.jsonl always excluded).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "feature_scaler_network.json",
        ),
        help="Output scaler JSON path.",
    )
    parser.add_argument(
        "--log-features",
        default=",".join(DEFAULT_LOG_FEATURES),
        help="Comma-separated features to log1p-transform before z-scoring "
        "(default: throughput_estimate_bps,last_segment_bitrate_bps). Pass '' to disable.",
    )
    parser.add_argument(
        "--extra-features",
        default="",
        help="Comma-separated DERIVED features to append (model v3): realized_throughput_bps. The scaler's "
        "feature_names then drive the model width; old 7-feature scalers/checkpoints are unchanged.",
    )
    args = parser.parse_args()

    for f in [f.strip() for f in args.extra_features.split(",") if f.strip()]:
        if f not in DERIVED_NETWORK_FEATURES:
            raise SystemExit("Unknown --extra-features: {} (known: {})".format(f, DERIVED_NETWORK_FEATURES))
        NETWORK_FEATURE_NAMES.append(f)

    log_features = [f.strip() for f in args.log_features.split(",") if f.strip()]
    unknown = [f for f in log_features if f not in NETWORK_FEATURE_NAMES]
    if unknown:
        raise SystemExit("Unknown --log-features: {}".format(unknown))

    scaler = fit(args.data_dir, args.glob_pattern, log_features)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(scaler, handle, indent=2)
        handle.write("\n")

    print("Fit network scaler from {} rows across {} files".format(
        scaler["n_rows"], scaler["n_files"]))
    print("log_features:", scaler["log_features"])
    for name, mu, sd in zip(scaler["feature_names"], scaler["mean"], scaler["std"]):
        print("  {:<26} mean={:>14.4f} std={:>14.4f}".format(name, mu, sd))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
