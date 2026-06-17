#!/usr/bin/env python3
"""
Fit the semantic-feature scaler for the LLM ABR policy.

Reads the raw ``semantic_raw`` block from training JSONL rows (or, equivalently, the
semantic columns of the metadata CSVs) and writes ``config/feature_scaler_semantic.json``
in the schema features.py expects (``feature_names``, ``mean``, ``std``) plus provenance.

Parity with ``fit_network_scaler.py`` — but semantic features are plain z-scored: no
log transform, no sentinel handling. Run this whenever the metadata/corpus changes so the
semantic normalization matches the data the model is trained and served on.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from datetime import datetime, timezone
from typing import Dict, List

# Fixed order — must equal features.SEMANTIC_FEATURE_NAMES.
SEMANTIC_FEATURE_NAMES: List[str] = [
    "motion_score",
    "saliency_score",
    "scene_cut_count",
    "object_score",
    "quality_sensitivity",
]


def _iter_semantic(data_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    paths = [p for p in paths if os.path.basename(p) != "all_runs.jsonl"]
    if not paths:
        raise FileNotFoundError(
            "No JSONL shards matching {} under {} (excluding all_runs.jsonl)".format(
                pattern, data_dir
            )
        )
    _iter_semantic.n_files = len(paths)  # type: ignore[attr-defined]
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sem = rec.get("semantic_raw")
                if isinstance(sem, dict):
                    yield sem


def fit(data_dir: str, pattern: str) -> Dict[str, object]:
    values: Dict[str, List[float]] = {name: [] for name in SEMANTIC_FEATURE_NAMES}
    n_rows = 0
    for sem in _iter_semantic(data_dir, pattern):
        n_rows += 1
        for name in SEMANTIC_FEATURE_NAMES:
            try:
                x = float(sem.get(name, 0.0))
            except (TypeError, ValueError):
                x = 0.0
            if math.isfinite(x):
                values[name].append(x)

    if n_rows == 0:
        raise RuntimeError("No semantic_raw rows found; nothing to fit.")

    means: List[float] = []
    stds: List[float] = []
    for name in SEMANTIC_FEATURE_NAMES:
        vals = values[name]
        mu = statistics.fmean(vals) if vals else 0.0
        sigma = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        if not math.isfinite(sigma) or sigma < 1e-12:
            sigma = 1.0
        means.append(mu)
        stds.append(sigma)

    return {
        "feature_names": list(SEMANTIC_FEATURE_NAMES),
        "mean": means,
        "std": stds,
        "source": os.path.join(data_dir, pattern),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": n_rows,
        "n_files": getattr(_iter_semantic, "n_files", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="train_jsonl",
                        help="Directory with per-run JSONL shards (default: train_jsonl).")
    parser.add_argument("--glob", default="*_run_*.jsonl", dest="glob_pattern",
                        help="Shard glob (all_runs.jsonl always excluded).")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "feature_scaler_semantic.json",
        ),
        help="Output scaler JSON path.",
    )
    args = parser.parse_args()

    scaler = fit(args.data_dir, args.glob_pattern)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(scaler, handle, indent=2)
        handle.write("\n")

    print("Fit semantic scaler from {} rows across {} files".format(
        scaler["n_rows"], scaler["n_files"]))
    for name, mu, sd in zip(scaler["feature_names"], scaler["mean"], scaler["std"]):
        print("  {:<22} mean={:>12.4f} std={:>12.4f}".format(name, mu, sd))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
