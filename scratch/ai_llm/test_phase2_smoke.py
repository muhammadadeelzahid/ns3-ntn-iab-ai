#!/usr/bin/env python3
"""
Phase-2 smoke checks for feature assembly and model forward pass.
"""

import math
import os
import sys

try:
    import torch
except ImportError:
    torch = None

from metadata_loader import load_metadata_index
from features import build_model_input
from model import create_model
from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NETWORK_DIM, NUM_LADDER_RUNGS


class DummyState:
    def __init__(self) -> None:
        self.throughput_estimate_bps = 2_500_000.0
        self.buffer_level_s = 8.0
        self.buffer_differential_s = 0.5
        self.segment_fetch_time_s = 1.3
        self.smoothed_rtt_ms = 22.5
        self.loss_rate_recent = 0.01
        # Actual encoded bitrate (VBR) of the last downloaded segment, not the nominal rung.
        self.last_segment_bitrate_bps = 5_800_000.0


def main() -> None:
    metadata_csv = "/home/adeel/projects/def-ekram/adeel/abr/output/metadata/hiking_segments.csv"
    idx = load_metadata_index(metadata_csv, DEFAULT_SEMANTIC_SCALER)
    segment = idx.get(0)
    if segment is None:
        raise RuntimeError("segment 0 not found in metadata")

    sample = build_model_input(
        state=DummyState(),
        segment_meta=segment,
        network_scaler_path=DEFAULT_NETWORK_SCALER,
        semantic_scaler_path=DEFAULT_SEMANTIC_SCALER,
    )
    if len(sample.network_norm) != NETWORK_DIM:
        raise RuntimeError(
            "expected network_norm length %d, got %d" % (NETWORK_DIM, len(sample.network_norm))
        )
    if len(sample.semantic_norm) != 5:
        raise RuntimeError("expected semantic_norm length 5, got %d" % len(sample.semantic_norm))
    if len(sample.ladder_context) != 6:
        raise RuntimeError("expected ladder_context length 6, got %d" % len(sample.ladder_context))
    if any(len(row) != 2 for row in sample.ladder_context):
        raise RuntimeError("expected ladder_context rows of width 2 [size_bytes, vmaf]")
    for value in sample.network_norm + sample.semantic_norm:
        if not math.isfinite(value):
            raise RuntimeError("non-finite normalized value found")

    if torch is None:
        print("feature checks passed; torch unavailable so skipping model forward")
        return

    model = create_model(network_dim=NETWORK_DIM, semantic_dim=5, ladder_dim=2, d_model=256)
    model.eval()
    with torch.inference_mode():
        logits, delay = model(
            torch.tensor([sample.network_norm], dtype=torch.float32),
            torch.tensor([sample.semantic_norm], dtype=torch.float32),
            torch.tensor([sample.ladder_context], dtype=torch.float32),
        )
    if logits.shape != (1, NUM_LADDER_RUNGS):
        raise RuntimeError("unexpected logits shape: %s" % (tuple(logits.shape),))
    if delay is not None:
        raise RuntimeError("delay head should be disabled, got shape: %s" % (tuple(delay.shape),))
    print("phase2 smoke passed (6-rung ladder, no delay head)")


if __name__ == "__main__":
    main()
