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
from features import (
    build_model_input,
    load_scaler,
    normalize_network_vector,
    normalize_network_vector_with_masks,
    SENTINEL_MASK_ORDER,
)
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


def mask_checks() -> None:
    """Missing-indicator wiring: 9-dim output, mask=1 + value~0 on a -1 sentinel,
    mask=0 on a real value, and train/inference channel-order parity (shared helper)."""
    scaler = load_scaler(DEFAULT_NETWORK_SCALER)
    n_mask = len(SENTINEL_MASK_ORDER)

    base = {
        "throughput_estimate_bps": 2_500_000.0,
        "buffer_level_s": 8.0,
        "buffer_differential_s": 0.5,
        "segment_fetch_time_s": 1.3,
        "last_segment_bitrate_bps": 5_800_000.0,
        "smoothed_rtt_ms": 22.5,
        "loss_rate_recent": 0.01,
    }
    present = normalize_network_vector_with_masks(base, scaler)
    if len(present) != NETWORK_DIM + n_mask:
        raise RuntimeError(
            "expected mask-aware length %d, got %d" % (NETWORK_DIM + n_mask, len(present))
        )
    # First n_mask trailing channels correspond to SENTINEL_MASK_ORDER; RTT present -> 0.
    rtt_mask_idx = NETWORK_DIM + SENTINEL_MASK_ORDER.index("smoothed_rtt_ms")
    if present[rtt_mask_idx] != 0.0:
        raise RuntimeError("expected RTT mask 0.0 when present, got %s" % present[rtt_mask_idx])

    missing = dict(base, smoothed_rtt_ms=-1.0)
    miss_vec = normalize_network_vector_with_masks(missing, scaler)
    if miss_vec[rtt_mask_idx] != 1.0:
        raise RuntimeError("expected RTT mask 1.0 when sentinel, got %s" % miss_vec[rtt_mask_idx])
    # RTT is imputed to the mean -> z-score 0; without the mask this is indistinguishable
    # from a real mean value. The mask is what recovers the difference.
    if abs(miss_vec[5]) > 1e-9:
        raise RuntimeError("expected imputed RTT z-score ~0.0, got %s" % miss_vec[5])

    # Parity: the 7-dim prefix of the mask-aware vector equals the mask-off vector exactly,
    # so train (with masks) and any mask-off path agree on the underlying features.
    plain = normalize_network_vector(missing, scaler)
    if any(abs(a - b) > 1e-12 for a, b in zip(plain, miss_vec[:NETWORK_DIM])):
        raise RuntimeError("mask-aware prefix diverges from mask-off vector")
    print("mask checks passed (network_dim %d -> %d, sentinel-aware)" % (NETWORK_DIM, len(present)))

    if torch is not None:
        # Forward pass with the wider network_dim over a [B, W, 9] window.
        model = create_model(network_dim=NETWORK_DIM + n_mask, semantic_dim=5, ladder_dim=2, d_model=256)
        model.eval()
        window = torch.tensor([[present, present]], dtype=torch.float32)  # [1, W=2, 9]
        with torch.inference_mode():
            logits, _ = model(
                window,
                torch.zeros((1, 5), dtype=torch.float32),
                torch.zeros((1, 6, 2), dtype=torch.float32),
            )
        if logits.shape != (1, NUM_LADDER_RUNGS):
            raise RuntimeError("unexpected masked logits shape: %s" % (tuple(logits.shape),))
        print("masked forward pass passed ([1, W, %d] -> [1, %d])" % (len(present), NUM_LADDER_RUNGS))


def main() -> None:
    mask_checks()

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
