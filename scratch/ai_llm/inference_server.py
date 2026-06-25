#!/usr/bin/env python3
"""Torch inference server for the live ns-3 ABR loop (runs under Python 3.11).

Why this exists
---------------
The ns3-ai binding (`ns3ai_abr_dash_ai_py`) is a `cpython-36m` extension — importable ONLY by
Python 3.6 — while the policy model needs torch 2.x + transformers (Python 3.11). The two cannot
live in one interpreter. So `run_ai_dash_llm.py` (3.6) keeps the ns-3 IPC loop and **spawns this
server** (3.11), exchanging one line-delimited JSON message per decision over stdin/stdout:

    client -> server (stdin) : {"node_id":.., "segment_id":.., <19 state fields>}
    server -> client (stdout): {"chosen_bitrate_bps":.., "delay_s":.., "reason":..}
                            or  {"fallback": true, "reason": ".."}   (client runs FDASH)

stdout carries ONLY JSON responses (one per line); all human logs go to stderr. The first stdout
line is a `{"ready": true}` handshake emitted once the (possibly slow) model load finishes.

Because the ns-3 simulation is PAUSED while it waits for the decision, the pipe round-trip is free
with respect to QoE — it only costs wall-clock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

# Make the ai_llm package importable whether run as a module or a bare script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # scratch/  -> enables `import ai_llm.*`
sys.path.insert(0, _HERE)                    # scratch/ai_llm/ -> enables inference.py's siblings

try:
    from ai_llm.inference import LlmAbrPolicy
    from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
except ImportError:  # run directly from scratch/ai_llm
    from inference import LlmAbrPolicy
    from metadata_loader import load_metadata_index, LADDER_LEVELS
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS


def _bps_to_rung(bps: float) -> int:
    """Nearest ladder rung for a chosen bitrate (bps)."""
    return min(range(len(NS3_RATE_LADDER_BPS)), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))


def log(msg: str) -> None:
    print("[infserver] %s" % msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Torch policy inference server (stdin/stdout JSON).")
    ap.add_argument("--checkpoint", default=None, help="Trained policy checkpoint (V0/V1/V2).")
    ap.add_argument("--metadata", required=True, help="Segment metadata CSV.")
    ap.add_argument("--network-scaler", default=DEFAULT_NETWORK_SCALER)
    ap.add_argument("--semantic-scaler", default=DEFAULT_SEMANTIC_SCALER)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default=None, help="cpu/cuda (auto if omitted).")
    ap.add_argument("--window", type=int, default=1, help="Overridden by the checkpoint if present.")
    ap.add_argument("--use-missing-masks", dest="use_missing_masks", action="store_true", default=True)
    ap.add_argument("--no-missing-masks", dest="use_missing_masks", action="store_false")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # The server owns metadata + scalers + model (reuses all existing inference code unchanged).
    metadata_index = load_metadata_index(args.metadata, args.semantic_scaler)
    log("metadata rows: %d from %s" % (len(metadata_index), args.metadata))
    # Segment-id convention: ns-3's AiDashClient reports 1-indexed segment ids (first segment = 1),
    # but the metadata index — and the training data (dataset_builder uses the 0-indexed trace
    # segment_id) — are 0-indexed (segments 0..N-1). So the live lookup must subtract 1, otherwise the
    # policy is fed the NEXT segment's encoding ladder and the last segment overflows. (The earlier
    # 'looping/modulo' patch misdiagnosed this off-by-one and is removed.)

    policy = LlmAbrPolicy(
        checkpoint_path=args.checkpoint,
        network_scaler_path=args.network_scaler or DEFAULT_NETWORK_SCALER,
        semantic_scaler_path=args.semantic_scaler or DEFAULT_SEMANTIC_SCALER,
        model_id=args.model_id,
        device=args.device,
        window=args.window,
        use_missing_masks=args.use_missing_masks,
    )
    log("policy loaded (checkpoint=%s, device=%s)" % (args.checkpoint, getattr(policy, "_device", "?")))

    # READY handshake (first stdout line) — tells the 3.6 client the model is loaded.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    # Per-UE last chosen rung, to reconstruct `last_segment_bitrate_bps` (the VBR-actual bitrate of
    # the previously downloaded segment) exactly as the offline trace pipeline does — the live struct
    # does NOT carry this field, so without this the model gets a constant 0.0 (a log-feature -> extreme
    # OOD z-score) and over-asks online. Mirrors extract_trace.py:_last_segment_bitrate.
    last_rung_by_node: dict = {}

    # One JSON request per line in; one JSON response per line out. EOF on stdin -> exit.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:  # pylint: disable=broad-except
            sys.stdout.write(json.dumps({"fallback": True, "reason": "bad_request:%s" % exc}) + "\n")
            sys.stdout.flush()
            continue

        seg_id = int(req.get("segment_id", -1))
        # Align ns-3's 1-indexed segment id to the 0-indexed metadata/training convention.
        segment_meta = metadata_index.get(seg_id - 1)
        if segment_meta is None:
            segment_meta = metadata_index.get(seg_id)  # tolerate an already-0-indexed source
        if segment_meta is None:
            resp = {"fallback": True, "reason": "missing_metadata"}
        else:
            # Inject last_segment_bitrate_bps = actual (VBR) bitrate of the PREVIOUS segment at the rung
            # the policy chose for it, matching the offline trace (extract_trace.py). The live struct
            # lacks this field; without it the model gets 0.0 -> OOD. 0.0 at startup (no prev segment).
            node = int(req.get("node_id", -1))
            if "last_segment_bitrate_bps" not in req:
                prev_meta = metadata_index.get(seg_id - 2)  # current 0-idx = seg_id-1; previous = -2
                last_rung = last_rung_by_node.get(node)
                if prev_meta is not None and last_rung is not None:
                    req["last_segment_bitrate_bps"] = float(
                        prev_meta.actual_bitrate_kbps.get(LADDER_LEVELS[last_rung], 0.0)) * 1000.0
                else:
                    req["last_segment_bitrate_bps"] = 0.0
            # Reconstruct a lightweight state object; LlmAbrPolicy reads fields via getattr(...),
            # so any field absent from the request defaults exactly as it would on the live struct.
            state = SimpleNamespace(**req)
            try:
                decision = policy.predict(state, segment_meta)
                last_rung_by_node[node] = _bps_to_rung(float(decision.chosen_bitrate_bps))
                resp = {
                    "chosen_bitrate_bps": int(decision.chosen_bitrate_bps),
                    "delay_s": float(decision.delay_s),
                    "reason": decision.reason,
                }
            except Exception as exc:  # pylint: disable=broad-except
                resp = {"fallback": True, "reason": "policy_error:%s" % exc}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    log("stdin closed; exiting.")


if __name__ == "__main__":
    main()
