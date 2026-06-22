#!/usr/bin/env python3
"""
LLM-oriented DASH ABR orchestrator (inference path).

Drives ns-3's AiDashClient over the ns3-ai shared-memory IPC: receives per-segment
player/network state, runs the trained policy (LlmAbrPolicy: V0 MLP head, or V1/V2 with the
LLM backbone) on the looked-up segment metadata, and returns a bitrate decision. Falls back to
the FDASH heuristic on missing metadata or any policy error.
"""

import argparse
import json
import os
import sys
import time

# Add src/ai/python_utils and build directory to PYTHONPATH.
NS3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NS3_ROOT, "src", "ai", "python_utils"))
sys.path.insert(0, os.path.join(NS3_ROOT, "build", "src", "ai", "examples", "abr-dash-ai", "use-msg"))

from ns3ai_utils import Experiment  # type: ignore
import ns3ai_abr_dash_ai_py as py_binding  # type: ignore

from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
from ai_llm.metadata_loader import load_metadata_index
from ai_llm.inference import LlmAbrPolicy
from run_ai_dash_heuristic import fdash_heuristic


def _bps_to_rung(bps: int) -> int:
    """Map a chosen bitrate (bps) to its ladder rung index (nearest, for robustness)."""
    try:
        return NS3_RATE_LADDER_BPS.index(int(bps))
    except ValueError:
        return min(
            range(len(NS3_RATE_LADDER_BPS)),
            key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - int(bps)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ns-3 AiDashClient with metadata-aware policy.")
    parser.add_argument(
        "--metadata",
        default="/home/adeel/projects/def-ekram/adeel/abr/output/metadata/hiking_segments.csv",
        help="Path to segment metadata CSV.",
    )
    parser.add_argument(
        "--semantic-scaler",
        "--scaler",
        dest="semantic_scaler",
        default=DEFAULT_SEMANTIC_SCALER,
        help="Path to semantic feature scaler JSON (default: feature_scaler_semantic.json).",
    )
    parser.add_argument(
        "--network-scaler",
        default=DEFAULT_NETWORK_SCALER,
        help="Path to network feature scaler JSON (default: feature_scaler_network.json).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint path for LLM policy weights.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Backbone model id for V1/V2 (ignored by V0). NOTE: a checkpoint's own model_id "
        "overrides this (inference.py), so this default only applies when none is recorded.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Time-series window (past segments per UE) for the network branch. "
        "Overridden by the checkpoint's window if present. 1 = snapshot.",
    )
    parser.add_argument(
        "--use-missing-masks",
        dest="use_missing_masks",
        action="store_true",
        default=True,
        help="Append RTT/loss missing-indicator channels (network_dim 7->9). Overridden by "
        "the checkpoint's flag if present. Default on.",
    )
    parser.add_argument(
        "--no-missing-masks",
        dest="use_missing_masks",
        action="store_false",
        help="Disable missing-indicator channels.",
    )
    parser.add_argument("--numUes", default="1", help="Number of UEs to simulate.")
    parser.add_argument("--run", default="1", help="RNG run index.")
    parser.add_argument(
        "--video",
        default=None,
        help="Video name for labeling + the decisions-log path (default: derived from --metadata "
        "filename, e.g. hiking_segments.csv -> hiking).",
    )
    parser.add_argument(
        "--decisions-out",
        default=None,
        help="Path to write the per-segment decisions JSONL (default: "
        "Quic_artifacts/AiDash/<video>/run_<run>/decisions.jsonl). Used for offline QoE replay.",
    )
    return parser.parse_args()


def _resolve_video_name(args) -> str:
    if args.video:
        return args.video
    base = os.path.basename(args.metadata)
    for suffix in ("_segments.csv", ".csv"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base or "video"


def _resolve_decisions_path(args, video: str) -> str:
    if args.decisions_out:
        return args.decisions_out
    return os.path.join(
        "Quic_artifacts", "AiDash", video, "run_%s" % args.run, "decisions.jsonl"
    )


def main() -> None:
    args = parse_args()
    os.environ["WAF_CMD"] = "python3 ./waf"

    metadata_index = load_metadata_index(args.metadata, args.semantic_scaler)
    policy = LlmAbrPolicy(
        checkpoint_path=args.checkpoint,
        network_scaler_path=args.network_scaler,
        semantic_scaler_path=args.semantic_scaler,
        model_id=args.model_id,
        device=args.device,
        window=args.window,
        use_missing_masks=args.use_missing_masks,
    )
    print(f"[Python] Loaded metadata rows: {len(metadata_index)} from {args.metadata}", flush=True)

    exp = Experiment(
        "ntn-iab-quic-dash-ai",
        os.path.abspath("."),
        py_binding,
        handleFinish=True,
        useVector=False,
        shmSize=4096,
    )

    msg = exp.run(
        setting={
            "aiEnable": "true",
            "numUes": str(args.numUes),
            "run": str(args.run),
            "algorithm": "ns3::AiDashClient",
            "dashMetadataCsv": str(args.metadata),
            "dashMetadataTrace": "dash_metadata_segments.csv",
        },
        show_output=True,
    )

    video = _resolve_video_name(args)
    decisions_path = _resolve_decisions_path(args, video)
    os.makedirs(os.path.dirname(decisions_path) or ".", exist_ok=True)
    decision_records = []  # one dict per decision, written to JSONL at the end (for QoE replay)

    print("[Python] LLM ABR orchestrator started. video=%s decisions_out=%s"
          % (video, decisions_path), flush=True)
    fallback_count = 0
    decisions = 0

    try:
        while True:
            # Bail out before blocking if the simulation subprocess is gone.
            if exp.proc is not None and exp.proc.poll() is not None:
                print(
                    f"[Python] ns-3 subprocess exited (rc={exp.proc.returncode}) "
                    f"before signalling finish; stopping loop.",
                    flush=True,
                )
                break

            msg.PyRecvBegin()
            if msg.PyGetFinished():
                print("[Python] Simulation finished!", flush=True)
                break

            state = msg.GetCpp2PyStruct()
            segment_meta = metadata_index.get(int(state.segment_id))

            t0 = time.perf_counter()
            if segment_meta is None:
                chosen_bitrate, delay_s = fdash_heuristic(state)
                reason = "fallback_missing_metadata"
                fallback_count += 1
            else:
                try:
                    decision = policy.predict(state, segment_meta)
                    chosen_bitrate = int(decision.chosen_bitrate_bps)
                    delay_s = float(decision.delay_s)
                    reason = decision.reason
                except Exception as exc:  # pylint: disable=broad-except
                    chosen_bitrate, delay_s = fdash_heuristic(state)
                    reason = f"fallback_policy_error:{exc}"
                    fallback_count += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0

            msg.PyRecvEnd()

            msg.PySendBegin()
            action = msg.GetPy2CppStruct()
            action.chosen_bitrate_bps = max(0, int(chosen_bitrate))
            # Phase A: inter-segment delay is not trained or inferred; always zero at runtime.
            action.delay_s = 0.0
            msg.PySendEnd()

            decisions += 1
            decision_records.append({
                "sim_time_s": float(getattr(state, "sim_time_s", 0.0)),
                "node_id": int(state.node_id),
                "video": video,
                "segment_id": int(state.segment_id),
                "chosen_bitrate_bps": int(action.chosen_bitrate_bps),
                "chosen_rung": _bps_to_rung(action.chosen_bitrate_bps),
                "reason": reason,
                "inference_latency_ms": latency_ms,
                "throughput_estimate_bps": float(getattr(state, "throughput_estimate_bps", 0.0)),
                "buffer_level_s": float(getattr(state, "buffer_level_s", 0.0)),
                "smoothed_rtt_ms": float(getattr(state, "smoothed_rtt_ms", -1.0)),
                "loss_rate_recent": float(getattr(state, "loss_rate_recent", -1.0)),
            })
            print(
                f"[Python][LLM] node={state.node_id} seg={state.segment_id} "
                f"choice={action.chosen_bitrate_bps} delay={action.delay_s:.4f}s "
                f"rtt_ms={getattr(state, 'smoothed_rtt_ms', -1.0):.3f} "
                f"loss={getattr(state, 'loss_rate_recent', -1.0):.4f} "
                f"lat_ms={latency_ms:.2f} reason={reason}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[Python] Interrupted by user.", flush=True)
    finally:
        exp.kill()
        # Persist the per-segment decisions for offline QoE replay (eval_qoe.py).
        try:
            with open(decisions_path, "w", encoding="utf-8") as fh:
                for rec in decision_records:
                    fh.write(json.dumps(rec) + "\n")
            print("[Python] Wrote %d decisions -> %s" % (len(decision_records), decisions_path),
                  flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print("[Python] WARN: failed to write decisions log: %s" % exc, flush=True)
        print(
            f"[Python] Experiment killed. decisions={decisions} fallbacks={fallback_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
