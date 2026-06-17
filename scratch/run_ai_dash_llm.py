#!/usr/bin/env python3
"""
LLM-oriented DASH ABR orchestrator (inference path).

Current phase uses a deterministic policy stub with metadata fusion and retains
FDASH fallback for robustness.
"""

import argparse
import os
import sys

# Add src/ai/python_utils and build directory to PYTHONPATH.
NS3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NS3_ROOT, "src", "ai", "python_utils"))
sys.path.insert(0, os.path.join(NS3_ROOT, "build", "src", "ai", "examples", "abr-dash-ai", "use-msg"))

from ns3ai_utils import Experiment  # type: ignore
import ns3ai_abr_dash_ai_py as py_binding  # type: ignore

from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER
from ai_llm.metadata_loader import load_metadata_index
from ai_llm.inference import LlmAbrPolicy
from run_ai_dash_heuristic import fdash_heuristic


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
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Backbone model id metadata for the policy wrapper.",
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
    parser.add_argument("--numUes", default="1", help="Number of UEs to simulate.")
    parser.add_argument("--run", default="1", help="RNG run index.")
    return parser.parse_args()


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

    print("[Python] LLM ABR orchestrator started.", flush=True)
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

            msg.PyRecvEnd()

            msg.PySendBegin()
            action = msg.GetPy2CppStruct()
            action.chosen_bitrate_bps = max(0, int(chosen_bitrate))
            # Phase A: inter-segment delay is not trained or inferred; always zero at runtime.
            action.delay_s = 0.0
            msg.PySendEnd()

            decisions += 1
            print(
                f"[Python][LLM] node={state.node_id} seg={state.segment_id} "
                f"choice={action.chosen_bitrate_bps} delay={action.delay_s:.4f}s "
                f"rtt_ms={getattr(state, 'smoothed_rtt_ms', -1.0):.3f} "
                f"loss={getattr(state, 'loss_rate_recent', -1.0):.4f} "
                f"reason={reason}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[Python] Interrupted by user.", flush=True)
    finally:
        exp.kill()
        print(
            f"[Python] Experiment killed. decisions={decisions} fallbacks={fallback_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
