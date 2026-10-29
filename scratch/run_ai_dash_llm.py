#!/usr/bin/env python3
"""
LLM-oriented DASH ABR orchestrator (inference path) - runs under Python 3.6.

Drives ns-3's AiDashClient over the ns3-ai shared-memory IPC: receives per-segment
player/network state and returns a bitrate decision. The policy itself (torch / transformers)
cannot live in this interpreter - the ns3-ai binding is a `cpython-36m` module, while torch
needs Python 3.11 - so this process is **torch-free** and delegates each decision to a
long-lived 3.11 inference server (`ai_llm/inference_server.py`) over a stdin/stdout JSON pipe.
Falls back to the in-process FDASH heuristic on missing metadata, a policy error, or no server.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Add src/ai/python_utils and build directory to PYTHONPATH.
NS3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(NS3_ROOT, "src", "ai", "python_utils"))
sys.path.insert(0, os.path.join(NS3_ROOT, "build", "src", "ai", "examples", "abr-dash-ai", "use-msg"))

from ns3ai_utils import Experiment  # type: ignore
import ns3ai_abr_dash_ai_py as py_binding  # type: ignore

# torch-free imports only (paths is pure pathlib; fdash_heuristic is pure-python math).
from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
from run_ai_dash_heuristic import fdash_heuristic

# The 19 Cpp2Py state fields populated in ai-dash-client.cc::CalcNextSegment (the order is not
# significant - they are sent by name). Serialized verbatim to the inference server each decision.
CPP2PY_FIELDS = (
    "node_id", "segment_id", "sim_time_s", "segment_fetch_time_s", "throughput_estimate_bps",
    "buffer_level_s", "buffer_differential_s", "current_bitrate_bps", "prev_bitrate_bps",
    "player_state", "interruption_time_s", "interruptions", "smoothed_rtt_ms", "loss_rate_recent",
    "throughput_ema_bps", "throughput_std_bps", "download_time_ratio", "last_decision_delay_s",
    "bitrate_switch_count_recent",
)


def _state_to_dict(state) -> dict:
    """Serialize the live Cpp2Py struct to a plain dict for the inference server."""
    return {f: getattr(state, f, 0.0) for f in CPP2PY_FIELDS}


def _bps_to_rung(bps: int) -> int:
    """Map a chosen bitrate (bps) to its ladder rung index (nearest, for robustness)."""
    try:
        return NS3_RATE_LADDER_BPS.index(int(bps))
    except ValueError:
        return min(
            range(len(NS3_RATE_LADDER_BPS)),
            key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - int(bps)),
        )


def _spawn_inference_server(args):
    """Launch the 3.11 torch inference server and block until its READY handshake.

    Returns the Popen handle, or None if no checkpoint was given (pure-FDASH mode)."""
    policy = getattr(args, "policy", "model")
    if not args.checkpoint and policy in ("model", "pensieve"):
        print("[Python] No --checkpoint: running FDASH-only (no inference server).", flush=True)
        return None
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_llm",
                                 "inference_server.py")
    cmd = [
        args.server_python, server_script,
        "--metadata", args.metadata,
        "--network-scaler", args.network_scaler,
        "--semantic-scaler", args.semantic_scaler,
        "--model-id", args.model_id,
        "--window", str(args.window),
        "--policy", policy,
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if policy == "greedy":
        if getattr(args, "w_stall", None) is not None:
            cmd += ["--w-stall", str(args.w_stall)]
        if getattr(args, "thr_safety", None) is not None:
            cmd += ["--thr-safety", str(args.thr_safety)]
    if args.device:
        cmd += ["--device", args.device]
    if not args.use_missing_masks:
        cmd += ["--no-missing-masks"]
    print("[Python] Spawning inference server: %s" % " ".join(cmd), flush=True)
    server = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        universal_newlines=True, bufsize=1,
    )
    ready_line = server.stdout.readline()  # blocks until the model is loaded
    try:
        if not ready_line or not json.loads(ready_line).get("ready"):
            raise RuntimeError("server did not signal READY (got %r)" % ready_line)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("inference server failed to start: %s" % exc)
    print("[Python] Inference server ready.", flush=True)
    return server


def _bridge_decision(server, state):
    """Send one state to the inference server and return (chosen_bitrate_bps, delay_s, reason,
    is_fallback). On any pipe/server problem, signal fallback so the caller runs FDASH."""
    if server is None or server.poll() is not None:
        return None, None, "fallback_no_server", True
    try:
        server.stdin.write(json.dumps(_state_to_dict(state)) + "\n")
        server.stdin.flush()
        resp_line = server.stdout.readline()
        if not resp_line:
            return None, None, "fallback_server_eof", True
        resp = json.loads(resp_line)
    except Exception as exc:  # pylint: disable=broad-except
        return None, None, "fallback_bridge_error:%s" % exc, True
    if resp.get("fallback"):
        return None, None, "fallback_%s" % resp.get("reason", "server"), True
    return int(resp["chosen_bitrate_bps"]), float(resp.get("delay_s", 0.0)), \
        resp.get("reason", "bridge"), False


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
        help="Torch device override (e.g., cpu, cuda) passed to the inference server.",
    )
    parser.add_argument(
        "--server-python",
        default=os.path.expanduser("~/abr-llm-venv/bin/python"),
        help="Python 3.11 interpreter (with torch+transformers) used to run the inference "
        "server. Default: ~/abr-llm-venv/bin/python.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Time-series window (past segments per UE) for the network branch. "
        "Overridden by the checkpoint's window if present. 1 = snapshot.",
    )
    parser.add_argument(
        "--policy", choices=["model", "greedy", "pensieve", "mpc", "qmpc"], default="model",
        help="'model' = trained LLM policy; 'greedy' = causal greedy-reward teacher rule (no LLM); "
             "'pensieve' = Pensieve A3C baseline (network-only); 'mpc' = RobustMPC on the linear bitrate "
             "QoE (network-only); 'qmpc' = RobustMPC on the benchmark scoring QoE (content-aware). "
             "mpc/qmpc/greedy need no --checkpoint and no GPU.",
    )
    parser.add_argument("--w-stall", type=float, default=None,
                        help="Override w_stall for --policy greedy.")
    parser.add_argument("--thr-safety", type=float, default=None,
                        help="Throughput safety factor for --policy greedy.")
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
        "--ns3-attr", action="append", default=[], metavar="ns3::Type::Attr=value",
        help="Set an ns-3 attribute default for this run through the scenario's CommandLine "
             "(repeatable), e.g. ns3::MmWaveFFTCodebookBeamforming::CacheExpiry=1000s.")
    parser.add_argument(
        "--sim-stop-time", default=None,
        help="Override the ns-3 simStopTime (seconds). Default: scenario default (90). "
        "Use a small value for a quick smoke.",
    )
    parser.add_argument(
        "--link-rate", default=None,
        help="Backhaul bottleneck data rate (e.g. 20Mbps). Default: scenario default (100Gb/s = "
        "uncongested). Narrow it to congest the link with few UEs for an ABR stress test.",
    )
    parser.add_argument(
        "--shm-id", default=None,
        help="Unique id for the ns3-ai shared-memory segment names (so multiple ns-3 IPC runs can "
        "share a node). Default: $SLURM_JOB_ID, else the PID.",
    )
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

    # The policy (torch) runs in a separate Python 3.11 server; this process stays torch-free.
    # The server owns the metadata + scalers + model and answers one decision per request.
    server = _spawn_inference_server(args)

    # UNIQUE ns3-ai shared-memory names per job so MULTIPLE ns-3 IPC runs can share ONE node without
    # colliding on the default "My Seg" (lets us use every GPU on a multi-GPU node). The Python side
    # passes the names to Experiment(...); the ns-3 side gets the SAME prefix via the --shmPrefix CLI
    # arg (in the setting dict) and rebuilds the matching names in ConfigureDashAiMsgInterface().
    # Default tag = SLURM job id, else PID.
    shm_id = args.shm_id or os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    shm_prefix = "abr_%s" % shm_id
    seg, c2p, p2c, lock = (
        shm_prefix + "_seg", shm_prefix + "_c2p", shm_prefix + "_p2c", shm_prefix + "_lock")
    print("[Python] ns3-ai shm prefix (job-unique): %s" % shm_prefix, flush=True)

    exp = Experiment(
        "ntn-iab-quic-dash-ai",
        os.path.abspath("."),
        py_binding,
        handleFinish=True,
        useVector=False,
        shmSize=4096,
        segName=seg,
        cpp2pyMsgName=c2p,
        py2cppMsgName=p2c,
        lockableName=lock,
    )

    setting = {
        "aiEnable": "true",
        "numUes": str(args.numUes),
        "run": str(args.run),
        "algorithm": "ns3::AiDashClient",
        "dashMetadataCsv": str(args.metadata),
        "dashMetadataTrace": "dash_metadata_segments.csv",
    }
    if args.sim_stop_time is not None:
        setting["simStopTime"] = str(args.sim_stop_time)
    if args.link_rate is not None:
        setting["linkRate"] = str(args.link_rate)
    for kv in args.ns3_attr:  # --ns3::Type::Attr=value is handled by ns-3's CommandLine (Config::SetDefault)
        key, _, value = kv.partition("=")
        if not key.startswith("ns3::") or not value:
            raise SystemExit("--ns3-attr expects ns3::Type::Attribute=value, got %r" % kv)
        setting[key] = value
    setting["shmPrefix"] = shm_prefix  # ns-3 side rebuilds the matching shm names from this
    msg = exp.run(setting=setting, show_output=True)

    video = _resolve_video_name(args)
    decisions_path = _resolve_decisions_path(args, video)
    os.makedirs(os.path.dirname(decisions_path) or ".", exist_ok=True)
    decision_records = []  # one dict per decision (kept for the run summary count)
    # Stream decisions to disk incrementally (flush per decision) so a SIGKILL / timeout / OOM
    # leaves a recoverable partial JSONL instead of losing the whole run (finally won't run on SIGKILL).
    decisions_fh = open(decisions_path, "w", encoding="utf-8")

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

            t0 = time.perf_counter()
            # Delegate the decision to the 3.11 inference server; on any fallback signal
            # (missing metadata, policy error, server gone) decide with the in-process FDASH.
            chosen_bitrate, delay_s, reason, is_fallback = _bridge_decision(server, state)
            if is_fallback:
                chosen_bitrate, delay_s = fdash_heuristic(state)
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
            rec = {
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
                # Full network observation (so decisions.jsonl can rebuild model-input rows for
                # DAgger relabelling / scoring; last_segment_bitrate_bps is reconstructed offline from
                # the previous segment's chosen rung + metadata, as inference_server does).
                "buffer_differential_s": float(getattr(state, "buffer_differential_s", 0.0)),
                "segment_fetch_time_s": float(getattr(state, "segment_fetch_time_s", 0.0)),
                # Actual playout state (cumulative) from ns-3 - the ground-truth stall for QoE.
                "interruption_time_s": float(getattr(state, "interruption_time_s", 0.0)),
                "interruptions": int(getattr(state, "interruptions", 0)),
                "smoothed_rtt_ms": float(getattr(state, "smoothed_rtt_ms", -1.0)),
                "loss_rate_recent": float(getattr(state, "loss_rate_recent", -1.0)),
            }
            decision_records.append(rec)
            decisions_fh.write(json.dumps(rec) + "\n")
            decisions_fh.flush()
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
        # Shut down the inference server (close its stdin -> EOF -> clean exit).
        if server is not None:
            try:
                if server.stdin:
                    server.stdin.close()
                server.terminate()
            except Exception:  # pylint: disable=broad-except
                pass
        # Decisions were streamed + flushed to disk per-decision above (crash-resilient);
        # just close the handle here so the file ends cleanly on a normal exit.
        try:
            decisions_fh.close()
            # fallback_count = decisions the server declined (FDASH decided instead; rows carry
            # reason=fallback_*). Printed here so a silently-FDASH cell is visible in the worker log.
            print("[Python] Wrote %d decisions (%d fallback->FDASH) -> %s" % (
                len(decision_records), fallback_count, decisions_path), flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print("[Python] WARN: failed to close decisions log: %s" % exc, flush=True)
        print(
            f"[Python] Experiment killed. decisions={decisions} fallbacks={fallback_count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
