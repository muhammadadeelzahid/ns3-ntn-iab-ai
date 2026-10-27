#!/usr/bin/env python3
"""Torch inference server for the live ns-3 ABR loop (runs under Python 3.11).

Why this exists
---------------
The ns3-ai binding (`ns3ai_abr_dash_ai_py`) is a `cpython-36m` extension - importable ONLY by
Python 3.6 - while the policy model needs torch 2.x + transformers (Python 3.11). The two cannot
live in one interpreter. So `run_ai_dash_llm.py` (3.6) keeps the ns-3 IPC loop and **spawns this
server** (3.11), exchanging one line-delimited JSON message per decision over stdin/stdout:

    client -> server (stdin) : {"node_id":.., "segment_id":.., <19 state fields>}
    server -> client (stdout): {"chosen_bitrate_bps":.., "delay_s":.., "reason":..}
                            or  {"fallback": true, "reason": ".."}   (client runs FDASH)

stdout carries ONLY JSON responses (one per line); all human logs go to stderr. The first stdout
line is a `{"ready": true}` handshake emitted once the (possibly slow) model load finishes.

Because the ns-3 simulation is PAUSED while it waits for the decision, the pipe round-trip is free
with respect to QoE - it only costs wall-clock.
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
    from ai_llm.inference import LlmAbrPolicy, PensievePolicy
    from ai_llm.metadata_loader import load_metadata_index, LADDER_LEVELS
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS
except ImportError:  # run directly from scratch/ai_llm
    from inference import LlmAbrPolicy, PensievePolicy
    from metadata_loader import load_metadata_index, LADDER_LEVELS
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NS3_RATE_LADDER_BPS


def _bps_to_rung(bps: float) -> int:
    """Nearest ladder rung for a chosen bitrate (bps)."""
    return min(range(len(NS3_RATE_LADDER_BPS)), key=lambda i: abs(NS3_RATE_LADDER_BPS[i] - bps))


def _feasible_rung(throughput_bps: float, segment_meta, safety: float,
                   buffer_s: float, seg_dur_s: float) -> int:
    """Highest rung whose download won't underrun the buffer, given current throughput.

    Download time of rung r = actual_bitrate[r]*seg_dur / throughput. To avoid a stall it must finish
    before the buffer empties, so the no-stall budget on the bitrate is

        actual_bitrate[r] <= throughput * max(buffer_s, seg_dur) / seg_dur * safety

    At low buffer this reduces to ~throughput (strictly sustainable); at high buffer it scales up
    (buffer can be spent), so steady-state buffer-spending is preserved while a single egregious
    over-ask that would drain even a healthy buffer is blocked. Uses actual (VBR) bitrate - the bytes
    really delivered - not the nominal ladder. Floor (0) if nothing fits.
    """
    head = max(buffer_s, seg_dur_s) / max(seg_dur_s, 1e-6)
    budget = throughput_bps * head * safety
    r = 0
    for i, lvl in enumerate(LADDER_LEVELS):
        if float(segment_meta.actual_bitrate_kbps.get(lvl, 0.0)) * 1000.0 <= budget:
            r = i
    return r


def _greedy_rung(thr, buf, sizes, vmafs, prev_vmaf, sem, cut, W) -> int:
    """Closed-loop greedy-reward CAUSAL teacher as a serving policy (mirrors benchmark/relabel_greedy.py
    greedy_rung EXACTLY, incl. the thr_safety download-time discount). Per segment, pick the rung that
    maximises the immediate QoE reward using only present info, so this is the deployable causal ceiling
    the LLM is trained to imitate. No no-stall guard (pure teacher)."""
    thr_eff = max(thr * W.get("thr_safety", 1.0), 1.0)
    best, bi = -1e18, 0
    for r in range(len(LADDER_LEVELS)):
        dl = 8.0 * sizes[r] / thr_eff
        rebuf = max(0.0, dl - buf)
        sw = 0.0 if prev_vmaf is None else W["w_switch"] * abs(vmafs[r] - prev_vmaf) * (1.0 - W["scene_cut_discount"] * cut)
        q = W["w_vmaf"] * sem * vmafs[r] + W.get("w_quality_sensitive", 0.0) * sem * vmafs[r]
        rew = q - W["w_stall"] * rebuf - sw
        if rew > best:
            best, bi = rew, r
    return bi


def log(msg: str) -> None:
    print("[infserver] %s" % msg, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Torch policy inference server (stdin/stdout JSON).")
    ap.add_argument("--checkpoint", default=None, help="Trained policy checkpoint (V0/V1/V2).")
    ap.add_argument("--policy", choices=["model", "greedy", "pensieve"], default="model",
                    help="'model' = trained LLM policy; 'greedy' = causal greedy-reward teacher rule "
                         "(no LLM, no guard) - the deployable causal ceiling the model imitates; "
                         "'pensieve' = Pensieve A3C baseline (network-only, no guard).")
    ap.add_argument("--reward-config", default=os.path.join(_HERE, "config", "oracle_reward.json"),
                    help="Reward weights + semantic-weight mixture for --policy greedy.")
    ap.add_argument("--w-stall", type=float, default=None, help="Override w_stall for --policy greedy.")
    ap.add_argument("--thr-safety", type=float, default=None,
                    help="Throughput safety factor for --policy greedy (discounts the estimate).")
    ap.add_argument("--metadata", required=True, help="Segment metadata CSV.")
    ap.add_argument("--network-scaler", default=DEFAULT_NETWORK_SCALER)
    ap.add_argument("--semantic-scaler", default=DEFAULT_SEMANTIC_SCALER)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default=None, help="cpu/cuda (auto if omitted).")
    ap.add_argument("--window", type=int, default=1, help="Overridden by the checkpoint if present.")
    ap.add_argument("--use-missing-masks", dest="use_missing_masks", action="store_true", default=True)
    ap.add_argument("--no-missing-masks", dest="use_missing_masks", action="store_false")
    # Buffer-dynamics-aware no-stall guard: cap the rung so its download won't underrun the buffer
    # (actual_bitrate <= throughput * max(buffer,seg_dur)/seg_dur * safety). Always on; low buffer =>
    # ~sustainable, high buffer => proportional buffer-spending. Catches the cold-start over-ask AND a
    # single over-ask that would drain a healthy buffer. Disable with --guard-safety 0.
    ap.add_argument("--guard-safety", type=float, default=0.9,
                    help="Safety factor on the guard's no-stall bitrate budget (0=guard off).")
    ap.add_argument("--guard-seg-dur-s", type=float, default=3.3,
                    help="Segment duration (s) used in the guard's download-feasibility budget.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # The server owns metadata + scalers + model (reuses all existing inference code unchanged).
    metadata_index = load_metadata_index(args.metadata, args.semantic_scaler)
    log("metadata rows: %d from %s" % (len(metadata_index), args.metadata))
    # Segment-id convention: ns-3's AiDashClient reports 1-indexed segment ids (first segment = 1),
    # but the metadata index - and the training data (dataset_builder uses the 0-indexed trace
    # segment_id) - are 0-indexed (segments 0..N-1). So the live lookup must subtract 1, otherwise the
    # policy is fed the NEXT segment's encoding ladder and the last segment overflows. (The earlier
    # 'looping/modulo' patch misdiagnosed this off-by-one and is removed.)

    # --policy greedy: the causal greedy-reward teacher rule as a serving policy (no LLM). Precompute the
    # per-segment semantic_weight (min-max mixture across this video) + reward weights EXACTLY as the
    # offline teacher (benchmark/relabel_greedy.py via dp_oracle), so serve == train.
    total_chunks = len(metadata_index)  # used by --policy pensieve for chunks-remaining
    policy = None
    greedy_W = None
    sem_weights = None
    if args.policy == "greedy":
        sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "scripts")))
        from dp_oracle import load_reward_config, compute_semantic_weights  # type: ignore
        cfg = load_reward_config(args.reward_config)
        greedy_W = dict(cfg["reward_weights"])
        if args.w_stall is not None:
            greedy_W["w_stall"] = args.w_stall
        if args.thr_safety is not None:
            greedy_W["thr_safety"] = args.thr_safety
        sem_weights = compute_semantic_weights(metadata_index, cfg["semantic_weight_mixture"])
        log("greedy teacher policy: weights=%s" % {k: round(greedy_W.get(k, 0.0), 3) for k in
            ("w_vmaf", "w_stall", "w_switch", "scene_cut_discount", "w_quality_sensitive", "thr_safety")})
    elif args.policy == "pensieve":
        policy = PensievePolicy(checkpoint_path=args.checkpoint, device=args.device)
        log("pensieve policy loaded (checkpoint=%s, device=%s, total_chunks=%d)" % (
            args.checkpoint, getattr(policy, "_device", "?"), total_chunks))
    else:
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

    prev_vmaf_by_node: dict = {}

    # READY handshake (first stdout line) - tells the 3.6 client the model is loaded.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    # Per-UE last chosen rung, to reconstruct `last_segment_bitrate_bps` (the VBR-actual bitrate of
    # the previously downloaded segment) exactly as the offline trace pipeline does - the live struct
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
        elif args.policy == "greedy":
            # Causal greedy-reward teacher rule (no LLM, no guard): argmax immediate QoE over the ladder.
            node = int(req.get("node_id", -1))
            seg0 = seg_id - 1 if metadata_index.get(seg_id - 1) is not None else seg_id
            thr = float(req.get("throughput_estimate_bps", 0.0))
            buf = float(req.get("buffer_level_s", 0.0))
            sizes = [float(segment_meta.size_bytes.get(lv, 0.0)) for lv in LADDER_LEVELS]
            vmafs = [float(segment_meta.vmaf.get(lv, 0.0)) for lv in LADDER_LEVELS]
            sem = float(sem_weights.get(seg0, 0.0))
            cut = 1 if float(segment_meta.semantic_raw.get("scene_cut_count", 0.0)) > 0 else 0
            rung = _greedy_rung(thr, buf, sizes, vmafs, prev_vmaf_by_node.get(node), sem, cut, greedy_W)
            prev_vmaf_by_node[node] = vmafs[rung]
            resp = {"chosen_bitrate_bps": int(NS3_RATE_LADDER_BPS[rung]), "delay_s": 0.0,
                    "reason": "greedy_teacher"}
        elif args.policy == "pensieve":
            # Pensieve A3C baseline (network-only, NO no-stall guard - keep it as-published).
            seg0 = seg_id - 1 if metadata_index.get(seg_id - 1) is not None else seg_id
            chunks_remaining = max(1, total_chunks - seg0)
            state = SimpleNamespace(**req)
            try:
                decision = policy.predict(state, segment_meta, chunks_remaining, total_chunks)
                resp = {"chosen_bitrate_bps": int(decision.chosen_bitrate_bps),
                        "delay_s": float(decision.delay_s), "reason": decision.reason}
            except Exception as exc:  # pylint: disable=broad-except
                resp = {"fallback": True, "reason": "policy_error:%s" % exc}
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
                chosen_bps = float(decision.chosen_bitrate_bps)
                reason = decision.reason
                # Buffer-dynamics-aware guard: clamp the rung so its download won't underrun the buffer
                # given current throughput (no-stall feasibility). Always on (the buffer is inside the
                # budget): low buffer => ~sustainable, high buffer => proportional spend. Catches both the
                # cold-start over-ask AND a single egregious over-ask that would drain a healthy buffer.
                buf = float(req.get("buffer_level_s", 0.0))
                thr = float(req.get("throughput_estimate_bps", 0.0))
                if args.guard_safety > 0.0 and thr > 0.0:
                    chosen_rung = _bps_to_rung(chosen_bps)
                    cap = _feasible_rung(thr, segment_meta, args.guard_safety, buf, args.guard_seg_dur_s)
                    if chosen_rung > cap:
                        chosen_bps = float(NS3_RATE_LADDER_BPS[cap])
                        reason = "%s+guard(buf%.1f:r%d->r%d)" % (reason, buf, chosen_rung, cap)
                last_rung_by_node[node] = _bps_to_rung(chosen_bps)
                resp = {
                    "chosen_bitrate_bps": int(chosen_bps),
                    "delay_s": float(decision.delay_s),
                    "reason": reason,
                }
            except Exception as exc:  # pylint: disable=broad-except
                resp = {"fallback": True, "reason": "policy_error:%s" % exc}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    log("stdin closed; exiting.")


if __name__ == "__main__":
    main()
