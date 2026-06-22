#!/usr/bin/env python3
"""
QoE + imitation evaluation for the ABR policy variants (V0/V1/V2).

This is **evaluation, not training**. It scores a trained policy's *deployed* behavior by
replaying its chosen rungs through the same player/buffer model the DP oracle uses, then
reports the combined QoE as a percentage of the per-session DP-oracle ceiling — the metric
the benchmark's claims actually rest on (docs/benchmark-plan.md §5). Imitation metrics
(macro-F1, per-class recall, confusion) are reported alongside but are *not* the headline.

Why replay (not just macro-F1): imitation accuracy != outcome, and ABR errors compound
across a session (a wrong rung changes the buffer, hence the next correct rung). Only a full
replay exposes that.

The QoE math is **reused** from scripts/dp_oracle.py (`_segment_transition`, `_segment_reward`,
`_switch_penalty`, `_planned_throughput`, `_safe_throughput`) so the policy QoE and the oracle
ceiling are computed with identical buffer/reward logic and the same weights
(config/oracle_reward.json). The oracle ceiling is obtained by replaying the **teacher's**
chosen rungs through the exact same function, guaranteeing apples-to-apples.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None

# Reuse the DP oracle's player/reward primitives (it self-adds scratch/ to sys.path).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from dp_oracle import (  # noqa: E402
    _planned_throughput,
    _safe_throughput,
    _segment_reward,
    _segment_transition,
    _switch_penalty,
    load_reward_config,
)

try:
    from ai_llm.train import AbrJsonlDataset, macro_f1_and_recall
    from ai_llm.model import create_model, DEFAULT_MODEL_ID
    from ai_llm.features import ladder_context_from_row, SENTINEL_MASK_ORDER
    from ai_llm.language import load_language_cache, DEFAULT_LANGUAGE_LEN
    from ai_llm.paths import (
        DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, DEFAULT_VIDEO_LANGUAGE,
        NETWORK_DIM, NUM_LADDER_RUNGS, NS3_RATE_LADDER_BPS,
    )
    from ai_llm.trace_schema import LABEL_FIELD_CLASS
except ImportError:  # run directly from scratch/ai_llm
    from train import AbrJsonlDataset, macro_f1_and_recall
    from model import create_model, DEFAULT_MODEL_ID
    from features import ladder_context_from_row, SENTINEL_MASK_ORDER
    from language import load_language_cache, DEFAULT_LANGUAGE_LEN
    from paths import (
        DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, DEFAULT_VIDEO_LANGUAGE,
        NETWORK_DIM, NUM_LADDER_RUNGS, NS3_RATE_LADDER_BPS,
    )
    from trace_schema import LABEL_FIELD_CLASS


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def replay_session(
    rungs: Sequence[int],
    rows: Sequence[Dict],
    ladders: Sequence[Sequence[Sequence[float]]],
    reward_cfg: Dict,
) -> Dict[str, float]:
    """Replay a rung sequence through the player model; return QoE + components.

    Mirrors scripts/dp_oracle.py's transition/reward exactly: download time uses the
    *planned* throughput, the first segment has no switch penalty, playback starts once the
    buffer crosses the startup threshold. `ladders[t][r] = [size_bytes, vmaf]`.
    """
    weights = reward_cfg["reward_weights"]
    player = reward_cfg["player_model"]
    solver = reward_cfg["solver"]
    seg_dur = float(player["segment_duration_s"])
    max_buffer = float(player["max_buffer_s"])
    startup = float(player["startup_threshold_s"])
    buffer_s = float(player["initial_buffer_s"])
    floor = float(solver.get("throughput_floor_bps", 1e4))
    planning_factor = float(solver.get("throughput_planning_factor", 1.0))

    playing = 0
    prev_vmaf: Optional[float] = None
    prev_rung: Optional[int] = None
    total_qoe = 0.0
    total_stall = 0.0
    switch_mag = 0.0
    switch_count = 0
    bitrate_mbps_sum = 0.0
    quality_sum = 0.0
    n = 0

    for t, row in enumerate(rows):
        rung = int(rungs[t])
        size_bytes = float(ladders[t][rung][0])
        vmaf_chosen = float(ladders[t][rung][1])
        bitrate_mbps_sum += NS3_RATE_LADDER_BPS[rung] / 1e6
        if prev_rung is not None and rung != prev_rung:
            switch_count += 1
        teacher = row.get("teacher", {}) or {}
        tp_observed = _safe_throughput(
            {"throughput_bps": float(teacher.get("throughput_bps_replay", 0.0))}, floor
        )
        tp_planned = _planned_throughput(tp_observed, floor, planning_factor)
        download_s = 8.0 * size_bytes / tp_planned
        rebuf, buffer_after, playing_after = _segment_transition(
            buffer_before_s=buffer_s,
            playing_before=playing,
            download_time_s=download_s,
            segment_duration_s=seg_dur,
            max_buffer_s=max_buffer,
            startup_threshold_s=startup,
        )
        weight = float(teacher.get("semantic_weight", 0.0))
        is_cut = int(teacher.get("is_scene_cut", 0))
        if prev_vmaf is None:
            switch_pen = 0.0
        else:
            switch_pen = _switch_penalty(vmaf_chosen, prev_vmaf, is_cut, weights)
            switch_mag += abs(vmaf_chosen - prev_vmaf)
        reward = _segment_reward(
            weight=weight,
            vmaf_chosen=vmaf_chosen,
            rebuffer_s=rebuf,
            switch_penalty_value=switch_pen,
            weights=weights,
            rung_idx=rung,
            n_rungs=NUM_LADDER_RUNGS,
        )
        total_qoe += reward
        total_stall += rebuf
        quality_sum += weight * vmaf_chosen
        buffer_s, playing, prev_vmaf = buffer_after, playing_after, vmaf_chosen
        prev_rung = rung
        n += 1

    return {
        "combined_qoe": total_qoe,
        "stall_s": total_stall,
        "switch_mag": switch_mag,
        "avg_quality": quality_sum / max(n, 1),
        "segments": float(n),
        # Core QoE pillars (aggregated corpus-wide in main()).
        "play_time_s": n * seg_dur,           # rebuffering pillar (denominator)
        "bitrate_mbps_sum": bitrate_mbps_sum,  # quality pillar (sum over segments)
        "switch_count": float(switch_count),   # smoothness pillar
    }


def _apply_language_mode(cache: Dict, mode: str, seed: int) -> Dict:
    """Rewrite a V3 language cache for the counterfactual-text probe.

    ``shuffle``: derange the video->text mapping (each video gets another video's tokens) so the
    text is well-formed but content-wrong. ``empty``: replace every entry with an all-pad sequence
    (mask all-zero) so the model's masked pool sees no real text. ``real``: unchanged.
    """
    if mode == "real":
        return cache
    if mode == "empty":
        return {
            vid: (torch.zeros_like(ids), torch.zeros_like(mask))
            for vid, (ids, mask) in cache.items()
        }
    # shuffle: deterministic derangement of the keys -> values mapping.
    import random

    keys = list(cache.keys())
    vals = list(cache.values())
    perm = list(range(len(keys)))
    rng = random.Random(seed)
    rng.shuffle(perm)
    # Avoid fixed points where possible (a true derangement) so no video keeps its own text.
    for i in range(len(perm)):
        if perm[i] == i and len(perm) > 1:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    return {keys[i]: vals[perm[i]] for i in range(len(keys))}


def load_policy_model(checkpoint_path: str, device: str):
    """Load a trained checkpoint into a variant-correct model (mirrors inference.py)."""
    import os

    ckpt = torch.load(checkpoint_path, map_location=device)
    variant = ckpt.get("variant", "v0")
    model_id = ckpt.get("model_id", DEFAULT_MODEL_ID)
    soft_tokens = int(ckpt.get("soft_tokens", 4))
    use_masks = bool(ckpt.get("use_missing_masks", True))
    network_dim = int(ckpt.get("network_dim", NETWORK_DIM + (len(SENTINEL_MASK_ORDER) if use_masks else 0)))
    window = int(ckpt.get("window", 1))

    model = create_model(
        network_dim=network_dim,
        semantic_dim=5,
        ladder_dim=2,
        d_model=256,
        num_classes=NUM_LADDER_RUNGS,
        variant=variant,
        model_id=model_id,
        soft_tokens=soft_tokens,
        # V1/V2 are trained in bf16 LoRA (train.py default, no bitsandbytes). create_model defaults
        # to 4-bit QLoRA, which needs bitsandbytes and would crash at eval — load in bf16 to match.
        qlora_4bit=False,
    ).to(device)
    model.eval()

    state = ckpt.get("state_dict", ckpt)
    if variant in ("v2", "v3"):
        adapter_dir = str(checkpoint_path) + ".lora"
        if os.path.isdir(adapter_dir) and hasattr(model.llm, "load_adapter"):
            try:
                model.llm.load_adapter(adapter_dir, adapter_name="default")
            except Exception:  # pylint: disable=broad-except
                pass
    model.load_state_dict(state, strict=False)

    # V3: rebuild the per-video language-token cache pinned by the checkpoint so eval embeds the
    # exact same text the model trained on.
    language_cache = None
    if variant == "v3":
        language_cache = load_language_cache(
            ckpt.get("language_file", DEFAULT_VIDEO_LANGUAGE),
            model_id=model_id,
            length=int(ckpt.get("language_len", DEFAULT_LANGUAGE_LEN)),
        )
    return model, window, use_masks, variant, language_cache


def main() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required for evaluate.py")

    parser = argparse.ArgumentParser(description="QoE + imitation evaluation for ABR variants.")
    parser.add_argument("--data-dir", default="train_jsonl")
    parser.add_argument("--glob", dest="glob_pattern", default="*.jsonl")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (omit with --policy oracle).")
    parser.add_argument("--policy", choices=["model", "oracle"], default="model",
                        help="'model' = run the checkpoint; 'oracle' = replay teacher rungs (sanity check, ~100%%).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reward-config", default=None, help="oracle_reward.json (default: config/).")
    parser.add_argument("--network-scaler", default=DEFAULT_NETWORK_SCALER)
    parser.add_argument("--semantic-scaler", default=DEFAULT_SEMANTIC_SCALER)
    parser.add_argument("--window", type=int, default=None, help="Override window (else read from checkpoint).")
    parser.add_argument("--include-video", action="append", default=None,
                        help="Only evaluate these video_names (repeatable) — e.g. an unseen-content split.")
    parser.add_argument("--exclude-video", action="append", default=None)
    parser.add_argument("--tag", default="", help="Label printed with the results (e.g. split name).")
    parser.add_argument(
        "--language-mode", choices=["real", "shuffle", "empty"], default="real",
        help="V3 counterfactual-text probe (eval-only, no retrain). real=correct per-video text; "
        "shuffle=each video gets a DIFFERENT video's text (wrong content, right format); empty=no "
        "text (all-pad). real==empty => the model ignores the channel; real==shuffle>empty => uses "
        "text presence but not content; real>shuffle>empty => uses content.",
    )
    parser.add_argument("--language-shuffle-seed", type=int, default=0,
                        help="Permutation seed for --language-mode shuffle (deterministic).")
    parser.add_argument(
        "--ablate", choices=["none", "semantic", "network"], default="none",
        help="Feature-ablation probe (eval-only): zero a whole input stream at inference. "
        "'semantic' zeros the 5-d semantic vector (mean after z-scoring = no info); 'network' zeros "
        "the network window. If macro-F1 is unchanged under 'semantic' but collapses under 'network', "
        "the model relies on network state and ignores the numeric semantics.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reward_cfg = load_reward_config(args.reward_config)

    # Resolve window + masks from the checkpoint when running a model.
    model = None
    window = args.window or 1
    use_masks = True
    variant = "v0"
    language_cache = None
    if args.policy == "model":
        if not args.checkpoint:
            raise SystemExit("--policy model requires --checkpoint")
        model, ckpt_window, use_masks, variant, language_cache = load_policy_model(
            args.checkpoint, device
        )
        if args.window is None:
            window = ckpt_window
        # Counterfactual-text probe: rewrite the V3 language cache before it reaches the dataset.
        if variant == "v3" and language_cache is not None and args.language_mode != "real":
            language_cache = _apply_language_mode(
                language_cache, args.language_mode, args.language_shuffle_seed
            )
            print("[language-mode=%s] V3 text channel rewritten for this eval" % args.language_mode)

    data_dir = Path(args.data_dir).resolve()
    paths = sorted(data_dir.glob(args.glob_pattern))
    if not paths:
        raise SystemExit("No shards matching %s under %s" % (args.glob_pattern, data_dir))

    dataset = AbrJsonlDataset(
        paths,
        network_scaler=args.network_scaler or None,
        semantic_scaler=args.semantic_scaler or None,
        window=window,
        use_missing_masks=use_masks,
        variant=variant,
        language_cache=language_cache,
    )

    include = set(args.include_video) if args.include_video else None
    exclude = set(args.exclude_video) if args.exclude_video else set()

    # Group rows by session, preserving the windowing already attached by the dataset.
    sessions: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in dataset._rows:  # pylint: disable=protected-access
        vid = row.get("video_name")
        if include is not None and vid not in include:
            continue
        if vid in exclude:
            continue
        sessions[(row.get("run_id"), row.get("node_id"))].append(row)

    if not sessions:
        raise SystemExit("No sessions after video filtering.")

    all_preds: List[int] = []
    all_labels: List[int] = []
    latencies_ms: List[float] = []
    policy_qoe_sum = 0.0
    oracle_qoe_sum = 0.0
    dp_reward_sum = 0.0  # Sum of teacher.immediate_reward = the DP's recorded optimum.
    stall_sum = 0.0
    switch_sum = 0.0
    quality_sum = 0.0
    playtime_sum = 0.0       # core: total play time (rebuffer-ratio denominator)
    bitrate_mbps_sum = 0.0   # core: sum of per-segment delivered bitrate (Mbps)
    switch_count_sum = 0.0   # core: total rung switches
    seg_total = 0.0          # total segments (delivered-bitrate denominator)
    n_sessions = 0

    for key, rows in sessions.items():
        rows = sorted(rows, key=lambda r: r.get("segment_id", 0))
        ladders = [ladder_context_from_row(r) for r in rows]
        labels = [int(r["teacher"][LABEL_FIELD_CLASS]) for r in rows]

        if args.policy == "oracle":
            policy_rungs = list(labels)
        else:
            policy_rungs = []
            for r in rows:
                net, sem, lad = dataset._features(r)  # pylint: disable=protected-access
                net_t = torch.tensor([net], dtype=torch.float32, device=device)
                sem_t = torch.tensor([sem], dtype=torch.float32, device=device)
                lad_t = torch.tensor([lad], dtype=torch.float32, device=device)
                # Feature-ablation probe: zero an input stream (z-scored 0 = uninformative mean).
                if args.ablate == "semantic":
                    sem_t = torch.zeros_like(sem_t)
                elif args.ablate == "network":
                    net_t = torch.zeros_like(net_t)
                lang_args = ()
                if variant == "v3":
                    li, lm = dataset._language(r)  # pylint: disable=protected-access
                    lang_args = (li.unsqueeze(0).to(device), lm.unsqueeze(0).to(device))
                t0 = time.perf_counter()
                with torch.inference_mode():
                    logits, _ = model(net_t, sem_t, lad_t, *lang_args)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                policy_rungs.append(int(torch.argmax(logits[0]).item()))

        all_preds.extend(policy_rungs)
        all_labels.extend(labels)

        pol = replay_session(policy_rungs, rows, ladders, reward_cfg)
        ora = replay_session(labels, rows, ladders, reward_cfg)
        policy_qoe_sum += pol["combined_qoe"]
        oracle_qoe_sum += ora["combined_qoe"]
        dp_reward_sum += sum(float((r.get("teacher", {}) or {}).get("immediate_reward", 0.0)) for r in rows)
        stall_sum += pol["stall_s"]
        switch_sum += pol["switch_mag"]
        quality_sum += pol["avg_quality"]
        playtime_sum += pol["play_time_s"]
        bitrate_mbps_sum += pol["bitrate_mbps_sum"]
        switch_count_sum += pol["switch_count"]
        seg_total += pol["segments"]
        n_sessions += 1

    macro_f1, recalls = macro_f1_and_recall(all_preds, all_labels, NUM_LADDER_RUNGS)
    acc = sum(int(p == t) for p, t in zip(all_preds, all_labels)) / max(len(all_labels), 1)
    pct_of_oracle = (policy_qoe_sum / oracle_qoe_sum * 100.0) if abs(oracle_qoe_sum) > 1e-9 else float("nan")

    # Core QoE pillars (quality / rebuffering / smoothness), aggregated corpus-wide.
    rebuffer_ratio = (stall_sum / playtime_sum) if playtime_sum > 1e-9 else float("nan")
    mean_bitrate_mbps = bitrate_mbps_sum / max(seg_total, 1.0)

    # 6x6 confusion (rows = true, cols = pred).
    confusion = [[0] * NUM_LADDER_RUNGS for _ in range(NUM_LADDER_RUNGS)]
    for p, t in zip(all_preds, all_labels):
        confusion[t][p] += 1

    tag = (" [%s]" % args.tag) if args.tag else ""
    print("=== ABR evaluation%s: policy=%s, %d sessions, %d decisions ===" % (
        tag, args.policy, n_sessions, len(all_labels)))
    print("-- Imitation --")
    print("  accuracy   : %.3f" % acc)
    print("  macro-F1   : %.3f" % macro_f1)
    print("  recall/rung: [%s]" % ", ".join("%.2f" % r for r in recalls))
    print("  confusion (true x pred):")
    for t in range(NUM_LADDER_RUNGS):
        print("    %d: %s" % (t, confusion[t]))
    print("-- QoE (replayed through the oracle's player model) --")
    print("  combined QoE (policy)      : %.2f" % policy_qoe_sum)
    print("  combined QoE (oracle replay): %.2f" % oracle_qoe_sum)
    print("  DP recorded reward (xcheck): %.2f  (Sum teacher.immediate_reward)" % dp_reward_sum)
    print("  %% of oracle ceiling        : %.1f%%" % pct_of_oracle)
    print("  total stall (s)      : %.2f" % stall_sum)
    print("  mean switch magnitude: %.3f" % (switch_sum / max(n_sessions, 1)))
    print("  mean weighted quality: %.3f" % (quality_sum / max(n_sessions, 1)))
    print("-- QoE (core) --")
    print("  rebuffer ratio        : %.4f  (stall / play time)" % rebuffer_ratio)
    print("  mean delivered bitrate: %.3f Mbps" % mean_bitrate_mbps)
    print("  switch count (total)  : %d  (mean %.2f / session)" % (
        int(switch_count_sum), switch_count_sum / max(n_sessions, 1)))
    if latencies_ms:
        print("-- Systems --")
        print("  decision latency p50/p95 (ms): %.2f / %.2f" % (
            _percentile(latencies_ms, 0.50), _percentile(latencies_ms, 0.95)))


if __name__ == "__main__":
    main()
