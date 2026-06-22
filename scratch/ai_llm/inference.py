#!/usr/bin/env python3
"""
Inference-time ABR policy wrappers.
"""

import os
from collections import deque
from typing import Deque, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from ai_llm.features import (
        build_model_input,
        load_scaler,
        network_observation,
        normalize_network_vector,
        normalize_network_vector_with_masks,
        SENTINEL_MASK_ORDER,
    )
    from ai_llm.model import create_model
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NETWORK_DIM, NS3_RATE_LADDER_BPS
except ImportError:
    from features import (
        build_model_input,
        load_scaler,
        network_observation,
        normalize_network_vector,
        normalize_network_vector_with_masks,
        SENTINEL_MASK_ORDER,
    )
    from model import create_model
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NETWORK_DIM, NS3_RATE_LADDER_BPS


class PolicyDecision:
    def __init__(self, chosen_bitrate_bps: int, delay_s: float, reason: str) -> None:
        self.chosen_bitrate_bps = chosen_bitrate_bps
        self.delay_s = delay_s
        self.reason = reason


class RuleBasedLlmPolicy:
    """
    Placeholder policy that consumes semantic + network state and emits a safe bitrate.
    Replace this class internals with LLM inference in later phases.
    """

    def __init__(self, ladder_bps: Optional[List[int]] = None, max_policy_rate_bps: int = 12_000_000) -> None:
        self.ladder_bps = ladder_bps or list(NS3_RATE_LADDER_BPS)
        self.max_policy_rate_bps = max_policy_rate_bps

    def predict(self, state, semantic_features: Dict[str, float]) -> PolicyDecision:
        throughput = float(
            getattr(state, "throughput_ema_bps", 0.0) or getattr(state, "throughput_estimate_bps", 0.0)
        )
        buffer_level = float(getattr(state, "buffer_level_s", 0.0))
        quality_sensitivity = float(semantic_features.get("quality_sensitivity", 0.0))
        scene_cut_count = float(semantic_features.get("scene_cut_count", 0.0))

        if throughput <= 0.0:
            return PolicyDecision(
                chosen_bitrate_bps=self.ladder_bps[0],
                delay_s=0.0,
                reason="fallback_low_throughput",
            )

        # Safety margin on estimated throughput to avoid over-shooting.
        target = 0.85 * throughput
        if buffer_level < 8.0:
            target *= 0.75
        elif buffer_level > 30.0:
            target *= 1.05

        # Semantic-driven cautious behavior around cuts or high sensitivity.
        if scene_cut_count > 0.0:
            target *= 0.95
        if quality_sensitivity > 25.0:
            target *= 1.10

        chosen = self.ladder_bps[0]
        for rate in self.ladder_bps:
            if target >= rate:
                chosen = rate
        chosen = min(chosen, self.max_policy_rate_bps)

        return PolicyDecision(
            chosen_bitrate_bps=int(chosen),
            delay_s=0.0,
            reason="rule_based_stub_capped",
        )


class LlmAbrPolicy:
    """
    Phase-2 policy wrapper:
    - build_model_input() for normalized tensors
    - lightweight model forward pass
    - deterministic fallback to rule-based policy
    """

    def __init__(
        self,
        ladder_bps: Optional[List[int]] = None,
        checkpoint_path: Optional[str] = None,
        network_scaler_path: str = DEFAULT_NETWORK_SCALER,
        semantic_scaler_path: str = DEFAULT_SEMANTIC_SCALER,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        window: int = 1,
        use_missing_masks: bool = True,
    ) -> None:
        self.ladder_bps = ladder_bps or list(NS3_RATE_LADDER_BPS)
        self.network_scaler_path = network_scaler_path
        self.semantic_scaler_path = semantic_scaler_path
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self._fallback = RuleBasedLlmPolicy(ladder_bps=self.ladder_bps)

        # Time-series window: per-UE ring buffer of raw network observations. The window
        # size is pinned from the checkpoint when present (so inference matches training);
        # the constructor value is the default/override.
        self._window = max(1, int(window))
        # Missing-indicator channels (RTT/loss); also pinned from the checkpoint so the
        # live network_dim matches the trained one.
        self._use_missing_masks = bool(use_missing_masks)
        self._net_scaler = load_scaler(network_scaler_path)
        self._history: Dict[int, Deque[Dict[str, float]]] = {}

        self._ready = False
        self._model = None
        self._device = "cpu"
        self._init_error = ""
        self._setup_model(device)

    def _setup_model(self, device: Optional[str]) -> None:
        if torch is None:
            self._init_error = "torch_unavailable"
            return
        try:
            if device:
                self._device = device
            elif torch.cuda.is_available():
                self._device = "cuda"
            # Read checkpoint metadata first so window + missing-mask flag (hence
            # network_dim) and the variant/backbone match training before construction.
            ckpt = None
            variant = "v0"
            soft_tokens = 4
            if self.checkpoint_path:
                ckpt = torch.load(self.checkpoint_path, map_location=self._device)
                if isinstance(ckpt, dict):
                    if "window" in ckpt:
                        self._window = max(1, int(ckpt["window"]))  # match training window
                    if "use_missing_masks" in ckpt:
                        self._use_missing_masks = bool(ckpt["use_missing_masks"])
                    variant = ckpt.get("variant", "v0")
                    soft_tokens = int(ckpt.get("soft_tokens", 4))
                    if ckpt.get("model_id"):
                        self.model_id = ckpt["model_id"]
            network_dim = NETWORK_DIM + (len(SENTINEL_MASK_ORDER) if self._use_missing_masks else 0)
            self._model = create_model(
                network_dim=network_dim,
                semantic_dim=5,
                ladder_dim=2,
                d_model=256,
                num_classes=len(self.ladder_bps),
                model_id=self.model_id,
                variant=variant,
                soft_tokens=soft_tokens,
            ).to(self._device)
            self._model.eval()
            if ckpt is not None:
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                # v2: reattach the trained LoRA adapter saved next to the checkpoint
                # (the frozen base was reloaded from HF inside create_model).
                if variant == "v2":
                    adapter_dir = str(self.checkpoint_path) + ".lora"
                    if os.path.isdir(adapter_dir) and hasattr(self._model.llm, "load_adapter"):
                        try:
                            self._model.llm.load_adapter(adapter_dir, adapter_name="default")
                        except Exception:  # pylint: disable=broad-except
                            self._model.llm.load_state_dict(  # fallback: merged weights in state
                                state, strict=False,
                            )
                # encoder + soft_proj + head (and v1 full LLM) load from the state dict.
                self._model.load_state_dict(state, strict=False)
            self._ready = True
        except Exception as exc:  # pylint: disable=broad-except
            self._init_error = "model_init_failed:%s" % exc
            self._ready = False
            self._model = None

    def _network_window(self, state) -> List[List[float]]:
        """Append this UE's raw observation to its ring buffer and return [W, 7] (or
        [W, 9] when missing-masks are enabled).

        Left-pads with the earliest available observation until the buffer holds W steps.
        Uses the same helper as the train loader so the channel order matches exactly.
        """
        node = int(getattr(state, "node_id", 0))
        buf = self._history.get(node)
        if buf is None:
            buf = deque(maxlen=self._window)
            self._history[node] = buf
        buf.append(network_observation(state))
        window = list(buf)
        if len(window) < self._window:
            window = [window[0]] * (self._window - len(window)) + window
        norm = normalize_network_vector_with_masks if self._use_missing_masks else normalize_network_vector
        return [norm(o, self._net_scaler) for o in window]

    def reset_history(self) -> None:
        """Clear per-UE windows (call between independent runs/episodes)."""
        self._history.clear()

    def predict(self, state, segment_meta) -> PolicyDecision:
        try:
            model_input = build_model_input(
                state=state,
                segment_meta=segment_meta,
                network_scaler_path=self.network_scaler_path,
                semantic_scaler_path=self.semantic_scaler_path,
            )
            if not self._ready or self._model is None:
                fallback = self._fallback.predict(state, segment_meta.semantic_norm)
                fallback.reason = "fallback_%s" % (self._init_error or "model_not_ready")
                return fallback

            if torch is None:
                raise RuntimeError("torch_unavailable")
            # Network branch uses the time-series window; semantic + ladder are current-segment.
            net_window = self._network_window(state)  # [W, 7] or [W, 9] with masks
            network_t = torch.tensor([net_window], dtype=torch.float32, device=self._device)  # [1, W, C]
            semantic_t = torch.tensor([model_input.semantic_norm], dtype=torch.float32, device=self._device)
            ladder_t = torch.tensor([model_input.ladder_context], dtype=torch.float32, device=self._device)
            with torch.inference_mode():
                logits, _delay_unused = self._model(network_t, semantic_t, ladder_t)

            if logits.ndim != 2 or logits.shape[1] != len(self.ladder_bps):
                raise RuntimeError("invalid_logits_shape:%s" % (tuple(logits.shape),))
            best_idx = int(torch.argmax(logits[0]).item())
            if best_idx < 0 or best_idx >= len(self.ladder_bps):
                raise RuntimeError("invalid_rate_index:%d" % best_idx)

            return PolicyDecision(
                chosen_bitrate_bps=int(self.ladder_bps[best_idx]),
                delay_s=0.0,
                reason="llm_policy_model",
            )
        except Exception as exc:  # pylint: disable=broad-except
            fallback = self._fallback.predict(state, segment_meta.semantic_norm)
            fallback.reason = "fallback_policy_error:%s" % exc
            return fallback
