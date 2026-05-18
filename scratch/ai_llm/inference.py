#!/usr/bin/env python3
"""
Inference-time ABR policy wrapper.

This keeps a deterministic policy stub while the full LLM policy is integrated.
"""

from typing import Dict, List, Optional


# ns-3 DASH representation ladder (must stay aligned with C++).
NS3_RATE_LADDER_BPS: List[int] = [
    45000,
    89000,
    131000,
    178000,
    221000,
    263000,
    334000,
    396000,
    522000,
    595000,
    791000,
    1033000,
    1245000,
    1547000,
    2134000,
    2484000,
    3079000,
    3527000,
    3840000,
    4220000,
    9500000,
    15000000,
    30000000,
    66000000,
    85000000,
]


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

    def __init__(self, ladder_bps: Optional[List[int]] = None, max_policy_rate_bps: int = 9500000) -> None:
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
