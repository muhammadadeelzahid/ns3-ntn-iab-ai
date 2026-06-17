#!/usr/bin/env python3
"""
Multimodal policy model for ABR decisions.
"""

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

try:
    from ai_llm.paths import NUM_LADDER_RUNGS
except ImportError:
    from paths import NUM_LADDER_RUNGS


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for LLM policy inference/training.")


if nn is not None:

    class MultimodalEncoder(nn.Module):
        def __init__(
            self,
            network_dim: int,
            semantic_dim: int,
            ladder_dim: int,
            d_model: int,
            hidden_net: int = 128,
            hidden_sem: int = 64,
            hidden_ladder: int = 32,
        ) -> None:
            super(MultimodalEncoder, self).__init__()
            # Network branch is a 1D-CNN over the time axis so it consumes either a
            # snapshot [B, network_dim] or a window [B, W, network_dim]. Conv over time
            # (channels = features) captures throughput/buffer/RTT trends; adaptive pool
            # makes it window-length agnostic (W is a loader hyperparameter, not a model
            # one). W=1 reduces to the snapshot baseline.
            self.network_cnn = nn.Sequential(
                nn.Conv1d(network_dim, hidden_net, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_net, hidden_net, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.network_pool = nn.AdaptiveAvgPool1d(1)
            self.network_ln = nn.LayerNorm(hidden_net)
            self.semantic_mlp = nn.Sequential(
                nn.Linear(semantic_dim, hidden_sem),
                nn.SiLU(),
                nn.LayerNorm(hidden_sem),
                nn.Dropout(0.1),
                nn.Linear(hidden_sem, hidden_sem),
                nn.SiLU(),
                nn.LayerNorm(hidden_sem),
            )
            self.ladder_proj = nn.Sequential(
                nn.Linear(ladder_dim, hidden_ladder),
                nn.SiLU(),
            )
            self.fuse = nn.Linear(hidden_net + hidden_sem + hidden_ladder, d_model)

        def forward(
            self,
            network_norm: "torch.Tensor",
            semantic_norm: "torch.Tensor",
            ladder_context: "torch.Tensor",
        ) -> "torch.Tensor":
            # network_norm: [B, network_dim] (snapshot) or [B, W, network_dim] (window).
            x = network_norm
            if x.dim() == 2:
                x = x.unsqueeze(1)            # [B, 1, C]
            x = x.transpose(1, 2)             # [B, C, W] for Conv1d
            x = self.network_cnn(x)           # [B, hidden_net, W]
            x = self.network_pool(x).squeeze(-1)  # [B, hidden_net]
            net_emb = self.network_ln(x)
            sem_emb = self.semantic_mlp(semantic_norm)

            # ladder_context: [batch, num_levels, ladder_dim]
            ladder_tokens = self.ladder_proj(ladder_context)
            ladder_emb = ladder_tokens.mean(dim=1)

            fused = torch.cat([net_emb, sem_emb, ladder_emb], dim=-1)
            return self.fuse(fused)


    class AbrPolicyHead(nn.Module):
        def __init__(self, d_model: int, num_classes: int = NUM_LADDER_RUNGS) -> None:
            super(AbrPolicyHead, self).__init__()
            self.trunk = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.SiLU(),
            )
            self.bitrate_head = nn.Linear(128, num_classes)

        def forward(self, state_hidden: "torch.Tensor") -> Tuple["torch.Tensor", None]:
            x = self.trunk(state_hidden)
            return self.bitrate_head(x), None


    class LlmAbrModel(nn.Module):
        """
        Phase-2 model:
        - Lightweight multimodal encoder
        - Placeholder backbone projection (kept explicit for future HF backbone integration)
        - ABR policy head
        """

        def __init__(
            self,
            network_dim: int = 7,
            semantic_dim: int = 5,
            ladder_dim: int = 2,
            d_model: int = 256,
            num_classes: int = NUM_LADDER_RUNGS,
            model_id: str = DEFAULT_MODEL_ID,
        ) -> None:
            super(LlmAbrModel, self).__init__()
            self.model_id = model_id
            self.encoder = MultimodalEncoder(
                network_dim=network_dim,
                semantic_dim=semantic_dim,
                ladder_dim=ladder_dim,
                d_model=d_model,
            )
            # Explicit backbone projection hook for later Qwen/Llama integration.
            self.backbone_projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
            )
            self.policy_head = AbrPolicyHead(d_model=d_model, num_classes=num_classes)

        def forward(
            self,
            network_norm: "torch.Tensor",
            semantic_norm: "torch.Tensor",
            ladder_context: "torch.Tensor",
        ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
            state_token = self.encoder(network_norm, semantic_norm, ladder_context)
            state_hidden = self.backbone_projector(state_token)
            return self.policy_head(state_hidden)


def create_model(**kwargs):
    _require_torch()
    return LlmAbrModel(**kwargs)
