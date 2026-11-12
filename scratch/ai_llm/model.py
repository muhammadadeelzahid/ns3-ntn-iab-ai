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


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Variants of the ABR policy (see docs/benchmark-plan.md §1):
#   v0 = no-LLM encoder -> placeholder projector -> head (the strict sub-network).
#   v1 = random-initialized LLM, full fine-tune (isolates pretrained knowledge).
#   v2 = pretrained LLM, frozen + LoRA, numeric soft-tokens (the Tier-1 workhorse).
#   v3 = v2 + content-derived language text concatenated with the soft-tokens (the language channel).
VARIANTS = ("v0", "v1", "v2", "v3")
DEFAULT_SOFT_TOKENS = 4

try:
    from ai_llm.paths import NUM_LADDER_RUNGS
except ImportError:
    from paths import NUM_LADDER_RUNGS


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for LLM policy inference/training.")


def build_llm_backbone(
    variant: str,
    model_id: str = DEFAULT_MODEL_ID,
    qlora_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Construct the transformer backbone for an LLM variant.

    Returns ``(llm_module, hidden_size)``. Heavy deps (transformers/peft/bitsandbytes)
    are imported lazily here so the v0 / CPU path never needs them.

    - v2/v3: pretrained ``model_id``, base frozen + LoRA adapters (QLoRA 4-bit optional).
      v3 is architecturally identical to v2 here - they differ only in the input sequence
      assembled in ``LlmAbrModel.forward`` (v3 concatenates text-token embeddings).
    - v1: same config as ``model_id`` but **random-initialized** and **fully trainable**.
    """
    _require_torch()
    if variant not in ("v1", "v2", "v3"):
        raise ValueError("build_llm_backbone is only for v1/v2/v3, got %r" % variant)

    from transformers import AutoConfig, AutoModelForCausalLM

    if variant in ("v2", "v3"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        quant_config = None
        if qlora_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
        )
        if qlora_4bit:
            base = prepare_model_for_kbit_training(base)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        llm = get_peft_model(base, lora_config)
        hidden_size = base.config.hidden_size
    else:  # v1: random init, all weights trainable
        config = AutoConfig.from_pretrained(model_id)
        llm = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        llm.gradient_checkpointing_enable()
        for param in llm.parameters():
            param.requires_grad = True
        hidden_size = config.hidden_size

    return llm, hidden_size


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


    class SoftTokenProjection(nn.Module):
        """Map the encoder's fused embedding into the LLM token-embedding space.

        e_fused [B, d_fuse] -> Linear(d_fuse -> k*H) -> LayerNorm -> soft_tokens [B, k, H],
        injected as a prefix via the LLM inputs_embeds. Separate from the encoder so the
        encoder stays identical across variants and v0 is a strict sub-network.

        The per-token LayerNorm is required: without it the raw projection emits tokens far
        larger than the frozen LLM's real embeddings, so the LLM saturates and training
        collapses to the majority class. LayerNorm rescales each token to embedding scale.
        """

        def __init__(self, d_fuse: int, num_tokens: int, hidden_size: int) -> None:
            super(SoftTokenProjection, self).__init__()
            self.num_tokens = num_tokens
            self.hidden_size = hidden_size
            self.proj = nn.Linear(d_fuse, num_tokens * hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, e_fused: "torch.Tensor") -> "torch.Tensor":
            batch = e_fused.shape[0]
            flat = self.proj(e_fused)                              # [B, k*H]
            tokens = flat.view(batch, self.num_tokens, self.hidden_size)  # [B, k, H]
            return self.norm(tokens)                               # per-token LayerNorm -> in-distribution scale


    class LlmAbrModel(nn.Module):
        """
        Multimodal ABR policy with four variants (docs/benchmark-plan.md §4):

        - **v0**: encoder -> backbone_projector (placeholder) -> head. No LLM. The
          strict sub-network; state-dict keys are unchanged from the original model.
        - **v1/v2**: encoder -> SoftTokenProjection -> LLM (inputs_embeds) -> pool over
          the k soft tokens -> head. v2 = pretrained frozen+LoRA, v1 = random full-FT.
        - **v3**: v2 + a per-video language string. The LLM input sequence is
          ``concat(soft_tokens[k], text_emb[L])`` (text embedded via the LLM's own
          ``get_input_embeddings()``), masked-mean-pooled over real positions -> head.
          Architecturally identical to v2 (frozen+LoRA); only the input sequence differs.
        """

        def __init__(
            self,
            network_dim: int = 7,
            semantic_dim: int = 5,
            ladder_dim: int = 2,
            d_model: int = 256,
            num_classes: int = NUM_LADDER_RUNGS,
            model_id: str = DEFAULT_MODEL_ID,
            variant: str = "v0",
            soft_tokens: int = DEFAULT_SOFT_TOKENS,
            qlora_4bit: bool = True,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            llm: Optional["nn.Module"] = None,
            hidden_size: Optional[int] = None,
            encoder_residual: bool = True,
        ) -> None:
            super(LlmAbrModel, self).__init__()
            if variant not in VARIANTS:
                raise ValueError("variant must be one of %s, got %r" % (VARIANTS, variant))
            self.model_id = model_id
            self.variant = variant
            self.soft_tokens = soft_tokens
            self.encoder = MultimodalEncoder(
                network_dim=network_dim,
                semantic_dim=semantic_dim,
                ladder_dim=ladder_dim,
                d_model=d_model,
            )

            if variant == "v0":
                # Explicit placeholder projection: keeps v0 a strict sub-network and its
                # checkpoint keys stable. No LLM is loaded.
                self.backbone_projector = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.LayerNorm(d_model),
                )
                self.llm = None
                self.soft_proj = None
                self.policy_head = AbrPolicyHead(d_model=d_model, num_classes=num_classes)
            else:
                # Allow a prebuilt llm/hidden_size to be injected (e.g. tiny config for
                # smoke tests); otherwise build the real backbone lazily.
                if llm is None or hidden_size is None:
                    llm, hidden_size = build_llm_backbone(
                        variant=variant,
                        model_id=model_id,
                        qlora_4bit=qlora_4bit,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                self.llm = llm
                self.hidden_size = hidden_size
                self.soft_proj = SoftTokenProjection(d_model, soft_tokens, hidden_size)
                # Encoder residual skip (default ON). The frozen LLM throttles gradient to the
                # encoder, so without this the encoder collapses to a constant and the head learns
                # the majority-rung prior. Concatenating the encoder's fused embedding with the
                # pooled LLM output gives the head a direct path to the encoder signal, while the
                # LLM+LoRA path still refines on top. Set encoder_residual=False to ablate.
                self.encoder_residual = encoder_residual
                self.d_model = d_model
                head_dim = hidden_size + d_model if encoder_residual else hidden_size
                self.policy_head = AbrPolicyHead(d_model=head_dim, num_classes=num_classes)

        def _llm_dtype(self) -> "torch.dtype":
            try:
                return next(self.llm.parameters()).dtype
            except StopIteration:
                return torch.float32

        def _combine(self, pooled: "torch.Tensor", e_fused: "torch.Tensor") -> "torch.Tensor":
            """Append the encoder's fused embedding to the pooled LLM output (residual skip).

            Gives the policy head a direct, gradient-un-throttled view of the discriminative
            encoder signal so it cannot collapse to the class prior. No-op if encoder_residual
            is False (the pure-LLM-path ablation).
            """
            if getattr(self, "encoder_residual", False):
                return torch.cat([pooled, e_fused.float()], dim=-1)
            return pooled

        def forward(
            self,
            network_norm: "torch.Tensor",
            semantic_norm: "torch.Tensor",
            ladder_context: "torch.Tensor",
            language_ids: Optional["torch.Tensor"] = None,
            language_mask: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
            e_fused = self.encoder(network_norm, semantic_norm, ladder_context)  # [B, d_fuse]

            if self.variant == "v0":
                state_hidden = self.backbone_projector(e_fused)
                return self.policy_head(state_hidden)

            # v1/v2/v3: inject soft tokens as a prefix via inputs_embeds (no input_ids).
            soft = self.soft_proj(e_fused)                       # [B, k, H]
            soft = soft.to(self._llm_dtype())
            batch, k = soft.shape[0], soft.shape[1]
            attn = torch.ones(batch, k, dtype=torch.long, device=soft.device)

            if self.variant == "v3" and language_ids is not None:
                # Concatenate per-video language-token embeddings after the k numeric soft tokens:
                #   seq = [soft_0 .. soft_{k-1}, text_0 .. text_{L-1}], attn = [1*k, language_mask].
                # Text is embedded via the LLM's own input embeddings (frozen weights), so the
                # world-knowledge tokens share the backbone's semantic space with the soft tokens.
                text_emb = self.llm.get_input_embeddings()(language_ids).to(soft.dtype)  # [B, L, H]
                inputs_embeds = torch.cat([soft, text_emb], dim=1)                       # [B, k+L, H]
                if language_mask is None:
                    language_mask = torch.ones(
                        batch, text_emb.shape[1], dtype=torch.long, device=soft.device
                    )
                attn = torch.cat([attn, language_mask.to(attn.dtype)], dim=1)            # [B, k+L]
                out = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
                last_hidden = out.hidden_states[-1]              # [B, k+L, H], final layer
                # Masked mean-pool over real (non-pad) positions so pad text tokens don't dilute
                # the pooled state.
                mask = attn.unsqueeze(-1).to(last_hidden.dtype)  # [B, k+L, 1]
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                return self.policy_head(self._combine(pooled.float(), e_fused))

            # v1/v2: soft tokens only, plain mean over the k tokens (unchanged from before).
            out = self.llm(
                inputs_embeds=soft,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = out.hidden_states[-1]                  # [B, k, H], final layer
            pooled = last_hidden.mean(dim=1).float()             # mean-pool over k tokens
            return self.policy_head(self._combine(pooled, e_fused))


def create_model(**kwargs):
    _require_torch()
    return LlmAbrModel(**kwargs)
