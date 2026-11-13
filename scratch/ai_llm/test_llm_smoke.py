#!/usr/bin/env python3
"""
Smoke checks for the V1/V2 LLM-backbone variants.

Uses a **tiny** transformer config (built via the real `transformers` stack) so the full
`inputs_embeds` forward + soft-token pooling + policy head + `loss.backward()` wiring is
exercised in seconds on CPU, before pulling the real Qwen2.5-1.5B on a GPU node. Skips
cleanly when transformers/peft are not installed (e.g. the CPU-only login venv).
"""

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

from model import LlmAbrModel, SoftTokenProjection
from paths import NUM_LADDER_RUNGS


def _have(mod_name: str) -> bool:
    import importlib
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:
        return False


def _tiny_qwen_llm(hidden: int = 64):
    """Build a tiny random causal LM mirroring the V1 construction path."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # Start from the real default config, then shrink so it runs on CPU instantly.
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    cfg.hidden_size = hidden
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.intermediate_size = 128
    llm = AutoModelForCausalLM.from_config(cfg)
    return llm, hidden


def soft_token_shape_check() -> None:
    sp = SoftTokenProjection(d_fuse=256, num_tokens=4, hidden_size=64)
    out = sp(torch.randn(3, 256))
    if tuple(out.shape) != (3, 4, 64):
        raise RuntimeError("SoftTokenProjection wrong shape: %s" % (tuple(out.shape),))
    print("soft-token projection shape ok: [3, 4, 64]")


def llm_variant_forward_backward(variant: str, with_language: bool = False) -> None:
    """Forward + backward through the injected tiny LLM; assert grads reach soft_proj/head.

    ``with_language`` exercises the V3 path: per-row language token ids are embedded by the LLM
    and concatenated with the soft tokens before pooling.
    """
    llm, hidden = _tiny_qwen_llm(64)
    model = LlmAbrModel(
        network_dim=9,
        variant=variant,
        soft_tokens=4,
        llm=llm,
        hidden_size=hidden,
    )
    model.train()

    net = torch.randn(2, 8, 9)
    sem = torch.randn(2, 5)
    lad = torch.randn(2, 6, 2)
    labels = torch.tensor([5, 3])

    lang_args = ()
    if with_language:
        vocab = model.llm.get_input_embeddings().num_embeddings
        L = 16
        lang_ids = torch.randint(0, vocab, (2, L))
        lang_mask = torch.ones(2, L, dtype=torch.long)
        lang_mask[1, L // 2:] = 0  # exercise the masked-pool with a padded row
        lang_args = (lang_ids, lang_mask)

    logits, delay = model(net, sem, lad, *lang_args)
    if tuple(logits.shape) != (2, NUM_LADDER_RUNGS):
        raise RuntimeError("[%s] logits shape %s" % (variant, tuple(logits.shape)))
    if delay is not None:
        raise RuntimeError("[%s] delay head should be disabled" % variant)

    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    if model.soft_proj.proj.weight.grad is None:
        raise RuntimeError("[%s] no gradient into soft-token projection" % variant)
    if model.policy_head.bitrate_head.weight.grad is None:
        raise RuntimeError("[%s] no gradient into policy head" % variant)
    if model.encoder.fuse.weight.grad is None:
        raise RuntimeError("[%s] no gradient into encoder" % variant)

    # State-dict must not contain backbone_projector for LLM variants.
    if any(k.startswith("backbone_projector") for k in model.state_dict()):
        raise RuntimeError("[%s] unexpected backbone_projector in state_dict" % variant)
    tag = "%s+lang" % variant if with_language else variant
    print("[%s] tiny-LLM forward+backward ok; grads flow to soft_proj/head/encoder" % tag)


def main() -> None:
    if torch is None:
        print("torch unavailable; skipping LLM smoke test")
        return
    if not _have("transformers"):
        print("transformers not installed; skipping LLM smoke test "
              "(run in the GPU/LLM venv per requirements-llm.txt)")
        return

    soft_token_shape_check()
    # V1 path: random init from config, all trainable. No peft needed.
    llm_variant_forward_backward("v1")
    # V2-style wiring uses the same injected-LLM path (LoRA/4-bit are GPU-only and are
    # covered by a separate GPU-side smoke test against the real model).
    # V3 path: soft tokens + language-token embeddings concatenated, masked-pooled.
    llm_variant_forward_backward("v3", with_language=True)
    print("LLM smoke passed")


if __name__ == "__main__":
    main()
