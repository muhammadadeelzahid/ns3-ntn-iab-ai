#!/usr/bin/env python3
"""Per-video language-token cache for the V3 ABR variant.

V3 feeds the frozen LLM a content-derived natural-language string per video (built offline by
`build_video_language.py` into `config/video_semantics.json`), concatenated with the numeric soft
tokens (see model.py). The text is the *same* for every decision of a given video, so it is
tokenized **once per video** here and cached as fixed-length `(input_ids[L], attention_mask[L])`
tensors. Both train.py and evaluate.py use this so training and evaluation embed identical tokens.

Tokenization is deferred to load time (rather than baked into the JSON) so the artifact stays
tokenizer-agnostic; the tokenizer is the same `model_id` the LLM backbone uses.
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from ai_llm.paths import DEFAULT_VIDEO_LANGUAGE
    from ai_llm.model import DEFAULT_MODEL_ID
except ImportError:
    from paths import DEFAULT_VIDEO_LANGUAGE
    from model import DEFAULT_MODEL_ID

# Fixed token budget per video string. The catalog strings are short (genre + title + a few tags),
# so 32 tokens comfortably covers them; longer strings are truncated, shorter ones right-padded.
DEFAULT_LANGUAGE_LEN = 32


def load_language_artifact(language_path: Optional[str] = None) -> Tuple[Dict[str, str], Optional[str]]:
    """Read `video_semantics.json` -> ({video_name: text}, provenance_model_id)."""
    path = language_path or DEFAULT_VIDEO_LANGUAGE
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    videos = data.get("videos", data)
    texts = {vid: (entry.get("text") or "") for vid, entry in videos.items()}
    model_id = (data.get("_provenance") or {}).get("model_id")
    return texts, model_id


def load_language_cache(
    language_path: Optional[str] = None,
    model_id: Optional[str] = None,
    length: int = DEFAULT_LANGUAGE_LEN,
) -> Dict[str, Tuple["torch.Tensor", "torch.Tensor"]]:
    """Build `{video_name: (input_ids[L], attention_mask[L])}` (fixed length, long tensors).

    `model_id` defaults to the tokenizer recorded in the artifact's provenance, falling back to the
    LLM default. Padding/truncation are to exactly `length` so every video yields the same shape
    and batches stack trivially.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to build the V3 language cache.")
    from transformers import AutoTokenizer

    texts, prov_model_id = load_language_artifact(language_path)
    tok_id = model_id or prov_model_id or DEFAULT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for vid, text in texts.items():
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=length,
            return_tensors="pt",
        )
        cache[vid] = (
            enc["input_ids"][0].long(),
            enc["attention_mask"][0].long(),
        )
    return cache
