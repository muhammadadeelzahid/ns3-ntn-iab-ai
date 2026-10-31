#!/usr/bin/env python3
"""
Supervised training loader for train_jsonl/ shards.

Reads model_input tensors and teacher.chosen_bitrate_class labels, then runs
a lightweight cross-entropy fit on LlmAbrModel (Phase-2 encoder + policy head).
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None

try:
    from ai_llm.features import (
        load_scaler,
        normalize_network_vector,
        normalize_network_vector_with_masks,
        normalize_semantic_vector,
        ladder_context_from_row,
        SENTINEL_MASK_ORDER,
    )
    from ai_llm.model import create_model, DEFAULT_MODEL_ID
    from ai_llm.paths import (
        DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, DEFAULT_VIDEO_LANGUAGE,
        NETWORK_DIM, NUM_LADDER_RUNGS,
    )
    from ai_llm.language import load_language_cache, DEFAULT_LANGUAGE_LEN
    from ai_llm.trace_schema import LABEL_FIELD_CLASS
except ImportError:
    from features import (
        load_scaler,
        normalize_network_vector,
        normalize_network_vector_with_masks,
        normalize_semantic_vector,
        ladder_context_from_row,
        SENTINEL_MASK_ORDER,
    )
    from model import create_model, DEFAULT_MODEL_ID
    from paths import (
        DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, DEFAULT_VIDEO_LANGUAGE,
        NETWORK_DIM, NUM_LADDER_RUNGS,
    )
    from language import load_language_cache, DEFAULT_LANGUAGE_LEN
    from trace_schema import LABEL_FIELD_CLASS


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    import hashlib
    import os

    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl_paths(data_dir: Path, pattern: str) -> List[Path]:
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            "No JSONL shards matching {} under {}".format(pattern, data_dir)
        )
    return paths


def _load_record(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


class AbrJsonlDataset(Dataset):
    """One row per ABR decision; label from teacher.

    Normalization is applied **at load time** from the raw `observation` / `semantic_raw`
    (the same scaler path inference uses). The network branch is fed a **time-series
    window** of the last `window` segments of the SAME session (`run_id`, `node_id`),
    assembled here at load time - so the window size is a hyperparameter that needs no
    dataset rebuild. `window=1` is the snapshot baseline. Semantic + ladder describe the
    current segment and are single-timestep.
    """

    def __init__(
        self,
        jsonl_paths: Sequence[Path],
        network_scaler: Optional[str] = None,
        semantic_scaler: Optional[str] = None,
        window: int = 1,
        use_missing_masks: bool = True,
        variant: str = "v0",
        language_cache: Optional[Dict[str, Tuple]] = None,
        ablate_semantic: bool = False,
        ladder_size_only: bool = False,
    ) -> None:
        self._network_scaler = network_scaler
        self._semantic_scaler = semantic_scaler
        self._window = max(1, int(window))
        self._use_missing_masks = bool(use_missing_masks)
        # V3 only: per-video (input_ids[L], attention_mask[L]) tensors looked up by video_name.
        self._variant = variant
        self._language_cache = language_cache
        # Input-ablation (content-channel study): zero the semantic branch and/or the ladder VMAF column
        # so the model must fit the (greedy-reward) labels from network-only / network+VMAF / full inputs.
        self._ablate_semantic = bool(ablate_semantic)
        self._ladder_size_only = bool(ladder_size_only)
        # Load scalers once (dicts), reused for every timestep.
        self._net_scaler = load_scaler(network_scaler or DEFAULT_NETWORK_SCALER)
        self._sem_scaler = load_scaler(semantic_scaler or DEFAULT_SEMANTIC_SCALER)

        self._rows: List[Dict] = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = _load_record(line)
                    if record is None:
                        continue
                    if "teacher" not in record:
                        continue
                    if record["teacher"].get(LABEL_FIELD_CLASS) is None:
                        continue
                    if not record.get("observation"):
                        continue
                    self._rows.append(record)

        if not self._rows:
            raise RuntimeError("No training rows with teacher labels and observations")

        self._attach_windows()

    def _attach_windows(self) -> None:
        """For each row, attach the last `window` observation dicts of its session.

        Windows never cross a (video_name, run_id, node_id) boundary. run_id/node_id repeat across
        videos (every video has run_1..run_10 and UEs 5-7), so keying on them alone mixed ~127 videos
        into each "session" and 99.9 % of training windows held other videos' observations (found
        2026-10-31; serving always used the true per-UE history). Sessions shorter than `window`
        are left-padded by repeating the earliest available observation. Observation dicts
        are stored by reference, so this survives split_dataset() which copies row refs.
        """
        from collections import defaultdict

        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, r in enumerate(self._rows):
            groups[(r.get("video_name"), r.get("run_id"), r.get("node_id"))].append(idx)

        W = self._window
        for idxs in groups.values():
            idxs.sort(key=lambda i: self._rows[i].get("segment_id", 0))
            for pos, i in enumerate(idxs):
                win = idxs[max(0, pos - W + 1): pos + 1]
                if len(win) < W:
                    win = [win[0]] * (W - len(win)) + win  # left-pad with earliest
                self._rows[i]["_window_obs"] = [
                    self._rows[j].get("observation", {}) or {} for j in win
                ]

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def network_dim(self) -> int:
        """Effective network channel count: 7, or 7 + len(SENTINEL_MASK_ORDER) with masks."""
        return NETWORK_DIM + (len(SENTINEL_MASK_ORDER) if self._use_missing_masks else 0)

    def _features(self, record: Dict):
        window_obs = record.get("_window_obs") or [record.get("observation", {}) or {}]
        # [W, 7] (masks off) or [W, 9] (masks on); mask bits appended after z-scoring.
        norm = normalize_network_vector_with_masks if self._use_missing_masks else normalize_network_vector
        network = [norm(o, self._net_scaler) for o in window_obs]
        semantic = normalize_semantic_vector(record.get("semantic_raw", {}) or {}, self._sem_scaler)
        ladder = ladder_context_from_row(record)
        return network, semantic, ladder

    def _language(self, record: Dict) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """V3: fixed-length (input_ids, attention_mask) for this row's video.

        Looked up by video_name from the precomputed cache. Videos absent from the cache (should
        not happen given 100% catalog coverage) degrade gracefully to an all-pad sequence, which
        the model's masked pool ignores - equivalent to V2 for that row.
        """
        cache = self._language_cache or {}
        ids_mask = cache.get(record.get("video_name"))
        if ids_mask is None:
            length = next(iter(cache.values()))[0].shape[0] if cache else DEFAULT_LANGUAGE_LEN
            return (
                torch.zeros(length, dtype=torch.long),
                torch.zeros(length, dtype=torch.long),
            )
        return ids_mask

    def __getitem__(self, index: int):
        record = self._rows[index]
        network_norm, semantic_norm, ladder_context = self._features(record)
        network = torch.tensor(network_norm, dtype=torch.float32)   # [W, 7] (W=1 -> [1,7])
        semantic = torch.tensor(semantic_norm, dtype=torch.float32)
        ladder = torch.tensor(ladder_context, dtype=torch.float32)   # [6, 2] = [size, vmaf] per rung
        if self._ablate_semantic:
            semantic = torch.zeros_like(semantic)        # content-channel ablation: no semantic features
        if self._ladder_size_only:
            ladder[..., 1] = 0.0                         # drop per-rung VMAF, keep size
        label = int(record["teacher"][LABEL_FIELD_CLASS])
        if self._variant == "v3":
            lang_ids, lang_mask = self._language(record)
            return network, semantic, ladder, label, lang_ids, lang_mask
        return network, semantic, ladder, label


def _collate_batch(batch: Sequence[Tuple]):
    """Stack a batch into ``(network, semantic, ladder, labels, lang_ids, lang_mask)``.

    ``lang_ids``/``lang_mask`` are ``None`` for v0/v1/v2 (4-tuple items) and stacked fixed-length
    long tensors for v3 (6-tuple items), so the training/eval loops have one uniform signature.
    """
    has_language = len(batch[0]) == 6
    if has_language:
        networks, semantics, ladders, labels, lang_ids, lang_masks = zip(*batch)
        lang_ids_t = torch.stack(lang_ids, dim=0)
        lang_mask_t = torch.stack(lang_masks, dim=0)
    else:
        networks, semantics, ladders, labels = zip(*batch)
        lang_ids_t = lang_mask_t = None
    return (
        torch.stack(networks, dim=0),
        torch.stack(semantics, dim=0),
        torch.stack(ladders, dim=0),
        torch.tensor(labels, dtype=torch.long),
        lang_ids_t,
        lang_mask_t,
    )


def compute_class_weights(
    rows: Sequence[Dict], num_classes: int, power: float = 1.0
) -> "torch.Tensor":
    """Inverse-frequency class weights computed on the **train split only**.

    Rung 5 is ~51% of decisions and rung 1 ~2%; without this the model collapses to
    the majority class (the 0.508 val-acc trap). Weights are normalized to mean 1 so the
    loss scale stays comparable to the unweighted run.

    ``power`` tempers the weighting: 1.0 = full inverse-frequency (can destabilize a small
    head - it over-rewards the rare rungs and the optimizer ping-pongs between single-class
    collapses); 0.5 = sqrt-inverse (gentler, usually more stable); 0.0 = uniform (no weighting).
    """
    counts = [0] * num_classes
    for r in rows:
        cls = r.get("teacher", {}).get(LABEL_FIELD_CLASS)
        if cls is not None and 0 <= int(cls) < num_classes:
            counts[int(cls)] += 1
    total = sum(counts)
    # inverse frequency ** power; classes absent from train get weight 0 (never penalized).
    weights = [
        ((total / (num_classes * c)) ** power) if c > 0 else 0.0
        for c in counts
    ]
    present = [w for w, c in zip(weights, counts) if c > 0]
    mean_w = sum(present) / max(len(present), 1)
    if mean_w > 0:
        weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def macro_f1_and_recall(
    preds: Sequence[int], labels: Sequence[int], num_classes: int
) -> Tuple[float, List[float]]:
    """Macro-F1 and per-class recall over `num_classes`.

    Uses sklearn when available; otherwise a pure-torch/python TP/FP/FN fallback so the
    V0/CPU path needs no extra dependency. macro-F1 (not top-1) is the honest metric here
    because rung 5 dominates.
    """
    try:
        from sklearn.metrics import f1_score, recall_score

        labs = list(range(num_classes))
        macro_f1 = float(f1_score(labels, preds, labels=labs, average="macro", zero_division=0))
        recalls = recall_score(labels, preds, labels=labs, average=None, zero_division=0)
        return macro_f1, [float(x) for x in recalls]
    except Exception:  # pragma: no cover - fallback path
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        for p, t in zip(preds, labels):
            if p == t:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1
        f1s: List[float] = []
        recalls = []
        for c in range(num_classes):
            denom_p = tp[c] + fp[c]
            denom_r = tp[c] + fn[c]
            prec = tp[c] / denom_p if denom_p else 0.0
            rec = tp[c] / denom_r if denom_r else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            f1s.append(f1)
            recalls.append(rec)
        return sum(f1s) / max(num_classes, 1), recalls


def train_epoch(
    model: "torch.nn.Module",
    loader: DataLoader,
    optimizer: "torch.optim.Optimizer",
    device: str,
    criterion: "torch.nn.Module",
    scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for network, semantic, ladder, labels, lang_ids, lang_mask in loader:
        network = network.to(device)
        semantic = semantic.to(device)
        ladder = ladder.to(device)
        labels = labels.to(device)
        if lang_ids is not None:
            lang_ids = lang_ids.to(device)
            lang_mask = lang_mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(network, semantic, ladder, lang_ids, lang_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_epoch(
    model: "torch.nn.Module",
    loader: DataLoader,
    device: str,
    criterion: "torch.nn.Module",
    num_classes: int = NUM_LADDER_RUNGS,
) -> Tuple[float, float, float, List[float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for network, semantic, ladder, labels, lang_ids, lang_mask in loader:
        network = network.to(device)
        semantic = semantic.to(device)
        ladder = ladder.to(device)
        labels = labels.to(device)
        if lang_ids is not None:
            lang_ids = lang_ids.to(device)
            lang_mask = lang_mask.to(device)
        logits, _ = model(network, semantic, ladder, lang_ids, lang_mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    macro_f1, recalls = macro_f1_and_recall(all_preds, all_labels, num_classes)
    return total_loss / max(total, 1), total_correct / max(total, 1), macro_f1, recalls


def split_dataset(
    dataset: AbrJsonlDataset,
    val_fraction: float,
    seed: int,
    group_keys: Tuple[str, ...] = ("video_name", "run_id", "node_id"),
    holdout_video: Optional[set] = None,
) -> Tuple[AbrJsonlDataset, AbrJsonlDataset]:
    """Group-aware train/val split.

    Whole sessions (default key ``(video_name, run_id, node_id)``) are assigned to one side, never
    split across train/val. This both prevents leakage and is required by the windowing
    (a window must not straddle the split). ``holdout_video`` (a set of ``video_name``s)
    pulls entire videos into the val/test side to build the §5.2 unseen-content split.
    """
    rows = dataset._rows  # pylint: disable=protected-access

    def _clone(selected: List[Dict]) -> "AbrJsonlDataset":
        ds = AbrJsonlDataset.__new__(AbrJsonlDataset)
        ds._rows = selected  # pylint: disable=protected-access
        ds._network_scaler = dataset._network_scaler  # pylint: disable=protected-access
        ds._semantic_scaler = dataset._semantic_scaler  # pylint: disable=protected-access
        ds._net_scaler = dataset._net_scaler  # pylint: disable=protected-access
        ds._sem_scaler = dataset._sem_scaler  # pylint: disable=protected-access
        ds._window = dataset._window  # pylint: disable=protected-access
        ds._use_missing_masks = dataset._use_missing_masks  # pylint: disable=protected-access
        ds._variant = dataset._variant  # pylint: disable=protected-access
        ds._language_cache = dataset._language_cache  # pylint: disable=protected-access
        return ds

    # Held-out videos go straight to val and are excluded from the grouped split.
    holdout_video = holdout_video or set()
    train_pool = [r for r in rows if r.get("video_name") not in holdout_video]
    video_holdout_rows = [r for r in rows if r.get("video_name") in holdout_video]

    # Group remaining rows by session and shuffle whole groups.
    from collections import OrderedDict

    groups: "OrderedDict[Tuple, List[Dict]]" = OrderedDict()
    for r in train_pool:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)
    group_keys_list = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys_list)

    target_val = int(len(train_pool) * val_fraction)
    val_rows: List[Dict] = []
    train_rows: List[Dict] = []
    for key in group_keys_list:
        grp = groups[key]
        if len(val_rows) < target_val:
            val_rows.extend(grp)
        else:
            train_rows.extend(grp)

    val_rows.extend(video_holdout_rows)
    return _clone(train_rows), _clone(val_rows)


def main() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for train.py")

    parser = argparse.ArgumentParser(description="Train LlmAbrModel from train_jsonl shards.")
    parser.add_argument(
        "--data-dir",
        default="train_jsonl",
        help="Directory containing per-run JSONL shards (default: train_jsonl).",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        dest="glob_pattern",
        help="Shard glob under --data-dir (default: *.jsonl).",
    )
    parser.add_argument(
        "--val-data-dir",
        default=None,
        help="Directory of held-out validation shards (persisted split). When given, train uses "
        "--data-dir and validates on --val-data-dir, bypassing the internal random split "
        "(--val-fraction/--holdout-video are then ignored). This is the leakage-free path: "
        "whole videos live in exactly one dir so no window crosses the boundary.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for projection/policy head.")
    parser.add_argument(
        "--lr-encoder", type=float, default=1e-2,
        help="LR for the multimodal encoder (HIGHER than --lr; default 10x). The deep encoder is "
        "gradient-starved vs the head, so it needs its own higher LR or the model collapses to the "
        "majority class. Validated range ~10-30x --lr.",
    )
    parser.add_argument(
        "--lr-softproj", type=float, default=None,
        help="LR for the V1/V2 soft-token projection (encoder->LLM bridge). Defaults to "
        "--lr-encoder; set a gentler value (e.g. 1e-2) if a high feature LR through the frozen LLM "
        "destabilizes V2.",
    )
    parser.add_argument(
        "--class-weight-power", type=float, default=1.0,
        help="Temper inverse-freq class weights: 1.0=full (default), 0.5=sqrt (gentler/stable), "
        "0.0=uniform. Lower this if a small head collapses to single-class predictions.",
    )
    parser.add_argument(
        "--balanced-sampling", action="store_true", default=False,
        help="Draw training rows with a WeightedRandomSampler (weight ~ 1/rung-frequency) so each "
        "batch is balanced across the 6 rungs. Directly counters majority-class collapse. When on, "
        "set --class-weight-power 0.0 to avoid double-correcting the imbalance.",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="CrossEntropy label smoothing (e.g. 0.1) to penalize over-confident single-class "
        "predictions. 0.0 = off (default).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    # --- LLM variant controls (defaults preserve the v0 baseline) ---
    parser.add_argument(
        "--variant", choices=["v0", "v1", "v2", "v3"], default="v0",
        help="v0=no-LLM encoder; v1=random LLM full-FT; v2=pretrained frozen+LoRA; "
        "v3=v2 + per-video language channel.",
    )
    parser.add_argument("--model-id", default=None, help="HF backbone id (v1/v2/v3); default Qwen2.5-1.5B.")
    parser.add_argument(
        "--language-file", default=DEFAULT_VIDEO_LANGUAGE,
        help="V3 per-video language strings JSON (built by build_video_language.py).",
    )
    parser.add_argument(
        "--language-len", type=int, default=DEFAULT_LANGUAGE_LEN,
        help="V3 fixed token length per video string (pad/truncate). Default 32.",
    )
    parser.add_argument("--soft-tokens", "-k", type=int, default=4, help="Number of soft tokens (v1/v2).")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr-llm", type=float, default=1e-4, help="LR for LLM (LoRA for v2, all for v1).")
    parser.add_argument("--qlora-4bit", action="store_true", help="Load v2 base in 4-bit (QLoRA).")
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="none")
    parser.add_argument(
        "--warmup-frac", type=float, default=0.0,
        help="Fraction of total steps for linear LR warmup before cosine decay (e.g. 0.05). "
        "Only applies with --scheduler cosine. A gentle start reduces seed-dependent collapse.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument(
        "--holdout-video", action="append", default=None,
        help="video_name to hold entirely out of train (repeatable); builds the unseen-content val set.",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/abr_policy.pt",
        help="Path to write state_dict checkpoint.",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda (auto-detect if omitted).")
    parser.add_argument(
        "--network-scaler",
        default=DEFAULT_NETWORK_SCALER,
        help="Network scaler JSON applied to raw observations at load time "
        "(default: config/feature_scaler_network.json). Pass '' to reuse baked v1 vectors.",
    )
    parser.add_argument(
        "--semantic-scaler",
        default=DEFAULT_SEMANTIC_SCALER,
        help="Semantic scaler JSON applied to raw semantic features at load time.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=8,
        help="Time-series window: number of past segments (same session) fed to the "
        "network 1D-CNN branch. 1 = snapshot baseline. Free to change (load-time).",
    )
    parser.add_argument(
        "--use-missing-masks",
        dest="use_missing_masks",
        action="store_true",
        default=True,
        help="Append 0/1 missing-indicator channels for sentinel features (RTT/loss) so the "
        "model distinguishes 'imputed-to-mean' from a genuine mean value (network_dim 7->9). "
        "Default on.",
    )
    parser.add_argument(
        "--no-missing-masks",
        dest="use_missing_masks",
        action="store_false",
        help="Disable missing-indicator channels (mask-blind Tier-0 baseline for ablation).",
    )
    parser.add_argument(
        "--no-semantic", dest="ablate_semantic", action="store_true", default=False,
        help="Content-channel ablation: zero the semantic-feature branch (net-only / +vmaf arms).",
    )
    parser.add_argument(
        "--ladder-size-only", dest="ladder_size_only", action="store_true", default=False,
        help="Content-channel ablation: zero the per-rung VMAF in the ladder (keep size) - net-only arm.",
    )
    args = parser.parse_args()

    network_scaler = args.network_scaler or None
    semantic_scaler = args.semantic_scaler or None

    # V3: tokenize each video's language string once (shared by train + val datasets).
    language_cache = None
    if args.variant == "v3":
        language_cache = load_language_cache(
            args.language_file,
            model_id=args.model_id or DEFAULT_MODEL_ID,
            length=args.language_len,
        )
        print("V3 language channel: %d videos tokenized (len=%d) from %s" % (
            len(language_cache), args.language_len, args.language_file))

    data_dir = Path(args.data_dir).resolve()
    paths = _iter_jsonl_paths(data_dir, args.glob_pattern)
    dataset = AbrJsonlDataset(
        paths,
        network_scaler=network_scaler,
        semantic_scaler=semantic_scaler,
        window=args.window,
        use_missing_masks=args.use_missing_masks,
        variant=args.variant,
        language_cache=language_cache,
        ablate_semantic=args.ablate_semantic,
        ladder_size_only=args.ladder_size_only,
    )
    if args.val_data_dir:
        # Persisted, leakage-free split: train on --data-dir, validate on --val-data-dir.
        # Each whole video is in exactly one directory, so windows never cross the boundary and
        # the internal random split is bypassed entirely.
        val_dir = Path(args.val_data_dir).resolve()
        val_ds = AbrJsonlDataset(
            _iter_jsonl_paths(val_dir, args.glob_pattern),
            network_scaler=network_scaler,
            semantic_scaler=semantic_scaler,
            window=args.window,
            use_missing_masks=args.use_missing_masks,
            variant=args.variant,
            language_cache=language_cache,
            ablate_semantic=args.ablate_semantic,
            ladder_size_only=args.ladder_size_only,
        )
        train_ds = dataset
        print("Using persisted split: train=%s (%d rows), val=%s (%d rows)" % (
            data_dir, len(train_ds), val_dir, len(val_ds)))
    else:
        holdout_video = set(args.holdout_video) if args.holdout_video else None
        train_ds, val_ds = split_dataset(
            dataset, args.val_fraction, args.seed, holdout_video=holdout_video,
        )

    network_dim = dataset.network_dim  # 7, or 9 when missing-masks are enabled
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.variant != "v0" and device != "cuda":
        raise SystemExit(
            "variant %s requires a CUDA GPU (bf16/4-bit); got device=%s. "
            "Run on a Grex GPU partition with the LLM venv (see requirements-llm.txt)."
            % (args.variant, device)
        )

    model = create_model(
        network_dim=network_dim,
        semantic_dim=5,
        ladder_dim=2,
        d_model=args.d_model,
        num_classes=NUM_LADDER_RUNGS,
        variant=args.variant,
        model_id=args.model_id or DEFAULT_MODEL_ID,
        soft_tokens=args.soft_tokens,
        qlora_4bit=args.qlora_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    # Three-LR param groups. The FEATURE PATH (multimodal encoder + the soft-token projection that
    # bridges it into the LLM) runs at --lr-encoder (HIGHER); the LLM/LoRA at --lr-llm; the policy
    # head at --lr. The deep residual-free encoder is gradient-starved (~40x smaller grad than the
    # head); without its own higher LR the head collapses to the majority class. For V1/V2 the
    # `soft_proj` (encoder 256-d -> LLM 6144-d soft tokens) sits behind the frozen LLM, which
    # attenuates its gradient the same way - so it joins the high-LR feature group, not the head.
    # (Encoder fix validated by the overfit harness: 30x head LR drives a batch to ~0.12 loss/0.95 acc.)
    # Only requires_grad params are handed to the optimizer (frozen 4-bit base excluded).
    enc_params, soft_params, llm_params, head_params = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder"):
            enc_params.append(p)
        elif name.startswith("soft_proj"):
            soft_params.append(p)
        elif name.startswith("llm."):
            llm_params.append(p)
        else:
            head_params.append(p)
    # soft_proj (the encoder->LLM bridge) gets its own LR: it needs a high LR like the encoder
    # (gradient-starved), but feeds the frozen LLM, so too-high a value can destabilize V2. Default
    # to --lr-encoder; --lr-softproj lets V1/V2 use a gentler value (e.g. 1e-2).
    lr_soft = args.lr_softproj if args.lr_softproj is not None else args.lr_encoder
    param_groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr_encoder})
    if soft_params:
        param_groups.append({"params": soft_params, "lr": lr_soft})
    if llm_params:
        param_groups.append({"params": llm_params, "lr": args.lr_llm})
    optimizer = torch.optim.AdamW(param_groups)

    # Class-weighted CE on the train split (fixes the rung-5 majority-class collapse).
    class_weights = compute_class_weights(  # pylint: disable=protected-access
        train_ds._rows, NUM_LADDER_RUNGS, power=args.class_weight_power
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    # Balanced sampling: oversample rare-rung rows so each batch is balanced across the 6 rungs
    # (the most direct counter to majority-class collapse). Mutually exclusive with shuffle.
    train_sampler = None
    if args.balanced_sampling:
        labels = [
            int(r.get("teacher", {}).get(LABEL_FIELD_CLASS, -1))
            for r in train_ds._rows  # pylint: disable=protected-access
        ]
        counts = [0] * NUM_LADDER_RUNGS
        for c in labels:
            if 0 <= c < NUM_LADDER_RUNGS:
                counts[c] += 1
        row_w = [
            (1.0 / counts[c]) if (0 <= c < NUM_LADDER_RUNGS and counts[c] > 0) else 0.0
            for c in labels
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(row_w, dtype=torch.double),
            num_samples=len(row_w),
            replacement=True,
        )
        print("Balanced sampling ON: per-rung row counts = %s" % counts)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    ) if len(val_ds) > 0 else None

    scheduler = None
    if args.scheduler == "cosine":
        steps = max(1, args.epochs * len(train_loader))
        warmup_steps = int(round(max(0.0, args.warmup_frac) * steps))
        if warmup_steps > 0:
            # Linear warmup (LR ramps 1% -> 100% of each group's base over warmup_steps), then
            # cosine decay over the rest. A gentle start lets the encoder and head co-adapt before
            # the head locks onto the majority class - reduces the seed-dependent collapse. All
            # param groups scale together, so the encoder:head LR ratio is preserved.
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, steps - warmup_steps)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    print(
        "Training variant={} on {} rows ({} val) from {} shards, network_dim={}, window={}, "
        "missing_masks={}".format(
            args.variant, len(train_ds), len(val_ds), len(paths), network_dim, args.window,
            args.use_missing_masks,
        )
    )

    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint() -> None:
        # Persist only non-LLM weights here (encoder + soft_proj + head); the frozen v2/v3
        # base reloads from HF and its LoRA adapter is saved separately below. For v1 the
        # full LLM has no frozen base to omit, so its weights are included.
        if args.variant in ("v2", "v3"):
            sd = {k: v for k, v in model.state_dict().items() if not k.startswith("llm.")}
        else:
            sd = model.state_dict()
        meta = {
            "state_dict": sd,
            "variant": args.variant,
            "model_id": args.model_id or DEFAULT_MODEL_ID,
            "soft_tokens": args.soft_tokens,
            "network_dim": network_dim,
            "num_classes": NUM_LADDER_RUNGS,
            "window": args.window,
            "use_missing_masks": args.use_missing_masks,
            # Pin the exact scalers used so deployment normalizes identically.
            "network_scaler": network_scaler,
            "semantic_scaler": semantic_scaler,
            "network_scaler_sha256": _file_sha256(network_scaler),
            "semantic_scaler_sha256": _file_sha256(semantic_scaler),
        }
        if args.variant == "v3":
            # Pin the language artifact + token length so eval embeds identical text.
            meta["language_file"] = args.language_file
            meta["language_len"] = args.language_len
            meta["language_file_sha256"] = _file_sha256(args.language_file)
        torch.save(meta, out_path)
        # v2/v3: save the LoRA adapter (few MB) alongside the checkpoint.
        if args.variant in ("v2", "v3") and hasattr(model.llm, "save_pretrained"):
            model.llm.save_pretrained(str(out_path) + ".lora")

    best_f1 = -1.0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, criterion, scheduler,
        )
        msg = "epoch {} train_loss={:.4f} train_acc={:.3f}".format(epoch, train_loss, train_acc)
        improved = False
        if val_loader is not None:
            val_loss, val_acc, val_f1, val_recalls = eval_epoch(
                model, val_loader, device, criterion, NUM_LADDER_RUNGS,
            )
            recall_str = ",".join("{:.2f}".format(r) for r in val_recalls)
            msg += " val_loss={:.4f} val_acc={:.3f} val_macroF1={:.3f} recall=[{}]".format(
                val_loss, val_acc, val_f1, recall_str,
            )
            if val_f1 > best_f1:
                best_f1, improved = val_f1, True
        print(msg)

        # Save the best macro-F1 checkpoint (or every epoch if no val set).
        if val_loader is None:
            _save_checkpoint()
        elif improved:
            _save_checkpoint()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
                print("Early stopping at epoch {} (best val_macroF1={:.3f})".format(epoch, best_f1))
                break

    print("Wrote checkpoint to {} (best val_macroF1={:.3f})".format(out_path, best_f1))


if __name__ == "__main__":
    main()
