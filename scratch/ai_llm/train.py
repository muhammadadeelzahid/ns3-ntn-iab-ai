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
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object

try:
    from ai_llm.features import (
        load_scaler,
        normalize_network_vector,
        normalize_semantic_vector,
        ladder_context_from_row,
    )
    from ai_llm.model import create_model
    from ai_llm.paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NETWORK_DIM, NUM_LADDER_RUNGS
    from ai_llm.trace_schema import LABEL_FIELD_CLASS
except ImportError:
    from features import (
        load_scaler,
        normalize_network_vector,
        normalize_semantic_vector,
        ladder_context_from_row,
    )
    from model import create_model
    from paths import DEFAULT_NETWORK_SCALER, DEFAULT_SEMANTIC_SCALER, NETWORK_DIM, NUM_LADDER_RUNGS
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
    assembled here at load time — so the window size is a hyperparameter that needs no
    dataset rebuild. `window=1` is the snapshot baseline. Semantic + ladder describe the
    current segment and are single-timestep.
    """

    def __init__(
        self,
        jsonl_paths: Sequence[Path],
        network_scaler: Optional[str] = None,
        semantic_scaler: Optional[str] = None,
        window: int = 1,
    ) -> None:
        self._network_scaler = network_scaler
        self._semantic_scaler = semantic_scaler
        self._window = max(1, int(window))
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

        Windows never cross a (run_id, node_id) boundary. Sessions shorter than `window`
        are left-padded by repeating the earliest available observation. Observation dicts
        are stored by reference, so this survives split_dataset() which copies row refs.
        """
        from collections import defaultdict

        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, r in enumerate(self._rows):
            groups[(r.get("run_id"), r.get("node_id"))].append(idx)

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

    def _features(self, record: Dict):
        window_obs = record.get("_window_obs") or [record.get("observation", {}) or {}]
        network = [normalize_network_vector(o, self._net_scaler) for o in window_obs]  # [W, 7]
        semantic = normalize_semantic_vector(record.get("semantic_raw", {}) or {}, self._sem_scaler)
        ladder = ladder_context_from_row(record)
        return network, semantic, ladder

    def __getitem__(self, index: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int]:
        record = self._rows[index]
        network_norm, semantic_norm, ladder_context = self._features(record)
        network = torch.tensor(network_norm, dtype=torch.float32)   # [W, 7] (W=1 -> [1,7])
        semantic = torch.tensor(semantic_norm, dtype=torch.float32)
        ladder = torch.tensor(ladder_context, dtype=torch.float32)
        label = int(record["teacher"][LABEL_FIELD_CLASS])
        return network, semantic, ladder, label


def _collate_batch(
    batch: Sequence[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int]]
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    networks, semantics, ladders, labels = zip(*batch)
    return (
        torch.stack(networks, dim=0),
        torch.stack(semantics, dim=0),
        torch.stack(ladders, dim=0),
        torch.tensor(labels, dtype=torch.long),
    )


def train_epoch(
    model: "torch.nn.Module",
    loader: DataLoader,
    optimizer: "torch.optim.Optimizer",
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    for network, semantic, ladder, labels in loader:
        network = network.to(device)
        semantic = semantic.to(device)
        ladder = ladder.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(network, semantic, ladder)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model: "torch.nn.Module", loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    for network, semantic, ladder, labels in loader:
        network = network.to(device)
        semantic = semantic.to(device)
        ladder = ladder.to(device)
        labels = labels.to(device)
        logits, _ = model(network, semantic, ladder)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())
        total += labels.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


def split_dataset(
    dataset: AbrJsonlDataset,
    val_fraction: float,
    seed: int,
) -> Tuple[AbrJsonlDataset, AbrJsonlDataset]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else 0
    val_idx = set(indices[:val_size])
    train_rows = [dataset._rows[i] for i in indices if i not in val_idx]  # pylint: disable=protected-access
    val_rows = [dataset._rows[i] for i in indices if i in val_idx]  # pylint: disable=protected-access

    def _clone(rows: List[Dict]) -> "AbrJsonlDataset":
        ds = AbrJsonlDataset.__new__(AbrJsonlDataset)
        ds._rows = rows  # pylint: disable=protected-access
        ds._network_scaler = dataset._network_scaler  # pylint: disable=protected-access
        ds._semantic_scaler = dataset._semantic_scaler  # pylint: disable=protected-access
        ds._net_scaler = dataset._net_scaler  # pylint: disable=protected-access
        ds._sem_scaler = dataset._sem_scaler  # pylint: disable=protected-access
        ds._window = dataset._window  # pylint: disable=protected-access
        return ds

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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
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
    args = parser.parse_args()

    network_scaler = args.network_scaler or None
    semantic_scaler = args.semantic_scaler or None

    data_dir = Path(args.data_dir).resolve()
    paths = _iter_jsonl_paths(data_dir, args.glob_pattern)
    dataset = AbrJsonlDataset(
        paths,
        network_scaler=network_scaler,
        semantic_scaler=semantic_scaler,
        window=args.window,
    )
    train_ds, val_ds = split_dataset(dataset, args.val_fraction, args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        network_dim=NETWORK_DIM,
        semantic_dim=5,
        ladder_dim=2,
        d_model=args.d_model,
        num_classes=NUM_LADDER_RUNGS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    ) if len(val_ds) > 0 else None

    print(
        "Training on {} rows ({} val) from {} shards, network_dim={}, window={}".format(
            len(train_ds), len(val_ds), len(paths), NETWORK_DIM, args.window
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        msg = "epoch {} train_loss={:.4f} train_acc={:.3f}".format(epoch, train_loss, train_acc)
        if val_loader is not None:
            val_loss, val_acc = eval_epoch(model, val_loader, device)
            msg += " val_loss={:.4f} val_acc={:.3f}".format(val_loss, val_acc)
        print(msg)

    out_path = Path(args.checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "network_dim": NETWORK_DIM,
            "num_classes": NUM_LADDER_RUNGS,
            "window": args.window,
            # Pin the exact scalers used so deployment normalizes identically.
            "network_scaler": network_scaler,
            "semantic_scaler": semantic_scaler,
            "network_scaler_sha256": _file_sha256(network_scaler),
            "semantic_scaler_sha256": _file_sha256(semantic_scaler),
        },
        out_path,
    )
    print("Wrote checkpoint to {}".format(out_path))


if __name__ == "__main__":
    main()
