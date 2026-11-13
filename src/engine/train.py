"""Training and evaluation loops for BreakHis baselines."""

from __future__ import annotations

import csv
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import LABEL_TO_INDEX
from src.engine.metrics import binary_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    dataset, batch_size: int, shuffle: bool, workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def compute_class_weights(rows: Sequence[Dict[str, str]]) -> torch.Tensor:
    counts = Counter(LABEL_TO_INDEX[row["label"]] for row in rows)
    total = sum(counts.values())
    weights = [
        total / (len(counts) * counts[index]) if counts[index] else 0.0
        for index in range(2)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc="train", leave=False)

    for step, (images, labels) in enumerate(progress):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        progress.set_postfix(loss=running_loss / max(seen, 1))

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probabilities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    total_loss = 0.0
    seen = 0

    for step, (images, labels) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        targets.append(labels.detach().cpu().numpy())
        seen += labels.size(0)

    y_prob = np.concatenate(probabilities) if probabilities else np.array([])
    y_true = np.concatenate(targets) if targets else np.array([])
    metrics = binary_metrics(y_true, y_prob) if y_true.size else {}
    if criterion is not None:
        metrics["loss"] = total_loss / max(seen, 1)
    return metrics, y_true, y_prob


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    out_dir: str | Path,
    class_weights: Optional[torch.Tensor] = None,
    max_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
) -> Tuple[List[Dict[str, float]], float]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: List[Dict[str, float]] = []
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, max_batches
        )
        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
            criterion=criterion,
            max_batches=max_eval_batches,
        )
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "time_s": time.time() - start}
        row.update({"val_" + key: value for key, value in val_metrics.items()})
        history.append(row)

        score = float(val_metrics.get("balanced_accuracy", -1.0))
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )

        print(
            "epoch %d/%d  train_loss=%.4f  val_bal_acc=%.4f  val_auc=%.4f  %.1fs"
            % (
                epoch,
                epochs,
                train_loss,
                val_metrics.get("balanced_accuracy", float("nan")),
                val_metrics.get("auc", float("nan")),
                row["time_s"],
            )
        )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return history, best_score


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint
