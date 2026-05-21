"""Training loop for the SentenceAttentionModel.

Manual PyTorch loop (no HuggingFace Trainer, per project spec). Implements:

    - AdamW optimizer on the model's trainable params only (attention + head;
      the FinBERT encoder is frozen and its embeddings are pre-computed).
    - Linear warmup over the first warmup_frac fraction of total steps.
    - Gradient clipping at grad_clip_norm.
    - Early stopping on validation MSE with `patience` epochs.
    - Per-epoch logging to stdout + a structured JSON metrics file.

Returns a TrainResult with the best validation MSE, best epoch, and the full
per-epoch history. The best model weights are also persisted to disk.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .schedule import linear_warmup
from .eval import regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters for the training loop. Mirrors configs/default.yaml.

    See configs/default.yaml for the canonical values; this dataclass is the
    runtime form that train() actually reads.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    epochs: int = 30
    warmup_fraction: float = 0.1
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 5


@dataclass
class TrainResult:
    """Summary of a completed training run."""

    best_epoch: int
    best_val_mse: float
    history: list[dict[str, Any]]


def _move_batch(batch: dict, device: torch.device) -> dict:
    """Move tensors in a batch dict to device; leave non-tensors as-is."""
    out = dict(batch)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
    return out


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    *,
    model_save_path: Path,
    metrics_save_path: Path,
) -> TrainResult:
    """Run the training loop.

    Args:
        model:             A SentenceAttentionModel (or anything with .forward()
                           returning a NamedTuple with `.prediction`).
        train_loader:      DataLoader yielding the collated batches from
                           data.dataset.collate_padded. NOTE: shuffle=False
                           by spec — temporal order is preserved.
        val_loader:        Validation DataLoader, also shuffle=False.
        config:            TrainConfig with all hyperparameters.
        device:            torch.device to train on.
        model_save_path:   Where to dump the best model's state_dict.
        metrics_save_path: Where to write per-epoch metrics JSON.

    Returns:
        TrainResult with best epoch / best val MSE / full per-epoch history.
    """
    model.to(device)

    # AdamW on only the parameters that require grad — protects us if any
    # extra frozen module ever ends up inside `model`.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = max(1, int(round(total_steps * config.warmup_fraction)))
    scheduler = linear_warmup(optimizer, warmup_steps)

    loss_fn = nn.MSELoss()

    history: list[dict[str, Any]] = []
    best_val_mse = math.inf
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        # ----- train -----
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["embeddings"], batch["mask"])
            loss = loss_fn(out.prediction, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        # ----- validate -----
        model.eval()
        val_preds: list[float] = []
        val_targets: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                out = model(batch["embeddings"], batch["mask"])
                val_preds.extend(out.prediction.detach().cpu().tolist())
                val_targets.extend(batch["target"].detach().cpu().tolist())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_metrics = regression_metrics(np.asarray(val_preds), np.asarray(val_targets))
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_row = {
            "epoch": epoch,
            "train_loss_mse": train_loss,
            "val_mse": val_metrics.mse,
            "val_r2": val_metrics.r2,
            "val_pearson_r": val_metrics.pearson_r,
            "lr": cur_lr,
        }
        history.append(epoch_row)
        logger.info(
            "epoch=%d train_mse=%.5f val_mse=%.5f val_r=%.3f lr=%.2e",
            epoch, train_loss, val_metrics.mse, val_metrics.pearson_r, cur_lr,
        )
        # Also print so SLURM .out captures it without configuring logging.
        print(json.dumps(epoch_row), flush=True)

        # ----- early stopping & best-model save -----
        if val_metrics.mse < best_val_mse - 1e-8:
            best_val_mse = val_metrics.mse
            best_epoch = epoch
            epochs_without_improvement = 0
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                logger.info(
                    "early stopping at epoch %d (no val improvement for %d epochs)",
                    epoch, config.early_stop_patience,
                )
                break

    # ----- persist metrics -----
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_save_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
                "history": history,
            },
            indent=2,
        )
    )

    return TrainResult(best_epoch=best_epoch, best_val_mse=best_val_mse, history=history)
