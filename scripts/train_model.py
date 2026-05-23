"""Main training entry point.

End-to-end flow:

    1. Load configs/default.yaml.
    2. Ensure the processed dataset exists (build it if not).
    3. Ensure the cached sentence embeddings exist (compute them if not).
    4. Carve temporal splits: train + val (early-stopping) + test1 + test2 + test3.
    5. Train SentenceAttentionModel on (train, val) with early stopping.
    6. Load best model weights and report metrics on val + each test segment.
    7. Save model state dict to outputs/model.pt and metrics to
       outputs/train_metrics.json.

The temporal-generalization regime analysis (R^2 degradation across the
inauguration-day boundaries, Chow test on residuals) lives in
scripts/run_regime_analysis.py, that script depends on the best model dumped
here.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from transcripts_fed_vix.data.build import build_processed_dataset
from transcripts_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import AttentionConfig
from transcripts_fed_vix.training import train
from transcripts_fed_vix.training.eval import (
    regression_metrics,
    binary_classification_metrics,
)
from transcripts_fed_vix.training.loop import TrainConfig
from transcripts_fed_vix.utils import set_seed, make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model")


def _resolve_device(spec: str) -> torch.device:
    """Translate 'auto'/'cuda'/'cpu' into a torch.device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _build_loader(df: pd.DataFrame, embeddings_path: Path, batch_size: int) -> DataLoader:
    """Build a non-shuffled DataLoader over the given documents.

    `shuffle=False` is required by the project spec: nothing in the pipeline
    may shuffle across time.
    """
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # tiny dataset; multi-worker overhead is not worth it
        collate_fn=collate_padded,
    )


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and return (predictions, targets) arrays."""
    model.eval()
    preds: list[float] = []
    tgts: list[float] = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            out = model(embeddings, mask)
            preds.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
    return np.asarray(preds), np.asarray(tgts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg["seed"]))

    # ----- paths -----
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    outputs_dir = Path(cfg["outputs"]["dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    model_save_path = outputs_dir / cfg["outputs"]["model_file"]
    metrics_save_path = outputs_dir / cfg["outputs"]["metrics_file"]
    final_report_path = outputs_dir / cfg["outputs"]["final_report_file"]

    # ----- data build (idempotent) -----
    if not documents_path.exists():
        logger.info("processed dataset missing, building from raw scrape")
        build_processed_dataset(raw_dir=raw_dir, processed_dir=processed_dir,
                                processed_filename=cfg["data"]["documents_file"])
    documents = pd.read_parquet(documents_path)
    logger.info("loaded %d processed documents", len(documents))

    # ----- embeddings (idempotent) -----
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"{embeddings_path} not found. Run scripts/precompute_embeddings.py first."
        )

    # ----- temporal splits -----
    splits = make_temporal_splits(
        documents,
        dates=SplitDates(
            train_end=date.fromisoformat(cfg["splits"]["train_end_date"]),
            regime2_start=date.fromisoformat(cfg["splits"]["regime2_start_date"]),
            regime3_start=date.fromisoformat(cfg["splits"]["regime3_start_date"]),
            val_fraction=float(cfg["splits"]["val_fraction"]),
        ),
    )
    logger.info(
        "splits: train=%d val=%d regime1=%d regime2=%d regime3=%d",
        len(splits.train), len(splits.val),
        len(splits.regime1), len(splits.regime2), len(splits.regime3),
    )

    # ----- model -----
    model_cfg = AttentionConfig(
        embed_dim=int(cfg["model"]["embed_dim"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = SentenceAttentionModel(model_cfg)
    logger.info("trainable params: %d",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ----- training -----
    train_cfg = TrainConfig(
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        batch_size=int(cfg["training"]["batch_size"]),
        epochs=int(cfg["training"]["epochs"]),
        warmup_fraction=float(cfg["training"]["warmup_fraction"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        early_stop_patience=int(cfg["training"]["early_stop_patience"]),
    )
    device = _resolve_device(cfg["model"].get("device", "auto"))

    train_loader = _build_loader(splits.train, embeddings_path, train_cfg.batch_size)
    val_loader = _build_loader(splits.val, embeddings_path, train_cfg.batch_size)

    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        model_save_path=model_save_path,
        metrics_save_path=metrics_save_path,
    )
    logger.info("best epoch %d with val MSE %.5f", result.best_epoch, result.best_val_mse)

    # ----- reload best weights & evaluate on val + each regime -----
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.to(device)

    # Training median as the binarization threshold. Computed on the train
    # split, not val/test, to avoid information leakage into evaluation.
    train_median = float(np.median(splits.train["target"].values))

    final_report: dict[str, dict] = {
        "split_dates": {
            "train_end_date": cfg["splits"]["train_end_date"],
            "regime2_start_date": cfg["splits"]["regime2_start_date"],
            "regime3_start_date": cfg["splits"]["regime3_start_date"],
        },
        "train_median_target": train_median,
        "best_epoch": result.best_epoch,
    }

    for name, split_df in [
        ("val", splits.val),
        ("regime1_2017_2021", splits.regime1),
        ("regime2_2021_2025", splits.regime2),
        ("regime3_2025_present", splits.regime3),
    ]:
        if len(split_df) == 0:
            logger.info("split %s is empty, skipping", name)
            continue
        loader = _build_loader(split_df, embeddings_path, train_cfg.batch_size)
        preds, tgts = _predict(model, loader, device)
        rm = regression_metrics(preds, tgts)
        bm = binary_classification_metrics(preds, tgts, threshold=train_median)
        final_report[name] = {"regression": rm.to_dict(), "binary": bm.to_dict()}

    final_report_path.write_text(json.dumps(final_report, indent=2))
    logger.info("final report written to %s", final_report_path)


if __name__ == "__main__":
    main()
