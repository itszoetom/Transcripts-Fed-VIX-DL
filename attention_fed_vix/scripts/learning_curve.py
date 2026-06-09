"""Learning curve: does pooled out-of-sample skill improve with more training data?

Motivation: the model predicts near-constant "no change" out of sample. Before
blaming dataset SIZE, test it directly. Train the same model on increasing
fractions of the existing training documents and measure pooled OOS Pearson r
at each size. If r stays flat as n_train grows, the bottleneck is signal, not
volume, so scraping more documents of the same kind would not help.

This reuses the cached FinBERT embeddings (no re-scrape, no re-embed), so it is
cheap: each point is one head-only training run on a few hundred documents.

The validation set (early-stopping signal) is held FIXED across all points; only
the training set is subsampled, taking the most recent k documents so every
subset ends just before the val boundary (temporal order preserved, no shuffle).

Run on Talapas from the repo root:

    python -m attention_fed_vix.scripts.learning_curve \
        --config outputs/winning_config.yaml \
        --out    outputs/final_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attention_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from attention_fed_vix.models import SentenceAttentionModel
from attention_fed_vix.models.attention import attention_config_from_dict
from attention_fed_vix.training import train
from attention_fed_vix.training.loop import TrainConfig
from attention_fed_vix.training.eval import regression_metrics
from attention_fed_vix.utils import set_seed, make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("learning_curve")

FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _loader(df: pd.DataFrame, emb_path: Path, bs: int) -> DataLoader:
    ds = EmbeddingDocDataset(df, embeddings_path=emb_path)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_padded)


def _predict(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["embeddings"].to(device), batch["mask"].to(device))
            preds.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
    return np.asarray(preds), np.asarray(tgts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("outputs/winning_config.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/final_eval"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = _resolve_device(cfg["model"].get("device", "auto"))
    processed_dir = Path(cfg["data"]["processed_dir"])
    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    bs = int(cfg["training"]["batch_size"])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    documents = pd.read_parquet(documents_path)
    splits = make_temporal_splits(
        documents,
        dates=SplitDates(
            train_end=date.fromisoformat(cfg["splits"]["train_end_date"]),
            regime2_start=date.fromisoformat(cfg["splits"]["regime2_start_date"]),
            regime3_start=date.fromisoformat(cfg["splits"]["regime3_start_date"]),
            val_fraction=float(cfg["splits"]["val_fraction"]),
        ),
    )
    pooled_test = pd.concat([splits.regime1, splits.regime2, splits.regime3]).reset_index(drop=True)
    val_loader = _loader(splits.val, embeddings_path, bs)
    test_loader = _loader(pooled_test, embeddings_path, bs)
    logger.info("full train=%d val=%d pooled_test=%d", len(splits.train), len(splits.val), len(pooled_test))

    train_cfg = TrainConfig(
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        batch_size=bs,
        epochs=int(cfg["training"]["epochs"]),
        warmup_fraction=float(cfg["training"]["warmup_fraction"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        early_stop_patience=int(cfg["training"]["early_stop_patience"]),
    )
    model_cfg = attention_config_from_dict(cfg["model"])

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for frac in FRACTIONS:
            set_seed(int(cfg["seed"]))  # same init each point; only data size varies
            k = max(1, int(round(frac * len(splits.train))))
            sub_train = splits.train.iloc[-k:].reset_index(drop=True)  # most-recent k, temporal
            model = SentenceAttentionModel(model_cfg).to(device)
            train(
                model=model,
                train_loader=_loader(sub_train, embeddings_path, bs),
                val_loader=val_loader,
                config=train_cfg,
                device=device,
                model_save_path=tmp / "m.pt",
                metrics_save_path=tmp / "m.json",
            )
            model.load_state_dict(torch.load(tmp / "m.pt", map_location=device, weights_only=True))
            preds, tgts = _predict(model, test_loader, device)
            rm = regression_metrics(preds, tgts)
            row = {"frac": frac, "n_train": k, "pooled_test_pearson_r": rm.pearson_r,
                   "pooled_test_pearson_p": rm.pearson_p, "pooled_test_mse": rm.mse,
                   "pooled_test_r2": rm.r2, "prediction_std": float(preds.std())}
            rows.append(row)
            logger.info("n_train=%d -> pooled r=%.3f (p=%.3f) pred_std=%.2f",
                        k, rm.pearson_r, rm.pearson_p, preds.std())

    (args.out / "learning_curve.json").write_text(json.dumps(rows, indent=2))

    # ---- plot pooled OOS r vs training-set size ----
    ns = [r["n_train"] for r in rows]
    rs = [r["pooled_test_pearson_r"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.plot(ns, rs, "o-", color="#4C72B0", linewidth=2, markersize=9)
    for r in rows:
        ax.annotate(f"r={r['pooled_test_pearson_r']:+.2f}",
                    (r["n_train"], r["pooled_test_pearson_r"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xlabel("Number of training documents")
    ax.set_ylabel("Pooled out-of-sample Pearson r")
    ax.set_title("Learning curve: out-of-sample skill vs training-set size")
    ax.set_ylim(-0.2, 0.5)
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "learning_curve.png", dpi=150)
    plt.close(fig)

    logger.info("wrote %s", args.out / "learning_curve.json")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
