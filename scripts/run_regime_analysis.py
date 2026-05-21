"""Temporal-generalization + regime-change analysis on the trained model.

What this script does:

  1. Loads the model trained by scripts/train_model.py (outputs/model.pt).
  2. Re-runs inference on each temporal segment (regime1 = 2017-01-20 to
     2021-01-20, regime2 = 2021-01-20 to 2025-01-20, regime3 = 2025-01-20+),
     producing predictions and residuals per document.
  3. Computes R^2 per segment to quantify performance degradation.
  4. Runs a Chow test on the residuals at the two breakpoint dates
     (2017-01-20 and 2025-01-20) to test whether the model's residual
     distribution shifts across political regimes.
  5. Saves the resulting report to outputs/regime_analysis.json.

The Chow test on residuals is the standard econometric procedure for detecting
a structural break in a regression's bias across a known event date
(Chow, 1960). We test at the inauguration-day boundaries because the secondary
research question is whether model performance degrades across U.S.
political-regime transitions.
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

from transcripts_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import AttentionConfig
from transcripts_fed_vix.training.eval import regression_metrics
from transcripts_fed_vix.utils import make_temporal_splits, SplitDates, chow_test_on_residuals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regime_analysis")


def _build_loader(df: pd.DataFrame, embeddings_path: Path, batch_size: int) -> DataLoader:
    """Non-shuffled loader over `df`."""
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_padded)


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Return (predictions, targets, release_dates) for a loader."""
    model.eval()
    preds: list[float] = []
    tgts: list[float] = []
    dates: list[pd.Timestamp] = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            out = model(embeddings, mask)
            preds.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
            dates.extend(batch["release_dates"])
    return np.asarray(preds), np.asarray(tgts), dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    processed_dir = Path(cfg["data"]["processed_dir"])
    outputs_dir = Path(cfg["outputs"]["dir"])
    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    model_path = outputs_dir / cfg["outputs"]["model_file"]
    report_path = outputs_dir / cfg["outputs"]["regime_report_file"]

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

    # Combine train + val (= everything pre-2017-01-20) with all test regimes
    # so we have a single chronologically-ordered residual series to feed Chow.
    all_post_train = pd.concat([splits.regime1, splits.regime2, splits.regime3],
                               ignore_index=True).sort_values("release_date")

    device_spec = cfg["model"].get("device", "auto")
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)

    model_cfg = AttentionConfig(
        embed_dim=int(cfg["model"]["embed_dim"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = SentenceAttentionModel(model_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    # Per-regime metrics.
    per_regime: dict[str, dict] = {}
    for name, df in [
        ("regime1_2017_2021", splits.regime1),
        ("regime2_2021_2025", splits.regime2),
        ("regime3_2025_present", splits.regime3),
    ]:
        if len(df) == 0:
            continue
        loader = _build_loader(df, embeddings_path, int(cfg["training"]["batch_size"]))
        preds, tgts, _ = _predict(model, loader, device)
        per_regime[name] = regression_metrics(preds, tgts).to_dict()
        logger.info("%s: R^2=%.3f Pearson=%.3f n=%d",
                    name, per_regime[name]["r2"], per_regime[name]["pearson_r"], per_regime[name]["n"])

    # Combined predictions (post-train range) for Chow test on residuals.
    if len(all_post_train) > 0:
        loader_all = _build_loader(all_post_train, embeddings_path, int(cfg["training"]["batch_size"]))
        preds_all, tgts_all, dates_all = _predict(model, loader_all, device)
        residuals = pd.Series(tgts_all - preds_all)
        dates_series = pd.Series(pd.to_datetime(dates_all))

        chow_results = []
        # Run a Chow test at each configured breakpoint, skipping any breakpoint
        # that doesn't have at least 2 docs on each side.
        for bp_str in cfg["regime_analysis"]["chow_breakpoints"]:
            bp = date.fromisoformat(bp_str)
            try:
                cr = chow_test_on_residuals(residuals, dates_series, bp)
                chow_results.append(cr.to_dict())
                logger.info("Chow @ %s: F=%.3f p=%.4f (n_pre=%d, n_post=%d)",
                            bp, cr.f_statistic, cr.p_value, cr.n_pre, cr.n_post)
            except ValueError as e:
                logger.warning("skipping Chow @ %s: %s", bp, e)
    else:
        chow_results = []

    report = {
        "per_regime_metrics": per_regime,
        "chow_tests": chow_results,
    }
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("regime analysis report written to %s", report_path)


if __name__ == "__main__":
    main()
