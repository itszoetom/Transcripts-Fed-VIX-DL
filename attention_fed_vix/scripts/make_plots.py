"""Generate the project's core 5 visualization figures.

Reads:
    configs/default.yaml
    data/processed/documents.parquet
    data/processed/sentence_embeddings.pt
    outputs/model.pt
    outputs/train_metrics.json
    outputs/final_report.json
    outputs/baseline_metrics.json

Writes:
    outputs/figures/training_curve.png
    outputs/figures/predicted_vs_actual.png
    outputs/figures/residuals_over_time.png
    outputs/figures/regression_comparison.png
    outputs/figures/attention_examples.png

The regime boundary lines on the residuals-over-time plot use the inauguration
dates from configs/default.yaml (train_end_date, regime2_start_date,
regime3_start_date), so they always match the configured temporal splits.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import torch
import yaml

from attention_fed_vix.utils import make_temporal_splits, SplitDates
from attention_fed_vix.utils.visualize import (
    _load_model_for_inference,
    plot_training_curve,
    plot_predicted_vs_actual,
    plot_residuals_over_time,
    plot_regression_comparison,
    plot_attention_examples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_plots")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    processed_dir = Path(cfg["data"]["processed_dir"])
    outputs_dir = Path(cfg["outputs"]["dir"])
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    model_path = outputs_dir / cfg["outputs"]["model_file"]
    metrics_path = outputs_dir / cfg["outputs"]["metrics_file"]
    final_report_path = outputs_dir / cfg["outputs"]["final_report_file"]
    baseline_metrics_path = outputs_dir / cfg["outputs"]["baselines_file"]

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

    device_spec = cfg["model"].get("device", "auto")
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)

    # The three regime boundaries: end of the training period (start of
    # regime 1), start of regime 2, and start of regime 3.
    regime_boundaries = [
        date.fromisoformat(cfg["splits"]["train_end_date"]),
        date.fromisoformat(cfg["splits"]["regime2_start_date"]),
        date.fromisoformat(cfg["splits"]["regime3_start_date"]),
    ]

    # 1. Training curve (no model needed).
    plot_training_curve(metrics_path, figures_dir / "training_curve.png")

    # 4. Per-regime regression comparison (JSON-only, no model needed).
    plot_regression_comparison(
        final_report_path, baseline_metrics_path,
        figures_dir / "regression_comparison.png",
    )

    # Model-driven plots: load once and reuse.
    model = _load_model_for_inference(model_path, cfg, device)
    batch_size = int(cfg["training"]["batch_size"])

    # 2. Predicted vs actual scatter, one panel per non-train segment.
    plot_predicted_vs_actual(
        splits, embeddings_path, model,
        figures_dir / "predicted_vs_actual.png",
        batch_size=batch_size, device=device,
    )

    # 3. Residuals over time with regime boundary lines.
    plot_residuals_over_time(
        splits, embeddings_path, model,
        figures_dir / "residuals_over_time.png",
        batch_size=batch_size, device=device,
        regime_boundaries=regime_boundaries,
    )

    # 5. Attention heatmap example for a handful of representative docs.
    plot_attention_examples(
        documents, embeddings_path, model,
        figures_dir / "attention_examples.png",
        batch_size=batch_size, device=device,
        n_examples=4,
    )

    logger.info("all figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
