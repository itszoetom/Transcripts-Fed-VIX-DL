"""Generate the project's standard set of visualization figures.

Reads:
    - configs/default.yaml
    - data/processed/documents.parquet
    - data/processed/sentence_embeddings.pt
    - outputs/model.pt
    - outputs/train_metrics.json
    - outputs/final_report.json
    - outputs/baseline_metrics.json

Writes:
    outputs/figures/*.png

All figures listed in the project's visualization-requirements memory:
    - training_curve.png        (train + val MSE per epoch)
    - training_pearson.png      (val Pearson r + R^2 per epoch)
    - lr_schedule.png           (realized LR schedule)
    - predicted_vs_actual.png   (4-panel scatter per temporal segment)
    - residuals_over_time.png   (residual scatter with Chow breakpoints)
    - regression_comparison.png (deep vs TF-IDF Ridge bar chart)
    - binary_comparison.png     (deep vs BoW Logistic bar chart)
    - target_distribution.png   (per-segment target histogram)
    - attention_examples.png    (per-sentence attention weights, 4 docs)
    - sentence_count_distribution.png (sanity check)
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import torch
import yaml

from transcripts_fed_vix.utils import make_temporal_splits, SplitDates
from transcripts_fed_vix.utils.visualize import (
    _load_model_for_inference,
    plot_training_curve,
    plot_training_pearson,
    plot_lr_schedule,
    plot_predicted_vs_actual,
    plot_residuals_over_time,
    plot_regression_comparison,
    plot_binary_comparison,
    plot_target_distribution,
    plot_attention_examples,
    plot_sentence_count_distribution,
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

    # ----- training-history plots (no model needed) -----
    plot_training_curve(metrics_path, figures_dir / "training_curve.png")
    plot_training_pearson(metrics_path, figures_dir / "training_pearson.png")
    plot_lr_schedule(metrics_path, figures_dir / "lr_schedule.png")

    # ----- corpus-level plots (no model needed) -----
    plot_target_distribution(splits, figures_dir / "target_distribution.png")
    plot_sentence_count_distribution(
        documents, figures_dir / "sentence_count_distribution.png",
        cap=int(cfg["data"]["sentence_cap"]),
    )

    # ----- comparison bar charts (read JSON only, no model) -----
    plot_regression_comparison(
        final_report_path, baseline_metrics_path,
        figures_dir / "regression_comparison.png",
    )
    plot_binary_comparison(
        final_report_path, baseline_metrics_path,
        figures_dir / "binary_comparison.png",
    )

    # ----- model-driven plots -----
    model = _load_model_for_inference(model_path, cfg, device)
    batch_size = int(cfg["training"]["batch_size"])
    breakpoints = [date.fromisoformat(b) for b in cfg["regime_analysis"]["chow_breakpoints"]]

    plot_predicted_vs_actual(
        splits, embeddings_path, model,
        figures_dir / "predicted_vs_actual.png",
        batch_size=batch_size, device=device,
    )
    plot_residuals_over_time(
        splits, embeddings_path, model,
        figures_dir / "residuals_over_time.png",
        batch_size=batch_size, device=device,
        breakpoints=breakpoints,
    )
    plot_attention_examples(
        documents, embeddings_path, model,
        figures_dir / "attention_examples.png",
        batch_size=batch_size, device=device,
        n_examples=4,
    )

    logger.info("all figures written to %s", figures_dir)


if __name__ == "__main__":
    main()
