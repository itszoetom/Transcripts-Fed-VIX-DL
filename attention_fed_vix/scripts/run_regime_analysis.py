"""Per-regime evaluation of the trained sentence-attention model.

What this script does:

  1. Loads the model trained by attention_fed_vix/scripts/train_model.py (outputs/model.pt).
  2. Re-runs inference on each temporal segment:
        regime1 = 2017-01-20 to 2021-01-20
        regime2 = 2021-01-20 to 2025-01-20
        regime3 = 2025-01-20 onward
  3. Reports regression metrics (MSE, R^2, Pearson r) per segment to
     quantify any performance degradation across political-regime
     boundaries.
  4. Saves the resulting report to outputs/regime_analysis.json.

The per-regime numbers answer the project's secondary research question:
does predictive performance degrade across U.S. political regimes
(Obama, Trump 1, Biden, Trump 2)?
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

from attention_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from attention_fed_vix.models import SentenceAttentionModel
from attention_fed_vix.models.attention import attention_config_from_dict
from attention_fed_vix.training.eval import regression_metrics
from attention_fed_vix.utils import make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regime_analysis")


def _build_loader(df: pd.DataFrame, embeddings_path: Path, batch_size: int) -> DataLoader:
    """Non-shuffled loader over `df`."""
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_padded)


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, targets) arrays for a loader."""
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

    device_spec = cfg["model"].get("device", "auto")
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)

    model_cfg = attention_config_from_dict(cfg["model"])
    model = SentenceAttentionModel(model_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    per_regime: dict[str, dict] = {}
    for name, df in [
        ("regime1_2017_2021", splits.regime1),
        ("regime2_2021_2025", splits.regime2),
        ("regime3_2025_present", splits.regime3),
    ]:
        if len(df) == 0:
            continue
        loader = _build_loader(df, embeddings_path, int(cfg["training"]["batch_size"]))
        preds, tgts = _predict(model, loader, device)
        per_regime[name] = regression_metrics(preds, tgts).to_dict()
        logger.info("%s: R^2=%.3f Pearson=%.3f n=%d",
                    name, per_regime[name]["r2"], per_regime[name]["pearson_r"], per_regime[name]["n"])

    report = {"per_regime_metrics": per_regime}
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("regime analysis report written to %s", report_path)


if __name__ == "__main__":
    main()
