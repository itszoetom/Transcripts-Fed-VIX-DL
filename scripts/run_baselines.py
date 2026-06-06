"""Linear-model baseline: TF-IDF + Ridge regression.

The project's primary research question (does the sentence-attention model
beat a linear bag-of-features baseline on Fed text?) requires an apples-to-
apples baseline on the same temporal splits.

Baseline:
    TF-IDF + Ridge regression
        Continuous regression target (3-day VIX change).
        Compared to the deep model on MSE, R^2, and Pearson r.

The baseline uses the same train / val / test1 / test2 / test3 segments as
the deep model. We deliberately do NOT do cross-validated hyperparameter
search inside the training segment; the baseline uses a fixed, conventional
default (Ridge alpha=1.0). This keeps the comparison honest: the deep model
has exactly one early-stopping signal, and the baseline has zero.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from transcripts_fed_vix.training.eval import regression_metrics
from transcripts_fed_vix.utils import make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_baselines")


def _doc_text(row: pd.Series) -> str:
    """Concatenate a document's first-80 sentences into one string for the vectorizer."""
    return " ".join(row["sentences"])


def _texts(df: pd.DataFrame) -> list[str]:
    return [_doc_text(r) for _, r in df.iterrows()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    processed_dir = Path(cfg["data"]["processed_dir"])
    outputs_dir = Path(cfg["outputs"]["dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    documents_path = processed_dir / cfg["data"]["documents_file"]
    out_path = outputs_dir / cfg["outputs"]["baselines_file"]

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

    # Concatenate train + val into the baseline's training set. Baselines don't
    # need a separate val split (no early stopping); using train+val matches the
    # data the deep model effectively had access to during training.
    train_df = pd.concat([splits.train, splits.val], ignore_index=True)
    train_df = train_df.sort_values("release_date").reset_index(drop=True)

    train_texts = _texts(train_df)
    train_targets = train_df["target"].values.astype(float)

    # TF-IDF + Ridge baseline.
    # min_df=2 to drop hapax-legomena that are dominated by noise.
    # Lowercase + sublinear TF are standard for this baseline.
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(train_texts)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_tfidf, train_targets)

    # Evaluate on each held-out segment.
    report: dict[str, dict] = {
        "n_train": int(len(train_df)),
    }

    for name, df in [
        ("regime1_2017_2021", splits.regime1),
        ("regime2_2021_2025", splits.regime2),
        ("regime3_2025_present", splits.regime3),
    ]:
        if len(df) == 0:
            continue
        texts = _texts(df)
        y = df["target"].values.astype(float)

        X_tfidf = tfidf.transform(texts)
        ridge_pred = ridge.predict(X_tfidf)
        rm = regression_metrics(ridge_pred, y)

        report[name] = {"tfidf_ridge": rm.to_dict()}
        logger.info("baseline %s | ridge R^2=%.3f Pearson=%.3f n=%d",
                    name, rm.r2, rm.pearson_r, rm.n)

    out_path.write_text(json.dumps(report, indent=2))
    logger.info("baselines report written to %s", out_path)


if __name__ == "__main__":
    main()
