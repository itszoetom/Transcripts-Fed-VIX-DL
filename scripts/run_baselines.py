"""Linear-model baselines: TF-IDF + Ridge regression, BoW + Logistic regression.

The project's primary research question (does the sentence-attention model beat
linear bag-of-features baselines on Fed text?) requires apples-to-apples
baselines on the *same* temporal splits.

Baselines:

    1. TF-IDF + Ridge regression
       - Continuous regression target (3-day VIX change).
       - Compared to the deep model on MSE and Pearson r.

    2. Bag-of-words counts + Logistic regression
       - Target binarized at the *training-set* median (no leakage).
       - Compared to the deep model on AUC-ROC and F1.

Both baselines use the same train / val / test1 / test2 / test3 segments the
deep model uses, and the same train-median binarization threshold.

We deliberately do NOT do cross-validated hyperparameter search inside the
training segment — the baselines use fixed, conventional defaults (Ridge
alpha=1.0, LR C=1.0). This keeps the comparison honest: the deep model has
exactly one early-stopping signal, and the baselines have zero.
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression

from transcripts_fed_vix.training.eval import (
    regression_metrics,
    binary_classification_metrics,
)
from transcripts_fed_vix.utils import make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_baselines")


def _doc_text(row: pd.Series) -> str:
    """Concatenate a document's first-80 sentences into one string for the vectorizers."""
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
    train_median = float(np.median(train_targets))

    # ---------------------------------------------------------------
    # Baseline 1: TF-IDF + Ridge
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Baseline 2: BoW + Logistic regression (binary)
    # ---------------------------------------------------------------
    bow = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.95,
    )
    X_train_bow = bow.fit_transform(train_texts)
    train_labels = (train_targets > train_median).astype(int)
    logreg = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
    # If all training labels are the same class (degenerate), logreg fails.
    if len(np.unique(train_labels)) < 2:
        logger.warning("training labels are all one class after binarization; skipping logreg")
        logreg = None
    else:
        logreg.fit(X_train_bow, train_labels)

    # ---------------------------------------------------------------
    # Evaluate on each held-out segment
    # ---------------------------------------------------------------
    report: dict[str, dict] = {
        "train_median_target": train_median,
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

        # TF-IDF + Ridge
        X_tfidf = tfidf.transform(texts)
        ridge_pred = ridge.predict(X_tfidf)
        rm = regression_metrics(ridge_pred, y)

        # BoW + Logistic
        if logreg is not None:
            X_bow = bow.transform(texts)
            # LR's decision_function gives a signed score; positive = predicts
            # "above-median". Use that as the continuous score for AUC, and
            # the >0 threshold maps it to a class prediction.
            scores = logreg.decision_function(X_bow)
            # Pass continuous scores + the median-binarized target into our
            # shared metric helper, with threshold 0 because decision_function
            # is centered at the LR boundary.
            #
            # We binarize the targets at train_median (so we're answering the
            # same question the deep-model binary metrics ask). For the
            # "score" we pass scores (continuous LR margin) and for "predict"
            # we use scores > 0.
            y_true = (y > train_median).astype(int)
            from sklearn.metrics import roc_auc_score, f1_score
            auc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, scores))
            f1 = float(f1_score(y_true, (scores > 0).astype(int), zero_division=0))
            bm = {
                "auc_roc": auc,
                "f1": f1,
                "threshold_used": train_median,
                "n": int(len(y)),
                "lr_decision_threshold": 0.0,
            }
        else:
            bm = {"skipped_reason": "training labels were all one class"}

        report[name] = {
            "tfidf_ridge": rm.to_dict(),
            "bow_logreg": bm,
        }
        logger.info("baseline %s | ridge R^2=%.3f Pearson=%.3f | logreg AUC=%s",
                    name, rm.r2, rm.pearson_r,
                    f"{bm.get('auc_roc', float('nan')):.3f}")

    out_path.write_text(json.dumps(report, indent=2))
    logger.info("baselines report written to %s", out_path)


if __name__ == "__main__":
    main()
