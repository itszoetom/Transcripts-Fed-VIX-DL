"""Evaluation metrics: regression + binarized classification.

Primary regression metrics:
    MSE       , direct comparison vs. the TF-IDF ridge baseline.
    R^2       , comparable across splits with different variance.
    Pearson r , direction-of-signal metric, scale-free.

Secondary binary metrics (target binarized at the *training* median):
    AUC-ROC   , comparable to the BoW logistic-regression baseline.
    F1        , at threshold 0.5 on the regression output (post-binarization).

The training median is the threshold rather than the per-split median so that
test-set evaluation does not leak information from the test split itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class RegressionMetrics:
    """Continuous-target metrics."""

    mse: float
    r2: float
    pearson_r: float
    pearson_p: float
    n: int

    def to_dict(self) -> dict:
        return {
            "mse": float(self.mse),
            "r2": float(self.r2),
            "pearson_r": float(self.pearson_r),
            "pearson_p": float(self.pearson_p),
            "n": int(self.n),
        }


@dataclass
class BinaryMetrics:
    """Binarized-target metrics."""

    auc_roc: float
    f1: float
    threshold_used: float
    n: int

    def to_dict(self) -> dict:
        return {
            "auc_roc": float(self.auc_roc),
            "f1": float(self.f1),
            "threshold_used": float(self.threshold_used),
            "n": int(self.n),
        }


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> RegressionMetrics:
    """Compute MSE, R^2, and Pearson r over a 1-D set of predictions/targets."""
    predictions = np.asarray(predictions, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()
    assert predictions.shape == targets.shape, "predictions and targets must match shape"

    n = int(predictions.size)
    err = predictions - targets
    mse = float(np.mean(err ** 2))

    # R^2 against the *target's* mean, the standard "fraction of variance
    # explained" definition. Guarded against zero-variance targets (degenerate
    # split): returns nan if total sum of squares is 0.
    tss = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / tss if tss > 0 else float("nan")

    # Pearson r, guard against constant inputs which scipy warns about.
    if predictions.std() == 0 or targets.std() == 0:
        pearson_r, pearson_p = float("nan"), float("nan")
    else:
        result = stats.pearsonr(predictions, targets)
        pearson_r, pearson_p = float(result.statistic), float(result.pvalue)

    return RegressionMetrics(mse=mse, r2=r2, pearson_r=pearson_r, pearson_p=pearson_p, n=n)


def binary_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
) -> BinaryMetrics:
    """Binarize regression outputs at `threshold` and report AUC-ROC + F1.

    The convention: positive class = "VIX went up by more than threshold" so
    label = (target > threshold). We feed the *continuous* prediction to
    roc_auc_score (it accepts a score) and the thresholded prediction to F1.

    Args:
        predictions: Continuous model outputs.
        targets:     Continuous target values.
        threshold:   Binarization threshold (typically the training-set
                     median of `targets`).
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()

    y_true = (targets > threshold).astype(int)
    y_score = predictions
    y_pred = (predictions > threshold).astype(int)

    # AUC is undefined when only one class is present in y_true; guard it.
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))

    # F1 with zero_division=0 returns 0 when there are no positive predictions
    # at all, preferable to crashing on a degenerate split.
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return BinaryMetrics(auc_roc=auc, f1=f1, threshold_used=float(threshold), n=int(y_true.size))
