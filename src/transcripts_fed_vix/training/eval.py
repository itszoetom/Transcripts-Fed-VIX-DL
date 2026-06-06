"""Evaluation metrics for the regression model.

Regression metrics:
    MSE       , direct comparison vs. the TF-IDF ridge baseline.
    R^2       , comparable across splits with different variance.
    Pearson r , direction-of-signal metric, scale-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


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
