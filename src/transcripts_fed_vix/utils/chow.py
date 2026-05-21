"""Chow test on prediction residuals for regime-change detection.

The Chow test asks: given a regression and a candidate breakpoint in the
time-ordered data, do the regression coefficients differ before vs. after the
breakpoint? Concretely we regress the model's prediction residuals
(`target - prediction`) on a constant (so the test reduces to a difference-
in-mean-residual test, equivalent to "does the model's bias shift across the
breakpoint?") and compute the F-statistic:

    F = ((RSS_pooled - (RSS_pre + RSS_post)) / k) / ((RSS_pre + RSS_post) / (N - 2k))

where k is the number of regressors (here k=1 — just the intercept), and N is
the total number of observations. Under H0 (no break), F ~ F(k, N - 2k).

We report the F-statistic, its p-value, and the per-segment mean residual so
the reader can interpret the direction of any detected shift.

This is the standard Chow (1960) procedure on residuals, the most common
formulation in the econometric regime-change literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ChowResult:
    """Result of one Chow test at a single breakpoint.

    Attributes:
        breakpoint:    Date used to split residuals into pre/post.
        f_statistic:   Chow F-statistic.
        p_value:       Two-sided p-value under F(k, N - 2k).
        n_pre:         Number of residuals in the pre-break segment.
        n_post:        Number of residuals in the post-break segment.
        mean_pre:      Mean residual in pre-break segment.
        mean_post:     Mean residual in post-break segment.
    """

    breakpoint: date
    f_statistic: float
    p_value: float
    n_pre: int
    n_post: int
    mean_pre: float
    mean_post: float

    def to_dict(self) -> dict:
        return {
            "breakpoint": self.breakpoint.isoformat(),
            "f_statistic": float(self.f_statistic),
            "p_value": float(self.p_value),
            "n_pre": int(self.n_pre),
            "n_post": int(self.n_post),
            "mean_pre": float(self.mean_pre),
            "mean_post": float(self.mean_post),
        }


def chow_test_on_residuals(
    residuals: pd.Series,
    dates: pd.Series,
    breakpoint: date,
) -> ChowResult:
    """Run a one-breakpoint Chow test on residuals.

    Args:
        residuals:  Series of model residuals (target - prediction).
        dates:      Series of release dates aligned with residuals.
        breakpoint: Date at which to split residuals into pre/post.

    Returns:
        ChowResult with F-stat, p-value, and segment summaries.

    Raises:
        ValueError if either segment has fewer than 2 observations (the test
        is undefined).
    """
    if len(residuals) != len(dates):
        raise ValueError("residuals and dates must have the same length")
    dates_ts = pd.to_datetime(dates).reset_index(drop=True)
    res = pd.Series(residuals.values, index=range(len(residuals)), dtype=float)

    bp_ts = pd.Timestamp(breakpoint)
    pre_mask = dates_ts < bp_ts
    post_mask = ~pre_mask

    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())
    if n_pre < 2 or n_post < 2:
        raise ValueError(
            f"Chow test undefined: n_pre={n_pre}, n_post={n_post}; need >= 2 in each segment"
        )

    # Regressing residuals on a constant => model prediction is the mean,
    # RSS = sum((y - mean)^2). With k=1.
    k = 1
    mean_pooled = float(res.mean())
    rss_pooled = float(((res - mean_pooled) ** 2).sum())

    mean_pre = float(res[pre_mask].mean())
    rss_pre = float(((res[pre_mask] - mean_pre) ** 2).sum())

    mean_post = float(res[post_mask].mean())
    rss_post = float(((res[post_mask] - mean_post) ** 2).sum())

    N = n_pre + n_post
    numerator = (rss_pooled - (rss_pre + rss_post)) / k
    denominator = (rss_pre + rss_post) / max(N - 2 * k, 1)
    f_stat = float(numerator / denominator) if denominator > 0 else float("inf")
    p_value = float(1.0 - stats.f.cdf(f_stat, k, N - 2 * k)) if np.isfinite(f_stat) else 0.0

    return ChowResult(
        breakpoint=breakpoint,
        f_statistic=f_stat,
        p_value=p_value,
        n_pre=n_pre,
        n_post=n_post,
        mean_pre=mean_pre,
        mean_post=mean_post,
    )
