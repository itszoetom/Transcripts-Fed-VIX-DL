"""VIX data acquisition and target construction.

Pulls the CBOE Volatility Index daily-close series (FRED ticker VIXCLS) and
aligns each Fed-document release date to its 3-day forward close-to-close VIX
change.

Why FRED VIXCLS:
    VIXCLS is the official CBOE closing value as redistributed by the St. Louis
    Fed. Pulling from FRED via `fredapi` is reproducible (API key + ticker
    fully determine the result) and avoids the slight close-time ambiguity of
    intraday tickers from third-party providers.

Why next-trading-day alignment:
    FOMC minutes are released at a fixed time (2:00 PM ET on a weekday) so
    those almost always have a same-day VIX close, but Humphrey-Hawkins
    testimony has historically taken place on weekdays whose closes are
    available, except in rare cases of a Saturday/Sunday release or a U.S.
    market holiday. When that happens we align the release to the *next*
    available trading day: this corresponds to the first VIX observation an
    investor could have reacted to. The alternative (previous trading day)
    would introduce look-ahead leakage. Documenting this choice is required
    for the academic write-up.

Target construction:
    Let r_t be the close on the aligned trading day t. The 3-day forward change
    is r_{t+3} - r_t, using *trading* days (so Friday + 3 trading days = the
    following Wednesday). Both close values are stored alongside the difference
    so the target can be re-derived or alternative targets explored without
    re-running this step.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series identifier for the CBOE VIX closing value.
VIX_SERIES_ID = "VIXCLS"

# Target horizon in *trading* days. Locked to 3 by the project spec.
TARGET_HORIZON_TRADING_DAYS = 3


@dataclass
class VixAlignment:
    """The VIX observations and target for a single document release.

    Attributes:
        release_date:         Original document release date (calendar).
        aligned_trading_date: Trading day used as t. Equal to release_date when
                              that's already a trading day; otherwise the next
                              available trading day from VIX series.
        vix_t:                VIX close on aligned_trading_date.
        vix_t_plus_h:         VIX close `TARGET_HORIZON_TRADING_DAYS` trading
                              days later.
        target:               vix_t_plus_h - vix_t (3-day forward change).
    """

    release_date: date
    aligned_trading_date: date
    vix_t: float
    vix_t_plus_h: float
    target: float


def fetch_vix(cache_path: Path | None = None) -> pd.Series:
    """Fetch the VIXCLS daily close series from FRED.

    Args:
        cache_path: If given and the file exists, read from it instead of
                    hitting the API. If it does not exist, fetch from FRED and
                    write the result to that path. Caching here keeps the
                    pipeline reproducible offline and avoids spamming FRED.

    Returns:
        A `pd.Series` indexed by date (sorted ascending) of VIX closing values.
        Non-trading days are absent from the index.

    Requires FRED_API_KEY in the environment (free key from
    https://fred.stlouisfed.org/docs/api/api_key.html).
    """
    if cache_path is not None and cache_path.exists():
        logger.info("loading cached VIX from %s", cache_path)
        df = pd.read_csv(cache_path, parse_dates=["date"])
        s = pd.Series(df["vix"].values, index=df["date"].dt.date, name="vix")
        return s.sort_index()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY environment variable not set. Get a free key from "
            "https://fred.stlouisfed.org/docs/api/api_key.html and `export FRED_API_KEY=...`."
        )

    # Import here so that consumers who only need the alignment helpers (e.g.,
    # tests) don't pay the fredapi import cost.
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series = fred.get_series(VIX_SERIES_ID)
    # fredapi returns a Series indexed by Timestamp; some daily series include
    # NaNs on official holidays, drop them so the index represents actual
    # trading days only.
    series = series.dropna()
    series.index = pd.to_datetime(series.index).date  # type: ignore[assignment]
    series = series.sort_index()
    series.name = "vix"

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = pd.DataFrame({"date": list(series.index), "vix": series.values})
        out.to_csv(cache_path, index=False)
        logger.info("cached VIX series to %s (%d rows)", cache_path, len(out))

    return series


def align_to_next_trading_day(release_date: date, vix: pd.Series) -> date | None:
    """Return the first trading-day date >= release_date that has a VIX close.

    Implements the next-trading-day convention described in the module
    docstring. Returns None if release_date is past the end of the series
    (no future VIX observation available).
    """
    # `vix.index` is sorted-ascending dates (trading days only). `searchsorted`
    # gives the insertion point; that's the first trading day >= release_date.
    idx = vix.index.searchsorted(release_date, side="left")
    if idx >= len(vix.index):
        return None
    return vix.index[idx]


def compute_forward_change(
    release_date: date,
    vix: pd.Series,
    horizon_trading_days: int = TARGET_HORIZON_TRADING_DAYS,
) -> VixAlignment | None:
    """Compute the h-day forward VIX change for one document release.

    Args:
        release_date:         Document release date.
        vix:                  VIX close series (trading-day-indexed).
        horizon_trading_days: Forward horizon in trading days.

    Returns:
        VixAlignment with aligned trading dates + target value, or None if
        either the alignment lookup or the +h-day lookup runs past the end of
        the series.
    """
    aligned = align_to_next_trading_day(release_date, vix)
    if aligned is None:
        return None
    idx_t = vix.index.searchsorted(aligned, side="left")
    idx_h = idx_t + horizon_trading_days
    if idx_h >= len(vix.index):
        # We don't have h trading days of future VIX after this release; skip.
        return None
    vix_t = float(vix.iloc[idx_t])
    vix_h = float(vix.iloc[idx_h])
    return VixAlignment(
        release_date=release_date,
        aligned_trading_date=aligned,
        vix_t=vix_t,
        vix_t_plus_h=vix_h,
        target=vix_h - vix_t,
    )
