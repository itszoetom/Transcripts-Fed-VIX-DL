"""Temporal splits for training and regime analysis.

The pipeline uses two layered split decisions:

  1. **Hold-out structure** (the primary modeling split):
     - train:  documents released  <  2017-01-20  (Trump 1 inauguration)
     - val:    last 15% of the train range chronologically, used only for
               early stopping. Carved AFTER the pre-2017 cutoff so that
               nothing later than 2017-01-20 ever sees gradients or
               hyperparameter signal.
     - test1:  2017-01-20 <= release_date < 2021-01-20  (Trump 1 era)
     - test2:  2021-01-20 <= release_date < 2025-01-20  (Biden era)
     - test3:  2025-01-20 <= release_date              (Trump 2 era; sparse)

     We anchor split dates to U.S. presidential inauguration days because the
     secondary research question is whether the model's accuracy degrades
     across political regimes. Using the exact inauguration date avoids the
     ambiguity of "January 2017" / "January 2021".

  2. **Chow-test breakpoints** (the regime-change inference):
     Same dates (2017-01-20 and 2025-01-20) used as breakpoints in
     utils.chow.

All splits are *temporal*, no random shuffling. Sort is by release_date,
ascending.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import pandas as pd


# Inauguration-day boundaries. Locked by the design discussion.
DEFAULT_TRAIN_END = date(2017, 1, 20)   # Trump 1 inauguration
DEFAULT_REGIME2_START = date(2021, 1, 20)  # Biden inauguration
DEFAULT_REGIME3_START = date(2025, 1, 20)  # Trump 2 inauguration


@dataclass(frozen=True)
class SplitDates:
    """The three chronological boundaries that define the four temporal segments.

    Documents are placed as follows:

        train_pool:  release_date <  train_end
        regime1:     train_end    <= release_date <  regime2_start
        regime2:     regime2_start <= release_date <  regime3_start
        regime3:     regime3_start <= release_date

    Within train_pool we further carve off the trailing val_fraction
    (chronologically last) for early stopping.
    """

    train_end: date = DEFAULT_TRAIN_END
    regime2_start: date = DEFAULT_REGIME2_START
    regime3_start: date = DEFAULT_REGIME3_START
    val_fraction: float = 0.15


class TemporalSplits(NamedTuple):
    """Per-split DataFrames, in chronological order within each."""

    train: pd.DataFrame
    val: pd.DataFrame
    regime1: pd.DataFrame
    regime2: pd.DataFrame
    regime3: pd.DataFrame


def make_temporal_splits(
    documents_df: pd.DataFrame,
    dates: SplitDates | None = None,
) -> TemporalSplits:
    """Carve a processed-dataset DataFrame into the five temporal segments.

    Args:
        documents_df: DataFrame with a 'release_date' column (datetime).
        dates:        SplitDates configuration. Defaults to inauguration-day
                      boundaries.

    Returns:
        TemporalSplits with (train, val, regime1, regime2, regime3) DataFrames.
        Each is sorted ascending by release_date.
    """
    dates = dates or SplitDates()
    df = documents_df.sort_values("release_date").reset_index(drop=True)

    # Convert configured `date` objects to pandas Timestamps for comparison.
    train_end = pd.Timestamp(dates.train_end)
    r2 = pd.Timestamp(dates.regime2_start)
    r3 = pd.Timestamp(dates.regime3_start)

    train_pool = df[df["release_date"] < train_end].reset_index(drop=True)
    regime1 = df[(df["release_date"] >= train_end) & (df["release_date"] < r2)].reset_index(drop=True)
    regime2 = df[(df["release_date"] >= r2) & (df["release_date"] < r3)].reset_index(drop=True)
    regime3 = df[df["release_date"] >= r3].reset_index(drop=True)

    # Chronological val carve-off: last `val_fraction` of train_pool.
    n_val = max(1, int(round(len(train_pool) * dates.val_fraction)))
    train = train_pool.iloc[:-n_val].reset_index(drop=True)
    val = train_pool.iloc[-n_val:].reset_index(drop=True)

    return TemporalSplits(train=train, val=val, regime1=regime1, regime2=regime2, regime3=regime3)
