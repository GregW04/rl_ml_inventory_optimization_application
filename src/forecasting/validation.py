"""Expanding-window time series cross-validation with embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesFold:
    """Single expanding-window fold."""

    fold_id: int
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_dates: np.ndarray
    val_dates: np.ndarray


def get_unique_dates(df: pd.DataFrame, date_col: str = "date") -> np.ndarray:
    """Sorted unique dates in the panel."""
    return np.sort(df[date_col].unique())


def generate_expanding_folds(
    dates: np.ndarray,
    n_folds: int = 5,
    fold_size_weeks: int = 13,
    embargo_weeks: int = 52,
    min_train_size_weeks: int = 52,
) -> list[TimeSeriesFold]:
    """
    Build expanding-window folds with embargo gap after train end.

    Fold k:
      Train:      dates[0 : train_end_idx]
      Embargo:    next `embargo_weeks` dates (excluded from train features leakage)
      Validation: next `fold_size_weeks` dates
    """
    dates = np.sort(dates)
    n = len(dates)
    folds: list[TimeSeriesFold] = []

    # anchor validation windows from the end of the timeline
    for k in range(n_folds):
        val_end_idx = n - 1 - k * fold_size_weeks
        val_start_idx = val_end_idx - fold_size_weeks + 1
        if val_start_idx < 0:
            break

        embargo_end_idx = val_start_idx - 1
        embargo_start_idx = embargo_end_idx - embargo_weeks + 1
        train_end_idx = embargo_start_idx - 1

        if train_end_idx + 1 < min_train_size_weeks:
            continue
        if embargo_start_idx < 0:
            continue

        train_dates = dates[: train_end_idx + 1]
        val_dates = dates[val_start_idx : val_end_idx + 1]

        folds.append(
            TimeSeriesFold(
                fold_id=len(folds),
                train_end=pd.Timestamp(dates[train_end_idx]),
                embargo_start=pd.Timestamp(dates[embargo_start_idx]),
                embargo_end=pd.Timestamp(dates[embargo_end_idx]),
                val_start=pd.Timestamp(dates[val_start_idx]),
                val_end=pd.Timestamp(dates[val_end_idx]),
                train_dates=train_dates,
                val_dates=val_dates,
            )
        )

    folds = folds[::-1]  # chronological order
    for i, fold in enumerate(folds):
        fold.fold_id = i
    return folds


def iter_fold_masks(
    df: pd.DataFrame,
    folds: list[TimeSeriesFold],
    date_col: str = "date",
) -> Iterator[tuple[TimeSeriesFold, pd.Series, pd.Series]]:
    """Yield (fold, train_mask, val_mask) for each fold."""
    for fold in folds:
        train_mask = df[date_col].isin(fold.train_dates)
        val_mask = df[date_col].isin(fold.val_dates)
        yield fold, train_mask, val_mask


def max_lag_from_config(config: dict) -> int:
    """Infer maximum lag for embargo sanity check."""
    lags = config.get("forecasting", {}).get("features", {}).get("lags", {}).get(
        "lags", [52]
    )
    group_lags = (
        config.get("forecasting", {})
        .get("features", {})
        .get("hierarchical", {})
        .get("group_lags", [52])
    )
    return max(max(lags), max(group_lags))
