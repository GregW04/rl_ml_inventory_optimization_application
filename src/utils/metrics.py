"""Forecast evaluation metrics: MASE, sMAPE, weighted quantile loss."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonal_period: int = 52,
) -> float:
    """
    Mean Absolute Scaled Error.

    Scale uses in-sample naive seasonal differences when `y_train` is provided,
    otherwise uses |y_true[t] - y_true[t-seasonal_period]| on the evaluation set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))

    if y_train is not None:
        y_train = np.asarray(y_train, dtype=float)
        y_train = y_train[np.isfinite(y_train)]
        if len(y_train) > seasonal_period:
            denom = np.mean(
                np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
            )
        else:
            denom = np.nan
    else:
        if len(y_true[mask]) > seasonal_period:
            yt = y_true[mask]
            denom = np.mean(np.abs(yt[seasonal_period:] - yt[:-seasonal_period]))
        else:
            denom = np.nan

    if not np.isfinite(denom) or denom == 0:
        return np.nan
    return float(mae / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    num = np.abs(y_true[mask] - y_pred[mask])
    denom = np.abs(y_true[mask]) + np.abs(y_pred[mask])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, num / denom, 0.0)
    return float(200.0 * np.mean(ratio))


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Pinball loss for a single quantile."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan

    err = y_true[mask] - y_pred[mask]
    q = quantile
    return float(np.mean(np.maximum(q * err, (q - 1) * err)))


def weighted_quantile_loss(
    y_true: np.ndarray,
    predictions: dict[float, np.ndarray],
    quantiles: list[float] | None = None,
) -> float:
    """
    Average pinball loss across quantiles (equal weights).

    `predictions` maps quantile level -> predicted array.
    """
    if quantiles is None:
        quantiles = sorted(predictions.keys())

    losses = []
    for q in quantiles:
        if q not in predictions:
            continue
        losses.append(quantile_loss(y_true, predictions[q], q))

    if not losses:
        return np.nan
    return float(np.mean(losses))


def evaluate_forecasts(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    quantile_pred_cols: dict[float, str] | None = None,
) -> dict[str, float]:
    """Compute standard metrics from a scored dataframe."""
    y = df[target_col].values
    y_hat = df[pred_col].values

    metrics = {
        "mae": float(np.nanmean(np.abs(y - y_hat))),
        "smape": smape(y, y_hat),
        "mase": mase(y, y_hat),
    }

    if quantile_pred_cols:
        preds = {q: df[col].values for q, col in quantile_pred_cols.items()}
        metrics["wql"] = weighted_quantile_loss(y, preds, list(quantile_pred_cols.keys()))

    return metrics
