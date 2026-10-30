"""Forecast scoring helpers."""

from __future__ import annotations

import numpy as np


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
