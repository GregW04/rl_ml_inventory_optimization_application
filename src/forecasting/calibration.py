"""Split-conformal calibration of quantile measures on out-of-fold forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.measures import Measure


def _transform(x: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(np.clip(x, 0.0, None)) if transform == "log1p" else np.asarray(x, dtype=float)


def fit_quantile_shifts(
    oof: pd.DataFrame,
    measures: list[Measure],
    transform: str = "log1p",
) -> dict[str, float]:
    """
    Shift per quantile measure so that P(y ≤ q̂ + c) = τ on the out-of-fold rows.

    The shift is taken in the model's target space (log1p by default), so it acts like a
    multiplicative correction that scales with the demand level. It uses the finite-sample
    conformal level ⌈(n + 1)τ⌉ / n, which guarantees coverage of at least τ on exchangeable
    data. Integer demand has ties, so realized coverage can sit slightly above τ.
    """
    shifts: dict[str, float] = {}
    for m in measures:
        tgt = f"target_h{m.horizon}"
        if m.kind != "quantile" or tgt not in oof.columns or m.id not in oof.columns:
            continue
        sub = oof[[tgt, m.id]].dropna()
        n = len(sub)
        if n == 0:
            continue
        scores = _transform(sub[tgt].to_numpy(float), transform) - _transform(
            sub[m.id].to_numpy(float), transform
        )
        level = min(1.0, np.ceil((n + 1) * m.tau) / n)
        # The epsilon keeps a shifted quantile that lands on an integer tie from
        # dropping just below it through floating-point round-off.
        shifts[m.id] = float(np.quantile(scores, level, method="higher")) + 1e-9
    return shifts
