"""Lag, rolling, and expanding features per store-product series (NaN-aware)."""

from __future__ import annotations

import numpy as np
import pandas as pd

_ROLL_AGGS = {"mean", "std", "min", "max", "median"}
_EXP_AGGS = {"mean", "std", "min", "max"}


def _rolling_agg(s: pd.Series, window: int, agg: str) -> pd.Series:
    rolled = s.shift(1).rolling(window=window, min_periods=1)
    return getattr(rolled, agg)()


def _expanding_agg(s: pd.Series, agg: str, min_periods: int) -> pd.Series:
    expanded = s.shift(1).expanding(min_periods=min_periods)
    return getattr(expanded, agg)()


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    group_cols: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add lag, rolling, and expanding features per series.

    Rolling and expanding stats use sales through t-1 only (shift before window).
    """
    out = df.sort_values(group_cols or ["store", "product"] + ["date"]).copy()
    group_cols = group_cols or ["store", "product"]

    feat_cfg = (config or {}).get("forecasting", {}).get("features", {})
    lag_cfg = feat_cfg.get("lags", {})

    if not lag_cfg.get("enabled", True):
        return out

    lags = lag_cfg.get("lags") or []
    rolling_windows = lag_cfg.get("rolling_windows") or []
    rolling_aggs = lag_cfg.get("rolling_aggs") or []
    expanding_aggs = lag_cfg.get("expanding_aggs") or []
    min_periods = lag_cfg.get("expanding_min_periods", 4)
    use_momentum = bool(lag_cfg.get("momentum", True))

    g = out.groupby(group_cols, observed=True)[value_col]

    for lag in lags:
        out[f"lag_{lag}w"] = g.shift(lag)

    for window in rolling_windows:
        for agg in rolling_aggs:
            if agg not in _ROLL_AGGS:
                continue
            col = f"roll_{agg}_{window}w"
            out[col] = g.transform(lambda s, w=window, a=agg: _rolling_agg(s, w, a))

    for agg in expanding_aggs:
        if agg not in _EXP_AGGS:
            continue
        col = f"exp_{agg}"
        out[col] = g.transform(
            lambda s, a=agg, mp=min_periods: _expanding_agg(s, a, mp)
        )

    if use_momentum:
        eps = 1.0
        chosen = [int(x) for x in list(lags) + list(rolling_windows)]
        peak_window = max(chosen) if chosen else 52
        lag1 = g.shift(1)
        roll4 = g.transform(lambda s: _rolling_agg(s, 4, "mean"))
        roll13 = g.transform(lambda s: _rolling_agg(s, 13, "mean"))
        roll_std4 = g.transform(lambda s: _rolling_agg(s, 4, "std"))
        out["demand_momentum"] = lag1 / (roll4 + eps)
        out["recent_spike"] = (lag1 > 2.0 * roll4).astype(float)
        out["demand_accel"] = roll4 / (roll13 + eps)
        out["demand_cv_4w"] = roll_std4 / (roll4 + eps)

        def _weeks_since_peak(s: pd.Series, window: int = peak_window) -> pd.Series:
            prev = s.shift(1)
            roll_max = prev.rolling(window, min_periods=1).max()
            at_peak = (prev >= roll_max) & prev.notna()
            idx = pd.Series(np.where(at_peak, np.arange(len(s)), np.nan), index=s.index)
            return (np.arange(len(s)) - idx.ffill()).astype(float)

        out["weeks_since_peak"] = g.transform(_weeks_since_peak)

    return out
