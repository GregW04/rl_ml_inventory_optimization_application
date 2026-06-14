"""Lag, rolling, and expanding features per store-product series (NaN-aware)."""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    group_cols: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add lag, rolling, and expanding features per series.

    Uses groupby.transform so indices stay aligned with the input frame.
    """
    out = df.sort_values(group_cols or ["store", "product"] + ["date"]).copy()
    group_cols = group_cols or ["store", "product"]

    feat_cfg = (config or {}).get("forecasting", {}).get("features", {})
    lag_cfg = feat_cfg.get("lags", {})

    if not lag_cfg.get("enabled", True):
        return out

    lags = lag_cfg.get("lags", [1, 2, 4, 8, 13, 26, 52])
    rolling_windows = lag_cfg.get("rolling_windows", [4, 8, 13, 26, 52])
    rolling_aggs = lag_cfg.get("rolling_aggs", ["mean", "std", "min", "max", "median"])
    expanding_aggs = lag_cfg.get("expanding_aggs", ["mean", "std", "min", "max"])
    min_periods = lag_cfg.get("expanding_min_periods", 4)

    g = out.groupby(group_cols, observed=True)[value_col]

    for lag in lags:
        out[f"lag_{lag}w"] = g.shift(lag)

    for window in rolling_windows:
        for agg in rolling_aggs:
            col = f"roll_{agg}_{window}w"
            if agg == "mean":
                out[col] = g.transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
                )
            elif agg == "std":
                out[col] = g.transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).std()
                )
            elif agg == "min":
                out[col] = g.transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).min()
                )
            elif agg == "max":
                out[col] = g.transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).max()
                )
            elif agg == "median":
                out[col] = g.transform(
                    lambda s, w=window: s.rolling(window=w, min_periods=1).median()
                )

    for agg in expanding_aggs:
        col = f"exp_{agg}"
        if agg == "mean":
            out[col] = g.transform(
                lambda s, mp=min_periods: s.expanding(min_periods=mp).mean()
            )
        elif agg == "std":
            out[col] = g.transform(
                lambda s, mp=min_periods: s.expanding(min_periods=mp).std()
            )
        elif agg == "min":
            out[col] = g.transform(
                lambda s, mp=min_periods: s.expanding(min_periods=mp).min()
            )
        elif agg == "max":
            out[col] = g.transform(
                lambda s, mp=min_periods: s.expanding(min_periods=mp).max()
            )

    return out
