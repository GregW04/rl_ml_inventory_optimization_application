"""Point-in-time demand profile per store-product series: intermittency, variability, demand pattern, volume tier."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Demand-pattern codes 0-3 (Syntetos-Boylan-Croston); -1 marks a series with too little
# history to classify. ``DEMAND_CLASSES`` holds the display names in code order.
SBC_CLASSES: tuple[str, ...] = ("smooth", "erratic", "intermittent", "lumpy")
DEMAND_CLASSES: tuple[str, ...] = ("Smooth", "Erratic", "Intermittent", "Lumpy")
UNCLASSIFIED = "Unclassified"

_DEFAULTS = {
    "window_weeks": 52,
    "min_periods": 13,
    "adi_cutoff": 1.32,
    "cv2_cutoff": 0.49,
    "volume_cutoffs": [0.8, 0.95],
}


def demand_class_name(code) -> str:
    """Display name of a demand-pattern code (``Unclassified`` for -1 or missing)."""
    try:
        i = int(code)
    except (TypeError, ValueError):
        return UNCLASSIFIED
    return DEMAND_CLASSES[i] if 0 <= i < len(DEMAND_CLASSES) else UNCLASSIFIED


def _trailing_sum(s: pd.Series, groups: list[pd.Series], window: int) -> pd.Series:
    """Per-series sum over the ``window`` weeks that end at t-1."""
    return s.groupby(groups, observed=True).transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
    ).fillna(0.0)


def _volume_tiers(df: pd.DataFrame, cutoffs: list[float]) -> pd.Series:
    """Volume tier per row (0 top, 1 middle, 2 tail) from the share of the date's total level held by larger series."""
    codes = pd.Series(-1, index=df.index, dtype=int)
    valid = df["dp_level"].notna()
    if not valid.any():
        return codes
    tmp = df.loc[valid, ["date", "dp_level"]].sort_values(
        ["date", "dp_level"], ascending=[True, False], kind="mergesort"
    )
    by_date = tmp.groupby("date")["dp_level"]
    share_before = (by_date.cumsum() - tmp["dp_level"]) / by_date.transform("sum").replace(0.0, np.nan)
    codes.loc[tmp.index] = np.select([share_before < cutoffs[0], share_before < cutoffs[1]], [0, 1], 2)
    return codes


def add_demand_profile_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    group_cols: list[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add each series' demand profile as of the week before the row.

    Every value comes from the trailing ``window_weeks`` that end at t-1, the same rule
    as the lag features, so a row never sees its own target. Weeks with masked
    (out-of-stock, NaN) sales are left out instead of counting as zero demand.

    - ``dp_adi``: average inter-demand interval, observed weeks / weeks with sales > 0
    - ``dp_cv2``: squared coefficient of variation of the non-zero weekly sales
    - ``dp_zero_share``: share of observed weeks with zero sales
    - ``dp_level``: mean weekly sales
    - ``dp_volume_pct``: percentile of ``dp_level`` among the series on the same date
    - ``dp_weeks_since_sale``: weeks since the last week with sales
    - ``dp_sbc_class``: Syntetos–Boylan–Croston class from the ADI and CV² cutoffs
      (0 smooth, 1 erratic, 2 intermittent, 3 lumpy)
    - ``dp_volume_tier``: volume tier on the date; 0 (top) while the larger series hold less
      than ``volume_cutoffs[0]`` of the date's total level, 1 (middle) below ``volume_cutoffs[1]``,
      else 2 (tail)

    The numeric features are NaN and the classes -1 until a series has ``min_periods``
    observed weeks (the SBC class also needs two weeks with sales): CatBoost rejects
    NaN categories. Per-date ranks and shares compare trailing values only.
    """
    cfg = {**_DEFAULTS, **((config or {}).get("forecasting", {}).get("features", {}).get("demand_profile") or {})}
    out = df.sort_values((group_cols or ["store", "product"]) + ["date"]).copy()
    if not cfg.get("enabled", True):
        return out
    group_cols = group_cols or ["store", "product"]
    window, min_periods = int(cfg["window_weeks"]), int(cfg["min_periods"])

    groups = [out[c] for c in group_cols]
    y = out[value_col].astype(float)
    size = y.where(y > 0, 0.0)
    n_obs = _trailing_sum(y.notna().astype(float), groups, window)
    n_pos = _trailing_sum((y > 0).astype(float), groups, window)
    total = _trailing_sum(size, groups, window)
    total_sq = _trailing_sum(size * size, groups, window)

    enough = n_obs >= min_periods
    pos = n_pos.where(n_pos > 0)
    mean_pos = total / pos
    var_pos = (total_sq / pos - mean_pos * mean_pos).clip(lower=0.0)

    out["dp_adi"] = (n_obs / pos).where(enough)
    out["dp_cv2"] = (var_pos / (mean_pos * mean_pos)).where(enough)
    out["dp_zero_share"] = ((n_obs - n_pos) / n_obs.where(n_obs > 0)).where(enough)
    out["dp_level"] = (total / n_obs.where(n_obs > 0)).where(enough)
    out["dp_volume_pct"] = out.groupby("date")["dp_level"].rank(pct=True)

    last_sale = out["date"].where(y > 0).groupby(groups, observed=True).transform(
        lambda s: s.shift(1).ffill()
    )
    out["dp_weeks_since_sale"] = (out["date"] - last_sale).dt.days / 7.0

    adi_low = out["dp_adi"] < float(cfg["adi_cutoff"])
    cv2_low = out["dp_cv2"] < float(cfg["cv2_cutoff"])
    sbc = np.select([adi_low & cv2_low, adi_low, cv2_low], [0, 1, 2], 3)
    out["dp_sbc_class"] = np.where(enough & (n_pos >= 2), sbc, -1).astype(int)
    # ``abc_cutoffs`` is the key older configs and saved setups used.
    cutoffs = cfg.get("volume_cutoffs") or cfg.get("abc_cutoffs") or _DEFAULTS["volume_cutoffs"]
    out["dp_volume_tier"] = _volume_tiers(out, [float(c) for c in cutoffs])
    return out
