"""Hierarchical / cannibalization features: group-level lags and SKU share."""

from __future__ import annotations

import pandas as pd


def _slug_level(level: list[str]) -> str:
    return "_".join(level)


def add_hierarchical_features(
    df: pd.DataFrame,
    value_col: str = "sales",
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add group-level lag and rolling features plus within-group sales share.

    Group levels default to store×product_group, store×division, etc.
    """
    out = df.sort_values("date").copy()
    feat_cfg = (config or {}).get("forecasting", {}).get("features", {})
    hier_cfg = feat_cfg.get("hierarchical", {})

    if not hier_cfg.get("enabled", True):
        return out

    group_levels = hier_cfg.get(
        "group_levels",
        [
            ["store", "product_group"],
            ["store", "division"],
            ["store", "department"],
            ["product_group"],
        ],
    )
    group_lags = hier_cfg.get("group_lags", [1, 2, 4, 8, 26, 52])
    group_rolling_windows = hier_cfg.get("group_rolling_windows", [4, 13, 52])
    add_share = hier_cfg.get("add_share_within_group", True)

    for level in group_levels:
        slug = _slug_level(level)
        grp_col = f"grp_sales_{slug}"

        # aggregate group demand per week
        group_ts = (
            out.groupby(level + ["date"], observed=True)[value_col]
            .sum()
            .reset_index()
            .rename(columns={value_col: grp_col})
        )
        group_ts = group_ts.sort_values(level + ["date"])

        g = group_ts.groupby(level, observed=True)[grp_col]
        for lag in group_lags:
            group_ts[f"grp_lag_{lag}w_{slug}"] = g.shift(lag)
        for window in group_rolling_windows:
            group_ts[f"grp_roll_mean_{window}w_{slug}"] = g.transform(
                lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
            )

        merge_cols = level + ["date"] + [
            c for c in group_ts.columns if c not in level + ["date", grp_col]
        ]
        out = out.merge(group_ts[merge_cols + [grp_col]], on=level + ["date"], how="left")

        if add_share:
            out[f"share_{slug}"] = out[value_col] / out[grp_col].replace(0, pd.NA)

    return out
