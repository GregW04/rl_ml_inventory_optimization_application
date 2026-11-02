"""Grouped CV metrics by product and store-product (optionally per fold)."""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.forecasting.forecaster import evaluate_measures
from src.forecasting.measures import HORIZONS, Measure, primary_metric_name
from src.utils import tracking


def ensure_oof_targets(oof: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the direct targets on an OOF panel (e.g. read back from parquet)."""
    out = oof.sort_values(["store", "product", "date"]).copy()
    g = out.groupby(["store", "product"], observed=True)["sales"]
    for h in HORIZONS:
        col = f"target_h{h}"
        if col not in out.columns:
            out[col] = g.shift(-h) if h else out["sales"]
    return out


def _group_keys_to_base(grain: str, group_cols: list[str], key: Any) -> dict:
    if not isinstance(key, tuple):
        key = (key,)
    base: dict[str, Any] = {"grain": grain}
    for col, val in zip(group_cols, key):
        if col == "fold_id":
            base["fold_id"] = int(val)
        else:
            base[col] = val
    if "fold_id" not in base:
        base["fold_id"] = None
    if grain == "product":
        base["store"] = None
    return base


def compute_cv_entity_metrics(oof: pd.DataFrame, measures: list[Measure]) -> pd.DataFrame:
    """
    Pooled per-measure metrics per entity group (same definitions as global CV).

    Grains: product (with/without fold_id), store+product (with/without fold_id).
    """
    if oof.empty:
        return pd.DataFrame()
    grain_specs: list[tuple[str, list[str]]] = [
        ("product", ["product"]),
        ("product", ["product", "fold_id"]),
        ("store_product", ["store", "product"]),
        ("store_product", ["store", "product", "fold_id"]),
    ]
    rows: list[dict] = []
    for grain, group_cols in grain_specs:
        if not set(group_cols) <= set(oof.columns):
            continue
        for key, gdf in oof.groupby(group_cols, observed=True):
            m = evaluate_measures(gdf, measures)
            if m:
                rows.append({**_group_keys_to_base(grain, group_cols, key), "n": len(gdf), **m})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def per_series_cv_scalars(oof: pd.DataFrame, primary: Measure) -> dict[str, float]:
    """Median and p90 of the primary metric across (store, product) series."""
    if oof.empty:
        return {}
    metric = primary_metric_name(primary)
    vals = []
    for _, gdf in oof.groupby(["store", "product"], observed=True):
        v = evaluate_measures(gdf, [primary]).get(metric)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=float)
    return {
        f"cv_series_{metric}_median": float(np.median(arr)),
        f"cv_series_{metric}_p90": float(np.percentile(arr, 90)),
    }


def log_cv_entity_metrics(oof: pd.DataFrame, measures: list[Measure], primary: Measure) -> pd.DataFrame:
    """Compute entity metrics and log table + per-series tail scalars to the active MLflow run."""
    entity_df = compute_cv_entity_metrics(oof, measures)
    if not entity_df.empty:
        tracking.log_dataframe(entity_df, "cv_by_entity", "cv_by_entity.csv")
        for key, val in per_series_cv_scalars(oof, primary).items():
            mlflow.log_metric(key, val)
    return entity_df
