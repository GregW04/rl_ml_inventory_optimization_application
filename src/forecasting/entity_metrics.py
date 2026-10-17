"""Grouped CV metrics by product and store-product (optionally per fold)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.forecasting.model import get_forecaster_class
from src.utils import tracking
from src.utils.config import load_config


def ensure_oof_targets(oof: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rebuild multi-horizon and lead-time targets on an OOF panel (e.g. from parquet)."""
    out = oof.sort_values(["store", "product", "date"]).copy()
    horizons = config["forecasting"]["model"]["horizons"]
    g = out.groupby(["store", "product"], observed=True)["sales"]

    for h in horizons:
        col = f"target_h{h}"
        if col not in out.columns:
            out[col] = g.shift(-h) if h else out["sales"]

    ltd_cfg = config["forecasting"]["model"].get("lead_time_demand", {})
    if ltd_cfg.get("enabled") and "target_S" not in out.columns:
        offsets = ltd_cfg.get("offsets", [0, 1, 2])
        out["target_S"] = sum(g.shift(-o) for o in offsets)

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


def compute_cv_entity_metrics(oof: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Pooled metrics per entity group (same definitions as global CV).

    Grains: product (with/without fold_id), store+product (with/without fold_id).
    """
    if oof.empty:
        return pd.DataFrame()

    evaluator = get_forecaster_class(config)(config)
    horizons = config["forecasting"]["model"]["horizons"]
    ltd_enabled = getattr(evaluator, "ltd_enabled", False)

    grain_specs: list[tuple[str, list[str]]] = [
        ("product", ["product"]),
        ("product", ["product", "fold_id"]),
        ("store_product", ["store", "product"]),
        ("store_product", ["store", "product", "fold_id"]),
    ]

    rows: list[dict] = []
    for grain, group_cols in grain_specs:
        for key, gdf in oof.groupby(group_cols, observed=True):
            base = _group_keys_to_base(grain, group_cols, key)

            for h in horizons:
                tgt = f"target_h{h}"
                pred = f"mu_h{h}"
                scored = gdf.dropna(subset=[tgt, pred])
                if scored.empty:
                    continue
                m = evaluator.evaluate_horizon(gdf, horizon=h)
                if not m:
                    continue
                rows.append({**base, "n": len(scored), **m})

            if ltd_enabled:
                m_ltd = evaluator.evaluate_ltd(gdf)
                if m_ltd:
                    scored_s = gdf.dropna(subset=["target_S"])
                    rows.append({**base, "n": len(scored_s), **m_ltd})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def per_series_cv_scalars(oof: pd.DataFrame, config: dict) -> dict[str, float]:
    """Median and p90 of per-(store, product) MAE and WQL on h=0 and lead-time S."""
    if oof.empty:
        return {}

    evaluator = get_forecaster_class(config)(config)
    ltd_enabled = getattr(evaluator, "ltd_enabled", False)

    mae_h0: list[float] = []
    wql_h0: list[float] = []
    mae_s: list[float] = []
    wql_s: list[float] = []

    for _, gdf in oof.groupby(["store", "product"], observed=True):
        m0 = evaluator.evaluate_horizon(gdf, horizon=0)
        if m0:
            if np.isfinite(m0.get("mae", np.nan)):
                mae_h0.append(float(m0["mae"]))
            wql = m0.get("wql")
            if wql is not None and np.isfinite(wql):
                wql_h0.append(float(wql))

        if ltd_enabled:
            m_s = evaluator.evaluate_ltd(gdf)
            if m_s:
                if np.isfinite(m_s.get("mae", np.nan)):
                    mae_s.append(float(m_s["mae"]))
                wql = m_s.get("wql")
                if wql is not None and np.isfinite(wql):
                    wql_s.append(float(wql))

    out: dict[str, float] = {}
    for name, vals in [
        ("mae_h0", mae_h0),
        ("wql_h0", wql_h0),
        ("mae_S", mae_s),
        ("wql_S", wql_s),
    ]:
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"cv_series_{name}_median"] = float(np.median(arr))
        out[f"cv_series_{name}_p90"] = float(np.percentile(arr, 90))
    return out


def log_cv_entity_metrics(oof: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute entity metrics and log table + per-series tail scalars to the active MLflow run."""
    entity_df = compute_cv_entity_metrics(oof, config)
    if not entity_df.empty:
        tracking.log_dataframe(entity_df, artifact_path="cv_by_entity")
        for key, val in per_series_cv_scalars(oof, config).items():
            mlflow.log_metric(key, val)
    return entity_df


def attach_cv_entity_metrics_to_run(
    run_id: str,
    oof_path: str | Path,
    config: dict,
    *,
    artifact_subdir: str = "cv_by_entity",
) -> pd.DataFrame:
    """Backfill entity metrics onto a FINISHED run (artifacts only)."""
    oof = ensure_oof_targets(pd.read_parquet(oof_path), config)
    entity_df = compute_cv_entity_metrics(oof, config)
    if entity_df.empty:
        return entity_df

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    entity_df.to_csv(tmp, index=False)
    client = MlflowClient()
    client.log_artifact(run_id, tmp, artifact_path=artifact_subdir)
    Path(tmp).unlink(missing_ok=True)
    return entity_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill CV entity metrics to an MLflow run")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--oof",
        default="artifacts/forecasts/oof_forecasts.parquet",
        help="Out-of-fold predictions parquet",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tracking.init_tracking(config, "forecasting")
    entity_df = attach_cv_entity_metrics_to_run(args.run_id, args.oof, config)
    print(f"Logged {len(entity_df)} rows to run {args.run_id}")


if __name__ == "__main__":
    main()
