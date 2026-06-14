"""End-to-end demand forecasting pipeline with rich MLflow tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.forecasting.model import (
    build_feature_matrix,
    get_feature_columns,
    get_forecaster_class,
    get_forecaster_type,
    get_primary_metric,
)
from src.forecasting.preprocessing import load_panel_from_config
from src.forecasting.validation import generate_expanding_folds, get_unique_dates, iter_fold_masks, max_lag_from_config
from src.utils import tracking
from src.utils.config import load_config


def run_cross_validation(
    df: pd.DataFrame,
    config: dict,
    model_name: str = "lightgbm",
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Run expanding-window CV; return OOF predictions, summary metrics, and per-fold details."""
    val_cfg = config["forecasting"]["validation"]
    dates = get_unique_dates(df)
    folds = generate_expanding_folds(
        dates,
        n_folds=val_cfg["n_folds"],
        fold_size_weeks=val_cfg["fold_size_weeks"],
        embargo_weeks=val_cfg["embargo_weeks"],
        min_train_size_weeks=val_cfg["min_train_size_weeks"],
    )

    feature_cols = get_feature_columns(df)
    ForecasterClass = get_forecaster_class(config)
    oof_parts = []
    fold_metrics = []

    for fold, train_mask, val_mask in iter_fold_masks(df, folds):
        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()

        forecaster = ForecasterClass(
            config, model_name=model_name, model_params=model_params
        )
        forecaster.fit(train_df, feature_cols=feature_cols)
        scored = forecaster.predict(val_df, feature_cols=feature_cols)
        scored["fold_id"] = fold.fold_id
        oof_parts.append(scored)

        for h in config["forecasting"]["model"]["horizons"]:
            m = forecaster.evaluate_horizon(scored, horizon=h)
            m["fold_id"] = fold.fold_id
            m["train_end"] = str(fold.train_end.date())
            m["val_start"] = str(fold.val_start.date())
            m["val_end"] = str(fold.val_end.date())
            fold_metrics.append(m)

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    metrics_summary = {}
    if fold_metrics:
        mdf = pd.DataFrame(fold_metrics)
        # Overall averages
        for metric in ["mae", "smape", "mase", "wql"]:
            if metric in mdf.columns:
                metrics_summary[f"cv_{metric}_mean"] = float(mdf[metric].mean())
        # Per-horizon averages
        for h in config["forecasting"]["model"]["horizons"]:
            hdf = mdf[mdf["horizon"] == h] if "horizon" in mdf.columns else mdf
            for metric in ["mae", "mase", "wql"]:
                if metric in hdf.columns:
                    metrics_summary[f"cv_{metric}_h{h}_mean"] = float(hdf[metric].mean())

    return oof, metrics_summary, fold_metrics


def _log_fold_metrics(fold_metrics: list[dict]) -> None:
    """Log per-fold, per-horizon metrics as MLflow step metrics."""
    mdf = pd.DataFrame(fold_metrics)
    if mdf.empty:
        return
    for _, row in mdf.iterrows():
        h = int(row.get("horizon", 0))
        fold = int(row.get("fold_id", 0))
        for metric in ["mae", "smape", "mase", "wql"]:
            val = row.get(metric)
            if val is not None and np.isfinite(float(val)):
                mlflow.log_metric(f"{metric}_h{h}", float(val), step=fold)


def _log_feature_importances(forecaster: Any, top_n: int = 30) -> None:
    """Log feature importance for horizon-0 model (Q50 or mean)."""
    models = forecaster.models
    if (0, 0.5) in models:
        model = models[(0, 0.5)]
        title_suffix = "Q50 h=0"
    elif 0 in models:
        model = models[0]
        title_suffix = "mean h=0"
    else:
        return
    try:
        if forecaster.model_name == "lightgbm":
            importances = model.booster_.feature_importance(importance_type="gain")
        else:
            importances = model.feature_importances_
    except AttributeError:
        return
    names = forecaster.feature_cols
    if len(names) != len(importances):
        return

    imp_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    tracking.log_dataframe(imp_df, artifact_path="feature_importance")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 3)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_xlabel("Gain (importance)")
    ax.set_title(f"Top feature importances — {title_suffix} model")
    tracking.log_figure(fig, artifact_path="feature_importance")


def _log_cv_chart(fold_metrics: list[dict], horizons: list[int]) -> None:
    """Log a CV metric chart (MAE per fold, per horizon)."""
    mdf = pd.DataFrame(fold_metrics)
    if mdf.empty or "mae" not in mdf.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for h in horizons:
        hdf = mdf[mdf["horizon"] == h].sort_values("fold_id")
        if not hdf.empty:
            ax.plot(hdf["fold_id"], hdf["mae"], marker="o", label=f"h={h}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("CV MAE per fold and horizon")
    ax.legend()
    tracking.log_figure(fig, artifact_path="cv_charts")


def _log_oof_sample_chart(oof: pd.DataFrame) -> None:
    """Log an OOF actual vs predicted chart for a sample SKU."""
    if oof.empty or "mu_h0" not in oof.columns:
        return
    # Pick the SKU with the most non-null OOF rows
    key = (
        oof.dropna(subset=["mu_h0"])
        .groupby(["store", "product"])
        .size()
        .idxmax()
    )
    sku = oof[(oof["store"] == key[0]) & (oof["product"] == key[1])].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sku["date"], sku["sales"], label="Actual", color="steelblue")
    ax.plot(sku["date"], sku["mu_h0"], label="Forecast mu (h=0)", color="orange", linestyle="--")
    if "pred_q10_h0" in sku.columns and "pred_q90_h0" in sku.columns:
        ax.fill_between(
            sku["date"],
            sku["pred_q10_h0"],
            sku["pred_q90_h0"],
            alpha=0.2,
            color="orange",
            label="Q10–Q90 interval",
        )
    ax.set_title(f"OOF forecast — store={key[0]}, product={key[1]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    tracking.log_figure(fig, artifact_path="cv_charts")


def _run_model_selection(
    df: pd.DataFrame,
    config: dict,
) -> list[tuple[str, float]]:
    """Run model selection step and log results to the active MLflow run."""
    from src.forecasting.model_selection import select_candidates

    primary = get_primary_metric(config)
    candidates = select_candidates(df, config)
    mlflow.log_param("selection_candidates", str([n for n, _ in candidates]))
    for name, score in candidates:
        mlflow.log_metric(f"selection_{primary}_{name}", score)
    return candidates


def _run_tuning(
    candidates: list[tuple[str, float]],
    df: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Tune each candidate and log Optuna results; return champion info."""
    from src.forecasting.tuning import tune_candidates

    primary = get_primary_metric(config)
    results = tune_candidates(candidates, df, config)
    champion = results[0]

    for r in results:
        n = r["model_name"]
        mlflow.log_metric(f"tuned_{primary}_{n}", r["best_value"])
        mlflow.log_metric(f"tuning_trials_{n}", r["n_trials_completed"])
        mlflow.log_params({f"best_{n}_{k}": v for k, v in r["best_params"].items()})

    mlflow.log_metric(f"champion_tuned_{primary}", champion["best_value"])
    return champion


def run_pipeline(config_path: str | None = None, force: bool = False) -> dict:
    """Full pipeline: load → features → CV → retrain → save artifacts."""
    config = load_config(config_path)
    root = Path(config["_project_root"])

    mlflow_cfg = config["forecasting"]["mlflow"]
    experiment_name = tracking.setup_mlflow(config, "forecasting")

    config_hash = tracking.config_hash(
        config["forecasting"], ignore_keys=("mlflow", "artifacts")
    )
    if mlflow_cfg.get("skip_duplicate_runs", True):
        tracking.check_duplicate_run(experiment_name, config_hash, force=force)

    panel = load_panel_from_config(config)
    df = build_feature_matrix(panel, config)

    max_lag = max_lag_from_config(config)
    embargo = config["forecasting"]["validation"]["embargo_weeks"]
    if embargo < max_lag:
        raise ValueError(f"embargo_weeks ({embargo}) must be >= max lag ({max_lag})")

    horizons = config["forecasting"]["model"]["horizons"]
    val_cfg = config["forecasting"]["validation"]
    sel_cfg = config.get("forecasting", {}).get("model_selection", {})
    tune_cfg = config.get("forecasting", {}).get("tuning", {})
    forecaster_type = get_forecaster_type(config)
    primary_metric = get_primary_metric(config)
    ForecasterClass = get_forecaster_class(config)

    with mlflow.start_run(run_name="demand_forecast_pipeline"):
        mlflow.set_tag("config_hash", config_hash)

        # --- params ---
        run_params = {
            "forecaster_type": forecaster_type,
            "primary_metric": primary_metric,
            "n_folds": val_cfg["n_folds"],
            "fold_size_weeks": val_cfg["fold_size_weeks"],
            "embargo_weeks": val_cfg["embargo_weeks"],
            "min_train_size_weeks": val_cfg["min_train_size_weeks"],
            "horizons": str(horizons),
            "n_features": len(get_feature_columns(df)),
            "n_skus": df.groupby(["store", "product"]).ngroups,
            "n_weeks": df["date"].nunique(),
            "oos_masking": True,
        }
        if forecaster_type == "quantile":
            run_params["quantiles"] = str(config["forecasting"]["model"]["quantiles"])
        mlflow.log_params(run_params)

        # --- Step 1: model selection (optional) ---
        champion_model_name = "lightgbm"
        champion_model_params: dict | None = None

        if sel_cfg.get("enabled", False):
            candidates = _run_model_selection(df, config)

            # --- Step 2: hypertune survivors (optional) ---
            if tune_cfg.get("enabled", False):
                champion = _run_tuning(candidates, df, config)
                champion_model_name = champion["model_name"]
                champion_model_params = champion["best_params"]
            else:
                # Use the best default-param model without tuning
                champion_model_name = candidates[0][0]
                champion_model_params = None
        elif tune_cfg.get("enabled", False):
            # Tune only the configured (or default) model, skip selection
            default_model = tune_cfg.get("model", "lightgbm")
            from src.forecasting.tuning import tune_model
            result = tune_model(default_model, df, config)
            champion_model_name = default_model
            champion_model_params = result["best_params"]
            mlflow.log_metric(f"champion_tuned_{primary_metric}", result["best_value"])
            mlflow.log_params(
                {f"best_{k}": v for k, v in champion_model_params.items()}
            )

        # Logged once, after every branch has settled on the champion.
        mlflow.log_param("champion_model", champion_model_name)

        # --- Step 3: full CV with champion ---
        oof, cv_metrics, fold_metrics = run_cross_validation(
            df, config,
            model_name=champion_model_name,
            model_params=champion_model_params,
        )

        for k, v in cv_metrics.items():
            mlflow.log_metric(k, v)

        _log_fold_metrics(fold_metrics)
        _log_cv_chart(fold_metrics, horizons)
        _log_oof_sample_chart(oof)

        if fold_metrics:
            tracking.log_dataframe(pd.DataFrame(fold_metrics), artifact_path="cv_detail")

        # --- Step 4: retrain on full data with champion ---
        feature_cols = get_feature_columns(df)
        forecaster = ForecasterClass(
            config,
            model_name=champion_model_name,
            model_params=champion_model_params,
        )
        forecaster.fit(df, feature_cols=feature_cols)
        full_scored = forecaster.predict(df, feature_cols=feature_cols)

        # Full-data metrics per horizon
        for h in horizons:
            m = forecaster.evaluate_horizon(full_scored, horizon=h)
            for k, v in m.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    mlflow.log_metric(f"train_{k}_h{h}", float(v))

        # Feature importances
        _log_feature_importances(forecaster)

        # --- Save artifacts ---
        model_dir = root / config["forecasting"]["artifacts"]["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = (
            "mean_forecaster.joblib"
            if forecaster_type == "mean"
            else "quantile_forecaster.joblib"
        )
        model_path = model_dir / artifact_name
        forecaster.save(model_path)
        mlflow.log_artifact(str(model_path))

        forecasts_dir = root / config["forecasting"]["artifacts"]["forecasts_dir"]
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        forecast_cols = (
            ["store", "product", "date", "sales"]
            + [c for c in full_scored.columns if c.startswith(("mu_h", "sigma_h", "pred_q"))]
        )
        full_scored[forecast_cols].to_parquet(forecasts_dir / "train_forecasts.parquet", index=False)
        if not oof.empty:
            oof_cols = [c for c in forecast_cols if c in oof.columns] + ["fold_id"]
            oof[[c for c in oof_cols if c in oof.columns]].to_parquet(
                forecasts_dir / "oof_forecasts.parquet", index=False
            )

        mlflow.log_dict(cv_metrics, "cv_metrics.json")
        return {"cv_metrics": cv_metrics, "model_path": str(model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demand forecasting pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if a FINISHED run exists for the same forecasting config",
    )
    args = parser.parse_args()
    result = run_pipeline(args.config, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
