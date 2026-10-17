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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.forecasting.model import (
    build_feature_matrix,
    get_feature_columns,
    get_forecaster_class,
    get_forecaster_type,
    get_primary_metric,
)
from src.forecasting.entity_metrics import log_cv_entity_metrics
from src.forecasting.periods import (
    build_setup_record,
    load_setup,
    resolve_periods,
    save_setup,
    setup_path,
)
from src.forecasting.preprocessing import load_panel_from_config
from src.forecasting.validation import TimeSeriesFold, get_unique_dates, iter_fold_masks, make_folds
from src.utils import tracking
from src.utils.config import load_config


def run_cross_validation(
    df: pd.DataFrame,
    config: dict,
    model_name: str = "lightgbm",
    model_params: dict | None = None,
    folds: list[TimeSeriesFold] | None = None,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Run expanding-window CV; return OOF predictions, summary metrics, and per-fold details."""
    if folds is None:
        folds = make_folds(get_unique_dates(df), config)

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

        if getattr(forecaster, "ltd_enabled", False):
            m_ltd = forecaster.evaluate_ltd(scored)
            if m_ltd:
                m_ltd["fold_id"] = fold.fold_id
                m_ltd["train_end"] = str(fold.train_end.date())
                m_ltd["val_start"] = str(fold.val_start.date())
                m_ltd["val_end"] = str(fold.val_end.date())
                fold_metrics.append(m_ltd)

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

        ltd_rows = [m for m in fold_metrics if m.get("target") == "S"]
        if ltd_rows:
            ldf = pd.DataFrame(ltd_rows)
            for metric in ["mae", "mase", "wql"]:
                if metric in ldf.columns:
                    metrics_summary[f"cv_{metric}_S_mean"] = float(ldf[metric].mean())
            if "picp_q83_S" in ldf.columns:
                metrics_summary["cv_picp_q83_S_mean"] = float(ldf["picp_q83_S"].mean())

    return oof, metrics_summary, fold_metrics


def _log_fold_metrics(fold_metrics: list[dict]) -> None:
    """Log per-fold, per-horizon metrics as MLflow step metrics."""
    mdf = pd.DataFrame(fold_metrics)
    if mdf.empty:
        return
    for _, row in mdf.iterrows():
        fold = int(row.get("fold_id", 0))
        if row.get("target") == "S":
            for metric in ["mae", "smape", "mase", "wql", "picp_q83_S"]:
                val = row.get(metric)
                if val is not None and np.isfinite(float(val)):
                    mlflow.log_metric(f"{metric}_S", float(val), step=fold)
            continue
        horizon = row.get("horizon")
        if horizon is None or not np.isfinite(float(horizon)):
            continue
        h = int(horizon)
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


def _fit_rl_block_forecasts(
    df: pd.DataFrame,
    config: dict,
    periods,
    ForecasterClass: type,
    model_name: str,
    model_params: dict | None,
) -> tuple[pd.DataFrame, dict[str, float], Any]:
    """Out-of-sample forecasts for each RL block (per-block refit when enabled)."""
    horizons = config["forecasting"]["model"]["horizons"]
    feature_cols = get_feature_columns(df)
    blocks = [
        ("rl_train", periods.rl_train_start, periods.rl_train_end, periods.fit_end_rl_train),
        ("rl_val", periods.rl_val_start, periods.rl_val_end, periods.fit_end_rl_val),
        ("rl_test", periods.rl_test_start, periods.rl_test_end, periods.fit_end_rl_test),
    ]
    if not periods.refit_per_block:
        fit_one = periods.fit_end_rl_train
        blocks = [(n, s, e, fit_one) for n, s, e, _ in blocks]

    parts: list[pd.DataFrame] = []
    block_metrics: dict[str, float] = {}
    last_forecaster = None

    for name, start, end, fit_end in blocks:
        train_df = df[df["date"] <= fit_end].copy()
        pred_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        if pred_df.empty:
            continue
        forecaster = ForecasterClass(
            config, model_name=model_name, model_params=model_params
        )
        forecaster.fit(train_df, feature_cols=feature_cols)
        scored = forecaster.predict(pred_df, feature_cols=feature_cols)
        parts.append(scored)
        last_forecaster = forecaster

        for h in horizons:
            m = forecaster.evaluate_horizon(scored, horizon=h)
            for k, v in m.items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    block_metrics[f"oos_{name}_{k}_h{h}"] = float(v)
        if getattr(forecaster, "ltd_enabled", False):
            m_ltd = forecaster.evaluate_ltd(scored)
            for k, v in (m_ltd or {}).items():
                if isinstance(v, (int, float)) and np.isfinite(float(v)):
                    block_metrics[f"oos_{name}_{k}"] = float(v)

    if not parts:
        return pd.DataFrame(), block_metrics, last_forecaster
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["store", "product", "date"], keep="last")
    return combined, block_metrics, last_forecaster


def _keep_previous_forecasts(config: dict) -> None:
    """Attach the on-disk forecasts to the run that wrote them if that run never logged them."""
    prev_id = (load_setup(config) or {}).get("run_id")
    path = setup_path(config).with_name("train_forecasts.parquet")
    if not prev_id or not path.exists():
        return
    client = MlflowClient()
    try:
        logged = {a.path for a in client.list_artifacts(prev_id, "forecasts")}
        if "forecasts/train_forecasts.parquet" not in logged:
            client.log_artifact(prev_id, str(path), artifact_path="forecasts")
    except MlflowException:  # run is in another tracking store
        pass


def run_pipeline(config_path: str | None = None, force: bool = False) -> dict:
    """Full pipeline: load → features → CV → retrain → save artifacts."""
    config = load_config(config_path)
    root = Path(config["_project_root"])

    mlflow_cfg = config["forecasting"]["mlflow"]
    experiment_name = tracking.setup_mlflow(config, "forecasting")

    hash_payload = {
        "forecasting": {
            k: v
            for k, v in config["forecasting"].items()
            if k not in ("mlflow", "artifacts")
        },
        "periods": config.get("periods"),
        "data": {
            k: config.get("data", {}).get(k)
            for k in ("sales_file", "initial_state_date")
        },
    }
    config_hash = tracking.config_hash(hash_payload)
    if mlflow_cfg.get("skip_duplicate_runs", True):
        tracking.check_duplicate_run(experiment_name, config_hash, force=force)

    periods = resolve_periods(config)
    if periods.errors:
        raise ValueError("; ".join(periods.errors))

    panel = load_panel_from_config(
        config, start=periods.data_start, end=periods.data_end
    )
    df = build_feature_matrix(panel, config)
    df_fc = df[df["date"] <= periods.forecaster_cv_end].copy()

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
            "gap_weeks": periods.gap_weeks,
            "label_gap_weeks": periods.label_gap_weeks,
            "min_train_size_weeks": val_cfg["min_train_size_weeks"],
            "horizons": str(horizons),
            "n_features": len(get_feature_columns(df)),
            "n_skus": df.groupby(["store", "product"]).ngroups,
            "n_weeks": df["date"].nunique(),
            "oos_masking": True,
            "rl_train_start": str(periods.rl_train_start.date()),
            "rl_train_end": str(periods.rl_train_end.date()),
            "rl_val_start": str(periods.rl_val_start.date()),
            "rl_val_end": str(periods.rl_val_end.date()),
            "rl_test_start": str(periods.rl_test_start.date()),
            "rl_test_end": str(periods.rl_test_end.date()),
            "refit_per_block": periods.refit_per_block,
        }
        if forecaster_type == "quantile":
            run_params["quantiles"] = str(config["forecasting"]["model"]["quantiles"])
            ltd_cfg = config["forecasting"]["model"].get("lead_time_demand", {})
            if ltd_cfg.get("enabled", False):
                run_params["ltd_enabled"] = True
                run_params["ltd_offsets"] = str(ltd_cfg.get("offsets", [0, 1, 2]))
                run_params["ltd_quantiles"] = str(ltd_cfg.get("quantiles", []))
        mlflow.log_params(run_params)

        # --- Step 1: model selection (optional) ---
        champion_model_name = "lightgbm"
        champion_model_params: dict | None = None

        if sel_cfg.get("enabled", False):
            candidates = _run_model_selection(df_fc, config)

            # --- Step 2: hypertune survivors (optional) ---
            if tune_cfg.get("enabled", False):
                champion = _run_tuning(candidates, df_fc, config)
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
            result = tune_model(default_model, df_fc, config)
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
            df_fc,
            config,
            model_name=champion_model_name,
            model_params=champion_model_params,
            folds=periods.folds,
        )

        for k, v in cv_metrics.items():
            mlflow.log_metric(k, v)

        _log_fold_metrics(fold_metrics)
        _log_cv_chart(fold_metrics, horizons)
        _log_oof_sample_chart(oof)

        if fold_metrics:
            tracking.log_dataframe(pd.DataFrame(fold_metrics), artifact_path="cv_detail")

        if not oof.empty:
            log_cv_entity_metrics(oof, config)

        # --- Step 4: out-of-sample forecasts per RL block ---
        full_scored, block_metrics, forecaster = _fit_rl_block_forecasts(
            df,
            config,
            periods,
            ForecasterClass,
            champion_model_name,
            champion_model_params,
        )
        for k, v in block_metrics.items():
            mlflow.log_metric(k, v)
        for w in periods.warnings:
            mlflow.set_tag(f"warning_{len(w)}", w[:250])

        if forecaster is not None:
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
        if forecaster is not None:
            forecaster.save(model_path)
            mlflow.log_artifact(str(model_path))

        forecasts_dir = root / config["forecasting"]["artifacts"]["forecasts_dir"]
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        forecast_cols = (
            ["store", "product", "date", "sales"]
            + [c for c in full_scored.columns if c.startswith(("mu_h", "sigma_h", "pred_q"))]
        )
        if not full_scored.empty:
            _keep_previous_forecasts(config)
            fc_path = forecasts_dir / "train_forecasts.parquet"
            full_scored[forecast_cols].to_parquet(fc_path, index=False)
            mlflow.log_artifact(str(fc_path), artifact_path="forecasts")
        run = mlflow.active_run()
        run_id = run.info.run_id if run is not None else None
        record = build_setup_record(config, periods, run_id=run_id)
        setup_path = save_setup(config, record)
        mlflow.log_artifact(str(setup_path), artifact_path="forecasts")
        if not oof.empty:
            oof_cols = [c for c in forecast_cols if c in oof.columns] + ["fold_id"]
            oof_export = oof[[c for c in oof_cols if c in oof.columns]]
            oof_path = forecasts_dir / "oof_forecasts.parquet"
            oof_export.to_parquet(oof_path, index=False)
            mlflow.log_artifact(str(oof_path), artifact_path="forecasts")

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
