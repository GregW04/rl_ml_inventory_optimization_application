"""End-to-end demand forecasting pipeline with rich MLflow tracking."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.forecasting.calibration import fit_quantile_shifts
from src.forecasting.entity_metrics import log_cv_entity_metrics
from src.forecasting.forecaster import (
    ENSEMBLE,
    EnsembleForecaster,
    MeasureForecaster,
    ensemble_settings,
    evaluate_measures,
    make_forecaster,
)
from src.forecasting.measures import (
    Measure,
    manifest_record,
    parse_measures,
    primary_measure,
    primary_metric_name,
    validate_measures,
)
from src.forecasting.model import build_feature_matrix, get_feature_columns
from src.forecasting.periods import (
    build_setup_record,
    current_setup_hash,
    resolve_periods,
    save_setup,
)
from src.forecasting.preprocessing import load_panel_from_config
from src.forecasting.validation import TimeSeriesFold, get_unique_dates, iter_fold_masks, make_folds
from src.utils import tracking
from src.utils.config import load_config


def run_cross_validation(
    df: pd.DataFrame,
    config: dict,
    measures: list[Measure],
    model_name: str = "lightgbm",
    model_params: dict | None = None,
    folds: list[TimeSeriesFold] | None = None,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """Run expanding-window CV; return OOF predictions, summary metrics, and per-fold details."""
    if folds is None:
        folds = make_folds(get_unique_dates(df), config)

    feature_cols = get_feature_columns(df)
    oof_parts = []
    fold_metrics = []

    for fold, train_mask, val_mask in iter_fold_masks(df, folds):
        forecaster = make_forecaster(config, model_name, model_params, measures)
        forecaster.fit(df.loc[train_mask], feature_cols=feature_cols)
        scored = forecaster.predict(df.loc[val_mask], feature_cols=feature_cols)
        scored["fold_id"] = fold.fold_id
        oof_parts.append(scored)
        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "train_end": str(fold.train_end.date()),
                "val_start": str(fold.val_start.date()),
                "val_end": str(fold.val_end.date()),
                **forecaster.evaluate(scored),
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    summary: dict[str, float] = {}
    if fold_metrics:
        mdf = pd.DataFrame(fold_metrics)
        for col in mdf.columns:
            if col in ("fold_id", "train_end", "val_start", "val_end"):
                continue
            val = float(mdf[col].mean(skipna=True))
            if np.isfinite(val):
                summary[f"cv_{col}"] = val
    return oof, summary, fold_metrics


def _log_fold_metrics(fold_metrics: list[dict]) -> None:
    """Log per-fold, per-measure metrics as MLflow step metrics (step = fold)."""
    for row in fold_metrics:
        fold = int(row.get("fold_id", 0))
        for key, val in row.items():
            if key in ("fold_id", "train_end", "val_start", "val_end"):
                continue
            if val is not None and np.isfinite(float(val)):
                mlflow.log_metric(key, float(val), step=fold)


def _importances(forecaster: MeasureForecaster, measure: Measure) -> pd.Series | None:
    """Gain importance of the booster behind ``measure``, indexed by feature."""
    model = forecaster.models.get(measure.id)
    if model is None:
        return None
    try:
        if forecaster.model_name == "lightgbm":
            values = model.booster_.feature_importance(importance_type="gain")
        else:
            values = model.feature_importances_
    except AttributeError:
        return None
    if len(values) != len(forecaster.feature_cols):
        return None
    return pd.Series(np.asarray(values, dtype=float), index=forecaster.feature_cols)


def _log_feature_importances(forecaster, measure: Measure, top_n: int = 30) -> None:
    """Log feature importance of the primary measure (ensembles: mean of normalized gains)."""
    if isinstance(forecaster, EnsembleForecaster):
        parts = [imp / imp.sum() for m in forecaster.members.values()
                 if (imp := _importances(m, measure)) is not None and imp.sum() > 0]
        importance = pd.concat(parts, axis=1).mean(axis=1) if parts else None
        label = f"{measure.id} model (ensemble mean of normalized gain)"
    else:
        importance = _importances(forecaster, measure)
        label = f"{measure.id} model"
    if importance is None:
        return

    imp_df = (
        importance.rename_axis("feature").reset_index(name="importance")
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    tracking.log_dataframe(imp_df, "feature_importance", "feature_importance.csv")

    fig, ax = plt.subplots(figsize=(8, max(4, top_n // 3)))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_xlabel("Gain (importance)")
    ax.set_title(f"Top feature importances — {label}")
    tracking.log_figure(fig, "feature_importance", "feature_importance.png")


def _log_cv_chart(fold_metrics: list[dict], metric: str) -> None:
    """Log the primary metric per CV fold."""
    mdf = pd.DataFrame(fold_metrics)
    if mdf.empty or metric not in mdf.columns:
        return
    mdf = mdf.sort_values("fold_id")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mdf["fold_id"], mdf[metric], marker="o")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric)
    ax.set_title(f"CV {metric} per fold")
    tracking.log_figure(fig, "cv_charts", f"cv_{metric}_per_fold.png")


def _log_oof_sample_chart(oof: pd.DataFrame, measures: list[Measure]) -> None:
    """Log an OOF actual vs forecast chart (h0 measures) for the SKU with most rows."""
    h0 = [m for m in measures if m.horizon == 0 and m.kind in ("mean", "quantile")]
    if oof.empty or not h0 or h0[0].id not in oof.columns:
        return
    key = oof.dropna(subset=[h0[0].id]).groupby(["store", "product"]).size().idxmax()
    sku = oof[(oof["store"] == key[0]) & (oof["product"] == key[1])].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sku["date"], sku["sales"], label="Actual", color="steelblue")
    for m in h0:
        ax.plot(sku["date"], sku[m.id], label=m.id, linestyle="--")
    ax.set_title(f"OOF forecast — store={key[0]}, product={key[1]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    tracking.log_figure(fig, "cv_charts", "oof_sample.png")


def _choose_model(
    df_fc: pd.DataFrame,
    config: dict,
    primary: Measure,
) -> tuple[str, dict | None]:
    """Model selection and/or tuning on the primary measure; return (backend, params)."""
    sel_cfg = config["forecasting"].get("model_selection", {})
    tune_cfg = config["forecasting"].get("tuning", {})
    metric = primary_metric_name(primary)
    name = config["forecasting"]["model"].get("name", "lightgbm")

    ens_on, members = ensemble_settings(config)
    if ens_on:
        # The ensemble replaces model selection: every member is kept and averaged.
        if sel_cfg.get("enabled", False):
            mlflow.set_tag("note_model_selection", "skipped: the ensemble averages all members")
        params: dict[str, dict | None] = {m: None for m in members}
        if tune_cfg.get("enabled", False):
            from src.forecasting.tuning import tune_model

            for member in members:
                result = tune_model(member, df_fc, config)
                mlflow.log_metric(f"tuned_{metric}_{member}", result["best_value"])
                mlflow.log_params({f"best_{member}_{k}": v for k, v in result["best_params"].items()})
                params[member] = result["best_params"]
        return ENSEMBLE, params

    if sel_cfg.get("enabled", False):
        from src.forecasting.model_selection import select_candidates

        candidates = select_candidates(df_fc, config)
        mlflow.log_param("selection_candidates", str([n for n, _ in candidates]))
        for n, score in candidates:
            mlflow.log_metric(f"selection_{metric}_{n}", score)
        if not tune_cfg.get("enabled", False):
            return candidates[0][0], None
        from src.forecasting.tuning import tune_candidates

        results = tune_candidates(candidates, df_fc, config)
        for r in results:
            n = r["model_name"]
            mlflow.log_metric(f"tuned_{metric}_{n}", r["best_value"])
            mlflow.log_metric(f"tuning_trials_{n}", r["n_trials_completed"])
            mlflow.log_params({f"best_{n}_{k}": v for k, v in r["best_params"].items()})
        mlflow.log_metric(f"champion_tuned_{metric}", results[0]["best_value"])
        return results[0]["model_name"], results[0]["best_params"]

    if tune_cfg.get("enabled", False):
        from src.forecasting.tuning import tune_model

        result = tune_model(name, df_fc, config)
        mlflow.log_metric(f"champion_tuned_{metric}", result["best_value"])
        mlflow.log_params({f"best_{k}": v for k, v in result["best_params"].items()})
        return name, result["best_params"]
    return name, None


def _fit_rl_block_forecasts(
    df: pd.DataFrame,
    config: dict,
    periods,
    measures: list[Measure],
    model_name: str,
    model_params: dict | None,
    calibration: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float], MeasureForecaster | EnsembleForecaster | None]:
    """
    Out-of-sample forecasts for the RL blocks, one forecaster fit per ``periods.block_fits``.

    Each fit uses rows up to its ``fit_end`` and writes forecasts for its own weeks only.
    The training and validation blocks get a fresh fit every ``refit_every_weeks``; the
    test block keeps one fit. Scores per block pool that block's fits.
    """
    feature_cols = get_feature_columns(df)
    parts: dict[str, list[pd.DataFrame]] = {}
    last_forecaster = None
    for fit in periods.block_fits:
        pred_df = df[(df["date"] >= fit.start) & (df["date"] <= fit.end)]
        if pred_df.empty:
            continue
        forecaster = make_forecaster(config, model_name, model_params, measures)
        forecaster.fit(df[df["date"] <= fit.fit_end], feature_cols=feature_cols)
        forecaster.calibration = dict(calibration)
        scored = forecaster.predict(pred_df, feature_cols=feature_cols)
        scored["fit_end"] = fit.fit_end
        parts.setdefault(fit.block, []).append(scored)
        last_forecaster = forecaster
        print(
            f"[block fit] {fit.block} {fit.index}: data to {fit.fit_end.date()}, "
            f"forecasts {fit.start.date()}..{fit.end.date()}",
            flush=True,
        )

    block_metrics: dict[str, float] = {}
    for block, block_parts in parts.items():
        scored = pd.concat(block_parts, ignore_index=True)
        for k, v in evaluate_measures(scored, measures).items():
            block_metrics[f"oos_{block}_{k}"] = v

    if not parts:
        return pd.DataFrame(), block_metrics, last_forecaster
    combined = pd.concat([p for block_parts in parts.values() for p in block_parts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["store", "product", "date"], keep="last")
    return combined, block_metrics, last_forecaster


def run_pipeline(config_path: str | None = None, force: bool = False) -> dict:
    """Full pipeline: load → features → model choice → CV → RL-block forecasts → artifacts."""
    config = load_config(config_path)

    measures = parse_measures(config)
    primary = primary_measure(config, measures)
    errors = validate_measures(measures, primary.id)
    if errors:
        raise ValueError("; ".join(errors))
    metric = primary_metric_name(primary)

    experiment_name = tracking.setup_mlflow(config, "forecasting")
    config_hash = current_setup_hash(config)
    if config["forecasting"].get("skip_duplicate_runs", True):
        tracking.check_duplicate_run(experiment_name, config_hash, force=force)

    periods = resolve_periods(config)
    if periods.errors:
        raise ValueError("; ".join(periods.errors))

    panel = load_panel_from_config(config, start=periods.data_start, end=periods.data_end)
    df = build_feature_matrix(panel, config)
    df_fc = df[df["date"] <= periods.forecaster_cv_end].copy()
    val_cfg = config["forecasting"]["validation"]
    model_cfg = config["forecasting"]["model"]
    calibrate = bool(config["forecasting"].get("calibrate_quantiles", False))

    with mlflow.start_run(run_name="demand_forecast_pipeline") as run:
        mlflow.set_tag("config_hash", config_hash)
        tracking.log_provenance(config)
        mlflow.log_params({
            "measures": ",".join(m.id for m in measures),
            "primary_measure": primary.id,
            "primary_metric": metric,
            "calibrate_quantiles": calibrate,
            "target_transform": model_cfg.get("target_transform", "none"),
            "mean_objective": model_cfg.get("mean_objective", "poisson"),
            "tweedie_variance_power": model_cfg.get("tweedie_variance_power", 1.5),
            "demand_profile": bool(
                config["forecasting"].get("features", {}).get("demand_profile", {}).get("enabled", False)
            ),
            "n_folds": val_cfg["n_folds"],
            "fold_size_weeks": val_cfg["fold_size_weeks"],
            "gap_weeks": periods.gap_weeks,
            "label_gap_weeks": periods.label_gap_weeks,
            "min_train_size_weeks": val_cfg["min_train_size_weeks"],
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
            "refit_every_weeks": periods.refit_every_weeks,
            "n_block_fits": len(periods.block_fits),
            "ensemble": ensemble_settings(config)[0],
            "ensemble_members": ",".join(ensemble_settings(config)[1]) if ensemble_settings(config)[0] else "",
        })

        # --- Steps 1–2: model selection / tuning on the primary measure (optional) ---
        model_name, model_params = _choose_model(df_fc, config, primary)
        mlflow.log_param("champion_model", model_name)

        # --- Step 3: full CV with the champion, every measure ---
        oof, cv_metrics, fold_metrics = run_cross_validation(
            df_fc, config, measures, model_name=model_name, model_params=model_params,
            folds=periods.folds,
        )
        if f"cv_{metric}" in cv_metrics:
            cv_metrics["cv_primary"] = cv_metrics[f"cv_{metric}"]
        for k, v in cv_metrics.items():
            mlflow.log_metric(k, v)
        _log_fold_metrics(fold_metrics)
        _log_cv_chart(fold_metrics, metric)
        _log_oof_sample_chart(oof, measures)
        if fold_metrics:
            tracking.log_dataframe(pd.DataFrame(fold_metrics), "cv_detail", "fold_metrics.csv")
        if not oof.empty:
            log_cv_entity_metrics(oof, measures, primary)

        # Conformal shifts come from the CV region only, so RL blocks stay out of sample.
        calibration: dict[str, float] = {}
        if calibrate and not oof.empty:
            transform = config["forecasting"]["model"].get("target_transform", "none")
            calibration = fit_quantile_shifts(oof, measures, transform)
            for k, v in calibration.items():
                mlflow.log_metric(f"calibration_shift_{k}", v)

        # --- Step 4: out-of-sample forecasts per RL block ---
        full_scored, block_metrics, forecaster = _fit_rl_block_forecasts(
            df, config, periods, measures, model_name, model_params, calibration
        )
        for k, v in block_metrics.items():
            mlflow.log_metric(k, v)
        if periods.warnings:
            mlflow.log_dict({"warnings": list(periods.warnings)}, "warnings.json")
        if forecaster is not None:
            _log_feature_importances(forecaster, primary)

        # --- Save artifacts ---
        # Files go to the run's artifact cache and are logged from there, so RL and the app read
        # this run without downloading. The fitted model is logged only (nothing reads it locally).
        if forecaster is not None:
            with tempfile.TemporaryDirectory() as tmp:
                model_path = Path(tmp) / "forecaster.joblib"
                forecaster.save(model_path)
                mlflow.log_artifact(str(model_path))

        forecasts_dir = tracking.run_cache_dir(config, run.info.run_id) / "forecasts"
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        forecast_cols = ["store", "product", "date", "sales"] + [m.id for m in measures]
        if not full_scored.empty:
            # ``fit_end`` records which fit wrote each row, so forecast age can be audited.
            full_scored[forecast_cols + ["fit_end"]].to_parquet(forecasts_dir / "train_forecasts.parquet", index=False)
        record = build_setup_record(
            config, periods, run_id=run.info.run_id, extra=manifest_record(measures, primary, calibration)
        )
        save_setup(forecasts_dir / "forecast_setup.json", record)
        if not oof.empty:
            oof[forecast_cols + ["fold_id"]].to_parquet(forecasts_dir / "oof_forecasts.parquet", index=False)
        mlflow.log_artifacts(str(forecasts_dir), artifact_path="forecasts")

        mlflow.log_dict(cv_metrics, "cv_metrics.json")
        return {"cv_metrics": cv_metrics, "run_id": run.info.run_id}


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
