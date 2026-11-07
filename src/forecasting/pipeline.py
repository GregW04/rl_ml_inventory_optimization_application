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
from src.forecasting.drift_refits import Retune, WalkState, walk_block
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
    MAX_BLOCK_FITS_WARNING,
    BlockFit,
    build_setup_record,
    current_setup_hash,
    periods_to_dict,
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


def _within_block(scored: pd.DataFrame, block_end: pd.Timestamp, horizons: list[int]) -> pd.DataFrame:
    """A copy whose ``target_h{h}`` is NaN where week + h falls after the block's last week."""
    out = scored.copy()
    for h in horizons:
        col = f"target_h{h}"
        if col in out.columns:
            out.loc[out["date"] + pd.Timedelta(weeks=h) > block_end, col] = np.nan
    return out


def _log_drift_alerts(config: dict, panel: pd.DataFrame, forecasts: pd.DataFrame, periods,
                      measures: list[Measure], forecasts_dir: Path) -> None:
    """The production bias alarm on the run's fits: an artifact, alert weeks per block, never an action."""
    from src.forecasting import drift_alerts
    from src.forecasting.drift import h0_level_measure

    level = h0_level_measure(measures)
    rules = drift_alerts.default_rules()
    if level is None or forecasts.empty:
        mlflow.set_tag("drift_monitoring", "false")
        return
    alerts = drift_alerts.fit_alerts(forecasts, drift_alerts.class_actuals(panel, config),
                                     periods_to_dict(periods), level.id, rules)
    alerts.to_parquet(forecasts_dir / "drift_alerts.parquet", index=False)
    mlflow.set_tag("drift_monitoring", "true")
    mlflow.log_params({"drift_rule": drift_alerts.describe(rules), "drift_level_measure": level.id})
    # The test block's count is left out of the metrics, like every test number before scoring.
    for block in ("rl_train", "rl_val"):
        mlflow.log_metric(f"drift_alert_weeks_{block}", drift_alerts.alert_weeks(alerts, block))


def _tune_all(df: pd.DataFrame, config: dict, model_name: str) -> dict | None:
    """Tune again on ``df``: the single backend's parameters, or each ensemble member's."""
    from src.forecasting.tuning import tune_model

    if model_name == ENSEMBLE:
        _, members = ensemble_settings(config)
        return {m: tune_model(m, df, config)["best_params"] for m in members}
    return tune_model(model_name, df, config)["best_params"]


def _retune_challenge(df, config, model_name, measures, primary, feature_cols, calibration, gap_weeks):
    """
    The retune step of a drift refit. The last ``monitoring.retune_holdout_weeks`` weeks known that week are a
    holdout; the rows before them (with the label gap) are the training rows. Tuning runs on the training rows
    only, with the run's own CV settings (``tuning_folds``, counted back from where those rows end), so the
    holdout stays unseen by the tuning. If the run's folds do not all fit in those rows the retune is skipped and
    says so, instead of tuning on fewer folds. Then the current and the new settings are fitted on the training
    rows and compared on the holdout by the primary measure. The new settings are adopted only if they are at
    least ``retune_min_gain_pct`` better.
    """
    from src.forecasting.tuning import expected_tuning_folds, tuning_folds

    mon = config.get("monitoring") or {}
    min_gain = float(mon.get("retune_min_gain_pct", 1.0))
    holdout_weeks = max(int(mon.get("retune_holdout_weeks", 8)), 1)
    metric = primary_metric_name(primary)
    week = pd.Timedelta(weeks=1)

    def retune(alert_week, fit_end, classes, params):
        val_start = fit_end - (holdout_weeks - 1) * week
        train = df[df["date"] <= val_start - (gap_weeks + 1) * week]
        val = df[(df["date"] >= val_start) & (df["date"] <= fit_end)]
        shown_classes = ",".join(classes)
        try:
            folds = tuning_folds(train, config) if len(train) else []
        except RuntimeError:  # not even one fold fits
            folds = []
        rows = dict(
            tuned_to=str(train["date"].max().date()) if len(train) else "", holdout_start=str(val_start.date()),
            holdout_end=str(fit_end.date()), folds=len(folds), fold_weeks=len(folds[-1].val_dates) if folds else 0)
        wanted = expected_tuning_folds(config)
        if len(folds) < wanted:
            note = f"only {len(folds)} of the run's {wanted} CV folds fit in the rows before the holdout"
            print(f"[retune] {alert_week.date()} {shown_classes}: skipped, {note}", flush=True)
            return Retune(alert_week, "", list(classes), None, None, None, False, **rows, skipped=True, note=note), params
        new_params = _tune_all(train, config, model_name)
        scores = {}
        for name, p in (("current", params), ("new", new_params)):
            forecaster = make_forecaster(config, model_name, p, [primary])
            forecaster.fit(train, feature_cols=feature_cols)
            forecaster.calibration = dict(calibration)
            scores[name] = evaluate_measures(forecaster.predict(val, feature_cols=feature_cols), [primary]).get(metric)
        cur, new = scores["current"], scores["new"]
        gain = 100.0 * (cur - new) / cur if cur and np.isfinite(cur) and new is not None and np.isfinite(new) else float("nan")
        outcome = Retune(alert_week, "", list(classes), cur, new, gain, bool(gain >= min_gain), **rows)
        shown = [f"{v:.4f}" if v is not None and np.isfinite(v) else "n/a" for v in (cur, new)]
        print(f"[retune] {alert_week.date()} {shown_classes}: {metric} {shown[0]} -> {shown[1]} ({gain:+.1f}%) on "
              f"{rows['holdout_start']}..{rows['holdout_end']} after tuning on {len(folds)} folds to {rows['tuned_to']}, "
              f"{'adopted' if outcome.adopted else 'kept the current settings'}", flush=True)
        return outcome, (new_params if outcome.adopted else params)

    return retune


def _fit_rl_block_forecasts(
    df: pd.DataFrame,
    config: dict,
    periods,
    measures: list[Measure],
    model_name: str,
    model_params: dict | None,
    calibration: dict[str, float],
    panel: pd.DataFrame | None = None,
    primary: Measure | None = None,
) -> tuple[pd.DataFrame, dict[str, float], MeasureForecaster | EnsembleForecaster | None, list[BlockFit], list[Retune]]:
    """
    Out-of-sample forecasts for the RL blocks, one forecaster fit per ``periods.block_fits``.

    Each fit uses rows up to its ``fit_end`` and writes forecasts for its own weeks only.
    The training and validation blocks get a fresh fit every ``refit_every_weeks``; the
    test block keeps one fit. With ``monitoring.drift_refits`` the training and validation fits are
    decided as the blocks are walked, so an alert adds a refit (src/forecasting/drift_refits.py);
    the fits actually made, and the retunes, are returned. Scores per block pool that block's fits,
    and count only targets inside the block: a validation week's h2 target can fall in the test block.
    """
    feature_cols = get_feature_columns(df)
    horizons = sorted({m.horizon for m in measures})
    parts: dict[str, list[pd.DataFrame]] = {}
    last_forecaster = None

    def fit_predict(fit_end, start, end, params):
        nonlocal last_forecaster
        print(f"[fit] data to {fit_end.date()}, forecasts from {start.date()}", flush=True)
        forecaster = make_forecaster(config, model_name, params, measures)
        forecaster.fit(df[df["date"] <= fit_end], feature_cols=feature_cols)
        forecaster.calibration = dict(calibration)
        last_forecaster = forecaster
        return forecaster.predict(df[(df["date"] >= start) & (df["date"] <= end)], feature_cols=feature_cols)

    from src.forecasting import drift_alerts
    from src.forecasting.drift import h0_level_measure

    mon = config.get("monitoring") or {}
    level = h0_level_measure(measures)
    walk = bool(mon.get("drift_refits")) and panel is not None and level is not None and primary is not None
    state = WalkState(params=model_params)
    fits: list[BlockFit] = []
    if walk:
        actuals, rules = drift_alerts.class_actuals(panel, config), drift_alerts.default_rules()
        tuning_on = bool(config["forecasting"].get("tuning", {}).get("enabled", False))
        retune = (_retune_challenge(df, config, model_name, measures, primary, feature_cols, calibration, periods.gap_weeks)
                  if mon.get("retune_on_return", True) and tuning_on else None)
        for block in ("rl_train", "rl_val"):
            static = [f for f in periods.block_fits if f.block == block]
            if not static:
                continue
            block_fits, block_parts = walk_block(
                block, static[0].start, static[-1].end, periods.gap_weeks, periods.refit_every_weeks, fit_predict,
                lambda chunk: drift_alerts.first_alert(chunk, actuals, level.id, rules), state, retune,
                int(mon.get("max_retunes", 1)), MAX_BLOCK_FITS_WARNING)
            fits += block_fits
            parts[block] = block_parts
    for fit in periods.block_fits:
        if walk and fit.block != "rl_test":
            continue
        scored = fit_predict(fit.fit_end, fit.start, fit.end, state.params)
        if scored.empty:
            continue
        parts.setdefault(fit.block, []).append(scored.assign(fit_end=fit.fit_end))
        fits.append(fit)
    for f in fits:
        print(f"[block fit] {f.block} {f.index}: data to {f.fit_end.date()}, forecasts {f.start.date()}..{f.end.date()} "
              f"({f.trigger}{' ' + f.classes if f.classes else ''}{', retuned' if f.retuned else ''})", flush=True)

    block_metrics: dict[str, float] = {}
    for block, block_parts in parts.items():
        scored = _within_block(pd.concat(block_parts, ignore_index=True), getattr(periods, f"{block}_end"), horizons)
        for k, v in evaluate_measures(scored, measures).items():
            block_metrics[f"oos_{block}_{k}"] = v

    if not parts:
        return pd.DataFrame(), block_metrics, last_forecaster, fits, state.retunes
    combined = pd.concat([p for block_parts in parts.values() for p in block_parts], ignore_index=True)
    combined = combined.drop_duplicates(subset=["store", "product", "date"], keep="last")
    return combined, block_metrics, last_forecaster, fits, state.retunes


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
    monitoring = config.get("monitoring") or {}

    with mlflow.start_run(run_name="demand_forecast_pipeline") as run:
        mlflow.set_tag("config_hash", config_hash)
        # Block scores count only targets inside their block (runs before this tag let the last
        # validation weeks' h1/h2 targets reach into the test block).
        mlflow.set_tag("scores_within_block", "true")
        mlflow.set_tag("drift_refits", str(bool(monitoring.get("drift_refits", False))).lower())
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
        full_scored, block_metrics, forecaster, fits, retunes = _fit_rl_block_forecasts(
            df, config, periods, measures, model_name, model_params, calibration, panel=panel, primary=primary
        )
        periods.block_fits = fits  # what was fitted: the fixed schedule, plus any alert-driven refits
        for k, v in block_metrics.items():
            mlflow.log_metric(k, v)
        if monitoring.get("drift_refits"):
            mlflow.log_metrics({
                "n_block_fits_actual": len(fits),
                "n_drift_refits": sum(f.trigger == "drift" for f in fits),
                "n_retunes": sum(not r.skipped for r in retunes),
                "n_retunes_adopted": sum(r.adopted for r in retunes),
            })
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
        if monitoring.get("drift_alerts", False) or monitoring.get("drift_refits", False):
            _log_drift_alerts(config, panel, full_scored[forecast_cols] if not full_scored.empty else full_scored,
                              periods, measures, forecasts_dir)
        else:
            mlflow.set_tag("drift_monitoring", "false")
        if monitoring.get("drift_refits"):
            (forecasts_dir / "retunes.json").write_text(json.dumps([r.to_dict() for r in retunes], indent=2), encoding="utf-8")
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
