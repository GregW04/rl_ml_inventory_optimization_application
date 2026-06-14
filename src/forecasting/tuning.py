"""
Optuna-based hyperparameter tuning for demand forecasters.

Objective: mean primary metric (wQL or MAE) across horizons, averaged over
the ``n_cv_folds`` most-recent CV folds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.model import (
    get_feature_columns,
    get_forecaster_class,
    get_primary_metric,
)
from src.forecasting.registry import get_spec, sample_params
from src.forecasting.validation import generate_expanding_folds, get_unique_dates

logger = logging.getLogger(__name__)


class _EarlyStoppingCallback:
    """Stop the Optuna study when the best value plateaus."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_value: float | None = None
        self._trials_since_improvement = 0

    def __call__(self, study: Any, trial: Any) -> None:
        if study.best_trial is None:
            return

        current_best = study.best_value
        if self._best_value is None:
            self._best_value = current_best
            return

        if current_best < self._best_value - self.min_delta:
            self._best_value = current_best
            self._trials_since_improvement = 0
        else:
            self._trials_since_improvement += 1

        if self._trials_since_improvement >= self.patience:
            logger.info(
                "Early stopping %s — no improvement >= %.6f for %d trials "
                "(best=%.4f)",
                study.study_name,
                self.min_delta,
                self.patience,
                study.best_value,
            )
            study.stop()


def _build_objective(
    model_name: str,
    fold_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    config: dict,
    feature_cols: list[str],
):
    """Return an Optuna objective minimizing mean primary metric over folds."""
    ForecasterClass = get_forecaster_class(config)
    primary = get_primary_metric(config)
    horizons = config["forecasting"]["model"]["horizons"]

    def objective(trial: Any) -> float:
        params = sample_params(trial, model_name)
        fold_scores: list[float] = []

        for train_df, val_df in fold_splits:
            try:
                forecaster = ForecasterClass(
                    config, model_name=model_name, model_params=params
                )
                forecaster.fit(train_df, feature_cols=feature_cols)
                scored = forecaster.predict(val_df, feature_cols=feature_cols)
            except Exception as exc:
                logger.debug("Trial raised during fit/predict: %s", exc)
                return float("inf")

            per_horizon: list[float] = []
            for h in horizons:
                m = forecaster.evaluate_horizon(scored, horizon=h)
                if primary in m and np.isfinite(m[primary]):
                    per_horizon.append(m[primary])

            if per_horizon:
                fold_scores.append(float(np.mean(per_horizon)))

        return float(np.mean(fold_scores)) if fold_scores else float("inf")

    return objective


def tune_model(
    model_name: str,
    df: pd.DataFrame,
    config: dict,
    n_trials: int = 50,
    seed: int | None = 42,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run an Optuna TPE study for ``model_name``."""
    import optuna

    primary = get_primary_metric(config)
    tune_cfg = config.get("forecasting", {}).get("tuning", {})
    n_trials = tune_cfg.get("n_trials", n_trials)
    seed = tune_cfg.get("seed", seed)
    timeout = tune_cfg.get("timeout_seconds", timeout)
    n_cv_folds_tune = tune_cfg.get("n_cv_folds", 1)
    n_trials_no_improve = tune_cfg.get("n_trials_no_improve")
    early_stopping_min_delta = tune_cfg.get("early_stopping_min_delta", 0.0001)

    val_cfg = config["forecasting"]["validation"]
    dates = get_unique_dates(df)
    folds = generate_expanding_folds(
        dates,
        n_folds=val_cfg["n_folds"],
        fold_size_weeks=val_cfg["fold_size_weeks"],
        embargo_weeks=val_cfg["embargo_weeks"],
        min_train_size_weeks=val_cfg["min_train_size_weeks"],
    )
    if not folds:
        raise RuntimeError("No CV folds generated; cannot run hyperparameter tuning.")

    tune_folds = folds[-n_cv_folds_tune:]
    feature_cols = get_feature_columns(df)

    fold_splits = [
        (
            df[df["date"].isin(f.train_dates)].copy(),
            df[df["date"].isin(f.val_dates)].copy(),
        )
        for f in tune_folds
    ]

    objective = _build_objective(model_name, fold_splits, config, feature_cols)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=f"tune_{model_name}",
    )

    callbacks: list[Any] = []
    if n_trials_no_improve is not None:
        callbacks.append(
            _EarlyStoppingCallback(
                patience=int(n_trials_no_improve),
                min_delta=float(early_stopping_min_delta),
            )
        )
        logger.info(
            "Early stopping enabled — patience=%d, min_delta=%.6f",
            n_trials_no_improve,
            early_stopping_min_delta,
        )

    logger.info(
        "Tuning %-12s — %d trials, %d fold(s), metric=%s ...",
        model_name,
        n_trials,
        n_cv_folds_tune,
        primary,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        callbacks=callbacks or None,
    )

    best_tuned = study.best_params
    logger.info(
        "  Best params: %s  (%s=%.4f)",
        best_tuned,
        primary,
        study.best_value,
    )

    spec = get_spec(model_name)
    fixed = {k: v for k, v in spec.default_params.items() if k not in spec.search_space}
    best_params = {**fixed, **best_tuned}

    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "n_trials_completed": len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        ),
    }


def tune_candidates(
    candidates: list[tuple[str, float]],
    df: pd.DataFrame,
    config: dict,
) -> list[dict[str, Any]]:
    """Tune each candidate; return results sorted by best post-tuning metric."""
    primary = get_primary_metric(config)
    results = []
    for model_name, default_score in candidates:
        logger.info(
            "Hypertuning %s (default %s=%.4f)...",
            model_name,
            primary,
            default_score,
        )
        result = tune_model(model_name, df, config)
        result["model_name"] = model_name
        result["default_score"] = default_score
        results.append(result)

    results.sort(key=lambda r: r["best_value"])
    return results
