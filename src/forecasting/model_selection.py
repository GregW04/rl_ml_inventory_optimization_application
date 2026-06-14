"""
Model selection: score candidate models on default params using the last CV fold.

Design rationale
----------------
- Only the **last (most recent) fold** is used so selection is fast.
- Models >``elimination_threshold`` worse than the best are dropped.
- The top ``n_candidates_to_tune`` survivors are passed to hypertuning.
- We do *not* cut to top-1 here — even a model 15 % worse on defaults
  can win after tuning (XGBoost defaults are notoriously conservative).
- Primary metric: wQL for ``forecaster_type: quantile``, MAE for ``mean``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.forecasting.model import get_feature_columns, get_forecaster_class, get_primary_metric
from src.forecasting.registry import available_candidates
from src.forecasting.validation import (
    generate_expanding_folds,
    get_unique_dates,
)

logger = logging.getLogger(__name__)


def _score_candidate(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    feature_cols: list[str],
) -> float:
    """Train candidate on one fold; return mean primary metric across horizons."""
    ForecasterClass = get_forecaster_class(config)
    primary = get_primary_metric(config)

    forecaster = ForecasterClass(config, model_name=model_name)
    forecaster.fit(train_df, feature_cols=feature_cols)
    scored = forecaster.predict(val_df, feature_cols=feature_cols)

    scores = []
    for h in config["forecasting"]["model"]["horizons"]:
        m = forecaster.evaluate_horizon(scored, horizon=h)
        if primary in m and np.isfinite(m[primary]):
            scores.append(m[primary])

    return float(np.mean(scores)) if scores else np.nan


def select_candidates(
    df: pd.DataFrame,
    config: dict,
    candidates: list[str] | None = None,
    n_keep: int = 2,
    elimination_threshold: float = 0.40,
) -> list[tuple[str, float]]:
    """
    Score candidates on default params and return the survivors for tuning.

    Returns list of ``(model_name, score)`` sorted best-first (lower is better).
    """
    sel_cfg = config.get("forecasting", {}).get("model_selection", {})
    if candidates is None:
        candidates = sel_cfg.get(
            "candidates", ["lightgbm", "xgboost", "catboost"]
        )
    n_keep = sel_cfg.get("n_candidates_to_tune", n_keep)
    elimination_threshold = sel_cfg.get(
        "elimination_threshold", elimination_threshold
    )
    primary = get_primary_metric(config)

    importable = available_candidates(candidates)
    if not importable:
        raise RuntimeError(
            "None of the requested model candidates could be imported: "
            f"{candidates}"
        )

    logger.info("Model selection — evaluating: %s (metric=%s)", importable, primary)

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
        raise RuntimeError("No CV folds generated; cannot run model selection.")

    last_fold = folds[-1]
    feature_cols = get_feature_columns(df)
    train_df = df[df["date"].isin(last_fold.train_dates)].copy()
    val_df = df[df["date"].isin(last_fold.val_dates)].copy()

    scores: dict[str, float] = {}
    for name in importable:
        try:
            logger.info("  Scoring %-12s (default params) ...", name)
            score = _score_candidate(name, train_df, val_df, config, feature_cols)
            scores[name] = score
            logger.info("    → %s = %.4f", primary, score)
        except Exception as exc:
            logger.warning("  %s failed during selection: %s", name, exc)
            scores[name] = np.nan

    ranked = sorted(
        [(n, s) for n, s in scores.items() if np.isfinite(s)],
        key=lambda x: x[1],
    )
    if not ranked:
        raise RuntimeError(
            "All candidates failed during model selection. "
            "Check logs for individual errors."
        )

    best_score = ranked[0][1]
    cutoff = best_score * (1.0 + elimination_threshold)
    survivors = [(n, s) for n, s in ranked if s <= cutoff][:n_keep]

    logger.info(
        "Model selection result (threshold=%.0f%%): %s",
        elimination_threshold * 100,
        [(n, f"{s:.4f}") for n, s in survivors],
    )
    return survivors
