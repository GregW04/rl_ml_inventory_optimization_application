"""Feature columns and the per-backend fit/predict plumbing shared by forecasters."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Forecast output columns (``mean_h0``, ``q83_h2``) are never features.
_MEASURE_COL = re.compile(r"^(?:mean|q\d{1,2})_h\d$")


def transform_values(y, transform: str) -> np.ndarray:
    """Target transform shared by every forecaster (``log1p`` or none)."""
    if transform == "log1p":
        return np.log1p(np.clip(y, 0.0, None))
    return np.asarray(y, dtype=float)


def inverse_values(pred, transform: str) -> np.ndarray:
    """Inverse of :func:`transform_values`."""
    if transform == "log1p":
        # Guard against log-space extrapolation blowing up under expm1
        # (real log targets are <=~5 here; 9.2 → ~1e4 is a safe finite ceiling).
        return np.expm1(np.minimum(pred, 9.21))
    return pred


def get_feature_columns(
    df: pd.DataFrame,
    target_prefix: str = "target_h",
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Select numeric / categorical feature columns, excluding targets and ids metadata."""
    exclude = set(
        exclude_cols
        or [
            "sales",
            "in_stock",
            "date",
            "target_h0",
            "target_h1",
            "target_h2",
        ]
    )
    # hierarchy raw columns used only for grouping (group_* features capture signal)
    exclude.update(
        c
        for c in df.columns
        if c.startswith("grp_sales_") or c.startswith("share_")
    )
    exclude.update(c for c in df.columns if c.startswith("target_"))

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith(("pred_", "mu_", "sigma_", "_")) or _MEASURE_COL.match(col):
            continue
        if df[col].dtype in ("object", "datetime64[ns]"):
            continue
        feature_cols.append(col)
    return feature_cols


def get_categorical_features(
    df: pd.DataFrame,
    config: dict,
    feature_cols: list[str],
) -> list[str]:
    """Intersect config categorical list with available feature columns."""
    cfg_cats = config.get("forecasting", {}).get("model", {}).get(
        "categorical_features", []
    )
    return [c for c in cfg_cats if c in feature_cols and c in df.columns]


class _BaseForecaster:
    """
    Backend wiring for direct boosted forecasters: parameter resolution, per-backend
    fit/predict with categorical encoding, target transform, and serialization.
    Subclasses define which boosters they fit and what they predict.
    """

    def __init__(
        self,
        config: dict,
        model_name: str = "lightgbm",
        model_params: dict | None = None,
    ):
        from src.forecasting.estimators import get_spec

        self.config = config
        self.model_name = model_name
        model_cfg = config["forecasting"]["model"]
        # "log1p" fits quantiles in log space — quantiles transform exactly under
        # monotonic maps, so this sharpens the fit on skewed, spiky demand.
        self.target_transform = model_cfg.get("target_transform", "none")
        self.model_params: dict = (
            model_params if model_params is not None else get_spec(model_name).default_params.copy()
        )
        self.models: dict = {}
        self.feature_cols: list[str] = []
        self.categorical_features: list[str] = []
        # XGBoost: store category levels fitted on train to reuse at predict time
        self._cat_levels: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Backend-specific fit / predict wrappers
    # ------------------------------------------------------------------

    def _fit_estimator(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        fit_cat_levels: bool = False,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if self.model_name == "lightgbm":
            model.fit(
                X, y,
                categorical_feature=self.categorical_features or "auto",
                sample_weight=sample_weight,
            )
        elif self.model_name == "xgboost":
            X_enc = self._encode_xgb(X, fit=fit_cat_levels)
            model.fit(X_enc, y, sample_weight=sample_weight)
        elif self.model_name == "catboost":
            cat_idx = [
                list(X.columns).index(c)
                for c in self.categorical_features
                if c in X.columns
            ]
            model.fit(X, y, cat_features=cat_idx, sample_weight=sample_weight)
        else:
            model.fit(X, y, sample_weight=sample_weight)

    def _predict_estimator(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if self.model_name == "xgboost":
            X_enc = self._encode_xgb(X, fit=False)
            return model.predict(X_enc)
        return model.predict(X)

    def _encode_xgb(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Convert categorical columns to pd.Categorical for XGBoost."""
        if not self.categorical_features:
            return X
        X = X.copy()
        for col in self.categorical_features:
            if col not in X.columns:
                continue
            if fit:
                X[col] = pd.Categorical(X[col])
                self._cat_levels[col] = list(X[col].cat.categories)
            else:
                X[col] = pd.Categorical(
                    X[col], categories=self._cat_levels.get(col)
                )
        return X

    def _runtime_params(self) -> dict:
        """Model params plus per-booster thread cap (not stored as tuned hyperparameters)."""
        params = dict(self.model_params)
        threads = int(
            self.config.get("forecasting", {}).get("compute", {}).get("booster_threads", 0)
            or 0
        )
        if threads > 0:
            key = "thread_count" if self.model_name == "catboost" else "n_jobs"
            params.setdefault(key, threads)
        return params

    def _fit_n_jobs(self) -> int:
        n = int(self.config.get("forecasting", {}).get("compute", {}).get("n_jobs", 1) or 1)
        if n < 0:
            n = os.cpu_count() or 1
        return max(1, n)

    def _prepare_fit(
        self, df: pd.DataFrame, feature_cols: list[str] | None
    ) -> None:
        """Resolve feature/categorical columns shared by every ``fit`` call."""
        self.feature_cols = feature_cols or get_feature_columns(df)
        self.categorical_features = get_categorical_features(
            df, self.config, self.feature_cols
        )

    # ------------------------------------------------------------------
    # Target transform (shared) — fit in transformed space, predict back
    # ------------------------------------------------------------------

    def _transform_target(self, y: pd.Series | np.ndarray) -> np.ndarray:
        return transform_values(y, self.target_transform)

    def _inverse_target(self, pred: np.ndarray) -> np.ndarray:
        return inverse_values(pred, self.target_transform)

    # ------------------------------------------------------------------
    # Serialization (shared)
    # ------------------------------------------------------------------

    def _extra_save_payload(self) -> dict:
        """Hook for subclass-specific fields added to the saved payload."""
        return {}

    def _payload(self) -> dict:
        """Everything :meth:`load` needs to rebuild this forecaster."""
        return {
            "models": self.models,
            "feature_cols": self.feature_cols,
            "categorical_features": self.categorical_features,
            "config": self.config,
            "model_name": self.model_name,
            "model_params": self.model_params,
            "cat_levels": self._cat_levels,
            "extra": self._extra_save_payload(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._payload(), path)

    @classmethod
    def load(cls, path: str | Path) -> _BaseForecaster:
        payload = joblib.load(path)
        forecaster = cls(
            payload["config"],
            model_name=payload.get("model_name", "lightgbm"),
            model_params=payload.get("model_params"),
        )
        forecaster.models = payload["models"]
        forecaster.feature_cols = payload["feature_cols"]
        forecaster.categorical_features = payload["categorical_features"]
        forecaster._cat_levels = payload.get("cat_levels", {})
        for key, val in payload.get("extra", {}).items():
            setattr(forecaster, key, val)
        forecaster._post_load()
        return forecaster

    def _post_load(self) -> None:
        """Hook to rebuild subclass fields from their saved (plain) form."""


def build_feature_matrix(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all enabled feature engineering steps."""
    from src.forecasting.features.calendar import add_calendar_features
    from src.forecasting.features.demand_profile import add_demand_profile_features
    from src.forecasting.features.hierarchical import add_hierarchical_features
    from src.forecasting.features.lags import add_lag_features

    out = df.copy()
    feat = config.get("forecasting", {}).get("features", {})
    if feat.get("calendar", {}).get("enabled", True) or feat.get("holidays", {}).get(
        "enabled", True
    ):
        out = add_calendar_features(out, config=config)
    if feat.get("lags", {}).get("enabled", True):
        out = add_lag_features(out, config=config)
    # Off unless configured, so setups saved before this family existed rebuild unchanged.
    if feat.get("demand_profile", {}).get("enabled", False):
        out = add_demand_profile_features(out, config=config)
    if feat.get("hierarchical", {}).get("enabled", True):
        out = add_hierarchical_features(out, config=config)
    return out
