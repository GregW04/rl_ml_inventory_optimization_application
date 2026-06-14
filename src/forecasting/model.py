"""Multi-backend quantile demand forecasting for multiple horizons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.metrics import evaluate_forecasts


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
    exclude.update(c for c in df.columns if c.startswith(target_prefix))

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("pred_") or col.startswith("mu_") or col.startswith("sigma_"):
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


def get_forecaster_type(config: dict) -> str:
    """``quantile`` (default) or ``mean``."""
    return config.get("forecasting", {}).get("forecaster_type", "quantile")


def get_primary_metric(config: dict) -> str:
    """Metric used for model selection and Optuna tuning."""
    return "wql" if get_forecaster_type(config) == "quantile" else "mae"


class _BaseForecaster:
    """
    Shared backend wiring for the direct multi-horizon forecasters.

    Holds the parts that are identical across the quantile and mean variants:
    parameter resolution, backend-specific fit/predict/categorical encoding, and
    serialization.  Subclasses define only the per-horizon model shape (``fit`` /
    ``predict``) and the metrics they report (``evaluate_horizon``).
    """

    def __init__(
        self,
        config: dict,
        model_name: str = "lightgbm",
        model_params: dict | None = None,
    ):
        self.config = config
        self.model_name = model_name
        model_cfg = config["forecasting"]["model"]
        self.horizons = model_cfg["horizons"]

        if model_params is not None:
            self.model_params: dict = model_params
        elif model_name == "lightgbm":
            # Backward-compatible: use the lightgbm block from config
            self.model_params = model_cfg.get("lightgbm", {})
        else:
            from src.forecasting.registry import get_spec
            self.model_params = get_spec(model_name).default_params.copy()

        self.models: dict = {}
        self.feature_cols: list[str] = []
        self.categorical_features: list[str] = []
        # XGBoost: store category levels fitted on train to reuse at predict time
        self._cat_levels: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Backend-specific fit / predict wrappers (shared by both variants)
    # ------------------------------------------------------------------

    def _fit_estimator(
        self, model: Any, X: pd.DataFrame, y: pd.Series, fit_cat_levels: bool = False
    ) -> None:
        if self.model_name == "lightgbm":
            model.fit(
                X, y,
                categorical_feature=self.categorical_features or "auto",
            )
        elif self.model_name == "xgboost":
            X_enc = self._encode_xgb(X, fit=fit_cat_levels)
            model.fit(X_enc, y)
        elif self.model_name == "catboost":
            cat_idx = [
                list(X.columns).index(c)
                for c in self.categorical_features
                if c in X.columns
            ]
            model.fit(X, y, cat_features=cat_idx)
        else:
            model.fit(X, y)

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

    def _prepare_fit(
        self, df: pd.DataFrame, feature_cols: list[str] | None
    ) -> None:
        """Resolve feature/categorical columns shared by every ``fit`` call."""
        self.feature_cols = feature_cols or get_feature_columns(df)
        self.categorical_features = get_categorical_features(
            df, self.config, self.feature_cols
        )

    # ------------------------------------------------------------------
    # Serialization (shared)
    # ------------------------------------------------------------------

    def _extra_save_payload(self) -> dict:
        """Hook for subclass-specific fields added to the saved payload."""
        return {}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self.models,
                "feature_cols": self.feature_cols,
                "categorical_features": self.categorical_features,
                "config": self.config,
                "model_name": self.model_name,
                "model_params": self.model_params,
                "cat_levels": self._cat_levels,
                **self._extra_save_payload(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "_BaseForecaster":
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
        return forecaster


class QuantileDemandForecaster(_BaseForecaster):
    """
    Direct multi-horizon quantile forecaster.

    Trains one model per (horizon, quantile) combination.  The backend
    (LightGBM / XGBoost / CatBoost) is selected via ``model_name``.
    """

    def __init__(
        self,
        config: dict,
        model_name: str = "lightgbm",
        model_params: dict | None = None,
    ):
        super().__init__(config, model_name=model_name, model_params=model_params)
        model_cfg = config["forecasting"]["model"]
        self.quantiles = model_cfg["quantiles"]
        self.sigma_divisor = model_cfg.get("sigma_divisor", 2.56)
        self.models: dict[tuple[int, float], Any] = {}

    def _make_estimator(self, quantile: float) -> Any:
        from src.forecasting.registry import get_spec
        return get_spec(self.model_name).make_estimator(quantile, self.model_params)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
        horizon: int | None = None,
    ) -> "QuantileDemandForecaster":
        """Fit all quantile models for one horizon (or all if target_col/horizon None)."""
        horizons = [horizon] if horizon is not None else self.horizons
        self._prepare_fit(df, feature_cols)

        first_model = True
        for h in horizons:
            tgt = target_col or f"target_h{h}"
            train_df = df.dropna(subset=[tgt])
            X = train_df[self.feature_cols]
            y = train_df[tgt]

            for q in self.quantiles:
                model = self._make_estimator(q)
                # Fit category levels only once (first model in first horizon)
                self._fit_estimator(model, X, y, fit_cat_levels=first_model)
                first_model = False
                self.models[(h, q)] = model

        return self

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add pred_q*, mu_h*, sigma_h* columns."""
        out = df.copy()
        cols = feature_cols or self.feature_cols
        X = out[cols]

        for h in self.horizons:
            for q in self.quantiles:
                key = (h, q)
                if key not in self.models:
                    continue
                out[f"pred_q{int(q*100)}_h{h}"] = self._predict_estimator(
                    self.models[key], X
                )

            q10 = out.get(f"pred_q10_h{h}")
            q50 = out.get(f"pred_q50_h{h}")
            q90 = out.get(f"pred_q90_h{h}")
            if q50 is not None:
                out[f"mu_h{h}"] = q50
            if q10 is not None and q90 is not None:
                out[f"sigma_h{h}"] = (q90 - q10) / self.sigma_divisor
                out[f"sigma_h{h}"] = out[f"sigma_h{h}"].clip(lower=0)

        return out

    def evaluate_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> dict[str, float]:
        """Metrics for a single horizon on scored dataframe."""
        tgt = f"target_h{horizon}"
        scored = df.dropna(subset=[tgt, f"mu_h{horizon}"])
        if scored.empty:
            return {}

        quantile_cols = {
            q: f"pred_q{int(q*100)}_h{horizon}"
            for q in self.quantiles
            if f"pred_q{int(q*100)}_h{horizon}" in scored.columns
        }
        metrics = evaluate_forecasts(
            scored,
            target_col=tgt,
            pred_col=f"mu_h{horizon}",
            quantile_pred_cols=quantile_cols,
        )
        metrics["horizon"] = horizon
        return metrics


class MeanDemandForecaster(_BaseForecaster):
    """
    Direct multi-horizon point forecaster (MSE).

    Trains one model per horizon. Outputs ``mu_h{h}`` only — for RL state
    without distributional uncertainty.
    """

    def _make_estimator(self) -> Any:
        from src.forecasting.registry import get_spec
        return get_spec(self.model_name).make_mean_estimator(self.model_params)

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
        horizon: int | None = None,
    ) -> "MeanDemandForecaster":
        """Fit one MSE model per horizon."""
        horizons = [horizon] if horizon is not None else self.horizons
        self._prepare_fit(df, feature_cols)

        first_model = True
        for h in horizons:
            tgt = target_col or f"target_h{h}"
            train_df = df.dropna(subset=[tgt])
            X = train_df[self.feature_cols]
            y = train_df[tgt]

            model = self._make_estimator()
            self._fit_estimator(model, X, y, fit_cat_levels=first_model)
            first_model = False
            self.models[h] = model

        return self

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Add ``mu_h{h}`` point-forecast columns only."""
        out = df.copy()
        cols = feature_cols or self.feature_cols
        X = out[cols]

        for h in self.horizons:
            if h not in self.models:
                continue
            out[f"mu_h{h}"] = self._predict_estimator(self.models[h], X)

        return out

    def evaluate_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
    ) -> dict[str, float]:
        """MAE / MASE / sMAPE for a single horizon (no wQL)."""
        tgt = f"target_h{horizon}"
        scored = df.dropna(subset=[tgt, f"mu_h{horizon}"])
        if scored.empty:
            return {}

        metrics = evaluate_forecasts(
            scored,
            target_col=tgt,
            pred_col=f"mu_h{horizon}",
            quantile_pred_cols=None,
        )
        metrics["horizon"] = horizon
        return metrics

    def _extra_save_payload(self) -> dict:
        return {"forecaster_type": "mean"}


def get_forecaster_class(config: dict) -> type:
    """Return ``QuantileDemandForecaster`` or ``MeanDemandForecaster``."""
    if get_forecaster_type(config) == "mean":
        return MeanDemandForecaster
    return QuantileDemandForecaster


def build_feature_matrix(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all enabled feature engineering steps."""
    from src.forecasting.features.calendar import add_calendar_features
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
    if feat.get("hierarchical", {}).get("enabled", True):
        out = add_hierarchical_features(out, config=config)
    return out
