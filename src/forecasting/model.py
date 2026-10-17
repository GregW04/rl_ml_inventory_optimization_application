"""Multi-backend quantile demand forecasting for multiple horizons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.metrics import evaluate_forecasts


def quantile_col_suffix(q: float) -> str:
    """Column suffix for quantile level, e.g. 0.83 → ``83``, 0.1 → ``10``."""
    return str(int(round(q * 100)))


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
        # Optional target transform. "log1p" fits in log space — quantiles transform
        # exactly under monotonic maps, so this sharpens fit on skewed/spiky demand.
        self.target_transform = model_cfg.get("target_transform", "none")
        # Optional sample weighting. "sku_mean" up-weights high-volume SKUs so the
        # forecast loss prioritizes the items that dominate inventory cost.
        self.sample_weight_mode = model_cfg.get("sample_weight", None)

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
        if self.target_transform == "log1p":
            return np.log1p(np.clip(y, 0.0, None))
        return np.asarray(y, dtype=float)

    def _inverse_target(self, pred: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            # Guard against log-space extrapolation blowing up under expm1
            # (real log targets are <=~5 here; 9.2 → ~1e4 is a safe finite ceiling).
            return np.expm1(np.minimum(pred, 9.21))
        return pred

    def _compute_weights(self, train_df: pd.DataFrame) -> np.ndarray | None:
        """Per-row sample weights. ``sku_mean`` = each SKU's mean weekly demand
        (floored at 1), so high-volume cost-driver SKUs get more loss weight."""
        if self.sample_weight_mode != "sku_mean" or "sales" not in train_df.columns:
            return None
        return (
            train_df.groupby(["store", "product"])["sales"]
            .transform("mean")
            .clip(lower=1.0)
            .to_numpy()
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
                "extra": self._extra_save_payload(),
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
        for key, val in payload.get("extra", {}).items():
            setattr(forecaster, key, val)
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
        ltd_cfg = model_cfg.get("lead_time_demand", {})
        self.ltd_enabled = bool(ltd_cfg.get("enabled", False))
        self.ltd_target_col = "target_S"
        self.ltd_quantiles: list[float] = (
            list(ltd_cfg.get("quantiles", [])) if self.ltd_enabled else []
        )
        self.ltd_models: dict[float, Any] = {}

    def _make_estimator(self, quantile: float) -> Any:
        from src.forecasting.registry import get_spec
        return get_spec(self.model_name).make_estimator(quantile, self._runtime_params())

    def _fit_quantiles(
        self,
        specs: list[tuple[float, pd.DataFrame, np.ndarray, np.ndarray | None]],
    ) -> list[Any]:
        """Fit one booster per quantile. Parallel when ``compute.n_jobs`` > 1."""

        def fit_one(q, X, y, w, fit_cats: bool):
            model = self._make_estimator(q)
            self._fit_estimator(model, X, y, fit_cat_levels=fit_cats, sample_weight=w)
            return model

        n_jobs = self._fit_n_jobs()
        if n_jobs <= 1 or len(specs) <= 1:
            return [
                fit_one(q, X, y, w, i == 0 and self.model_name == "xgboost")
                for i, (q, X, y, w) in enumerate(specs)
            ]

        from joblib import Parallel, delayed

        # XGBoost category levels must be learned once before the other fits read them.
        if self.model_name == "xgboost":
            first = fit_one(*specs[0], True)
            rest = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(fit_one)(q, X, y, w, False) for q, X, y, w in specs[1:]
            )
            return [first, *rest]
        return Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(fit_one)(q, X, y, w, False) for q, X, y, w in specs
        )

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

        specs: list[tuple[float, pd.DataFrame, np.ndarray, np.ndarray | None]] = []
        keys: list[tuple] = []
        for h in horizons:
            tgt = target_col or f"target_h{h}"
            train_df = df.dropna(subset=[tgt])
            X = train_df[self.feature_cols]
            y = self._transform_target(train_df[tgt])
            w = self._compute_weights(train_df)
            for q in self.quantiles:
                keys.append(("h", h, q))
                specs.append((q, X, y, w))

        if self.ltd_enabled and self.ltd_quantiles:
            tgt = self.ltd_target_col
            train_df = df.dropna(subset=[tgt])
            if not train_df.empty:
                X = train_df[self.feature_cols]
                y = self._transform_target(train_df[tgt])
                w = self._compute_weights(train_df)
                for q in self.ltd_quantiles:
                    keys.append(("S", q))
                    specs.append((q, X, y, w))

        for key, model in zip(keys, self._fit_quantiles(specs)):
            if key[0] == "h":
                self.models[(key[1], key[2])] = model
            else:
                self.ltd_models[key[1]] = model

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
                # Invert any target transform, then clip (demand is non-negative).
                out[f"pred_q{int(q*100)}_h{h}"] = np.clip(
                    self._inverse_target(self._predict_estimator(self.models[key], X)),
                    0.0, None,
                )

            # Enforce non-crossing quantiles (independent regressors can invert).
            qcols = [
                f"pred_q{int(q*100)}_h{h}"
                for q in sorted(self.quantiles)
                if f"pred_q{int(q*100)}_h{h}" in out.columns
            ]
            if len(qcols) > 1:
                out[qcols] = np.sort(out[qcols].values, axis=1)

            q10 = out.get(f"pred_q10_h{h}")
            q50 = out.get(f"pred_q50_h{h}")
            q90 = out.get(f"pred_q90_h{h}")
            if q50 is not None:
                out[f"mu_h{h}"] = q50
            if q10 is not None and q90 is not None:
                out[f"sigma_h{h}"] = (q90 - q10) / self.sigma_divisor
                out[f"sigma_h{h}"] = out[f"sigma_h{h}"].clip(lower=0)

        if self.ltd_enabled and self.ltd_models:
            for q in self.ltd_quantiles:
                if q not in self.ltd_models:
                    continue
                suffix = quantile_col_suffix(q)
                out[f"pred_q{suffix}_S"] = np.clip(
                    self._inverse_target(
                        self._predict_estimator(self.ltd_models[q], X)
                    ),
                    0.0,
                    None,
                )
            ltd_cols = [
                f"pred_q{quantile_col_suffix(q)}_S"
                for q in sorted(self.ltd_quantiles)
                if f"pred_q{quantile_col_suffix(q)}_S" in out.columns
            ]
            if len(ltd_cols) > 1:
                out[ltd_cols] = np.sort(out[ltd_cols].values, axis=1)

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

    def evaluate_ltd(self, df: pd.DataFrame) -> dict[str, float]:
        """Metrics for lead-time demand target ``target_S`` on a scored dataframe."""
        if not self.ltd_enabled:
            return {}

        tgt = self.ltd_target_col
        mu_col = f"pred_q{quantile_col_suffix(0.5)}_S"
        scored = df.dropna(subset=[tgt])
        if scored.empty or mu_col not in scored.columns:
            return {}

        quantile_cols = {
            q: f"pred_q{quantile_col_suffix(q)}_S"
            for q in self.ltd_quantiles
            if f"pred_q{quantile_col_suffix(q)}_S" in scored.columns
        }
        metrics = evaluate_forecasts(
            scored,
            target_col=tgt,
            pred_col=mu_col,
            quantile_pred_cols=quantile_cols,
        )
        y = scored[tgt].to_numpy(dtype=float)
        for q, col in quantile_cols.items():
            pred = scored[col].to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(pred)
            if mask.sum() > 0:
                metrics[f"picp_q{quantile_col_suffix(q)}_S"] = float(
                    np.mean(y[mask] <= pred[mask])
                )
        metrics["target"] = "S"
        return metrics

    def _extra_save_payload(self) -> dict:
        return {
            "ltd_models": self.ltd_models,
            "ltd_enabled": self.ltd_enabled,
            "ltd_quantiles": self.ltd_quantiles,
            "ltd_target_col": self.ltd_target_col,
            "quantiles": self.quantiles,
            "sigma_divisor": self.sigma_divisor,
        }


class MeanDemandForecaster(_BaseForecaster):
    """
    Direct multi-horizon point forecaster (MSE).

    Trains one model per horizon. Outputs ``mu_h{h}`` only — for RL state
    without distributional uncertainty.
    """

    def _make_estimator(self) -> Any:
        from src.forecasting.registry import get_spec
        return get_spec(self.model_name).make_mean_estimator(self._runtime_params())

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
            y = self._transform_target(train_df[tgt])
            w = self._compute_weights(train_df)

            model = self._make_estimator()
            self._fit_estimator(
                model, X, y, fit_cat_levels=first_model, sample_weight=w
            )
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
            # Invert any target transform, then clip (demand is non-negative).
            out[f"mu_h{h}"] = np.clip(
                self._inverse_target(self._predict_estimator(self.models[h], X)), 0.0, None
            )

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
