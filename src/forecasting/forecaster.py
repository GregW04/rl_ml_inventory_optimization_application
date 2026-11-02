"""Direct multi-horizon forecaster for a configurable set of measures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.forecasting.measures import HORIZONS, Measure, parse_measures
from src.forecasting.model import _BaseForecaster, inverse_values, transform_values
from src.utils.metrics import quantile_loss

MEAN_OBJECTIVES: tuple[str, ...] = ("poisson", "tweedie")
ENSEMBLE = "ensemble"
BACKENDS: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")


def finalize_measures(
    out: pd.DataFrame,
    measures: list[Measure],
    calibration: dict[str, float],
    transform: str,
) -> pd.DataFrame:
    """
    Apply conformal shifts to quantile columns and sort each horizon's quantiles.

    Shifts act in the target-transform space (log1p by default), like a multiplicative
    correction. Independent quantile boosters can cross; sorting restores monotonicity.
    """
    for m in measures:
        shift = calibration.get(m.id, 0.0)
        if m.kind == "quantile" and shift and m.id in out.columns:
            q = out[m.id].to_numpy(dtype=float)
            out[m.id] = np.clip(inverse_values(transform_values(q, transform) + shift, transform), 0.0, None)
    for h in HORIZONS:
        cols = [m.id for m in sorted(measures, key=lambda m: m.tau or 0.0)
                if m.kind == "quantile" and m.horizon == h and m.id in out.columns]
        if len(cols) > 1:
            out[cols] = np.sort(out[cols].to_numpy(dtype=float), axis=1)
    return out


class MeasureForecaster(_BaseForecaster):
    """
    One booster per measure, all sharing the backend's hyperparameters.

    - ``mean_h{h}``: count objective on raw demand, set by ``forecasting.model.mean_objective``:
      ``poisson`` (default) or ``tweedie``, whose variance grows as ``mean^p`` with
      ``p = tweedie_variance_power`` in (1, 2) and suits zero-heavy, overdispersed demand.
    - ``q{τ}_h{h}``: quantile objective in the target-transform space (log1p by default);
      the quantiles of a horizon are sorted afterwards so they never cross.

    ``calibration`` holds optional conformal shifts per quantile id, added in the
    transformed space before inverting (see :mod:`src.forecasting.calibration`).
    """

    def __init__(
        self,
        config: dict,
        model_name: str = "lightgbm",
        model_params: dict | None = None,
        measures: list[Measure] | None = None,
    ):
        super().__init__(config, model_name=model_name, model_params=model_params)
        self.measures: list[Measure] = (
            list(measures) if measures is not None else parse_measures(config)
        )
        self.calibration: dict[str, float] = {}
        model_cfg = config["forecasting"]["model"]
        self.mean_objective = str(model_cfg.get("mean_objective", "poisson")).lower()
        self.tweedie_power = float(model_cfg.get("tweedie_variance_power", 1.5))
        if self.mean_objective not in MEAN_OBJECTIVES:
            raise ValueError(f"mean_objective must be one of {MEAN_OBJECTIVES}, got '{self.mean_objective}'.")
        if self.mean_objective == "tweedie" and not 1.0 < self.tweedie_power < 2.0:
            raise ValueError(f"tweedie_variance_power must be between 1 and 2, got {self.tweedie_power}.")

    # ------------------------------------------------------------------
    # Boosters
    # ------------------------------------------------------------------

    def _make(self, kind: str, tau: float | None) -> Any:
        from src.forecasting.estimators import get_spec

        spec = get_spec(self.model_name)
        params = self._runtime_params()
        if kind == "quantile":
            return spec.make_quantile_estimator(tau, params)
        if self.mean_objective == "tweedie":
            return spec.make_tweedie_estimator(self.tweedie_power, params)
        return spec.make_poisson_estimator(params)

    def _target(self, df: pd.DataFrame, kind: str, horizon: int) -> np.ndarray:
        y = df[f"target_h{horizon}"].astype(float).clip(lower=0.0).to_numpy()
        return self._transform_target(y) if kind == "quantile" else y

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        only: list[str] | None = None,
    ) -> MeasureForecaster:
        """Fit the boosters for every measure, or only for the measure ids in ``only``."""
        self._prepare_fit(df, feature_cols)
        wanted = self.measures if only is None else [m for m in self.measures if m.id in only]
        specs = []
        for m in wanted:
            train_df = df.dropna(subset=[f"target_h{m.horizon}"])
            if train_df.empty:
                continue
            specs.append((m.id, m.kind, m.tau, train_df[self.feature_cols], self._target(train_df, m.kind, m.horizon)))
        for key, model in zip([s[0] for s in specs], self._fit_all(specs)):
            self.models[key] = model
        return self

    def _fit_all(self, specs: list[tuple]) -> list[Any]:
        """Fit one booster per spec; threads when ``compute.n_jobs`` > 1."""

        def fit_one(kind, tau, X, y, fit_cats: bool):
            model = self._make(kind, tau)
            self._fit_estimator(model, X, y, fit_cat_levels=fit_cats)
            return model

        n_jobs = self._fit_n_jobs()
        xgb = self.model_name == "xgboost"
        if n_jobs <= 1 or len(specs) <= 1:
            return [fit_one(o, t, X, y, i == 0 and xgb) for i, (_, o, t, X, y) in enumerate(specs)]

        from joblib import Parallel, delayed

        # XGBoost category levels must be learned once before the other fits read them.
        first = [fit_one(*specs[0][1:], True)] if xgb else []
        rest = specs[1:] if xgb else specs
        return first + Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(fit_one)(o, t, X, y, False) for _, o, t, X, y in rest
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        *,
        finalize: bool = True,
    ) -> pd.DataFrame:
        """Add one column per fitted measure (its id).

        ``finalize=False`` skips calibration and quantile sorting, so an ensemble can
        average raw member forecasts and finalize once.
        """
        out = df.copy()
        X = out[feature_cols or self.feature_cols]
        raw = {key: self._predict_estimator(model, X) for key, model in self.models.items()}
        for m in self.measures:
            if m.kind == "quantile" and m.id in raw:
                out[m.id] = np.clip(self._inverse_target(raw[m.id]), 0.0, None)
            elif m.kind == "mean" and m.id in raw:
                out[m.id] = np.clip(raw[m.id], 0.0, None)
        if finalize:
            finalize_measures(out, self.measures, self.calibration, self.target_transform)
        return out

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(self, scored: pd.DataFrame) -> dict[str, float]:
        """Per-measure metrics keyed ``{metric}_{id}`` on rows with a known target."""
        return evaluate_measures(scored, self.measures)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _extra_save_payload(self) -> dict:
        return {
            "measures": [m.to_dict() for m in self.measures],
            "calibration": dict(self.calibration),
        }

    def _post_load(self) -> None:
        self.measures = [m if isinstance(m, Measure) else Measure.from_dict(m) for m in self.measures]


class EnsembleForecaster:
    """
    Equal-weight average of several :class:`MeasureForecaster` backends.

    Every member fits the same measures on the same rows. ``predict`` averages each
    measure column, then applies calibration once and re-sorts the quantiles.
    """

    def __init__(
        self,
        config: dict,
        members: list[str],
        member_params: dict[str, dict | None] | None = None,
        measures: list[Measure] | None = None,
    ):
        if len(members) < 2:
            raise ValueError("An ensemble needs at least two backends.")
        unknown = [m for m in members if m not in BACKENDS]
        if unknown:
            raise ValueError(f"Unknown ensemble backends {unknown}; choose from {list(BACKENDS)}.")
        self.config = config
        self.model_name = ENSEMBLE
        self.measures: list[Measure] = list(measures) if measures is not None else parse_measures(config)
        self.members: dict[str, MeasureForecaster] = {
            name: MeasureForecaster(
                config, model_name=name, model_params=(member_params or {}).get(name), measures=self.measures
            )
            for name in members
        }
        self.calibration: dict[str, float] = {}
        self.target_transform = config["forecasting"]["model"].get("target_transform", "none")

    @property
    def feature_cols(self) -> list[str]:
        return next(iter(self.members.values())).feature_cols

    def fit(self, df: pd.DataFrame, feature_cols: list[str] | None = None, only: list[str] | None = None):
        for member in self.members.values():
            member.fit(df, feature_cols=feature_cols, only=only)
        return self

    def predict(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
        preds = [m.predict(df, feature_cols, finalize=False) for m in self.members.values()]
        out = df.copy()
        for m in self.measures:
            cols = [p[m.id].to_numpy(dtype=float) for p in preds if m.id in p.columns]
            if cols:
                out[m.id] = np.mean(cols, axis=0)
        return finalize_measures(out, self.measures, self.calibration, self.target_transform)

    def evaluate(self, scored: pd.DataFrame) -> dict[str, float]:
        return evaluate_measures(scored, self.measures)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": ENSEMBLE,
                "members": {name: m._payload() for name, m in self.members.items()},
                "measures": [m.to_dict() for m in self.measures],
                "calibration": dict(self.calibration),
            },
            path,
        )

    @classmethod
    def from_payload(cls, payload: dict) -> EnsembleForecaster:
        """Rebuild a fitted ensemble from what :meth:`save` writes."""
        members = {name: MeasureForecaster.from_payload(p) for name, p in payload["members"].items()}
        first = next(iter(members.values()))
        ens = cls(first.config, list(members), measures=[Measure.from_dict(m) for m in payload["measures"]])
        ens.members = members
        ens.calibration = dict(payload.get("calibration", {}))
        return ens


def load_forecaster(path: str | Path) -> MeasureForecaster | EnsembleForecaster:
    """A forecaster saved by ``save`` (single backend or ensemble), ready to predict."""
    payload = joblib.load(path)
    if payload.get("kind") == ENSEMBLE:
        return EnsembleForecaster.from_payload(payload)
    return MeasureForecaster.from_payload(payload)


def ensemble_settings(config: dict) -> tuple[bool, list[str]]:
    """(enabled, members) from ``forecasting.ensemble``."""
    ens = config.get("forecasting", {}).get("ensemble") or {}
    members = [m for m in (ens.get("members") or ["catboost", "lightgbm"]) if m in BACKENDS]
    return bool(ens.get("enabled", False)), members


def make_forecaster(
    config: dict,
    model_name: str,
    model_params: dict | None,
    measures: list[Measure],
):
    """A single-backend forecaster, or the ensemble when ``model_name`` is ``ensemble``.

    For the ensemble, ``model_params`` maps each member to its parameters (or None).
    """
    if model_name == ENSEMBLE:
        _, members = ensemble_settings(config)
        return EnsembleForecaster(config, members, model_params, measures)
    return MeasureForecaster(config, model_name=model_name, model_params=model_params, measures=measures)


def evaluate_measures(scored: pd.DataFrame, measures: list[Measure]) -> dict[str, float]:
    """
    Metrics per measure: mean → RMSE (its proper score), MAE and bias;
    quantile → pinball loss and coverage P(y ≤ q).
    """
    out: dict[str, float] = {}
    for m in measures:
        cols = [f"target_h{m.horizon}", m.id]
        if not all(c in scored.columns for c in cols):
            continue
        sub = scored[cols].dropna()
        if sub.empty:
            continue
        y = sub[cols[0]].to_numpy(dtype=float)
        p = sub[m.id].to_numpy(dtype=float)
        if m.kind == "mean":
            err = y - p
            out[f"rmse_{m.id}"] = float(np.sqrt(np.mean(err * err)))
            out[f"mae_{m.id}"] = float(np.mean(np.abs(err)))
            out[f"bias_{m.id}"] = float(np.mean(err))
        elif m.kind == "quantile":
            out[f"pinball_{m.id}"] = quantile_loss(y, p, m.tau)
            out[f"coverage_{m.id}"] = float(np.mean(y <= p))
    return {k: v for k, v in out.items() if np.isfinite(v)}
