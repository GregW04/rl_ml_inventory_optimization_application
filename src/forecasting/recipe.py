"""
Production forecaster: a frozen *recipe*, refit on all data up to an as-of week, and the
inference frame that forecasts the next decision week.

A recipe is what defines *how* the forecaster is trained: the feature switches, the measures,
the backend or ensemble and its hyperparameters, the mean objective and the target transform.
It says nothing about data or dates, so ``Recipe.key()`` stays the same across refits; refits of
one recipe are told apart by the data they saw (``panel_digest``).

Timing follows the research block fits (``pipeline._fit_rl_block_forecasts``): the row dated t
forecasts h0 = t, h1 = t+1, h2 = t+2 from sales up to t−1, so at ``as_of`` (the last week with
sales) the decision week is ``as_of + 1 week``, and a refit uses every row whose longest-horizon
label is already known: rows up to ``as_of − max(HORIZONS)`` weeks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.forecasting.forecaster import (
    ENSEMBLE,
    EnsembleForecaster,
    MeasureForecaster,
    ensemble_settings,
    make_forecaster,
)
from src.forecasting.measures import HORIZONS, Measure, parse_measures
from src.forecasting.model import build_feature_matrix, get_feature_columns
from src.forecasting.preprocessing import build_target_columns

# The parts of config["forecasting"] that shape a fitted model. Validation, tuning, model
# selection and compute settings are research or runtime choices, not part of the recipe.
RECIPE_SECTIONS = ("features", "model", "ensemble")
PANEL_COLUMNS = ["store", "product", "date", "sales", "in_stock"]

Forecaster = MeasureForecaster | EnsembleForecaster


@dataclass(frozen=True)
class Recipe:
    forecasting: dict
    measures: tuple[dict, ...]  # resolved (numeric tau), so the recipe doesn't depend on vn2 costs
    model_name: str  # a backend, or "ensemble"
    model_params: dict | None = None  # per member for the ensemble; None → the backend's defaults

    @classmethod
    def from_config(cls, config: dict, model_params: dict | None = None) -> Recipe:
        fc = config["forecasting"]
        enabled, _ = ensemble_settings(config)
        return cls(
            forecasting={k: copy.deepcopy(fc[k]) for k in RECIPE_SECTIONS if k in fc},
            measures=tuple(m.to_dict() for m in parse_measures(config)),
            model_name=ENSEMBLE if enabled else fc["model"]["name"],
            model_params=copy.deepcopy(model_params),
        )

    @classmethod
    def from_dict(cls, d: dict) -> Recipe:
        return cls(d["forecasting"], tuple(d["measures"]), d["model_name"], d.get("model_params"))

    def to_dict(self) -> dict:
        return {
            "forecasting": self.forecasting,
            "measures": list(self.measures),
            "model_name": self.model_name,
            "model_params": self.model_params,
        }

    def key(self) -> str:
        """Short hash of the recipe: equal for every refit of the same recipe."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def measure_list(self) -> list[Measure]:
        return [Measure.from_dict(d) for d in self.measures]

    def apply(self, config: dict) -> dict:
        """``config`` with its forecasting settings replaced by the recipe's (runtime settings stay)."""
        cfg = copy.deepcopy(config)
        cfg["forecasting"].update(copy.deepcopy(self.forecasting))
        return cfg


def fit_end(as_of: pd.Timestamp | str) -> pd.Timestamp:
    """Last training row of a refit at ``as_of``: its longest-horizon label is the as-of week."""
    return pd.Timestamp(as_of) - pd.Timedelta(weeks=max(HORIZONS))


def fit_recipe(features: pd.DataFrame, recipe: Recipe, config: dict, as_of: pd.Timestamp | str) -> Forecaster:
    """Fit ``recipe`` on the feature rows whose labels are all known by ``as_of``."""
    cfg = recipe.apply(config)
    train = features[features["date"] <= fit_end(as_of)]
    if train.empty:
        raise ValueError(f"No training rows up to {fit_end(as_of).date()}.")
    forecaster = make_forecaster(cfg, recipe.model_name, recipe.model_params, recipe.measure_list())
    forecaster.fit(train, feature_cols=get_feature_columns(features))
    return forecaster


def panel_from_history(history: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """
    The long panel (``load_sales_panel`` layout) from raw weekly history: hierarchy attributes
    from ``master``, out-of-stock weeks masked to NaN (censored demand), targets per horizon.
    """
    panel = history[PANEL_COLUMNS].merge(master, on=["store", "product"], how="left")
    panel["date"] = pd.to_datetime(panel["date"])
    panel.loc[~panel["in_stock"].fillna(True).astype(bool), "sales"] = np.nan
    panel = panel.sort_values(["store", "product", "date"]).reset_index(drop=True)
    return build_target_columns(panel, horizons=list(HORIZONS))


def panel_digest(panel: pd.DataFrame) -> str:
    """Content hash of the sales the panel holds (what a refit saw), independent of row order."""
    rows = panel[PANEL_COLUMNS].sort_values(["store", "product", "date"])
    return hashlib.sha256(pd.util.hash_pandas_object(rows, index=False).to_numpy().tobytes()).hexdigest()[:16]


def make_inference_frame(panel: pd.DataFrame, as_of: pd.Timestamp | str, config: dict) -> pd.DataFrame:
    """
    Feature rows of the decision week (``as_of + 1 week``), one per SKU with sales at ``as_of``.

    One row per SKU is appended with unknown sales. Every feature of a row uses sales up to the
    week before it, so the appended row gets exactly the features a training row had
    (``tests/test_recipe.py`` checks this against a panel that already holds that week).
    """
    as_of = pd.Timestamp(as_of)
    hist = panel[panel["date"] <= as_of].drop(columns=[c for c in panel.columns if c.startswith("target_h")])
    last = hist[hist["date"] == as_of]
    if last.empty:
        raise ValueError(f"The panel has no week {as_of.date()}.")
    decision = as_of + pd.Timedelta(weeks=1)
    future = last.assign(date=decision, sales=np.nan, in_stock=True)
    full = pd.concat([hist, future], ignore_index=True).sort_values(["store", "product", "date"])
    full = build_target_columns(full.reset_index(drop=True), horizons=list(HORIZONS))
    features = build_feature_matrix(full, config)
    return features[features["date"] == decision].reset_index(drop=True)


def forecast(forecaster: Forecaster, panel: pd.DataFrame, as_of: pd.Timestamp | str, config: dict) -> pd.DataFrame:
    """Forecasts issued at ``as_of`` for the decision week: one row per SKU, one column per measure."""
    frame = make_inference_frame(panel, as_of, config)
    scored = forecaster.predict(frame, feature_cols=forecaster.feature_cols)
    return scored[["store", "product", "date", *[m.id for m in forecaster.measures]]].reset_index(drop=True)
