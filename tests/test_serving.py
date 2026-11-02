"""The forecaster served from the Model Registry predicts exactly what it predicted in process."""

from __future__ import annotations

import copy

import mlflow
import pandas as pd

from src.forecasting import recipe as rec
from src.forecasting.model import build_feature_matrix
from src.forecasting.preprocessing import load_master, load_panel_from_config
from src.ops import registry
from src.serving.forecaster import log_forecaster
from src.utils.config import load_config
from tests.test_recipe import AS_OF, CATBOOST


def test_registered_forecaster_serves_the_in_process_forecasts():
    cfg = copy.deepcopy(load_config())
    cfg["forecasting"]["ensemble"]["enabled"] = False
    cfg["forecasting"]["model"]["name"] = "catboost"  # CatBoost may share a process with torch
    panel = load_panel_from_config(cfg)
    panel = panel.merge(panel[["store", "product"]].drop_duplicates().head(12), on=["store", "product"])
    hist = panel[panel["date"] <= AS_OF]
    recipe = rec.Recipe.from_config(cfg, model_params=CATBOOST)
    used = recipe.apply(cfg)
    model = rec.fit_recipe(build_feature_matrix(hist, used), recipe, cfg, AS_OF)
    expected = rec.forecast(model, hist, AS_OF, used)

    name = registry.model_name("dev", "forecaster")
    history = hist[rec.PANEL_COLUMNS]
    with mlflow.start_run():
        info = log_forecaster(model, recipe, cfg, history, load_master(cfg["data"]["master_file"]), AS_OF,
                              rec.panel_digest(hist))
    version = registry.register(info.model_uri, name, {"recipe_key": recipe.key(), "trigger": "bootstrap"})
    registry.promote(name, version.version)

    served = mlflow.pyfunc.load_model(f"models:/{name}@{registry.CHAMPION}")
    pd.testing.assert_frame_equal(served.predict(history, params={"as_of": str(AS_OF.date())}), expected,
                                  check_dtype=False)
    # Without the param, the latest week in the input is the as-of week.
    pd.testing.assert_frame_equal(served.predict(history), expected, check_dtype=False)
    champion = registry.resolve(name, registry.CHAMPION)
    assert champion.tags == {"recipe_key": recipe.key(), "trigger": "bootstrap"}
    logged = mlflow.get_logged_model(info.model_id)
    assert logged.params["recipe_key"] == recipe.key() and logged.params["as_of"] == str(AS_OF.date())
