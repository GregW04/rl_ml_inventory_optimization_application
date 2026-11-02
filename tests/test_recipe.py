"""Production forecaster path: recipe identity, the refit cut, the inference frame, forecasts."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.forecasting import recipe as rec
from src.forecasting.model import build_feature_matrix, get_feature_columns
from src.forecasting.preprocessing import load_panel_from_config
from src.utils.config import load_config

AS_OF = pd.Timestamp("2023-12-04")  # decision week 2023-12-11, a week before Christmas Eve's
CATBOOST = {"iterations": 30, "depth": 3, "learning_rate": 0.2, "allow_writing_files": False}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def panel(cfg):
    p = load_panel_from_config(cfg)
    return p.merge(p[["store", "product"]].drop_duplicates().head(30), on=["store", "product"])


def test_recipe_key_ignores_runtime_settings_but_not_the_model(cfg):
    base = rec.Recipe.from_config(cfg)
    runtime = copy.deepcopy(cfg)
    runtime["forecasting"]["compute"]["n_jobs"] = 1
    runtime["forecasting"]["tuning"]["n_trials"] = 3
    runtime["periods"]["refit_every_weeks"] = 4
    runtime["data"]["sales_file"] = "elsewhere.csv"
    assert rec.Recipe.from_config(runtime).key() == base.key()
    assert rec.Recipe.from_config(cfg, model_params={"depth": 4}).key() != base.key()
    lags = copy.deepcopy(cfg)
    lags["forecasting"]["features"]["lags"]["lags"] = [1, 2]
    assert rec.Recipe.from_config(lags).key() != base.key()
    assert rec.Recipe.from_dict(base.to_dict()) == base


def test_refit_uses_every_row_whose_labels_are_known(cfg, panel, monkeypatch):
    seen = {}

    class Stub:
        def fit(self, df, feature_cols=None):
            seen["last"], seen["cols"] = df["date"].max(), feature_cols
            return self

    monkeypatch.setattr(rec, "make_forecaster", lambda *a, **k: Stub())
    features = build_feature_matrix(panel[panel["date"] <= AS_OF], cfg)
    rec.fit_recipe(features, rec.Recipe.from_config(cfg), cfg, AS_OF)
    # The last training row's h2 label is the as-of week's sales: the research block-fit cut.
    assert seen["last"] == AS_OF - pd.Timedelta(weeks=2)
    assert "target_h2" not in seen["cols"] and "sales" not in seen["cols"]


def test_inference_rows_equal_the_training_rows_of_that_week(cfg, panel):
    frame = rec.make_inference_frame(panel[panel["date"] <= AS_OF], AS_OF, cfg)
    full = build_feature_matrix(panel, cfg)
    cols = get_feature_columns(full)
    want = full[full["date"] == AS_OF + pd.Timedelta(weeks=1)]
    got, want = (d.sort_values(["store", "product"])[cols].reset_index(drop=True) for d in (frame, want))
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_categorical=False)


def test_forecast_issues_one_row_per_sku_for_the_decision_week(cfg, panel):
    single = copy.deepcopy(cfg)
    single["forecasting"]["ensemble"]["enabled"] = False
    single["forecasting"]["model"]["name"] = "catboost"
    recipe = rec.Recipe.from_config(single, model_params=CATBOOST)
    hist = panel[panel["date"] <= AS_OF]
    used = recipe.apply(single)
    model = rec.fit_recipe(build_feature_matrix(hist, used), recipe, single, AS_OF)
    out = rec.forecast(model, hist, AS_OF, used)
    ids = [m.id for m in recipe.measure_list()]
    assert len(out) == hist[["store", "product"]].drop_duplicates().shape[0]
    assert (out["date"] == AS_OF + pd.Timedelta(weeks=1)).all()
    assert out[ids].notna().all().all() and (out[ids] >= 0).all().all()
    assert (out["q50_h2"] <= out["q83_h2"]).all()
    assert rec.panel_digest(hist) == rec.panel_digest(hist.sample(frac=1.0, random_state=0))
    assert rec.panel_digest(hist) != rec.panel_digest(panel[panel["date"] <= AS_OF - pd.Timedelta(weeks=1)])
