"""Unbiased training forecasts: factors per fit and class, small-cell fallback, lookup copy, spec key."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.forecasting.measures import Measure
from src.rl.debias import (
    MIN_CELL_WEEKS,
    SHRINK_UNITS,
    debiased_lookup,
    training_bias_factors,
    training_lookup,
    training_rows,
)
from src.rl.spec import PolicySpec, validate_spec
from src.simulation.anchor import AnchorSpec

DATES = pd.date_range("2023-01-02", periods=20, freq="W-MON")
MEASURES = [Measure("mean", 0), Measure("mean", 1), Measure("quantile", 2, 5 / 6)]
RESOLVED = {
    "rl_train_start": str(DATES[2].date()), "rl_train_end": str(DATES[15].date()),
    "rl_val_start": str(DATES[16].date()), "rl_val_end": str(DATES[19].date()),
    "block_fits": [
        {"block": "rl_train", "index": 1, "start": str(DATES[2].date()), "end": str(DATES[8].date()),
         "fit_end": str(DATES[0].date())},
        {"block": "rl_train", "index": 2, "start": str(DATES[9].date()), "end": str(DATES[15].date()),
         "fit_end": str(DATES[6].date())},
        {"block": "rl_val", "index": 1, "start": str(DATES[16].date()), "end": str(DATES[19].date()),
         "fit_end": str(DATES[13].date())},
    ],
}


def _world(n_big: int = 90, n_small: int = 3):
    """Smooth SKUs forecast 20% low in fit 1 and 10% high in fit 2; a few Lumpy SKUs forecast right."""
    sales, lookup, classes = [], {}, []
    skus = [(s, "Smooth") for s in range(n_big)] + [(1000 + s, "Lumpy") for s in range(n_small)]
    for store, cls in skus:
        for d in DATES:
            sales.append({"store": store, "product": 1, "date": d, "sales": 10.0})
            classes.append({"store": store, "product": 1, "date": d, "demand_class": cls})
            fit1 = DATES[2] <= d <= DATES[8]
            fit2 = DATES[9] <= d <= DATES[15]
            f = 10.0 * (0.8 if (fit1 and cls == "Smooth") else 1.1 if (fit2 and cls == "Smooth") else 1.0)
            lookup[(store, 1, pd.Timestamp(d))] = {"mean_h0": f, "mean_h1": f, "q83_h2": 1.5 * f}
    return lookup, pd.DataFrame(sales), pd.DataFrame(classes)


def test_factor_per_fit_removes_the_average_error():
    lookup, panel, _ = _world(n_small=0)
    rows = training_rows(lookup, panel, RESOLVED, MEASURES)
    f = training_bias_factors(rows, MEASURES, by_class=False).set_index("fit")
    assert list(f.index) == ["training 1/2", "training 2/2"]
    a1, n1 = f.loc["training 1/2", "actual"], f.loc["training 1/2", "sku_weeks"]
    assert f.loc["training 1/2", "factor"] == pytest.approx((a1 + SHRINK_UNITS) / (0.8 * a1 + SHRINK_UNITS))
    assert f.loc["training 2/2", "factor"] < 1.0                     # forecasts too high → scaled down
    assert n1 > 0 and f.loc["training 1/2", "error_pct"] == pytest.approx(20.0)


def test_small_class_cells_use_the_fit_factor():
    lookup, panel, classes = _world()
    rows = training_rows(lookup, panel, RESOLVED, MEASURES, classes)
    f = training_bias_factors(rows, MEASURES, by_class=True).set_index(["fit", "demand_class"])
    smooth, lumpy = f.loc[("training 1/2", "Smooth")], f.loc[("training 1/2", "Lumpy")]
    assert smooth["sku_weeks"] >= MIN_CELL_WEEKS and not smooth["uses_fit_factor"]
    assert lumpy["sku_weeks"] < MIN_CELL_WEEKS and lumpy["uses_fit_factor"]
    fit_level = training_bias_factors(rows, MEASURES, by_class=False).set_index("fit").loc["training 1/2", "factor"]
    assert lumpy["factor"] == pytest.approx(fit_level)


def test_lookup_copy_scales_training_dates_only():
    lookup, panel, classes = _world()
    rows = training_rows(lookup, panel, RESOLVED, MEASURES, classes)
    factors = training_bias_factors(rows, MEASURES, by_class=True)
    new = debiased_lookup(lookup, rows, factors)
    key_train = (0, 1, pd.Timestamp(DATES[3]))
    f = factors.set_index(["fit", "demand_class"]).loc[("training 1/2", "Smooth"), "factor"]
    assert new[key_train]["mean_h0"] == pytest.approx(8.0 * f)
    assert new[key_train]["q83_h2"] == pytest.approx(12.0 * f)       # quantiles scale with the level
    for d in (DATES[0], DATES[17]):                                   # before training and validation
        key = (0, 1, pd.Timestamp(d))
        assert new[key] == lookup[key]
    assert lookup[key_train]["mean_h0"] == pytest.approx(8.0)          # the original is untouched


def test_training_lookup_on_run_data():
    lookup, panel, _ = _world(n_small=0)
    data = SimpleNamespace(lookup=lookup, panel=panel, setup={"resolved": RESOLVED}, measures=MEASURES)
    new, factors = training_lookup(data, {}, by_class=False)
    corrected = [new[(s, 1, pd.Timestamp(DATES[3]))]["mean_h0"] for s in range(5)]
    assert np.mean(corrected) == pytest.approx(10.0, rel=0.01)       # right on average after correction
    with pytest.raises(ValueError):
        training_lookup(SimpleNamespace(lookup=lookup, panel=panel, setup={"resolved": RESOLVED},
                                        measures=[Measure("quantile", 2, 5 / 6)]), {}, by_class=False)


def _spec(**kw) -> PolicySpec:
    return PolicySpec("run", "residual", "ppo", ["on_hand", "in_transit_w1", "in_transit_w2"],
                      AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 5 / 6), **kw)


def test_spec_key_unchanged_for_as_is_and_validated():
    as_is = _spec()
    d = as_is.to_dict()
    d.pop("training_forecasts")                                      # a spec saved before the option
    assert PolicySpec.from_dict(d).config_key() == as_is.config_key()
    unbiased = _spec(training_forecasts="unbiased_by_class")
    assert unbiased.config_key() != as_is.config_key()
    assert "unbiased_by_class" in unbiased.label()
    ids = ["mean_h0", "mean_h1", "q83_h2"]
    assert validate_spec(unbiased, ids) == []
    assert any("Training forecasts" in e for e in validate_spec(_spec(training_forecasts="nope"), ids))
    assert any("mean forecast" in e for e in validate_spec(unbiased, ["q50_h0", "q50_h1", "q83_h2"]))
