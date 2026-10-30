"""MeasureForecaster on a synthetic Poisson panel, plus conformal calibration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.calibration import fit_quantile_shifts
from src.forecasting.forecaster import MeasureForecaster
from src.forecasting.measures import Measure

MEASURES = [
    Measure("mean", 0),
    Measure("quantile", 1, 0.5),
    Measure("quantile", 1, 0.9),
    Measure("quantile", 2, 0.25),
    Measure("quantile", 2, 0.8333),
]


def _config():
    return {
        "forecasting": {
            "model": {"target_transform": "log1p", "categorical_features": []},
            "compute": {"n_jobs": 1, "booster_threads": 1},
        },
        "vn2": {},
    }


def _panel(n_sku: int = 6, n_weeks: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sku):
        level = 1.0 + 3.0 * s
        lam = level * (1.0 + 0.4 * np.sin(np.arange(n_weeks + 3) * 2 * np.pi / 13))
        y = rng.poisson(lam)
        for t in range(n_weeks):
            rows.append(
                {
                    "store": 0, "product": s, "date": pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=t),
                    "sales": y[t], "level": level, "season": np.sin(t * 2 * np.pi / 13),
                    "target_h0": y[t], "target_h1": y[t + 1], "target_h2": y[t + 2],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def scored():
    df = _panel()
    f = MeasureForecaster(_config(), model_name="lightgbm",
                          model_params={"n_estimators": 60, "num_leaves": 8, "learning_rate": 0.1,
                                        "min_child_samples": 5},
                          measures=MEASURES)
    f.fit(df[df["date"] < "2023-04-01"], feature_cols=["level", "season"])
    return f, f.predict(df[df["date"] >= "2023-04-01"])


def test_columns_match_measures(scored):
    _, out = scored
    for m in MEASURES:
        assert m.id in out.columns
        assert (out[m.id] >= 0).all()


def test_quantiles_do_not_cross(scored):
    _, out = scored
    assert (out["q50_h1"] <= out["q90_h1"]).all()
    assert (out["q25_h2"] <= out["q83_h2"]).all()


def test_metrics_per_measure(scored):
    f, out = scored
    m = f.evaluate(out)
    for key in ("rmse_mean_h0", "pinball_q83_h2", "coverage_q83_h2"):
        assert key in m, key
    assert 0.0 <= m["coverage_q83_h2"] <= 1.0


def test_fit_only_primary(scored):
    df = _panel()
    f = MeasureForecaster(_config(), model_name="lightgbm",
                          model_params={"n_estimators": 20, "num_leaves": 4}, measures=MEASURES)
    f.fit(df, feature_cols=["level", "season"], only=["q83_h2"])
    assert list(f.models) == ["q83_h2"]
    assert "q83_h2" in f.predict(df).columns


def test_conformal_shift_reaches_tau():
    rng = np.random.default_rng(1)
    y = rng.poisson(5.0, size=4000)
    oof = pd.DataFrame({"target_h2": y, "q83_h2": np.full(len(y), 3.0)})  # under-covers
    shift = fit_quantile_shifts(oof, [Measure("quantile", 2, 0.8333)], "log1p")["q83_h2"]
    q = np.expm1(np.log1p(3.0) + shift)
    cov = float(np.mean(y <= q))
    assert cov >= 0.8333 and cov < 0.95


def check_tweedie_mean(backend: str, params: dict) -> None:
    """Poisson and Tweedie means both estimate E[y] in units, so on Poisson data they agree."""
    df = _panel()
    train, test = df[df["date"] < "2023-04-01"], df[df["date"] >= "2023-04-01"]
    preds = {}
    for objective in ("poisson", "tweedie"):
        cfg = _config()
        cfg["forecasting"]["model"].update({"mean_objective": objective, "tweedie_variance_power": 1.5})
        f = MeasureForecaster(cfg, model_name=backend, model_params=dict(params), measures=[Measure("mean", 0)])
        f.fit(train, feature_cols=["level", "season"])
        preds[objective] = f.predict(test)["mean_h0"].to_numpy()
    assert preds["tweedie"].mean() == pytest.approx(test["target_h0"].mean(), rel=0.15)
    assert preds["tweedie"].mean() == pytest.approx(preds["poisson"].mean(), rel=0.10)
    assert np.corrcoef(preds["poisson"], preds["tweedie"])[0, 1] > 0.95


@pytest.mark.parametrize(
    "backend, params",
    [
        ("lightgbm", {"n_estimators": 200, "num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 5}),
        ("xgboost", {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1}),
        ("catboost", {"iterations": 200, "depth": 3, "learning_rate": 0.1}),
    ],
)
def test_tweedie_mean_is_on_the_demand_scale(backend, params):
    if backend != "xgboost":
        check_tweedie_mean(backend, params)
        return
    # XGBoost and torch bundle clashing libomp copies (XGBoost segfaults after torch loads,
    # torch hangs after XGBoost loads), and the RL tests load torch into this process.
    code = f"from tests.test_forecaster import check_tweedie_mean; check_tweedie_mean('xgboost', {params!r})"
    res = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]


def test_bad_mean_objective_is_rejected():
    cfg = _config()
    cfg["forecasting"]["model"]["mean_objective"] = "gamma"
    with pytest.raises(ValueError, match="mean_objective"):
        MeasureForecaster(cfg, measures=MEASURES)
    cfg["forecasting"]["model"].update({"mean_objective": "tweedie", "tweedie_variance_power": 2.5})
    with pytest.raises(ValueError, match="tweedie_variance_power"):
        MeasureForecaster(cfg, measures=MEASURES)
