"""Ensemble forecaster: equal-weight average, calibration applied once, quantiles never cross."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.forecaster import ENSEMBLE, EnsembleForecaster, finalize_measures, make_forecaster
from src.forecasting.measures import Measure

ROOT = Path(__file__).resolve().parents[1]
MEASURES = [Measure("mean", 0), Measure("quantile", 2, 0.5), Measure("quantile", 2, 0.8333)]


def test_finalize_shifts_in_log_space_then_sorts():
    out = pd.DataFrame({"q50_h2": [5.0, 3.0], "q83_h2": [4.0, 6.0]})
    finalize_measures(out, MEASURES, {"q83_h2": float(np.log(2.0))}, "log1p")
    # (1 + 4) * 2 - 1 = 9 then sorted per row; second row: (1 + 6) * 2 - 1 = 13
    assert out.loc[0].tolist() == pytest.approx([5.0, 9.0])
    assert out.loc[1].tolist() == pytest.approx([3.0, 13.0])
    crossing = pd.DataFrame({"q50_h2": [7.0], "q83_h2": [2.0]})
    finalize_measures(crossing, MEASURES, {}, "log1p")
    assert crossing.loc[0].tolist() == [2.0, 7.0]


def test_ensemble_needs_two_known_backends():
    cfg = {"forecasting": {"model": {"target_transform": "log1p"}}, "vn2": {}}
    for members in (["catboost"], ["catboost", "prophet"]):
        try:
            EnsembleForecaster(cfg, members, None, MEASURES)
        except ValueError:
            continue
        raise AssertionError(f"{members} should be rejected")


def check_ensemble_fit() -> None:
    """Real CatBoost + LightGBM fit; runs in a subprocess (see the test below)."""
    from tests.test_forecaster import _config, _panel

    df = _panel()
    cfg = _config()
    cfg["forecasting"]["ensemble"] = {"enabled": True, "members": ["catboost", "lightgbm"]}
    params = {
        "catboost": {"iterations": 80, "depth": 3, "learning_rate": 0.1, "allow_writing_files": False},
        "lightgbm": {"n_estimators": 80, "num_leaves": 8, "learning_rate": 0.1, "min_child_samples": 5},
    }
    ens = make_forecaster(cfg, ENSEMBLE, params, MEASURES)
    assert isinstance(ens, EnsembleForecaster) and list(ens.members) == ["catboost", "lightgbm"]
    train, test = df[df["date"] < "2023-04-01"], df[df["date"] >= "2023-04-01"]
    ens.fit(train, feature_cols=["level", "season"])
    out = ens.predict(test)
    raw = [m.predict(test, finalize=False) for m in ens.members.values()]
    assert np.allclose(out["mean_h0"], (raw[0]["mean_h0"] + raw[1]["mean_h0"]) / 2)
    assert (out["q50_h2"] <= out["q83_h2"]).all() and (out["q83_h2"] >= 0).all()
    ens.calibration = {"q83_h2": 0.3}
    shifted = ens.predict(test)
    assert (shifted["q83_h2"] >= out["q83_h2"] - 1e-9).all() and (shifted["q83_h2"] > out["q83_h2"]).any()
    assert set(ens.evaluate(out)) >= {"rmse_mean_h0", "pinball_q83_h2", "coverage_q83_h2"}


def test_ensemble_fit_in_subprocess(tmp_path):
    # LightGBM must not share a process with torch, which other test modules import.
    code = "from tests.test_ensemble import check_ensemble_fit; check_ensemble_fit()"
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
