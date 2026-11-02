"""Saved forecasters load back and predict exactly what they predicted before saving."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.forecaster import ENSEMBLE, EnsembleForecaster, MeasureForecaster, load_forecaster, make_forecaster
from tests.test_ensemble import MEASURES
from tests.test_forecaster import _config, _panel

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["level", "season"]


def _split():
    df = _panel()
    return df[df["date"] < "2023-04-01"], df[df["date"] >= "2023-04-01"]


def _same_predictions(model, loaded, test: pd.DataFrame) -> None:
    ids = [m.id for m in MEASURES]
    np.testing.assert_array_equal(model.predict(test)[ids].to_numpy(), loaded.predict(test)[ids].to_numpy())


def test_single_backend_round_trip(tmp_path):
    # CatBoost may share a process with torch (other test modules import it); LightGBM may not.
    train, test = _split()
    model = make_forecaster(_config(), "catboost", {"iterations": 40, "depth": 3, "allow_writing_files": False}, MEASURES)
    model.fit(train, feature_cols=FEATURES)
    model.calibration = {"q83_h2": 0.1}
    model.save(tmp_path / "f.joblib")
    loaded = load_forecaster(tmp_path / "f.joblib")
    assert isinstance(loaded, MeasureForecaster) and loaded.calibration == {"q83_h2": 0.1}
    _same_predictions(model, loaded, test)


def check_ensemble_round_trip(path: str) -> None:
    """CatBoost + LightGBM ensemble save/load; runs in a subprocess (see the test below)."""
    cfg = _config()
    cfg["forecasting"]["ensemble"] = {"enabled": True, "members": ["catboost", "lightgbm"]}
    params = {
        "catboost": {"iterations": 40, "depth": 3, "allow_writing_files": False},
        "lightgbm": {"n_estimators": 40, "num_leaves": 8, "min_child_samples": 5},
    }
    train, test = _split()
    model = make_forecaster(cfg, ENSEMBLE, params, MEASURES)
    model.fit(train, feature_cols=FEATURES)
    model.save(path)
    loaded = load_forecaster(path)
    assert isinstance(loaded, EnsembleForecaster) and list(loaded.members) == ["catboost", "lightgbm"]
    _same_predictions(model, loaded, test)


def test_ensemble_round_trip_in_subprocess(tmp_path):
    code = f"from tests.test_forecaster_io import check_ensemble_round_trip; check_ensemble_round_trip({str(tmp_path / 'e.joblib')!r})"
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
