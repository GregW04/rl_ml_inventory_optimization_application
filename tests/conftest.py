"""
Shared test setup: every test runs against its own throwaway MLflow store.

``tracking.init_tracking`` prefers ``MLFLOW_TRACKING_URI`` over the config, so pointing that
variable at a temp SQLite file also overrides a URI exported in the shell (the local tracking
server). New experiments in a SQLite store keep their artifacts next to the database
(``tracking.experiment_artifact_location``), so nothing lands in the live ``mlruns/``.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from src.utils.config import load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long end-to-end test (deselect with -m 'not slow')")
    config.addinivalue_line(
        "markers", "needs_local_run: needs a forecasting run in the local artifact cache (skipped in CI)"
    )


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path_factory, monkeypatch):
    store = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{store / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def local_forecast_run() -> str:
    """
    Id of the newest forecasting run whose forecasts sit in the local artifact cache
    (``mlflow.cache_dir``); the test is skipped when there is none (e.g. in CI). Tests read
    that run from the cache only, since their MLflow store is the isolated temp one.
    """
    cfg = load_config()
    cache = Path(cfg["_project_root"]) / cfg["mlflow"]["cache_dir"]
    setups = [p for p in cache.glob("*/forecasts/forecast_setup.json") if (p.parent / "train_forecasts.parquet").exists()]
    if not setups:
        pytest.skip("needs a forecasting run in the local artifact cache")
    return max(setups, key=lambda p: p.stat().st_mtime).parent.parent.name
