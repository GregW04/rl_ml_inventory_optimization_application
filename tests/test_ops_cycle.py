"""
End to end: a short ops environment bootstraps, refits at its max age, promotes with the previous
forecaster running in shadow, skips finished weeks, and replays bit for bit. Isolated: a temp
project root (ops state, artifact cache) and the per-test MLflow store from conftest.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.ops import registry
from src.ops.settings import OpsConfig, cycle_dir, weeks
from src.ops.steps import run_env
from src.utils.config import load_config

ENV = {
    "kind": "sim", "start": "2023-10-02", "end": "2023-11-06",
    "model_params": {"iterations": 30, "depth": 3, "learning_rate": 0.2, "allow_writing_files": False},
    "forecasting": {"model": {"name": "catboost"}, "ensemble": {"enabled": False}},  # CatBoost: fine next to torch
    "rules": {"strategy": "fixed", "max_age_weeks": 2},
}


@pytest.mark.slow
def test_environment_bootstraps_refits_promotes_and_replays(tmp_path):
    config = load_config()
    config["_project_root"] = str(tmp_path)
    ops = OpsConfig.model_validate({"envs": {"a": ENV, "b": ENV}})

    assert run_env(config, ops, "a") == ["done"] * 6
    name = registry.model_name("a", "forecaster")
    # Bootstrap on 10-02 (v1); age 2 refits on 10-16 (v2) and 10-30 (v3), each promoted.
    assert str(registry.resolve(name, registry.CHAMPION).version) == "3"
    assert str(registry.resolve(name, registry.PREVIOUS).version) == "2"
    decisions = [pd.read_json(cycle_dir(config, "a", w) / "decision.json", typ="series")["action"]
                 for w in weeks(ENV["start"], ENV["end"])]
    assert decisions == ["bootstrap", "keep", "refit", "keep", "refit", "keep"]
    # The previous forecaster runs in shadow while the new champion is on probation.
    shadow = [cycle_dir(config, "a", w).joinpath("costs_previous.parquet").exists() for w in weeks(ENV["start"], ENV["end"])]
    assert shadow == [False, False, False, True, True, True]

    client = MlflowClient()
    exp = client.get_experiment_by_name("vn2-ops-a")
    runs = client.search_runs([exp.experiment_id])
    stages = sorted(r.data.tags["stage"] for r in runs)
    assert stages == sorted(["timeline"] + ["cycle"] * 6 + ["bootstrap", "refit", "refit"])
    assert not client.search_runs([mlflow.get_experiment_by_name("Default").experiment_id])  # nothing leaks there
    assert all(r.info.status == "FINISHED" for r in runs)

    assert run_env(config, ops, "a") == ["skipped"] * 6

    # A second environment with the same settings replays the same weeks bit for bit.
    run_env(config, ops, "b")
    for w in weeks(ENV["start"], ENV["end"])[1:]:
        a = pd.read_parquet(cycle_dir(config, "a", w) / "costs_live.parquet")
        b = pd.read_parquet(cycle_dir(config, "b", w) / "costs_live.parquet")
        pd.testing.assert_frame_equal(a, b)
        assert (cycle_dir(config, "a", w) / "orders.csv").read_text() == (cycle_dir(config, "b", w) / "orders.csv").read_text()
