"""
End to end: a short ops environment bootstraps, refits at its max age, promotes with the previous
forecaster running in shadow, skips finished weeks, and replays bit for bit. Isolated: a temp
project root (ops state, artifact cache) and the per-test MLflow store from conftest.
"""

from __future__ import annotations

import json

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
    # Six weeks are far too few to train RL on: the anchor (policy version 1) orders throughout.
    policy = [pd.read_json(cycle_dir(config, "a", w) / "decision.json", typ="series")["policy"]["action"]
              for w in weeks(ENV["start"], ENV["end"])]
    assert policy == ["bootstrap"] + ["keep"] * 5
    # The previous forecaster runs in shadow while the new champion is on probation.
    shadow = [cycle_dir(config, "a", w).joinpath("costs_previous.parquet").exists() for w in weeks(ENV["start"], ENV["end"])]
    assert shadow == [False, False, False, True, True, True]

    client = MlflowClient()
    exp = client.get_experiment_by_name("vn2-ops-a")
    runs = client.search_runs([exp.experiment_id])
    stages = sorted(r.data.tags["stage"] for r in runs)
    assert stages == sorted(["timeline"] + ["cycle"] * 6 + ["bootstrap", "refit", "refit", "policy-bootstrap"])
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


RL_ENV = {
    "kind": "sim", "start": "2023-06-05", "end": "2023-11-13",
    "model_params": {"iterations": 30, "depth": 3, "learning_rate": 0.2, "allow_writing_files": False},
    "forecasting": {"model": {"name": "catboost"}, "ensemble": {"enabled": False}},
    "rules": {"strategy": "never", "policy": {
        "train_weeks": 9, "val_weeks": 8, "seeds": [42, 43],
        "agent_params": {"iterations": 3, "batch_windows": 64, "minibatch_size": 64, "epochs": 1, "early_stop": False},
        # Permissive, so the promotion path runs: any challenger that won some demand class passes.
        "promote_margin_pct": -100.0, "units_band_pct": 1000.0, "min_seed_share": 0.0,
    }},
}


@pytest.mark.slow
def test_rl_challenger_trains_on_the_ledger_and_serves_with_an_anchor_shadow(tmp_path):
    config = load_config()
    config["_project_root"] = str(tmp_path)
    ops = OpsConfig.model_validate({"envs": {"rl": RL_ENV}})
    assert set(run_env(config, ops, "rl")) == {"done"}

    days = [str(w.date()) for w in weeks(RL_ENV["start"], RL_ENV["end"])]
    policy_actions = {d: pd.read_json(cycle_dir(config, "rl", pd.Timestamp(d)) / "decision.json", typ="series")["policy"]["action"]
                      for d in days}
    # Bootstrap the anchor, wait for 4 burn-in + 9 training + 8 validation ledger weeks, train once.
    assert policy_actions[days[0]] == "bootstrap"
    assert [d for d, a in policy_actions.items() if a == "retrain"] == ["2023-10-30"]

    name = registry.model_name("rl", "policy")
    champion = registry.resolve(name, registry.CHAMPION)
    assert str(champion.version) == "2" and champion.tags["kind"] == "ppo" and champion.tags["validation_status"] == "passed"
    assert json.loads(champion.tags["gate"]) and champion.tags["validated_with_forecaster"] == "1"
    # Once RL serves, the anchor runs in shadow and RL decides the orders of its gated classes.
    # The shadow starts from the live state at promotion: RL's own first order arrives on 11-20 (lead
    # time 2), so both trajectories cost the same in the two weeks before.
    for d in ("2023-11-06", "2023-11-13"):
        folder = cycle_dir(config, "rl", pd.Timestamp(d))
        pd.testing.assert_frame_equal(pd.read_parquet(folder / "costs_anchor.parquet"),
                                      pd.read_parquet(folder / "costs_live.parquet"))
    orders = pd.read_csv(cycle_dir(config, "rl", pd.Timestamp("2023-11-13")) / "orders.csv")
    assert orders["rl"].any() and not orders["rl"].all()

    client = MlflowClient()
    exp = client.get_experiment_by_name("vn2-ops-rl")
    runs = client.search_runs([exp.experiment_id])
    training = [r for r in runs if r.data.tags.get("stage") == "rl-train"]
    cycle_30 = [r for r in runs if r.data.tags.get("stage") == "cycle" and r.data.tags.get("as_of") == "2023-10-30"][0]
    assert len(training) == 2 and all(r.data.tags["mlflow.parentRunId"] == cycle_30.info.run_id for r in training)
    assert not client.search_runs([mlflow.get_experiment_by_name("Default").experiment_id])


RECIPES = {
    "cheap": {"forecasting": {"model": {"name": "catboost"}, "ensemble": {"enabled": False}},
              "model_params": {"iterations": 30, "depth": 3, "learning_rate": 0.2, "allow_writing_files": False}},
    "cheap-deeper": {"forecasting": {"model": {"name": "catboost"}, "ensemble": {"enabled": False}},
                     "model_params": {"iterations": 60, "depth": 5, "learning_rate": 0.1, "allow_writing_files": False}},
}
CHALLENGE_ENV = {"kind": "sim", "start": "2023-08-07", "end": "2023-10-23", "recipe": "cheap", "rules": {"strategy": "never"}}


@pytest.mark.slow
def test_an_operator_challenge_scores_a_recipe_and_a_won_one_is_refit_next_week(tmp_path):
    from src.forecasting.recipe import Recipe
    from src.ops import retune
    from src.ops.settings import env_dir, recipe_config

    config = load_config()
    config["_project_root"] = str(tmp_path)
    ops = OpsConfig.model_validate({"recipes": RECIPES, "envs": {"c": CHALLENGE_ENV}})
    run_env(config, ops, "c", end="2023-10-16")
    out = retune.challenge(config, ops, "c", "cheap-deeper", holdout_weeks=8)
    # Cycle A = 08-21; its orders reach the shelves from 09-11, so 6 of the 8 holdout weeks can differ.
    assert out["evidence_weeks"] == 6 and out["as_of"] == "2023-10-16"
    pending = env_dir(config, "c") / retune.PENDING
    assert pending.exists() == out["passed"]

    # A won challenge, as the operator's run would leave it (whatever this holdout said):
    recipe = Recipe.from_config(recipe_config(config, ops, "cheap-deeper"), RECIPES["cheap-deeper"]["model_params"])
    pending.write_text(json.dumps({**out, "passed": True, "delta_pct": -1.0, "ci_lo": -2.0, "ci_hi": -0.1,
                                   "recipe_dict": recipe.to_dict()}), encoding="utf-8")
    run_env(config, ops, "c")
    decision = pd.read_json(cycle_dir(config, "c", pd.Timestamp("2023-10-23")) / "decision.json", typ="series")
    assert decision["action"] == "retune"
    champion = registry.resolve(registry.model_name("c", "forecaster"), registry.CHAMPION)
    assert str(champion.version) == "2" and champion.tags["trigger"] == "retune" and champion.tags["recipe"] == "cheap-deeper"
    assert champion.tags["challenge_run_id"] == out["run_id"] and champion.tags["recipe_key"] == recipe.key()
    assert not pending.exists()
    runs = MlflowClient().search_runs([MlflowClient().get_experiment_by_name("vn2-ops-c").experiment_id],
                                      "tags.stage = 'challenge'")
    assert len(runs) == 1 and runs[0].info.run_id == out["run_id"]
