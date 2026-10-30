"""
RL protocol on real forecasts (read-only), with MLflow and run folders in a temp dir:
training loads only rl_train and rl_val, seeds group into one configuration, the
champion is scored once on rl_test, and a second test score needs an override.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from src.rl import evaluate as rl_eval
from src.rl import train as rl_train
from src.rl.selection import champion, leaderboard
from src.rl.agents import model_file
from src.rl.spec import build_spec, default_state
from src.utils.config import get_project_root, load_config

ROOT = get_project_root()


@pytest.mark.skipif(
    not (ROOT / "artifacts" / "forecasts" / "forecast_setup.json").exists(),
    reason="needs a forecasting run on disk",
)
@pytest.mark.parametrize("agent", ["dqn", "ppo", "grpo"])
def test_train_validate_select_and_lock_test(tmp_path: Path, monkeypatch, agent: str):
    config = load_config()
    db = f"sqlite:///{tmp_path / 'mlruns.db'}"
    config["rl"]["mlflow"]["tracking_uri"] = db
    config["rl"]["artifacts"] = {"runs_dir": str(tmp_path / "runs")}
    mlflow.set_tracking_uri(db)

    blocks: list[str] = []
    real_block = rl_train.block_episodes

    def few_episodes(cfg, data, block):
        blocks.append(block)
        eps = real_block(cfg, data, block)
        return {k: eps[k] for k in list(eps)[:15]}

    monkeypatch.setattr(rl_train, "block_episodes", few_episodes)
    monkeypatch.setattr(rl_eval, "block_episodes", few_episodes)

    data = rl_train.load_run_data(config, None)
    spec = build_spec(
        config, str(data.run_id), "residual", agent,
        default_state("residual", data.measure_ids), data.default_anchor,
        total_timesteps=300 if agent == "dqn" else None, iterations=None if agent == "dqn" else 2,
    )
    if agent == "dqn":
        spec.agent_params["learning_starts"] = 50
    else:
        spec.agent_params.update({"batch_windows": 8, "minibatch_size": 64, "group_size": 2})
    results = rl_train.train_seeds(spec, [42, 7], config)

    assert blocks == ["rl_train", "rl_val"]
    assert len({r["config_key"] for r in results}) == 1
    for r in results:
        run_dir = Path(r["run_dir"])
        assert {model_file(agent), "spec.json", "val_episodes.parquet"} <= {p.name for p in run_dir.iterdir()}
        assert "val_anchor_cost_eur_per_sku" in r and "final_insample_vs_anchor_pct" in r

    board = leaderboard(config, str(data.run_id))
    rl_row = board[board["kind"] == "rl"].iloc[0]
    assert rl_row["n_seeds"] == 2 and rl_row["deploy_run_id"] == results[0]["rl_run_id"]
    assert set(board["kind"]) == {"rl", "anchor"}
    assert champion(board) is not None

    first = rl_eval.score_test(config, str(data.run_id))
    assert blocks[-1] == "rl_test" and first["post_hoc"] is False
    with pytest.raises(rl_eval.LockedTestError):
        rl_eval.score_test(config, str(data.run_id))
    again = rl_eval.score_test(config, str(data.run_id), override_lock=True)
    assert again["post_hoc"] is True
    tags = mlflow.get_run(again["test_run_id"]).data.tags
    assert tags["stage"] == "test" and tags["post_hoc"] == "true"
