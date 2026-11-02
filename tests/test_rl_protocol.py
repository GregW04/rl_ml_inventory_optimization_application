"""
RL protocol on real forecasts (read-only), with MLflow (conftest's isolated store) and run
folders in a temp dir:
training loads only rl_train and rl_val, seeds group into one configuration, the
champion is scored once on rl_test, and a second test score needs an override.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from src.rl import evaluate as rl_eval
from src.rl import train as rl_train
from src.rl.agents import model_file
from src.rl.selection import champion, leaderboard
from src.rl.spec import build_spec, default_state
from src.utils.config import load_config


@pytest.mark.slow
@pytest.mark.needs_local_run
@pytest.mark.parametrize(
    ("agent", "training_forecasts"),
    [("ppo", "unbiased_by_class"), ("grpo", "as_is")],
)
def test_train_validate_select_and_lock_test(
    tmp_path: Path, monkeypatch, local_forecast_run: str, agent: str, training_forecasts: str
):
    config = load_config()
    # RL run folders go to a temp cache; the real forecasting run is linked in read-only.
    real_cache = Path(config["_project_root"]) / config["mlflow"]["cache_dir"]
    config["mlflow"]["cache_dir"] = str(tmp_path / "cache")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / local_forecast_run).symlink_to(real_cache / local_forecast_run)

    blocks: list[str] = []
    real_block = rl_train.block_episodes

    def few_episodes(cfg, data, block):
        blocks.append(block)
        eps = real_block(cfg, data, block)
        return {k: eps[k] for k in list(eps)[:15]}

    monkeypatch.setattr(rl_train, "block_episodes", few_episodes)
    monkeypatch.setattr(rl_eval, "block_episodes", few_episodes)

    data = rl_train.load_run_data(config, local_forecast_run)
    spec = build_spec(
        config, str(data.run_id), "residual", agent,
        default_state("residual", data.measure_ids), data.default_anchor,
        iterations=2,
        training_forecasts=training_forecasts,
    )
    spec.agent_params.update({"batch_windows": 8, "minibatch_size": 64, "group_size": 2})
    results = rl_train.train_seeds(spec, [42, 7], config)

    assert blocks == ["rl_train", "rl_val"]
    assert len({r["config_key"] for r in results}) == 1
    for r in results:
        run_dir = Path(r["run_dir"])
        files = {p.name for p in run_dir.iterdir()}
        assert {model_file(agent), "spec.json", "val_episodes.parquet", "insample_episodes.parquet"} <= files
        assert "val_anchor_cost_eur_per_sku" in r and "final_insample_vs_anchor_pct" in r
        assert "val_units_vs_anchor_pct" in r
        assert (run_dir / "training_bias_factors.csv").exists() == (training_forecasts != "as_is")
        # The saved training-sample episodes are the curve's last point, on the final weights.
        ins = pd.read_parquet(run_dir / "insample_episodes.parquet")
        pct = 100.0 * (ins["cost_eur"].sum() / ins["anchor_cost_eur"].sum() - 1.0)
        assert pct == pytest.approx(r["final_insample_vs_anchor_pct"], abs=1e-6)
        history = mlflow.tracking.MlflowClient().get_metric_history(r["rl_run_id"], "insample_units_vs_anchor_pct")
        assert len(history) >= 2

    board = leaderboard(config, str(data.run_id))
    rl_row = board[(board["kind"] == "rl") & board["gate"].isna()].iloc[0]
    assert rl_row["n_seeds"] == 2 and rl_row["deploy_run_id"] == results[0]["rl_run_id"]
    assert rl_row["training_forecasts"] == training_forecasts and pd.notna(rl_row["units_vs_anchor"])
    assert pd.notna(rl_row["insample_pct"]) and pd.notna(rl_row["val_pct"]) and pd.notna(rl_row["val_pct_sd"])
    assert set(board["kind"]) == {"rl", "anchor"}
    # A gated candidate exists only when the configuration won some demand classes and lost others.
    for _, g in board[board["gate"].notna()].iterrows():
        assert g["config_key"] == f"{rl_row['config_key']}+gate" and g["val_mean"] <= rl_row["val_mean"]
    champ = champion(board)
    assert champ is not None

    first = rl_eval.score_test(config, str(data.run_id))
    assert blocks[-1] == "rl_test" and first["post_hoc"] is False
    assert first["gate"] == (champ["gate"] if champ["kind"] == "rl" else None)
    with pytest.raises(rl_eval.LockedTestError):
        rl_eval.score_test(config, str(data.run_id))
    again = rl_eval.score_test(config, str(data.run_id), override_lock=True)
    assert again["post_hoc"] is True
    tags = mlflow.get_run(again["test_run_id"]).data.tags
    assert tags["stage"] == "test" and tags["post_hoc"] == "true"
