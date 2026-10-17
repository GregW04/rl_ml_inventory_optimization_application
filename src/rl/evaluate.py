"""Evaluate RL agents against classical baselines on the inventory simulation.

Rolls each policy deterministically through the *same* :class:`InventoryEnv`
across every SKU, reporting raw (un-normalized) KPIs so "did the RL agent
actually beat a base-stock heuristic?" is answerable. Logs a comparison table
and bar chart to MLflow.

Usage::

    python -m src.rl.evaluate                 # all available policies
    python -m src.rl.evaluate --agents ppo    # subset
    python -m src.rl.evaluate --forecast-run <run_id>   # test block of that run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.rl.train import agent_meta, forecast_source, prepare_eval_data
from src.simulation.baselines import BaseStockPolicy, LeadTimeBaseStockPolicy
from src.simulation.env import InventoryEnv
from src.utils import tracking
from src.utils.config import load_config


class NoOpPolicy:
    """Never orders — a floor showing what the starting inventory alone achieves."""

    def predict(self, obs, deterministic: bool = True):
        return np.array([0.0], dtype=np.float32), None


def rollout_policy(
    policy,
    episodes: dict[tuple[int, int], pd.DataFrame],
    lookup: dict,
    config: dict,
    order_caps: dict[tuple[int, int], float] | None = None,
    normalize_action: bool = False,
    residual_mode: bool = False,
    discrete_buckets: list[float] | None = None,
) -> dict[str, float]:
    """Run ``policy`` over all SKU episodes; return aggregate raw KPIs.

    ``holding_cost_per_sku`` and ``shortage_cost_per_sku`` are the two terms of
    the objective. Their sum is ``-mean_reward_per_sku``.

    The action-mode flags must match how the policy emits actions: RL agents use the
    same mode they were trained with (residual or normalized); classical baselines
    emit absolute units and rely on the env's per-SKU cap to clip.
    """
    env_cfg = config["rl"]["environment"]
    holding_cost = float(env_cfg["holding_cost"])
    shortage_cost = float(env_cfg["shortage_cost"])
    order_caps = order_caps or {}
    tot_reward = tot_demand = tot_shortage = tot_leftover = 0.0
    n_steps = 0

    for (store, product), ep in episodes.items():
        env = InventoryEnv(
            episode_data=ep,
            forecast_lookup=lookup,
            holding_cost=env_cfg["holding_cost"],
            shortage_cost=env_cfg["shortage_cost"],
            lead_time_weeks=env_cfg["lead_time_weeks"],
            max_order=env_cfg["max_order"],
            order_cap=order_caps.get((store, product)),
            normalize_action=normalize_action,
            residual_mode=residual_mode,
            residual_floor_frac=env_cfg.get("residual_floor_frac", 0.25),
            discrete_buckets=discrete_buckets,
            scale_obs_by_cap=bool(env_cfg.get("scale_obs_by_cap", False)),
        )
        obs, info = env.reset()
        if hasattr(policy, "update_info"):
            policy.update_info(info)
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if hasattr(policy, "update_info"):
                policy.update_info(info)
            # Recompute RAW reward from info so eval is invariant to any training-time
            # reward shaping (e.g. per-SKU normalization).
            tot_reward += -(holding_cost * info["leftover"] + shortage_cost * info["shortage"])
            tot_demand += float(info["true_demand"])
            tot_shortage += float(info["shortage"])
            tot_leftover += float(info["leftover"])
            n_steps += 1
            done = terminated or truncated

    n_sku = max(len(episodes), 1)
    fill_rate = 1.0 - tot_shortage / tot_demand if tot_demand > 0 else 1.0
    holding_cost_per_sku = holding_cost * tot_leftover / n_sku
    shortage_cost_per_sku = shortage_cost * tot_shortage / n_sku
    return {
        "mean_reward_per_sku": tot_reward / n_sku,
        "holding_cost_per_sku": holding_cost_per_sku,
        "shortage_cost_per_sku": shortage_cost_per_sku,
        "fill_rate": fill_rate,
        "shortage_rate": tot_shortage / tot_demand if tot_demand > 0 else 0.0,
        "avg_leftover": tot_leftover / max(n_steps, 1),
        "total_reward": tot_reward,
    }


def _load_sb3_agent(agent_type: str, config: dict):
    """Load a saved PPO/DQN agent if its artifact exists, else None."""
    from stable_baselines3 import DQN, PPO

    path = Path(config["_project_root"]) / "artifacts" / "rl" / f"{agent_type}_inventory.zip"
    if not path.exists():
        return None
    # norm_obs=False during training, so policies act on raw observations —
    # no VecNormalize reload is needed for deterministic action selection.
    return (DQN if agent_type == "dqn" else PPO).load(str(path))


def evaluate_all(
    config_path: str | None = None,
    agents: list[str] | None = None,
    forecast_run_id: str | None = None,
) -> dict:
    config = load_config(config_path)
    if forecast_run_id:
        config["rl"]["forecast_run_id"] = forecast_run_id
    setup, _ = forecast_source(config)
    episodes, lookup, order_caps = prepare_eval_data(config)
    if not episodes:
        raise RuntimeError("No RL episodes; run the forecasting pipeline first.")

    env_cfg = config["rl"]["environment"]
    residual = bool(env_cfg.get("residual_mode", False))
    normalize = bool(env_cfg.get("normalize_action", False))
    dqn_cfg = config["rl"]["agents"]["dqn"]
    dqn_buckets = dqn_cfg["residual_buckets"] if residual else dqn_cfg["order_buckets"]
    agents = agents or ["ppo", "dqn"]

    # Policy set as (name, policy, normalize_action, residual_mode, discrete_buckets):
    # classical baselines emit absolute units and rely on the env's per-SKU cap;
    # RL agents use the same action mode (residual/normalized) they were trained with.
    policies: list[tuple[str, object, bool, bool, list[float] | None]] = [
        ("no_op", NoOpPolicy(), False, False, None),
        (
            "base_stock",
            BaseStockPolicy(
                holding_cost=env_cfg["holding_cost"],
                shortage_cost=env_cfg["shortage_cost"],
                lead_time_weeks=env_cfg["lead_time_weeks"],
                max_order=float("inf"),  # env applies the per-SKU cap
            ),
            False,
            False,
            None,
        ),
        (
            "base_stock_ltd",
            LeadTimeBaseStockPolicy(
                ltd_quantile=float(env_cfg.get("ltd_base_stock_quantile", 0.83)),
                max_order=float("inf"),
            ),
            False,
            False,
            None,
        ),
    ]
    for a in agents:
        agent = _load_sb3_agent(a, config)
        if agent is not None:
            trained_on = agent_meta(config, a).get("forecast_run_id")
            if trained_on and trained_on != setup.get("run_id"):
                print(
                    f"Warning: {a} was trained on forecasting run {trained_on}, "
                    f"but is scored on the test block of {setup.get('run_id')}."
                )
            policies.append(
                (a, agent, normalize, residual, dqn_buckets if a == "dqn" else None)
            )

    rows = []
    for name, policy, norm, resid, buckets in policies:
        kpis = rollout_policy(
            policy, episodes, lookup, config,
            order_caps=order_caps, normalize_action=norm,
            residual_mode=resid, discrete_buckets=buckets,
        )
        rows.append({"policy": name, **kpis})
    results = pd.DataFrame(rows)

    tracking.setup_mlflow(config, "rl")
    with mlflow.start_run(run_name="policy_evaluation"):
        mlflow.log_params({
            "n_skus": len(episodes),
            "lead_time_weeks": env_cfg["lead_time_weeks"],
            "forecast_run_id": setup.get("run_id") or "",
            "rl_test_start": setup["resolved"]["rl_test_start"],
            "rl_test_end": setup["resolved"]["rl_test_end"],
        })
        for _, r in results.iterrows():
            for metric in [
                "mean_reward_per_sku",
                "holding_cost_per_sku",
                "shortage_cost_per_sku",
                "fill_rate",
                "shortage_rate",
                "avg_leftover",
            ]:
                mlflow.log_metric(f"{r['policy']}_{metric}", float(r[metric]))
        tracking.log_dataframe(results, artifact_path="evaluation")
        _log_comparison_chart(results)

    return {"results": results.to_dict(orient="records")}


def _log_comparison_chart(results: pd.DataFrame) -> None:
    """Bar chart: reward-per-SKU and fill rate per policy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(results["policy"], results["mean_reward_per_sku"], color="steelblue")
    ax1.set_ylabel("Mean reward / SKU")
    ax1.set_title("Reward by policy (higher = better)")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax2.bar(results["policy"], results["fill_rate"], color="seagreen")
    ax2.set_ylabel("Fill rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Fill rate by policy")
    fig.autofmt_xdate(rotation=30)
    tracking.log_figure(fig, artifact_path="evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL agents vs baselines")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--agents", nargs="*", default=["ppo", "dqn"])
    parser.add_argument(
        "--forecast-run",
        default=None,
        help="Forecasting run id whose test block to score (default: latest run on disk)",
    )
    args = parser.parse_args()
    out = evaluate_all(args.config, agents=args.agents, forecast_run_id=args.forecast_run)
    print(json.dumps(out["results"], indent=2, default=float))


if __name__ == "__main__":
    main()
