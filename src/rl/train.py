"""Train PPO or DQN inventory agents with MLflow logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from src.forecasting.preprocessing import load_initial_state, load_panel_from_config
from src.rl.agents import build_agent, wrap_vec_env
from src.simulation.env import InventoryEnv
from src.utils import tracking
from src.utils.config import load_config

# Columns needed for simulation (no feature-engineering matrix in RL path)
_PANEL_COLS = ["store", "product", "date", "sales"]
_FORECAST_COLS = [
    "store",
    "product",
    "date",
    "mu_h0",
    "sigma_h0",
    "mu_h1",
    "sigma_h1",
    "mu_h2",
    "sigma_h2",
]


class MultiSKUInventoryEnv(gym.Env):
    """
    Wrapper that samples a random store-product episode each reset.

    Episodes are pre-built once at init to avoid repeated filters on large frames.
    """

    def __init__(
        self,
        episodes: dict[tuple[int, int], pd.DataFrame],
        forecast_lookup: dict[tuple[int, int, pd.Timestamp], dict[str, float]],
        config: dict,
        discrete_buckets: list[float] | None = None,
    ):
        super().__init__()
        self.episodes = episodes
        self.forecast_lookup = forecast_lookup
        self.config = config
        self.discrete_buckets = discrete_buckets
        self.sku_keys = list(episodes.keys())
        if not self.sku_keys:
            raise ValueError("No RL episodes available; run forecasting pipeline first.")

        self._inner: InventoryEnv | None = None
        env_cfg = config["rl"]["environment"]
        max_order = float(env_cfg["max_order"])

        # Finite bounds — SB3 can crash with observation_space high=np.inf
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.array(
                [
                    max_order * 4,
                    max_order * 4,
                    max_order * 4,
                    500.0,
                    100.0,
                    500.0,
                    100.0,
                    500.0,
                    100.0,
                    50.0,
                    1000.0,
                ],
                dtype=np.float32,
            ),
            shape=(11,),
            dtype=np.float32,
        )

        if discrete_buckets is not None:
            self.action_space = gym.spaces.Discrete(len(discrete_buckets))
        else:
            self.action_space = gym.spaces.Box(
                low=0.0,
                high=max_order,
                shape=(1,),
                dtype=np.float32,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        store, product = self.sku_keys[int(rng.integers(0, len(self.sku_keys)))]
        episode_data = self.episodes[(store, product)]
        env_cfg = self.config["rl"]["environment"]
        self._inner = InventoryEnv(
            episode_data=episode_data,
            forecast_lookup=self.forecast_lookup,
            holding_cost=env_cfg["holding_cost"],
            shortage_cost=env_cfg["shortage_cost"],
            lead_time_split=env_cfg["lead_time_split"],
            max_order=env_cfg["max_order"],
            discrete_buckets=self.discrete_buckets,
        )
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)


def load_forecasts(config: dict) -> pd.DataFrame:
    """Load slim forecast parquet (mu/sigma per horizon)."""
    root = Path(config["_project_root"])
    path = root / config["forecasting"]["artifacts"]["forecasts_dir"] / "train_forecasts.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    keep = [c for c in _FORECAST_COLS if c in df.columns]
    return df[keep]


def build_forecast_lookup(forecasts: pd.DataFrame) -> dict[tuple, dict[str, float]]:
    """O(1) forecast lookup keyed by (store, product, date)."""
    if forecasts.empty:
        return {}

    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    lookup: dict[tuple, dict[str, float]] = {}
    for row in fc.itertuples(index=False):
        key = (int(row.store), int(row.product), pd.Timestamp(row.date))
        lookup[key] = {
            f"mu_h{h}": float(getattr(row, f"mu_h{h}", 0.0) or 0.0)
            for h in range(3)
        }
        for h in range(3):
            lookup[key][f"sigma_h{h}"] = float(
                getattr(row, f"sigma_h{h}", 1.0) or 1.0
            )
    return lookup


def build_episode_cache(
    panel: pd.DataFrame,
    initial_state: pd.DataFrame,
    config: dict,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Pre-build simulation episodes per SKU (avoids per-reset DataFrame filtering)."""
    env_cfg = config["rl"]["environment"]
    sim_weeks = int(env_cfg.get("simulation_weeks", 8))
    split_date = pd.Timestamp(env_cfg.get("train_split_date", "2024-04-08"))

    slim = panel[_PANEL_COLS].copy()
    slim["date"] = pd.to_datetime(slim["date"])

    init = initial_state.rename(
        columns={
            "Store": "store",
            "Product": "product",
        }
        if "Store" in initial_state.columns
        else {}
    )

    episodes: dict[tuple[int, int], pd.DataFrame] = {}
    for (store, product), grp in slim.groupby(["store", "product"], observed=True):
        ep = grp[grp["date"] >= split_date].tail(sim_weeks)
        if ep.empty:
            continue
        ep = ep.copy()
        row_init = init[
            (init["store"] == store) & (init["product"] == product)
        ]
        if not row_init.empty:
            r = row_init.iloc[0]
            ep.loc[ep.index[0], "start_inventory"] = float(
                r.get("start_inventory", r.get("end_inventory", 0))
            )
            ep.loc[ep.index[0], "in_transit_1w"] = float(r.get("in_transit_1w", 0))
            ep.loc[ep.index[0], "in_transit_2w"] = float(r.get("in_transit_2w", 0))
        episodes[(int(store), int(product))] = ep.reset_index(drop=True)

    return episodes


def prepare_rl_data(config: dict) -> tuple[dict, dict]:
    """Load slim panel + forecasts and build episode / lookup caches."""
    panel = load_panel_from_config(config)[_PANEL_COLS]
    forecasts = load_forecasts(config)
    initial_state = load_initial_state(config["data"]["initial_state_file"])
    episodes = build_episode_cache(panel, initial_state, config)
    lookup = build_forecast_lookup(forecasts)
    return episodes, lookup


class InventoryMetricsCallback(BaseCallback):
    """
    Log per-episode inventory KPIs to MLflow after each episode ends.

    Tracks: cumulative reward, fill rate, shortage rate, avg holding cost.
    """

    def __init__(self, log_freq: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards: list[float] = []
        self._episode_shortages: list[float] = []
        self._episode_leftovers: list[float] = []
        self._episode_demands: list[float] = []
        self._episode_count = 0
        self._step_rewards: list[float] = []
        self._step_shortages: list[float] = []
        self._step_leftovers: list[float] = []
        self._step_demands: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        for i, info in enumerate(infos):
            r = float(rewards[i]) if i < len(rewards) else 0.0
            self._step_rewards.append(r)
            self._step_shortages.append(float(info.get("shortage", 0)))
            self._step_leftovers.append(float(info.get("leftover", 0)))
            self._step_demands.append(float(info.get("true_demand", 0)))

            if dones[i]:
                ep_reward = float(np.sum(self._step_rewards))
                total_demand = float(np.sum(self._step_demands))
                total_shortage = float(np.sum(self._step_shortages))
                total_leftover = float(np.sum(self._step_leftovers))

                fill_rate = (
                    1.0 - total_shortage / total_demand
                    if total_demand > 0
                    else 1.0
                )
                shortage_rate = total_shortage / total_demand if total_demand > 0 else 0.0
                avg_holding = total_leftover / max(len(self._step_leftovers), 1)

                self._episode_rewards.append(ep_reward)
                self._episode_shortages.append(shortage_rate)
                self._episode_leftovers.append(avg_holding)
                self._episode_count += 1

                if self._episode_count % self.log_freq == 0:
                    n = self.log_freq
                    mlflow.log_metrics(
                        {
                            "ep_reward_mean": float(np.mean(self._episode_rewards[-n:])),
                            "fill_rate_mean": float(1.0 - np.mean(self._episode_shortages[-n:])),
                            "shortage_rate_mean": float(np.mean(self._episode_shortages[-n:])),
                            "avg_leftover_mean": float(np.mean(self._episode_leftovers[-n:])),
                        },
                        step=self._episode_count,
                    )

                # Reset step accumulators for next episode
                self._step_rewards.clear()
                self._step_shortages.clear()
                self._step_leftovers.clear()
                self._step_demands.clear()

        return True

    def summary(self) -> dict:
        if not self._episode_rewards:
            return {}
        return {
            "final_ep_reward_mean": float(np.mean(self._episode_rewards[-50:])),
            "final_fill_rate": float(1.0 - np.mean(self._episode_shortages[-50:])),
            "final_shortage_rate": float(np.mean(self._episode_shortages[-50:])),
            "total_episodes": self._episode_count,
        }

    def log_training_chart(self, agent_type: str) -> None:
        """Log reward and fill-rate curves over episodes."""
        if len(self._episode_rewards) < 2:
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        episodes = range(1, len(self._episode_rewards) + 1)

        # Smoothed with rolling average
        window = max(1, len(self._episode_rewards) // 20)
        smooth = lambda x: pd.Series(x).rolling(window, min_periods=1).mean().values

        ax1.plot(episodes, smooth(self._episode_rewards), color="steelblue")
        ax1.set_ylabel("Episode reward")
        ax1.set_title(f"{agent_type.upper()} — Training curves")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

        fill_rates = [1.0 - s for s in self._episode_shortages]
        ax2.plot(episodes, smooth(fill_rates), color="seagreen")
        ax2.set_ylabel("Fill rate")
        ax2.set_xlabel("Episode")
        ax2.set_ylim(0, 1.05)

        tracking.log_figure(fig, artifact_path="rl_charts")


def train_agent(agent_type: str, config_path: str | None = None) -> dict:
    config = load_config(config_path)
    seed = config.get("project", {}).get("seed", 42)

    episodes, forecast_lookup = prepare_rl_data(config)

    tracking.setup_mlflow(config, "rl")

    discrete_buckets = None
    if agent_type.lower() == "dqn":
        discrete_buckets = config["rl"]["agents"]["dqn"]["order_buckets"]

    def _make_env():
        return MultiSKUInventoryEnv(
            episodes=episodes,
            forecast_lookup=forecast_lookup,
            config=config,
            discrete_buckets=discrete_buckets,
        )

    vec_env = wrap_vec_env(_make_env, n_envs=1)
    agent = build_agent(agent_type, vec_env, config, seed=seed)

    timesteps = config["rl"]["agents"][agent_type.lower()].get(
        "total_timesteps", 100_000
    )

    env_cfg = config["rl"]["environment"]
    agent_cfg = config["rl"]["agents"][agent_type.lower()]

    with mlflow.start_run(run_name=f"inventory_{agent_type}"):
        mlflow.log_params({
            "agent": agent_type,
            "holding_cost": env_cfg["holding_cost"],
            "shortage_cost": env_cfg["shortage_cost"],
            "lead_time_split": str(env_cfg["lead_time_split"]),
            "max_order": env_cfg["max_order"],
            "simulation_weeks": env_cfg.get("simulation_weeks", 8),
            "total_timesteps": timesteps,
            "n_skus": len(episodes),
            "learning_rate": agent_cfg.get("learning_rate"),
            "gamma": agent_cfg.get("gamma", 0.99),
            "reward_holding_cost": env_cfg["holding_cost"],
            "reward_shortage_cost": env_cfg["shortage_cost"],
        })
        if agent_type.lower() == "dqn":
            mlflow.log_params({
                "order_buckets": str(agent_cfg.get("order_buckets")),
                "buffer_size": agent_cfg.get("buffer_size"),
            })

        callback = InventoryMetricsCallback(log_freq=50)
        agent.learn(total_timesteps=timesteps, callback=callback)

        # Summary metrics
        summary = callback.summary()
        if summary:
            mlflow.log_metrics(summary)

        # Training curves chart
        callback.log_training_chart(agent_type)

        model_dir = Path(config["_project_root"]) / "artifacts" / "rl"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{agent_type}_inventory.zip"
        agent.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        return {"model_path": str(model_path), "agent": agent_type, **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inventory RL agent")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--agent", choices=["ppo", "dqn"], default="ppo")
    args = parser.parse_args()
    result = train_agent(args.agent, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
