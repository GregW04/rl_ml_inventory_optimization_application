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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from src.forecasting.periods import fetch_run_file, load_run_setup
from src.forecasting.preprocessing import load_initial_state, load_panel_from_config
from src.rl.agents import build_agent, wrap_vec_env
from src.simulation.env import InventoryEnv, OBS_DIM, _obs_high
from src.utils import tracking
from src.utils.config import load_config

# Columns needed for simulation (no feature-engineering matrix in RL path)
_PANEL_COLS = ["store", "product", "date", "sales"]
_FORECAST_COLS = ["store", "product", "date"] + [
    f"pred_q{q}_h{h}" for h in range(3) for q in (10, 50, 90)
] + [
    "pred_q50_S",
    "pred_q83_S",
    "pred_q90_S",
    "pred_q95_S",
]


def _ltd_lookup_key(pred_col: str) -> str:
    """``pred_q83_S`` → ``q83_S`` for env lookup / info."""
    return pred_col.replace("pred_", "", 1)


class MultiSKUInventoryEnv(gym.Env):
    """
    Wrapper that samples a random store-product episode each reset.

    Episodes are pre-built once at init to avoid repeated filters on large frames.
    Per-SKU order caps let one shared (normalized) action space cover SKUs whose
    demand spans two orders of magnitude.
    """

    def __init__(
        self,
        episodes: dict[tuple[int, int], pd.DataFrame],
        forecast_lookup: dict[tuple[int, int, pd.Timestamp], dict[str, float]],
        config: dict,
        order_caps: dict[tuple[int, int], float] | None = None,
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

        env_cfg = config["rl"]["environment"]
        self.order_caps = order_caps or {}
        self.normalize_action = bool(env_cfg.get("normalize_action", False))
        self.residual_mode = bool(env_cfg.get("residual_mode", False))
        # Per-SKU reward scale = that SKU's mean weekly demand (floored), so each SKU
        # contributes comparably to the gradient regardless of volume.
        self.reward_scales: dict[tuple[int, int], float] = {}
        if env_cfg.get("per_sku_reward_norm", False):
            self.reward_scales = {
                k: max(1.0, float(ep["sales"].mean() or 0.0))
                for k, ep in episodes.items()
            }
        # Bound the shared spaces by the largest per-SKU cap.
        self._global_cap = max(
            [float(env_cfg["max_order"])] + list(self.order_caps.values())
        )
        self.scale_obs_by_cap = bool(env_cfg.get("scale_obs_by_cap", False))
        # Scaled clip bounds are largest for the smallest cap (200/cap, 1000/cap).
        bound_cap = self._global_cap
        if self.scale_obs_by_cap and self.order_caps:
            bound_cap = min(float(c) for c in self.order_caps.values())

        self._inner: InventoryEnv | None = None

        # Finite bounds — SB3 can crash with observation_space high=np.inf
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=_obs_high(bound_cap, scale_by_cap=self.scale_obs_by_cap),
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        if discrete_buckets is not None:
            self.action_space = gym.spaces.Discrete(len(discrete_buckets))
        elif self.residual_mode:
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            high = 1.0 if self.normalize_action else self._global_cap
            self.action_space = gym.spaces.Box(
                low=0.0, high=high, shape=(1,), dtype=np.float32
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
            lead_time_weeks=env_cfg["lead_time_weeks"],
            max_order=env_cfg["max_order"],
            order_cap=self.order_caps.get((store, product)),
            normalize_action=self.normalize_action,
            residual_mode=self.residual_mode,
            residual_floor_frac=env_cfg.get("residual_floor_frac", 0.25),
            reward_scale=self.reward_scales.get((store, product), 1.0),
            discrete_buckets=self.discrete_buckets,
            scale_obs_by_cap=self.scale_obs_by_cap,
        )
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)


def load_forecasts(path: Path) -> pd.DataFrame:
    """Load slim forecast parquet (mu/sigma per horizon)."""
    df = pd.read_parquet(path)
    ltd_cols = [c for c in df.columns if c.startswith("pred_q") and c.endswith("_S")]
    keep = [c for c in _FORECAST_COLS if c in df.columns] + [
        c for c in ltd_cols if c not in _FORECAST_COLS
    ]
    return df[keep]


def build_forecast_lookup(forecasts: pd.DataFrame) -> dict[tuple, dict[str, float]]:
    """O(1) forecast lookup keyed by (store, product, date)."""
    if forecasts.empty:
        return {}

    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    lookup: dict[tuple, dict[str, float]] = {}
    ltd_cols = [c for c in fc.columns if c.startswith("pred_q") and c.endswith("_S")]
    for row in fc.itertuples(index=False):
        key = (int(row.store), int(row.product), pd.Timestamp(row.date))
        entry = {
            f"q{q}_h{h}": float(getattr(row, f"pred_q{q}_h{h}", 0.0) or 0.0)
            for h in range(3)
            for q in (10, 50, 90)
        }
        for col in ltd_cols:
            entry[_ltd_lookup_key(col)] = float(getattr(row, col, 0.0) or 0.0)
        lookup[key] = entry
    return lookup


def compute_order_caps(
    panel: pd.DataFrame,
    config: dict,
    cap_end: pd.Timestamp | None = None,
) -> dict[tuple[int, int], float]:
    """Per-SKU order cap set independently from each SKU's own demand history.

    cap = max(min_order_cap, ceil(mult * that SKU's peak historical weekly sales)).

    With ``order_cap_mult = 1`` the cap is simply each SKU's busiest historical
    week, so the normalized [0, 1] action maps to a sensibly scaled order range for
    every SKU rather than sharing one flat ceiling.
    """
    env_cfg = config["rl"]["environment"]
    mult = float(env_cfg.get("order_cap_mult", 1.0))
    min_cap = float(env_cfg.get("min_order_cap", 1.0))
    slim = panel[_PANEL_COLS].copy()
    slim["date"] = pd.to_datetime(slim["date"])
    if cap_end is not None:
        slim = slim[slim["date"] <= pd.Timestamp(cap_end)]
    caps: dict[tuple[int, int], float] = {}
    for (store, product), grp in slim.groupby(["store", "product"], observed=True):
        peak = float(grp["sales"].max() or 0.0)
        caps[(int(store), int(product))] = max(min_cap, float(np.ceil(mult * peak)))
    return caps


def _attach_demand_shape_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Past-only zero-week share and CV of positive weeks (strictly before each row)."""
    grp = grp.sort_values("date").reset_index(drop=True)
    n = len(grp)
    zero_share = np.zeros(n, dtype=np.float64)
    pos_cv = np.zeros(n, dtype=np.float64)
    n_past = 0
    n_zero = 0
    pos_vals: list[float] = []
    for i in range(n):
        if n_past > 0:
            zero_share[i] = n_zero / n_past
            if len(pos_vals) >= 2:
                mu = float(np.mean(pos_vals))
                if mu > 0:
                    pos_cv[i] = min(5.0, float(np.std(pos_vals)) / mu)
        sales = float(grp.loc[i, "sales"]) if pd.notna(grp.loc[i, "sales"]) else 0.0
        n_past += 1
        if sales <= 0:
            n_zero += 1
        else:
            pos_vals.append(sales)
    grp = grp.copy()
    grp["zero_share"] = zero_share
    grp["pos_cv"] = pos_cv
    return grp


def build_episodes(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lead: int,
    initial_state: pd.DataFrame | None = None,
    use_initial_state: bool = False,
    warm_weeks: int = 8,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Build per-SKU episodes for an inclusive date block."""
    slim = panel[_PANEL_COLS].copy()
    slim["date"] = pd.to_datetime(slim["date"])
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    init = None
    if use_initial_state and initial_state is not None:
        init = initial_state.rename(
            columns={"Store": "store", "Product": "product"}
            if "Store" in initial_state.columns
            else {}
        )

    episodes: dict[tuple[int, int], pd.DataFrame] = {}
    for (store, product), grp in slim.groupby(["store", "product"], observed=True):
        grp = _attach_demand_shape_features(grp)
        ep = grp[(grp["date"] >= start) & (grp["date"] <= end)]
        if len(ep) < lead + 2:
            continue
        ep = ep.copy()
        if use_initial_state and init is not None:
            row_init = init[(init["store"] == store) & (init["product"] == product)]
            if not row_init.empty:
                r = row_init.iloc[0]
                ep.loc[ep.index[0], "start_inventory"] = float(
                    r.get("start_inventory", r.get("end_inventory", 0))
                )
                ep.loc[ep.index[0], "in_transit_1w"] = float(r.get("in_transit_1w", 0))
                ep.loc[ep.index[0], "in_transit_2w"] = float(r.get("in_transit_2w", 0))
        else:
            hist = grp[grp["date"] < start].tail(warm_weeks)
            mean_d = float(hist["sales"].mean() or 0.0) if not hist.empty else 0.0
            ep.loc[ep.index[0], "start_inventory"] = mean_d
            ep.loc[ep.index[0], "in_transit_1w"] = mean_d
            ep.loc[ep.index[0], "in_transit_2w"] = mean_d
        episodes[(int(store), int(product))] = ep.reset_index(drop=True)
    return episodes


def forecast_source(config: dict) -> tuple[dict, Path]:
    """Setup record and forecasts of ``rl.forecast_run_id`` (None → latest run on disk)."""
    run_id = config["rl"].get("forecast_run_id")
    setup = load_run_setup(config, run_id)
    forecasts = fetch_run_file(config, run_id, "train_forecasts.parquet")
    if setup is None or forecasts is None:
        which = f"forecasting run {run_id}" if run_id else "the latest forecasting run on disk"
        raise RuntimeError(
            f"No period setup or forecasts found for {which}. "
            "Run the forecasting pipeline from the app or CLI first."
        )
    return setup, forecasts


def agent_meta(config: dict, agent_type: str) -> dict:
    """Sidecar of a saved agent: the forecasting run it was trained on."""
    path = Path(config["_project_root"]) / "artifacts" / "rl" / f"{agent_type}_inventory.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def prepare_block(config: dict, block: str) -> tuple[dict, dict, dict]:
    """Episodes, forecast lookup, and order caps for ``rl_train``, ``rl_val``, or ``rl_test``."""
    setup, forecasts_file = forecast_source(config)
    resolved = setup["resolved"]
    start = pd.Timestamp(resolved[f"{block}_start"])
    end = pd.Timestamp(resolved[f"{block}_end"])
    data_start = pd.Timestamp(resolved["data_start"])
    data_end = pd.Timestamp(resolved["data_end"])
    cap_end = pd.Timestamp(resolved["rl_train_end"])

    panel = load_panel_from_config(config, start=data_start, end=data_end)[_PANEL_COLS]
    forecasts = load_forecasts(forecasts_file)
    lead = int(config["rl"]["environment"]["lead_time_weeks"])

    init_date = config.get("data", {}).get("initial_state_date")
    use_init = bool(
        init_date and pd.Timestamp(init_date) == start and block == "rl_test"
    )
    initial_state = (
        load_initial_state(config["data"]["initial_state_file"]) if use_init else None
    )
    episodes = build_episodes(
        panel,
        start,
        end,
        lead,
        initial_state=initial_state,
        use_initial_state=use_init,
    )
    lookup = build_forecast_lookup(forecasts)
    caps = compute_order_caps(panel, config, cap_end=cap_end)
    return episodes, lookup, caps


def prepare_rl_data(config: dict) -> tuple[dict, dict, dict]:
    return prepare_block(config, "rl_train")


def prepare_eval_data(config: dict) -> tuple[dict, dict, dict]:
    return prepare_block(config, "rl_test")


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


class BestCheckpointCallback(BaseCallback):
    """
    Periodically score the policy on the held-out window and keep the *best* model.

    This is checkpoint selection, NOT early stopping: training always runs the full
    ``total_timesteps``; we just remember the best-on-held-out snapshot, since longer
    training here tends to drift past the good intermediate policy.
    """

    def __init__(
        self,
        eval_episodes: dict,
        eval_lookup: dict,
        order_caps: dict,
        config: dict,
        save_path: str,
        normalize_action: bool,
        residual_mode: bool,
        discrete_buckets: list[float] | None,
        eval_freq: int = 100_000,
        n_eval_skus: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_lookup = eval_lookup
        self.order_caps = order_caps
        self.config = config
        self.save_path = save_path
        self.normalize_action = normalize_action
        self.residual_mode = residual_mode
        self.discrete_buckets = discrete_buckets
        self.eval_freq = eval_freq
        self.best_reward = -np.inf
        self.best_step = -1
        self.best_holding = None
        self.best_shortage = None
        # Evaluate on the FULL SKU set by default: a random subsample under-represents
        # the few high-volume SKUs that dominate total cost, giving a misleading score.
        keys = list(eval_episodes.keys())
        if n_eval_skus is not None and len(keys) > n_eval_skus:
            rng = np.random.default_rng(0)
            keys = [keys[i] for i in rng.choice(len(keys), n_eval_skus, replace=False)]
        self._eval_subset = {k: eval_episodes[k] for k in keys}

        from src.rl.evaluate import rollout_policy
        from src.simulation.baselines import BaseStockPolicy

        env_cfg = config["rl"]["environment"]
        base_stock = BaseStockPolicy(
            holding_cost=env_cfg["holding_cost"],
            shortage_cost=env_cfg["shortage_cost"],
            lead_time_weeks=env_cfg["lead_time_weeks"],
            max_order=float("inf"),
        )
        bs_kpis = rollout_policy(
            base_stock,
            self._eval_subset,
            self.eval_lookup,
            self.config,
            order_caps=self.order_caps,
            normalize_action=False,
            residual_mode=False,
            discrete_buckets=None,
        )
        self.base_stock_reward = float(bs_kpis["mean_reward_per_sku"])
        self.best_reward = self.base_stock_reward

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        from src.rl.evaluate import rollout_policy

        kpis = rollout_policy(
            self.model, self._eval_subset, self.eval_lookup, self.config,
            order_caps=self.order_caps, normalize_action=self.normalize_action,
            residual_mode=self.residual_mode, discrete_buckets=self.discrete_buckets,
        )
        r = kpis["mean_reward_per_sku"]
        holding = kpis["holding_cost_per_sku"]
        shortage = kpis["shortage_cost_per_sku"]
        mlflow.log_metric("heldout_reward", r, step=self.num_timesteps)
        mlflow.log_metric("heldout_fill", kpis["fill_rate"], step=self.num_timesteps)
        mlflow.log_metric("heldout_holding", holding, step=self.num_timesteps)
        mlflow.log_metric("heldout_shortage", shortage, step=self.num_timesteps)
        mlflow.log_metric(
            "heldout_base_stock_reward", self.base_stock_reward, step=self.num_timesteps
        )
        mlflow.log_metric(
            "heldout_minus_base_stock", r - self.base_stock_reward, step=self.num_timesteps
        )
        if r > self.best_reward:
            self.best_reward = r
            self.best_step = self.num_timesteps
            self.best_holding = holding
            self.best_shortage = shortage
            self.model.save(self.save_path)
            if self.verbose:
                print(
                    f"[best] step={self.num_timesteps} heldout_reward={r:.3f} "
                    f"holding={holding:.3f} shortage={shortage:.3f}"
                )
        return True


def train_agent(
    agent_type: str,
    config_path: str | None = None,
    forecast_run_id: str | None = None,
) -> dict:
    config = load_config(config_path)
    if forecast_run_id:
        config["rl"]["forecast_run_id"] = forecast_run_id
    seed = config.get("project", {}).get("seed", 42)

    setup, _ = forecast_source(config)
    episodes, forecast_lookup, order_caps = prepare_rl_data(config)

    tracking.setup_mlflow(config, "rl")

    discrete_buckets = None
    if agent_type.lower() == "dqn":
        dqn_cfg = config["rl"]["agents"]["dqn"]
        discrete_buckets = (
            dqn_cfg["residual_buckets"]
            if config["rl"]["environment"].get("residual_mode", False)
            else dqn_cfg["order_buckets"]
        )

    def _make_env():
        return MultiSKUInventoryEnv(
            episodes=episodes,
            forecast_lookup=forecast_lookup,
            config=config,
            order_caps=order_caps,
            discrete_buckets=discrete_buckets,
        )

    base_vec_env = wrap_vec_env(_make_env, n_envs=1)
    # Normalize rewards only (observations are already bounded/clipped in the env).
    # Per-SKU reward scale varies wildly (sales std ≫ mean), so reward normalization
    # is the main lever for taming the noisy training signal.
    gamma = config["rl"]["agents"][agent_type.lower()].get("gamma", 0.99)
    vec_env = VecNormalize(
        base_vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=gamma,
    )
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
            "lead_time_weeks": env_cfg["lead_time_weeks"],
            "max_order": env_cfg["max_order"],
            "order_cap_mult": env_cfg.get("order_cap_mult", 2.0),
            "normalize_action": env_cfg.get("normalize_action", False),
            "scale_obs_by_cap": env_cfg.get("scale_obs_by_cap", False),
            "residual_mode": env_cfg.get("residual_mode", False),
            "rl_train_start": setup["resolved"]["rl_train_start"],
            "rl_train_end": setup["resolved"]["rl_train_end"],
            "rl_val_start": setup["resolved"]["rl_val_start"],
            "rl_val_end": setup["resolved"]["rl_val_end"],
            "forecast_run_id": setup.get("run_id") or "",
            "forecast_config_hash": setup.get("config_hash") or "",
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

        model_dir = Path(config["_project_root"]) / "artifacts" / "rl"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{agent_type}_inventory.zip"
        best_path = model_dir / f"{agent_type}_inventory_best.zip"
        # A leftover checkpoint from a previous run must not count as this run's winner.
        if best_path.exists():
            best_path.unlink()

        # Held-out checkpoint selection: keep the best-on-eval snapshot (no early stop).
        eval_episodes, eval_lookup, _ = prepare_block(config, "rl_val")
        eval_freq = max(50_000, timesteps // 40)
        best_cb = BestCheckpointCallback(
            eval_episodes, eval_lookup, order_caps, config, str(best_path),
            normalize_action=env_cfg.get("normalize_action", False),
            residual_mode=env_cfg.get("residual_mode", False),
            discrete_buckets=discrete_buckets,
            eval_freq=eval_freq,
            verbose=1,
        )
        metrics_cb = InventoryMetricsCallback(log_freq=50)
        agent.learn(
            total_timesteps=timesteps,
            callback=CallbackList([metrics_cb, best_cb]),
        )

        # Summary metrics
        summary = metrics_cb.summary()
        if summary:
            mlflow.log_metrics(summary)

        # Training curves chart
        metrics_cb.log_training_chart(agent_type)

        # Deploy only if this run saved a checkpoint that beat base-stock.
        beat_base_stock = best_cb.best_step >= 0 and best_path.exists()
        deployed_policy = agent_type if beat_base_stock else "base_stock"
        mlflow.log_param("deployed_policy", deployed_policy)
        mlflow.log_metric("heldout_base_stock_reward", best_cb.base_stock_reward)

        if beat_base_stock:
            best_agent = type(agent).load(str(best_path))
            best_agent.save(str(model_path))
            mlflow.log_metric("best_heldout_reward", best_cb.best_reward)
            mlflow.log_metric("best_heldout_step", best_cb.best_step)
            summary["best_heldout_reward"] = best_cb.best_reward
            summary["best_heldout_step"] = best_cb.best_step
            if best_cb.best_holding is not None:
                mlflow.log_metric("best_heldout_holding", best_cb.best_holding)
                mlflow.log_metric("best_heldout_shortage", best_cb.best_shortage)
                summary["best_heldout_holding"] = best_cb.best_holding
                summary["best_heldout_shortage"] = best_cb.best_shortage
            mlflow.log_artifact(str(model_path))
        elif model_path.exists():
            model_path.unlink()

        sidecar = {
            "forecast_run_id": setup.get("run_id"),
            "rl_run_id": mlflow.active_run().info.run_id,
            "total_timesteps": timesteps,
            "deployed_policy": deployed_policy,
            "heldout_base_stock_reward": best_cb.base_stock_reward,
        }
        if beat_base_stock:
            sidecar["best_heldout_reward"] = best_cb.best_reward
            sidecar["best_heldout_step"] = best_cb.best_step
        model_path.with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2),
            encoding="utf-8",
        )
        summary["deployed_policy"] = deployed_policy
        summary["heldout_base_stock_reward"] = best_cb.base_stock_reward

        # Persist reward-normalization stats so eval/inference reload the same scale.
        vecnorm_path = model_dir / f"{agent_type}_vecnormalize.pkl"
        vec_env.save(str(vecnorm_path))
        mlflow.log_artifact(str(vecnorm_path))

        return {
            "model_path": str(model_path),
            "vecnormalize_path": str(vecnorm_path),
            "agent": agent_type,
            **summary,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inventory RL agent")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--agent", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument(
        "--forecast-run",
        default=None,
        help="Forecasting run id to train on (default: latest run on disk)",
    )
    args = parser.parse_args()
    result = train_agent(args.agent, args.config, forecast_run_id=args.forecast_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
