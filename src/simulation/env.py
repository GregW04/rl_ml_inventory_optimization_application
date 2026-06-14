"""Gymnasium inventory environment for VN2 weekly replenishment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.simulation.reward import compute_reward


def _finite_obs(
    inventory: float,
    in_transit: list[float],
    fc: dict[str, float],
    store: float,
    product: float,
    max_order: float,
) -> np.ndarray:
    """Build observation vector with NaN/Inf replaced and values clipped."""
    obs = np.array(
        [
            inventory,
            in_transit[0],
            in_transit[1],
            fc.get("mu_h0", 0.0),
            fc.get("sigma_h0", 1.0),
            fc.get("mu_h1", 0.0),
            fc.get("sigma_h1", 1.0),
            fc.get("mu_h2", 0.0),
            fc.get("sigma_h2", 1.0),
            store,
            product,
        ],
        dtype=np.float32,
    )
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=500.0, neginf=0.0)
    high = np.array(
        [max_order * 4, max_order * 4, max_order * 4]
        + [500.0, 100.0] * 3
        + [50.0, 1000.0],
        dtype=np.float32,
    )
    return np.clip(obs, 0.0, high)


class InventoryEnv(gym.Env):
    """
    Single store-product weekly inventory environment.

    State: inventory, in_transit_1w/2w, demand mu/std for h=0,1,2, store_id, product_id
    Action: order quantity (continuous) or discrete bucket index when using DQN wrapper
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_data: pd.DataFrame,
        forecasts: pd.DataFrame | None = None,
        forecast_lookup: dict[tuple[int, int, pd.Timestamp], dict[str, float]] | None = None,
        holding_cost: float = 0.2,
        shortage_cost: float = 1.0,
        lead_time_split: list[float] | None = None,
        max_order: float = 50.0,
        discrete_buckets: list[float] | None = None,
    ):
        super().__init__()
        self.episode_data = episode_data.sort_values("date").reset_index(drop=True)
        self.forecasts = forecasts
        self.forecast_lookup = forecast_lookup or {}
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.lead_time_split = lead_time_split or [0.5, 0.5]
        self.max_order = max_order
        self.discrete_buckets = discrete_buckets

        self._t = 0
        self.inventory = 0.0
        self.in_transit = [0.0, 0.0]

        obs_high = np.array(
            [max_order * 4, max_order * 4, max_order * 4]
            + [500.0, 100.0] * 3
            + [50.0, 1000.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=obs_high, shape=(11,), dtype=np.float32
        )

        if discrete_buckets is not None:
            self.action_space = spaces.Discrete(len(discrete_buckets))
            self._continuous = False
        else:
            self.action_space = spaces.Box(
                low=0.0, high=max_order, shape=(1,), dtype=np.float32
            )
            self._continuous = True

    def _get_forecast_row(self, date: pd.Timestamp) -> dict[str, float]:
        store = int(self.episode_data.loc[self._t, "store"])
        product = int(self.episode_data.loc[self._t, "product"])
        key = (store, product, pd.Timestamp(date))
        defaults = {f"mu_h{h}": 0.0 for h in range(3)}
        defaults.update({f"sigma_h{h}": 1.0 for h in range(3)})

        if key in self.forecast_lookup:
            return {**defaults, **self.forecast_lookup[key]}

        if self.forecasts is not None and not self.forecasts.empty:
            row = self.forecasts[
                (self.forecasts["store"] == store)
                & (self.forecasts["product"] == product)
                & (self.forecasts["date"] == date)
            ]
            if not row.empty:
                r = row.iloc[0]
                for h in range(3):
                    defaults[f"mu_h{h}"] = float(r.get(f"mu_h{h}", 0.0) or 0.0)
                    defaults[f"sigma_h{h}"] = float(r.get(f"sigma_h{h}", 1.0) or 1.0)
        return defaults

    def _obs(self) -> np.ndarray:
        fc = self._get_forecast_row(self.episode_data.loc[self._t, "date"])
        row = self.episode_data.loc[self._t]
        return _finite_obs(
            self.inventory,
            self.in_transit,
            fc,
            float(row["store"]),
            float(row["product"]),
            self.max_order,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        row = self.episode_data.iloc[0]
        self.inventory = float(row.get("start_inventory", row.get("end_inventory", 0)))
        self.in_transit = [
            float(row.get("in_transit_1w", 0)),
            float(row.get("in_transit_2w", 0)),
        ]
        return self._obs(), {}

    def _resolve_order(self, action: np.ndarray | int) -> float:
        if self._continuous:
            return float(np.clip(action[0], 0, self.max_order))
        return float(self.discrete_buckets[int(action)])

    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        order_qty = self._resolve_order(action)
        split = self.lead_time_split
        self.in_transit[0] += order_qty * split[0]
        self.in_transit[1] += order_qty * split[1]

        arriving = self.in_transit[0]
        self.in_transit[0] = self.in_transit[1]
        self.in_transit[1] = 0.0

        available = self.inventory + arriving
        row = self.episode_data.loc[self._t]
        true_demand = float(row["sales"]) if pd.notna(row["sales"]) else 0.0

        sales = min(available, true_demand)
        leftover = max(0.0, available - sales)
        shortage = max(0.0, true_demand - available)

        reward = compute_reward(
            leftover, shortage, self.holding_cost, self.shortage_cost
        )

        self.inventory = leftover
        self._t += 1
        terminated = self._t >= len(self.episode_data)
        truncated = False

        obs = self._obs() if not terminated else np.zeros(11, dtype=np.float32)
        info = {
            "leftover": leftover,
            "shortage": shortage,
            "sales": sales,
            "true_demand": true_demand,
            "order_qty": order_qty,
        }
        return obs, reward, terminated, truncated, info


def make_vectorized_inventory_env(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: dict,
    store: int,
    product: int,
) -> InventoryEnv:
    """Build env for one store-product series over simulation window."""
    env_cfg = config["rl"]["environment"]
    ep = panel[(panel["store"] == store) & (panel["product"] == product)].copy()
    return InventoryEnv(
        episode_data=ep,
        forecasts=forecasts,
        holding_cost=env_cfg["holding_cost"],
        shortage_cost=env_cfg["shortage_cost"],
        lead_time_split=env_cfg["lead_time_split"],
        max_order=env_cfg["max_order"],
    )
