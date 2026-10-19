"""Gymnasium inventory environment for VN2 weekly replenishment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.simulation.reward import compute_reward


OBS_DIM = 15
# On-hand, two pipeline slots, and q10/q50/q90 × h0/h1/h2. These share units
# with the fraction action when ``scale_obs_by_cap`` is on.
OBS_SCALED_DIM = 12
# Tail: base-stock order / cap, past zero-week share, past positive-week CV.
OBS_TAIL_DIM = 3
# Forecast quantiles exposed in the observation (per horizon h0/h1/h2).
_FC_QUANTILES = (10, 50, 90)
# Lead-time demand quantiles (info only — not part of the 14-dim observation).
_LTD_QUANTILE_KEYS = ("q50_S", "q83_S", "q90_S", "q95_S")


def _obs_high(order_cap: float, scale_by_cap: bool = False) -> np.ndarray:
    """Upper bounds for the observation Box, sized to the (per-SKU) order cap.

    With ``scale_by_cap``, inventory, pipeline, and weekly quantiles are in
    units of the cap, so the same raw ceiling is divided by that cap.
    """
    cap = max(float(order_cap), 1e-6)
    inv_high = max(cap * 4.0, 200.0)
    demand_high = max(cap * 4.0, 1000.0)
    if scale_by_cap:
        inv_high /= cap
        demand_high /= cap
    return np.array(
        [inv_high, inv_high, inv_high]   # inventory + 2 in-transit slots
        + [demand_high] * 9              # q10/q50/q90 for h0,h1,h2
        + [1.0, 1.0, 5.0],              # base_order/cap, zero_share, pos_cv
        dtype=np.float32,
    )


def _inventory_forecast_obs(
    inventory: float,
    in_transit: list[float],
    fc: dict[str, float],
    order_cap: float,
    scale_by_cap: bool = False,
) -> np.ndarray:
    """Build observation vector with NaN/Inf replaced and values clipped.

    The forecast block is the predicted quantiles (q10/q50/q90) for h0/h1/h2 —
    fed directly rather than collapsed to mu/sigma, so the policy sees the full
    (skewed) predicted distribution. In-transit exposes the first two pipeline slots.

    When ``scale_by_cap`` is set, on-hand inventory, the pipeline slots, and those
    weekly quantiles are divided by the SKU order cap. The fraction action is
    already in that unit, so a value of 1 means "one cap" of stock or demand.
    """
    it0 = in_transit[0] if len(in_transit) > 0 else 0.0
    it1 = in_transit[1] if len(in_transit) > 1 else 0.0
    quants = [
        fc.get(f"q{q}_h{h}", 0.0) for h in range(3) for q in _FC_QUANTILES
    ]
    obs = np.array([inventory, it0, it1] + quants, dtype=np.float32)
    np.nan_to_num(obs, copy=False, nan=0.0, posinf=500.0, neginf=0.0)
    if scale_by_cap:
        obs[:OBS_SCALED_DIM] /= max(float(order_cap), 1e-6)
    high = _obs_high(order_cap, scale_by_cap=scale_by_cap)
    return np.clip(obs, 0.0, high[:OBS_SCALED_DIM])


class InventoryEnv(gym.Env):
    """
    Single store-product weekly inventory environment.

    State: inventory, pipeline, forecast quantiles (12), then base-stock order/cap,
    past zero-week share, and past positive-week CV (3).
    With ``scale_obs_by_cap``, the first twelve entries are fractions of the SKU cap.
    Action: order quantity (continuous) or discrete bucket index when using DQN wrapper

    Replenishment: an order placed at week t arrives **in full** at t + lead_time_weeks
    (no same-week receipt, no partial split). The pipeline is a FIFO of length
    lead_time_weeks; the new order enters the tail and is received once it reaches the head.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode_data: pd.DataFrame,
        forecasts: pd.DataFrame | None = None,
        forecast_lookup: dict[tuple[int, int, pd.Timestamp], dict[str, float]] | None = None,
        holding_cost: float = 0.2,
        shortage_cost: float = 1.0,
        lead_time_weeks: int = 2,
        max_order: float = 50.0,
        discrete_buckets: list[float] | None = None,
        order_cap: float | None = None,
        normalize_action: bool = False,
        residual_mode: bool = False,
        residual_floor_frac: float = 0.25,
        reward_scale: float = 1.0,
        scale_obs_by_cap: bool = False,
    ):
        super().__init__()
        self.reward_scale = float(reward_scale) if reward_scale else 1.0
        self.episode_data = episode_data.sort_values("date").reset_index(drop=True)
        self.forecasts = forecasts
        self.forecast_lookup = forecast_lookup or {}
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.lead_time_weeks = int(lead_time_weeks)
        if self.lead_time_weeks < 1:
            raise ValueError("lead_time_weeks must be >= 1")
        self.max_order = max_order
        # Per-SKU ceiling on order quantity; defaults to the flat max_order.
        self.order_cap = float(order_cap) if order_cap is not None else float(max_order)
        self.normalize_action = normalize_action
        self.discrete_buckets = discrete_buckets
        self.residual_mode = residual_mode
        self.residual_floor_frac = float(residual_floor_frac)
        self.scale_obs_by_cap = bool(scale_obs_by_cap)
        self._obs_scale = float(self.order_cap if self.scale_obs_by_cap else 1.0)

        from src.simulation.baselines import BaseStockPolicy

        # Base-stock order is both an observation feature and the residual anchor.
        self._baseline = BaseStockPolicy(
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
            lead_time_weeks=self.lead_time_weeks,
            max_order=float("inf"),  # env applies the per-SKU cap
        )
        self._baseline.update_info({"obs_scale": self._obs_scale})

        self._t = 0
        self.inventory = 0.0
        self.in_transit = [0.0] * self.lead_time_weeks

        self.observation_space = spaces.Box(
            low=0.0,
            high=_obs_high(self.order_cap, scale_by_cap=self.scale_obs_by_cap),
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        if discrete_buckets is not None:
            self.action_space = spaces.Discrete(len(discrete_buckets))
            self._continuous = False
        elif residual_mode:
            # Signed correction centered on base-stock: action in [-1, 1].
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self._continuous = True
        else:
            # Normalized: action is a fraction in [0, 1] scaled by order_cap.
            high = 1.0 if normalize_action else self.order_cap
            self.action_space = spaces.Box(
                low=0.0, high=high, shape=(1,), dtype=np.float32
            )
            self._continuous = True

    def _get_forecast_row(self, date: pd.Timestamp) -> dict[str, float]:
        store = int(self.episode_data.loc[self._t, "store"])
        product = int(self.episode_data.loc[self._t, "product"])
        key = (store, product, pd.Timestamp(date))
        defaults = {
            f"q{q}_h{h}": 0.0 for h in range(3) for q in _FC_QUANTILES
        }
        for k in _LTD_QUANTILE_KEYS:
            defaults[k] = 0.0

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
                    for q in _FC_QUANTILES:
                        defaults[f"q{q}_h{h}"] = float(
                            r.get(f"pred_q{q}_h{h}", 0.0) or 0.0
                        )
                for k in _LTD_QUANTILE_KEYS:
                    pred_col = f"pred_{k}"
                    if pred_col in r.index:
                        defaults[k] = float(r.get(pred_col, 0.0) or 0.0)
        return defaults

    def _ltd_info(self, fc: dict[str, float]) -> dict[str, float]:
        """Lead-time demand quantiles for policies that read ``info``, not ``obs``.

        Quantiles here stay in units. ``obs_scale`` is the divisor applied to the
        observation, so a policy can multiply the scaled entries back to units.
        """
        info = {k: float(fc.get(k, 0.0)) for k in _LTD_QUANTILE_KEYS}
        info["obs_scale"] = self._obs_scale
        return info

    def _obs(self) -> np.ndarray:
        fc = self._get_forecast_row(self.episode_data.loc[self._t, "date"])
        row = self.episode_data.loc[self._t]
        core = _inventory_forecast_obs(
            self.inventory,
            self.in_transit,
            fc,
            self.order_cap,
            scale_by_cap=self.scale_obs_by_cap,
        )
        pad = np.zeros(OBS_TAIL_DIM, dtype=np.float32)
        base_qty = self._baseline.order_quantity(np.concatenate([core, pad]))
        cap = max(float(self.order_cap), 1e-6)
        base_frac = float(np.clip(base_qty / cap, 0.0, 1.0))
        zero_share = float(row.get("zero_share", 0.0) or 0.0)
        pos_cv = float(row.get("pos_cv", 0.0) or 0.0)
        tail = np.array([base_frac, zero_share, min(5.0, pos_cv)], dtype=np.float32)
        high = _obs_high(self.order_cap, scale_by_cap=self.scale_obs_by_cap)
        return np.clip(np.concatenate([core, tail]), 0.0, high)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        row = self.episode_data.iloc[0]
        self.inventory = float(row.get("start_inventory", row.get("end_inventory", 0)))
        # Seed the FIFO pipeline: slot 0 = arriving in 1 week, slot 1 = in 2 weeks, ...
        self.in_transit = [0.0] * self.lead_time_weeks
        if self.lead_time_weeks > 0:
            self.in_transit[0] = float(row.get("in_transit_1w", 0))
        if self.lead_time_weeks > 1:
            self.in_transit[1] = float(row.get("in_transit_2w", 0))
        fc = self._get_forecast_row(self.episode_data.loc[self._t, "date"])
        return self._obs(), self._ltd_info(fc)

    def _resolve_order(self, action: np.ndarray | int) -> float:
        """Map a raw policy action to an order quantity (units), capped per SKU.

        - ``residual_mode``: action is a signed correction on the base-stock order:
          ``order = base + action * scale``, ``scale = max(base, floor_frac*cap)``.
          A zero action reproduces the base-stock policy exactly.
        - ``normalize_action``: continuous/discrete actions are fractions of order_cap.
        - otherwise: actions are absolute unit counts.
        """
        if self.residual_mode:
            base = self._baseline.order_quantity(self._obs())
            scale = max(base, self.residual_floor_frac * self.order_cap)
            frac = float(action[0]) if self._continuous else float(
                self.discrete_buckets[int(action)]
            )
            qty = base + frac * scale
        elif self._continuous:
            raw = float(action[0])
            qty = raw * self.order_cap if self.normalize_action else raw
        else:
            bucket = float(self.discrete_buckets[int(action)])
            qty = bucket * self.order_cap if self.normalize_action else bucket
        return float(np.clip(qty, 0.0, self.order_cap))

    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        order_qty = self._resolve_order(action)

        # Receive the shipment due this week (ordered lead_time_weeks ago), then
        # advance the pipeline and enqueue the new order at the tail so it arrives
        # in FULL after the full lead time — never the same week.
        arriving = self.in_transit[0]
        self.in_transit = self.in_transit[1:] + [0.0]
        self.in_transit[-1] += order_qty

        available = self.inventory + arriving
        row = self.episode_data.loc[self._t]
        true_demand = float(row["sales"]) if pd.notna(row["sales"]) else 0.0

        sales = min(available, true_demand)
        leftover = max(0.0, available - sales)
        shortage = max(0.0, true_demand - available)

        reward = compute_reward(
            leftover, shortage, self.holding_cost, self.shortage_cost
        )
        # Per-SKU reward normalization: scale so each SKU contributes comparably to
        # the gradient regardless of its demand magnitude. `info` stays raw so eval
        # can always recover true cost.
        reward /= self.reward_scale

        self.inventory = leftover
        self._t += 1
        terminated = self._t >= len(self.episode_data)
        truncated = False

        obs = self._obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        fc = self._get_forecast_row(self.episode_data.loc[self._t, "date"]) if not terminated else {}
        info = {
            "leftover": leftover,
            "shortage": shortage,
            "sales": sales,
            "true_demand": true_demand,
            "order_qty": order_qty,
            **self._ltd_info(fc),
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
        lead_time_weeks=env_cfg["lead_time_weeks"],
        max_order=env_cfg["max_order"],
        scale_obs_by_cap=bool(env_cfg.get("scale_obs_by_cap", False)),
    )
