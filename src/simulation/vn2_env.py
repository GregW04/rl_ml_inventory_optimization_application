"""Gymnasium environment: one VN2 8-week window, residual on the anchor or raw orders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.simulation.anchor import AnchorDecision, LostSalesAnchor
from src.simulation.vn2 import InventoryState, VN2Rules, step_one_week

if TYPE_CHECKING:
    from src.rl.spec import PolicySpec

_OBS_HIGH = 10.0


def class_one_hot(code) -> dict[str, float]:
    """``class_smooth`` … ``class_lumpy`` from a demand-pattern code (all 0 when unknown)."""
    try:
        c = int(code)
    except (TypeError, ValueError):
        c = -1
    return {
        "class_smooth": float(c == 0),
        "class_erratic": float(c == 1),
        "class_intermittent": float(c == 2),
        "class_lumpy": float(c == 3),
    }


def _fc(fc: dict, key: str) -> float:
    v = fc.get(key, 0.0)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


class VN2Env(gym.Env):
    """
    Six order rounds of one SKU window; reward is the negative euro cost of weeks 3–8.

    Actions (``spec.method``):

    - ``residual``: ``order = max(0, round(anchor + f · max(anchor, floor_frac · scale)))``
      with ``f ∈ [−1, 1]`` (PPO) or a bucket value (DQN); ``f = 0`` is the anchor.
    - ``raw``: ``order = round(f · order_cap_mult · scale)`` with ``f ∈ [0, 1]``; PPO's
      ``a ∈ [−1, 1]`` maps to ``f = (a + 1) / 2``, DQN picks a bucket value.

    ``scale`` is the SKU's peak weekly sales in the training history. The observation
    lists ``spec.state`` in order; features in units of demand are divided by ``scale``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        spec: PolicySpec,
        episode_data: pd.DataFrame,
        forecast_lookup: dict[tuple[int, int, pd.Timestamp], dict[str, float]],
        demand_scale: float = 1.0,
        rules: VN2Rules | None = None,
    ):
        super().__init__()
        from src.rl.spec import SIGNED_FEATURES, UNITLESS_FEATURES

        self.spec = spec
        self.rules = rules or VN2Rules()
        self.anchor = LostSalesAnchor(spec.anchor, self.rules)
        self.episode_data = episode_data.sort_values("date").reset_index(drop=True)
        if len(self.episode_data) != self.rules.window_weeks:
            raise ValueError(
                f"episode must have {self.rules.window_weeks} weeks, got {len(self.episode_data)}"
            )
        self.forecast_lookup = forecast_lookup
        self.demand_scale = max(float(demand_scale), 1.0)
        self._unitless = [name in UNITLESS_FEATURES for name in spec.state]

        self._t = 0
        self.inventory = 0.0
        self.in_transit = [0.0] * self.rules.lead_time_weeks
        self._scored_cost = 0.0
        self._last_h0 = 0.0
        self._prev_week_sales = 0.0
        self._decision: AnchorDecision | None = None

        low = np.array([-_OBS_HIGH if n in SIGNED_FEATURES else 0.0 for n in spec.state], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=np.full(len(spec.state), _OBS_HIGH, dtype=np.float32), dtype=np.float32
        )
        if spec.agent == "dqn":
            self.action_space = spaces.Discrete(len(spec.buckets))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _sku_date(self) -> tuple[int, int, pd.Timestamp]:
        row = self.episode_data.iloc[self._t]
        return int(row["store"]), int(row["product"]), pd.Timestamp(row["date"])

    def _get_fc(self) -> dict[str, float]:
        return self.forecast_lookup.get(self._sku_date(), {})

    def _arrivals(self) -> tuple[float, float]:
        t = self.in_transit
        return (t[0] if t else 0.0), (t[1] if len(t) > 1 else 0.0)

    def _decide(self, fc: dict) -> AnchorDecision:
        """Anchor decision for the current week, computed once per round."""
        a0, a1 = self._arrivals()
        self._decision = self.anchor.decide(self.inventory, a0, a1, fc)
        return self._decision

    def _obs(self, fc: dict) -> np.ndarray:
        row = self.episode_data.iloc[self._t]
        dec = self._decision
        a0, a1 = self._arrivals()
        values = {
            "on_hand": self.inventory,
            "in_transit_w1": a0,
            "in_transit_w2": a1,
            "anchor_order": float(dec.order),
            "projected_stock": dec.projected_stock_at_arrival,
            "projected_lost": dec.projected_lost_sales_before_arrival,
            "forecast_error": self._prev_week_sales - self._last_h0 if self._t > 0 else 0.0,
            "round_frac": (self._t + 1) / self.rules.n_order_rounds,
            "zero_share": float(row.get("zero_share", 0.0) or 0.0),
            "pos_cv": min(5.0, float(row.get("pos_cv", 0.0) or 0.0)),
            **class_one_hot(row.get("dp_sbc_class", -1)),
        }
        scale = self.demand_scale
        vec = np.array(
            [
                (values[n] if n in values else _fc(fc, n)) / (1.0 if unitless else scale)
                for n, unitless in zip(self.spec.state, self._unitless)
            ],
            dtype=np.float32,
        )
        vec = np.nan_to_num(vec, nan=0.0, posinf=_OBS_HIGH, neginf=-_OBS_HIGH)
        return np.clip(vec, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = 0
        self._scored_cost = 0.0
        row = self.episode_data.iloc[0]
        self.inventory = float(row.get("end_inventory", row.get("on_hand", 0)) or 0)
        lead = self.rules.lead_time_weeks
        self.in_transit = [0.0] * lead
        if lead > 0:
            self.in_transit[0] = float(row.get("in_transit_1w", 0) or 0)
        if lead > 1:
            self.in_transit[1] = float(row.get("in_transit_2w", 0) or 0)
        self._prev_week_sales = 0.0
        fc = self._get_fc()
        self._last_h0 = _fc(fc, self.spec.anchor.h0)
        self._decide(fc)
        return self._obs(fc), {"demand_scale": self.demand_scale}

    def unrounded_order(self, action) -> float:
        """Order for ``action`` in the current round, before rounding and the floor at zero."""
        if self.spec.agent == "dqn":
            f = float(self.spec.buckets[int(np.asarray(action).reshape(-1)[0])])
        else:
            f = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        if self.spec.method == "raw":
            if self.spec.agent != "dqn":
                f = (f + 1.0) / 2.0
            return f * self.spec.order_cap_mult * self.demand_scale
        anchor_q = float(self._decision.order)
        return anchor_q + f * max(anchor_q, self.spec.floor_frac * self.demand_scale)

    def resolve_order(self, action) -> float:
        """Order quantity for ``action`` in the current round."""
        return float(max(0, round(self.unrounded_order(action))))

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        fc = self._get_fc()
        self._last_h0 = _fc(fc, self.spec.anchor.h0)
        round_idx = self._t + 1
        order = self.resolve_order(action) if round_idx <= self.rules.n_order_rounds else 0.0
        row = self.episode_data.iloc[self._t]
        demand = float(row["sales"]) if pd.notna(row["sales"]) else 0.0

        new_state, wr = step_one_week(InventoryState(self.inventory, list(self.in_transit)), demand, order, self.rules)
        self.inventory = new_state.on_hand
        self.in_transit = new_state.pipeline
        self._prev_week_sales = demand

        scored = round_idx >= self.rules.score_from_week
        if scored:
            self._scored_cost += wr.total_eur
        self._t += 1
        terminated = self._t >= len(self.episode_data)
        info = {
            "scored_cost_eur": self._scored_cost,
            "week_cost_eur": wr.total_eur,
            "order_qty": order,
            "demand": wr.demand,
            "leftover": wr.leftover,
            "shortage": wr.shortage,
        }
        if terminated:
            obs = np.zeros(len(self.spec.state), dtype=np.float32)
        else:
            fc_next = self._get_fc()
            self._decide(fc_next)
            obs = self._obs(fc_next)
        return obs, (-wr.total_eur if scored else 0.0), terminated, False, info
