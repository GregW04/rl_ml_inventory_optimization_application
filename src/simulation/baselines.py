"""Classical inventory baseline policies for benchmarking the RL agents.

These operate on the same observation vector the RL agents see, so they can be
dropped into the identical :class:`~src.simulation.env.InventoryEnv` for an
apples-to-apples comparison.

Observation layout (see ``InventoryEnv._obs``)::

    [0] inventory            [1] in_transit_1w     [2] in_transit_2w
    [3] q10_h0 [4] q50_h0 [5] q90_h0  ... h1, h2
    [12] base_stock_order/cap  [13] zero_share  [14] pos_cv

Policies read only ``[0:12]``; the tail is for the RL agent.

When the env is constructed with ``scale_obs_by_cap``, entries ``[0:12]`` are
fractions of that SKU's order cap. ``info["obs_scale"]`` is the cap (or 1 when
scaling is off); multiplying those entries by it restores units. Lead-time
demand quantiles (``q50_S``, ``q83_S``, …) stay in units on ``info``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.simulation.env import OBS_SCALED_DIM


def restore_obs_units(obs: np.ndarray, obs_scale: float) -> np.ndarray:
    """Copy ``obs`` and put inventory, pipeline, and weekly quantiles back in units."""
    out = np.asarray(obs, dtype=np.float64).reshape(-1).copy()
    scale = float(obs_scale)
    if scale != 1.0:
        out[:OBS_SCALED_DIM] *= scale
    return out


def critical_ratio(holding_cost: float, shortage_cost: float) -> float:
    """Newsvendor critical ratio Cs / (Cs + Ch) — the optimal in-stock probability."""
    denom = shortage_cost + holding_cost
    if denom <= 0:
        return 0.5
    return float(shortage_cost / denom)


def ltd_quantile_key(quantile: float) -> str:
    """Map quantile level to lookup/info key, e.g. 0.83 → ``q83_S``."""
    return f"q{int(round(quantile * 100))}_S"


class BaseStockPolicy:
    """
    Order-up-to-S (base-stock) policy with a newsvendor safety stock.

    Targets demand over the protection period (lead time + one review week) using
    the forecast mean, plus a safety term ``z * sqrt(sum sigma_h^2)`` where ``z``
    is the critical-ratio quantile of the standard normal. Orders the gap between
    the target level S and the current inventory position (on-hand + in-transit).

    Exposes a ``predict`` shim matching the Stable-Baselines3 agent API so the
    evaluation harness can treat it uniformly.
    """

    def __init__(
        self,
        holding_cost: float = 0.2,
        shortage_cost: float = 1.0,
        lead_time_weeks: int = 2,
        max_order: float = 50.0,
    ):
        self.lead_time_weeks = int(lead_time_weeks)
        self.max_order = float(max_order)
        self.cr = critical_ratio(holding_cost, shortage_cost)
        # Order-up-to the critical-ratio quantile of CUMULATIVE demand over the
        # protection window. Per-horizon upper quantiles don't sum to the cumulative
        # quantile, so we take sum(median) + safety with spread combined in quadrature.
        # (Q90-Q50) ≈ 1.2816·σ_upper for a normal, so weight = z_cr / 1.2816.
        self._wq = float(norm.ppf(self.cr)) / 1.2816
        self._obs_scale = 1.0

    def update_info(self, info: dict) -> None:
        """Read ``obs_scale`` so a cap-scaled observation is restored to units."""
        self._obs_scale = float(info.get("obs_scale", self._obs_scale))

    def order_quantity(self, obs: np.ndarray) -> float:
        """Order up to the cumulative critical-ratio quantile over the protection period."""
        obs = restore_obs_units(obs, self._obs_scale)
        inventory = float(obs[0])
        in_transit = float(obs[1]) + float(obs[2])
        # Protection period = lead time + review (1 week) → cover h0..h(lead_time).
        n_cover = min(self.lead_time_weeks + 1, 3)  # obs exposes h0,h1,h2
        med_sum, spread_sq = 0.0, 0.0
        for h in range(n_cover):
            base = 3 + 3 * h  # [inv,it1,it2, q10_h0,q50_h0,q90_h0, ...]
            q50, q90 = float(obs[base + 1]), float(obs[base + 2])
            med_sum += q50
            spread_sq += max(0.0, q90 - q50) ** 2
        target_S = med_sum + self._wq * np.sqrt(spread_sq)
        order = target_S - (inventory + in_transit)
        return float(np.clip(order, 0.0, self.max_order))

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """SB3-compatible interface: returns (action, state)."""
        obs = np.asarray(obs).reshape(-1)
        qty = self.order_quantity(obs)
        return np.array([qty], dtype=np.float32), None


class LeadTimeBaseStockPolicy:
    """
    Order-up-to policy using a direct quantile forecast of lead-time demand S_t.

    Reads ``q{τ}_S`` from the env ``info`` dict (populated from LTD pinball models).
    Order quantity: ``(q̂_τ(S) − inventory_position)+``.
    """

    def __init__(
        self,
        ltd_quantile: float = 0.83,
        max_order: float = 50.0,
    ):
        self.ltd_quantile = float(ltd_quantile)
        self.ltd_key = ltd_quantile_key(self.ltd_quantile)
        self.max_order = float(max_order)
        self._info: dict = {}

    def update_info(self, info: dict) -> None:
        """Refresh LTD quantile forecasts from the latest env step."""
        self._info = info

    def order_quantity(self, obs: np.ndarray) -> float:
        scale = float(self._info.get("obs_scale", 1.0))
        inventory = float(obs[0]) * scale
        in_transit = (float(obs[1]) + float(obs[2])) * scale
        target_S = float(self._info.get(self.ltd_key, 0.0))
        order = target_S - (inventory + in_transit)
        return float(np.clip(order, 0.0, self.max_order))

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        obs = np.asarray(obs).reshape(-1)
        qty = self.order_quantity(obs)
        return np.array([qty], dtype=np.float32), None
