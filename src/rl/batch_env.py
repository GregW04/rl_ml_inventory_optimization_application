"""
Batched replay of VN2 windows for training PPO and GRPO.

``BatchVN2Env`` is a tensor twin of :class:`src.rl.vn2_env.VN2Env`: the same
observation features, scaling and clipping, the same lost-sales anchor, the same residual
and raw order rules (rounded half to even, as Python's ``round``), and the same weekly
costs scored from week 3. It steps thousands of SKU-windows at once and runs a shadow
anchor on the same demand, so a policy's cost can be paired with the anchor's.

It is only used for training. Validation and test go through ``VN2Env`` and
:func:`src.rl.rollout.rollout`, so every agent is scored by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.forecasting.features.demand_profile import UNCLASSIFIED
from src.forecasting.measures import poisson_quantile
from src.rl.spec import (
    ANCHOR_FEATURES,
    CLASS_FEATURES,
    CONTEXT_FEATURES,
    INVENTORY_FEATURES,
    SIGNED_FEATURES,
    UNITLESS_FEATURES,
    PolicySpec,
)
from src.simulation.vn2 import VN2Rules

OBS_HIGH = 10.0
_BUILTIN = set(INVENTORY_FEATURES) | set(ANCHOR_FEATURES) | set(CONTEXT_FEATURES) | set(CLASS_FEATURES)


def _num(v) -> float:
    """``VN2Env``'s reading of a forecast value: missing, None or NaN is 0."""
    if v is None:
        return 0.0
    v = float(v)
    return 0.0 if np.isnan(v) else v


@dataclass
class EpisodeArrays:
    """One row per SKU-window, one column per week of the window."""

    keys: list[tuple]
    sales: np.ndarray                 # [N, W] demand, >= 0
    forecasts: dict[str, np.ndarray]  # measure id -> [N, W]
    target: np.ndarray                # [N, W] anchor's delivery-week target level
    init: np.ndarray                  # [N, 3] on hand, arriving this week, arriving next week
    scale: np.ndarray                 # [N] demand scale (>= 1)
    zero_share: np.ndarray            # [N, W]
    pos_cv: np.ndarray                # [N, W] (already capped at 5, as the env does)
    sbc: np.ndarray                   # [N, W] demand-pattern code, -1 unknown
    demand_class: list[str]           # class at the window start (reporting)

    def __len__(self) -> int:
        return len(self.keys)


def build_episode_arrays(
    episodes: dict,
    lookup: dict,
    spec: PolicySpec,
    scales: dict,
    rules: VN2Rules,
) -> EpisodeArrays:
    """Stack ``block_episodes`` output into arrays for :class:`BatchVN2Env`."""
    anchor = spec.anchor
    ids = sorted({n for n in spec.state if n not in _BUILTIN} | anchor.inputs())
    keys = list(episodes.keys())
    n, w = len(keys), rules.window_weeks
    sales = np.zeros((n, w))
    fc = {m: np.zeros((n, w)) for m in ids}
    init = np.zeros((n, 3))
    scale = np.ones(n)
    zero_share = np.zeros((n, w))
    pos_cv = np.zeros((n, w))
    sbc = np.full((n, w), -1, dtype=np.int64)
    classes: list[str] = []
    for i, key in enumerate(keys):
        ep = episodes[key].sort_values("date").reset_index(drop=True)
        rows = ep.to_dict("records")
        r0 = rows[0]
        init[i] = [
            float(r0.get("end_inventory", r0.get("on_hand", 0)) or 0),
            float(r0.get("in_transit_1w", 0) or 0),
            float(r0.get("in_transit_2w", 0) or 0),
        ]
        scale[i] = max(float(scales.get((key[0], key[1]), 1.0)), 1.0)
        classes.append(str(r0.get("demand_class") or UNCLASSIFIED))
        for t, row in enumerate(rows[:w]):
            s = row.get("sales")
            sales[i, t] = max(0.0, float(s)) if pd.notna(s) else 0.0
            zero_share[i, t] = float(row.get("zero_share", 0.0) or 0.0)
            pos_cv[i, t] = min(5.0, float(row.get("pos_cv", 0.0) or 0.0))
            code = row.get("dp_sbc_class", -1)
            sbc[i, t] = int(code) if code is not None and pd.notna(code) else -1
            f = lookup.get((int(key[0]), int(key[1]), pd.Timestamp(row["date"])), {})
            for m in ids:
                fc[m][i, t] = _num(f.get(m, 0.0))
    target_src = fc[anchor.target_ids[0]]
    if anchor.target == "quantile":
        target = np.maximum(0.0, target_src)
    else:
        target = poisson_quantile(np.maximum(0.0, target_src), anchor.tau).astype(float)
    return EpisodeArrays(keys, sales, fc, target, init, scale, zero_share, pos_cv, sbc, classes)


class BatchVN2Env:
    """Step a batch of SKU-windows of ``arrays`` at once (see the module docstring)."""

    def __init__(self, spec: PolicySpec, arrays: EpisodeArrays, rules: VN2Rules):
        if spec.agent == "dqn":
            raise ValueError("BatchVN2Env replays continuous actions; DQN was retired.")
        if rules.lead_time_weeks != 2:
            raise ValueError("BatchVN2Env implements the VN2 lead time of 2 weeks.")
        f64 = torch.float64
        self.spec, self.rules = spec, rules
        self.sales = torch.as_tensor(arrays.sales, dtype=f64)
        self.fc = {k: torch.as_tensor(v, dtype=f64) for k, v in arrays.forecasts.items()}
        self.target = torch.as_tensor(arrays.target, dtype=f64)
        self.init = torch.as_tensor(arrays.init, dtype=f64)
        self.scale = torch.as_tensor(arrays.scale, dtype=f64)
        self.zero_share = torch.as_tensor(arrays.zero_share, dtype=f64)
        self.pos_cv = torch.as_tensor(arrays.pos_cv, dtype=f64)
        self.sbc = torch.as_tensor(arrays.sbc, dtype=torch.long)
        self.n = len(arrays)
        self.state = list(spec.state)
        self.unitless = [name in UNITLESS_FEATURES for name in self.state]
        self.low = torch.tensor(
            [-OBS_HIGH if name in SIGNED_FEATURES else 0.0 for name in self.state], dtype=torch.float32
        )
        self.high = torch.full((len(self.state),), OBS_HIGH, dtype=torch.float32)
        self.h0, self.h1 = spec.anchor.h0, spec.anchor.h1
        self.rounds, self.weeks = rules.n_order_rounds, rules.window_weeks

    @property
    def obs_dim(self) -> int:
        return len(self.state)

    # -- anchor ---------------------------------------------------------------------
    def _decide(self, inv, a0, a1, t: int):
        """``LostSalesAnchor.decide`` for every row: (order, projected stock, projected lost)."""
        idx = self.idx
        d0 = torch.clamp(self.fc[self.h0][idx, t], min=0.0)
        d1 = torch.clamp(self.fc[self.h1][idx, t], min=0.0)
        start_k = inv + a0
        end_k = torch.clamp(start_k - d0, min=0.0)
        lost_k = torch.clamp(d0 - start_k, min=0.0)
        start_k1 = end_k + a1
        end_k1 = torch.clamp(start_k1 - d1, min=0.0)
        lost_k1 = torch.clamp(d1 - start_k1, min=0.0)
        order = torch.clamp(torch.round(self.target[idx, t] - end_k1), min=0.0)
        return order, end_k1, lost_k + lost_k1

    # -- episode ----------------------------------------------------------------------
    def reset(self, idx: torch.Tensor) -> torch.Tensor:
        self.idx = idx
        self.t = 0
        init = self.init[idx]
        self.inv, self.p0, self.p1 = init[:, 0].clone(), init[:, 1].clone(), init[:, 2].clone()
        self.sh_inv, self.sh_p0, self.sh_p1 = self.inv.clone(), self.p0.clone(), self.p1.clone()
        self.prev_sales = torch.zeros_like(self.inv)
        self.last_h0 = torch.zeros_like(self.inv)
        self.dec = self._decide(self.inv, self.p0, self.p1, 0)
        return self._obs()

    def _obs(self) -> torch.Tensor:
        idx, t = self.idx, self.t
        order, proj_stock, proj_lost = self.dec
        code = self.sbc[idx, t]
        zeros = torch.zeros_like(self.inv)
        values = {
            "on_hand": self.inv,
            "in_transit_w1": self.p0,
            "in_transit_w2": self.p1,
            "anchor_order": order,
            "projected_stock": proj_stock,
            "projected_lost": proj_lost,
            "forecast_error": (self.prev_sales - self.last_h0) if t > 0 else zeros,
            "round_frac": torch.full_like(self.inv, (t + 1) / self.rounds),
            "zero_share": self.zero_share[idx, t],
            "pos_cv": self.pos_cv[idx, t],
            "class_smooth": (code == 0).to(torch.float64),
            "class_erratic": (code == 1).to(torch.float64),
            "class_intermittent": (code == 2).to(torch.float64),
            "class_lumpy": (code == 3).to(torch.float64),
        }
        scale = self.scale[idx]
        cols = []
        for name, unitless in zip(self.state, self.unitless):
            v = values[name] if name in values else self.fc[name][idx, t]
            cols.append(v if unitless else v / scale)
        vec = torch.stack(cols, dim=1).to(torch.float32)
        vec = torch.nan_to_num(vec, nan=0.0, posinf=OBS_HIGH, neginf=-OBS_HIGH)
        return torch.maximum(torch.minimum(vec, self.high), self.low)

    def resolve(self, action: torch.Tensor) -> torch.Tensor:
        """Order quantity for ``action`` (``VN2Env.resolve_order`` for continuous agents)."""
        f = torch.clamp(action.reshape(-1).to(torch.float64), -1.0, 1.0)
        scale = self.scale[self.idx]
        if self.spec.method == "raw":
            f = (f + 1.0) / 2.0
            return torch.clamp(torch.round(f * self.spec.order_cap_mult * scale), min=0.0)
        anchor_q = self.dec[0]
        step = torch.maximum(anchor_q, self.spec.floor_frac * scale)
        return torch.clamp(torch.round(anchor_q + f * step), min=0.0)

    def _week(self, inv, p0, p1, order, demand):
        """``step_one_week`` with lead time 2: (cost, on hand, arriving next, arriving after)."""
        start = inv + p0
        sold = torch.minimum(start, demand)
        left = torch.clamp(start - sold, min=0.0)
        short = torch.clamp(demand - sold, min=0.0)
        cost = self.rules.holding_cost * left + self.rules.shortage_cost * short
        return cost, left, p1, order

    def step(self, action: torch.Tensor | None):
        """Advance one week. Returns (obs or None, cost, shadow-anchor cost), both scored only from week 3."""
        t, idx = self.t, self.idx
        zeros = torch.zeros_like(self.inv)
        order = self.resolve(action) if (t < self.rounds and action is not None) else zeros
        sh_order = self._decide(self.sh_inv, self.sh_p0, self.sh_p1, t)[0] if t < self.rounds else zeros
        self.last_order, self.last_shadow_order = order, sh_order
        demand = self.sales[idx, t]
        cost, self.inv, self.p0, self.p1 = self._week(self.inv, self.p0, self.p1, order, demand)
        sh_cost, self.sh_inv, self.sh_p0, self.sh_p1 = self._week(self.sh_inv, self.sh_p0, self.sh_p1, sh_order, demand)
        scored = (t + 1) >= self.rules.score_from_week
        if not scored:
            cost, sh_cost = zeros, zeros.clone()
        self.prev_sales = demand
        self.last_h0 = self.fc[self.h0][idx, t]
        self.t += 1
        if self.t >= self.weeks:
            return None, cost, sh_cost
        self.dec = self._decide(self.inv, self.p0, self.p1, self.t)
        return self._obs(), cost, sh_cost
