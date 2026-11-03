"""
Why a policy orders what it orders, for one decision of a :class:`VN2Env`.

``explain_order`` splits the order into the anchor's arithmetic (order up to the
delivery-week target, minus the stock projected to be left when the order arrives) and the
policy's adjustment. ``drivers`` raises one quantity input at a time and reports how the
policy's unrounded correction moves, with the anchor order held fixed. It is a local
what-if: inputs interact, so the effects don't add up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rl.spec import UNITLESS_FEATURES
from src.rl.vn2_env import VN2Env


def explain_order(env: VN2Env, agent, obs) -> dict:
    """Anchor arithmetic and the policy's order for the env's current round."""
    dec = env._decision
    action, _ = agent.predict(obs, deterministic=True)
    order = env.resolve_order(action)
    out = {
        "target": float(env.anchor.target_level(env._get_fc())),
        "projected_stock": float(dec.projected_stock_at_arrival),
        "projected_lost": float(dec.projected_lost_sales_before_arrival),
        "anchor_order": int(dec.order),
        "rl_order": order,
        "adjustment": order - dec.order,
    }
    if env.spec.method == "residual":
        out["step"] = max(float(dec.order), env.spec.floor_frac * env.demand_scale)
    return out


def _correction(env: VN2Env, agent, obs: np.ndarray) -> float:
    """The policy's unrounded change to the anchor order (raw: its whole order), in units."""
    action, _ = agent.predict(obs, deterministic=True)
    return env.unrounded_order(action) - (0.0 if env.spec.method == "raw" else float(env._decision.order))


def drivers(env: VN2Env, agent, obs, rel: float = 0.1) -> pd.DataFrame:
    """
    Change of the policy's correction (units) when one quantity input rises by ``rel`` times
    the SKU's demand scale, the others fixed: a central difference within the observation
    bounds. Unitless inputs (round, demand pattern, class) are left out. Largest effect first.
    """
    base = np.asarray(obs, dtype=np.float32)
    low, high = env.observation_space.low, env.observation_space.high
    rows = []
    for i, name in enumerate(env.spec.state):
        if name in UNITLESS_FEATURES:
            continue
        up, down = base.copy(), base.copy()
        up[i], down[i] = min(high[i], base[i] + rel), max(low[i], base[i] - rel)
        width = float(up[i] - down[i])
        if width <= 0:
            continue
        effect = (_correction(env, agent, up) - _correction(env, agent, down)) * rel / width
        rows.append({"input": name, "value": float(base[i]) * env.demand_scale,
                     "change": rel * env.demand_scale, "effect_units": effect})
    out = pd.DataFrame(rows, columns=["input", "value", "change", "effect_units"])
    return out.sort_values("effect_units", key=np.abs, ascending=False, ignore_index=True)
