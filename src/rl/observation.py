"""
The policy's observation and the action → order mapping as plain functions, so the training
environment (:class:`src.rl.vn2_env.VN2Env`) and the served policy (``src/serving/policy_model.py``)
compute them one way. :class:`src.rl.batch_env.BatchVN2Env` is their tensor twin (tested to match).
"""

from __future__ import annotations

import numpy as np

from src.rl.spec import SIGNED_FEATURES, UNITLESS_FEATURES, PolicySpec
from src.simulation.anchor import AnchorDecision

OBS_HIGH = 10.0
POS_CV_CAP = 5.0


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


def forecast_value(fc: dict, key: str) -> float:
    """``fc[key]``, or 0 when the forecast is missing or NaN."""
    v = fc.get(key, 0.0)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


def effective_scale(scale: float) -> float:
    """The SKU's demand scale as the policy sees it (at least 1 unit)."""
    return max(float(scale), 1.0)


def state_values(on_hand: float, transit_1: float, transit_2: float, decision: AnchorDecision,
                 zero_share, pos_cv, demand_class) -> dict[str, float]:
    """The named state features of one SKU-week; the VN2-window ones are the environment's own."""
    return {
        "on_hand": on_hand,
        "in_transit_w1": transit_1,
        "in_transit_w2": transit_2,
        "anchor_order": float(decision.order),
        "projected_stock": decision.projected_stock_at_arrival,
        "projected_lost": decision.projected_lost_sales_before_arrival,
        "zero_share": float(zero_share or 0.0),
        "pos_cv": min(POS_CV_CAP, float(pos_cv or 0.0)),
        **class_one_hot(demand_class),
    }


def observation_bounds(spec: PolicySpec) -> tuple[np.ndarray, np.ndarray]:
    low = np.array([-OBS_HIGH if n in SIGNED_FEATURES else 0.0 for n in spec.state], dtype=np.float32)
    return low, np.full(len(spec.state), OBS_HIGH, dtype=np.float32)


def observation(spec: PolicySpec, values: dict, fc: dict, scale: float) -> np.ndarray:
    """
    The policy's input: ``spec.state`` in order, taken from ``values`` (inventory, anchor and
    context features) or else from the forecasts ``fc``; features in demand units are divided
    by ``scale``; the vector is clipped to the observation bounds.
    """
    vec = np.array(
        [(values[n] if n in values else forecast_value(fc, n)) / (1.0 if n in UNITLESS_FEATURES else scale)
         for n in spec.state],
        dtype=np.float32,
    )
    vec = np.nan_to_num(vec, nan=0.0, posinf=OBS_HIGH, neginf=-OBS_HIGH)
    return np.clip(vec, *observation_bounds(spec))


def order_before_rounding(spec: PolicySpec, anchor_order: float, scale: float, action) -> float:
    """Order for ``action``: a correction of the anchor (residual) or a share of the cap (raw)."""
    f = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
    if spec.method == "raw":
        return (f + 1.0) / 2.0 * spec.order_cap_mult * scale
    return anchor_order + f * max(anchor_order, spec.floor_frac * scale)


def order_for_action(spec: PolicySpec, anchor_order: float, scale: float, action) -> float:
    return float(max(0, round(order_before_rounding(spec, anchor_order, scale, action))))
