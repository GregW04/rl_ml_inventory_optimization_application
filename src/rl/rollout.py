"""Roll a policy through VN2 episodes; the anchor baseline is the zero residual."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.forecasting.features.demand_profile import DEMAND_CLASSES, UNCLASSIFIED
from src.rl.spec import PolicySpec
from src.simulation.vn2 import VN2Rules
from src.simulation.vn2_env import VN2Env


class ZeroResidual:
    """Residual of 0 on every round: orders exactly the anchor."""

    def predict(self, obs, deterministic: bool = True):
        return np.array([0.0], dtype=np.float32), None


def anchor_baseline(spec: PolicySpec) -> tuple[PolicySpec, ZeroResidual]:
    """Spec and policy that reproduce ``spec``'s anchor with no learned correction."""
    return replace(spec, method="residual", agent="ppo", buckets=None), ZeroResidual()


def rollout(
    policy,
    spec: PolicySpec,
    episodes: dict[tuple[int, int, pd.Timestamp], pd.DataFrame],
    lookup: dict,
    demand_scales: dict[tuple[int, int], float],
    rules: VN2Rules,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Run ``policy`` deterministically over every episode.

    Returns aggregate KPIs in euros and one row per episode with its scored cost
    (weeks 3–8), holding and shortage cost over the whole window, and the SKU's demand
    class at the window start (when the episodes carry it).
    """
    rows = []
    for (store, product, wstart), ep in episodes.items():
        env = VN2Env(spec, ep, lookup, demand_scales.get((store, product), 1.0), rules)
        obs, _ = env.reset()
        done = False
        holding = shortage = 0.0
        info: dict = {}
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            holding += rules.holding_cost * float(info.get("leftover", 0))
            shortage += rules.shortage_cost * float(info.get("shortage", 0))
            done = terminated or truncated
        rows.append(
            {
                "store": store,
                "product": product,
                "window_start": pd.Timestamp(wstart),
                "demand_class": (
                    str(ep.sort_values("date")["demand_class"].iloc[0]) if "demand_class" in ep.columns else None
                ),
                "cost_eur": float(info.get("scored_cost_eur", 0.0)),
                "holding_eur": holding,
                "shortage_eur": shortage,
            }
        )
    per_ep = pd.DataFrame(rows)
    n = max(len(per_ep), 1)
    total = float(per_ep["cost_eur"].sum()) if len(per_ep) else 0.0
    summary = {
        "window_cost_eur": total,
        "cost_eur_per_sku": total / n,
        "holding_eur_per_sku": float(per_ep["holding_eur"].sum()) / n if len(per_ep) else 0.0,
        "shortage_eur_per_sku": float(per_ep["shortage_eur"].sum()) / n if len(per_ep) else 0.0,
        "n_episodes": len(per_ep),
    }
    return summary, per_ep


def per_class_delta(per_ep: pd.DataFrame) -> dict[str, float]:
    """
    Cost change against the anchor in % per demand class, in display order.

    ``per_ep`` needs ``cost_eur``, ``anchor_cost_eur`` and ``demand_class``; episodes saved
    before classes existed give an empty dict.
    """
    if per_ep.empty or "demand_class" not in per_ep or per_ep["demand_class"].isna().all():
        return {}
    out: dict[str, float] = {}
    for cls in (*DEMAND_CLASSES, UNCLASSIFIED):
        g = per_ep[per_ep["demand_class"] == cls]
        base = float(g["anchor_cost_eur"].sum())
        if len(g) and base > 0:
            out[cls] = 100.0 * (float(g["cost_eur"].sum()) / base - 1.0)
    return out


def format_class_delta(deltas: dict[str, float]) -> str:
    """'Smooth −1.2% · Erratic +0.4% · …', or '—' when there are no classes."""
    if not deltas:
        return "—"
    return " · ".join(f"{cls} {'+' if v >= 0 else '−'}{abs(v):.1f}%" for cls, v in deltas.items())


def paired_bootstrap_ci(
    delta: np.ndarray,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of paired per-episode differences."""
    delta = np.asarray(delta, dtype=float)
    if len(delta) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
