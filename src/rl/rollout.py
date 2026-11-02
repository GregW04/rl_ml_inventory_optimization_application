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
    gate: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Run ``policy`` deterministically over every episode.

    Returns aggregate KPIs in euros and one row per episode with its scored cost
    (weeks 3–8), holding and shortage cost over the whole window, the units it ordered,
    and the SKU's demand class at the window start (when the episodes carry it).

    ``gate`` (demand classes) keeps ``policy`` only on episodes whose class at the window
    start is listed; the others, and episodes without a class, order the anchor.
    """
    base_spec, base_policy = anchor_baseline(spec)
    rows = []
    for (store, product, wstart), ep in episodes.items():
        cls = str(ep.sort_values("date")["demand_class"].iloc[0]) if "demand_class" in ep.columns else None
        on = gate is None or cls in gate
        env = VN2Env(spec if on else base_spec, ep, lookup, demand_scales.get((store, product), 1.0), rules)
        pol = policy if on else base_policy
        obs, _ = env.reset()
        done = False
        holding = shortage = units = 0.0
        info: dict = {}
        while not done:
            action, _ = pol.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            holding += rules.holding_cost * float(info.get("leftover", 0))
            shortage += rules.shortage_cost * float(info.get("shortage", 0))
            units += float(info.get("order_qty", 0))
            done = terminated or truncated
        rows.append(
            {
                "store": store,
                "product": product,
                "window_start": pd.Timestamp(wstart),
                "demand_class": cls,
                "cost_eur": float(info.get("scored_cost_eur", 0.0)),
                "holding_eur": holding,
                "shortage_eur": shortage,
                "units": units,
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


def units_vs_anchor(per_ep: pd.DataFrame) -> float | None:
    """Units ordered against the anchor's on the same episodes, in %; None without unit columns."""
    if per_ep.empty or not {"units", "anchor_units"} <= set(per_ep.columns):
        return None
    base = float(per_ep["anchor_units"].sum())
    return 100.0 * (float(per_ep["units"].sum()) / base - 1.0) if base > 0 else None


def per_class_units(per_ep: pd.DataFrame) -> dict[str, float]:
    """:func:`units_vs_anchor` per demand class, in display order (empty without classes or units)."""
    if units_vs_anchor(per_ep) is None or "demand_class" not in per_ep or per_ep["demand_class"].isna().all():
        return {}
    out: dict[str, float] = {}
    for cls in (*DEMAND_CLASSES, UNCLASSIFIED):
        value = units_vs_anchor(per_ep[per_ep["demand_class"] == cls])
        if value is not None:
            out[cls] = value
    return out


def vs_anchor(per_ep: pd.DataFrame, by) -> pd.DataFrame:
    """
    Per group of ``by`` (a column name or a series): windows, cost and units against the anchor
    in %, and the euro difference. Units are NaN for episodes saved before units were recorded.
    """
    g = per_ep.groupby(by)
    cost, base = g["cost_eur"].sum(), g["anchor_cost_eur"].sum()
    has_units = {"units", "anchor_units"} <= set(per_ep.columns)
    out = pd.DataFrame({
        "windows": g.size(),
        "cost_vs_anchor_pct": 100.0 * (cost / base - 1.0),
        "units_vs_anchor_pct": 100.0 * (g["units"].sum() / g["anchor_units"].sum() - 1.0) if has_units else np.nan,
        "delta_eur": cost - base,
    })
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


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
