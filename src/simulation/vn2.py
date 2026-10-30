"""Official VN2 inventory replay: lost sales, lead time, scoring in euros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

OrderFn = Callable[[int, "InventoryState", dict], int]


@dataclass(frozen=True)
class VN2Rules:
    """Competition parameters (costs may be overridden for sensitivity studies)."""

    holding_cost: float = 0.2
    shortage_cost: float = 1.0
    lead_time_weeks: int = 2
    n_order_rounds: int = 6
    window_weeks: int = 8
    score_from_week: int = 3  # 1-indexed within the window

    @classmethod
    def from_config(cls, config: dict) -> VN2Rules:
        vn2 = config.get("vn2") or {}
        return cls(
            holding_cost=float(vn2.get("holding_cost", 0.2)),
            shortage_cost=float(vn2.get("shortage_cost", 1.0)),
            lead_time_weeks=int(vn2.get("lead_time_weeks", 2)),
            n_order_rounds=int(vn2.get("n_order_rounds", 6)),
            window_weeks=int(vn2.get("window_weeks", 8)),
            score_from_week=int(vn2.get("score_from_week", 3)),
        )

    @property
    def critical_ratio(self) -> float:
        denom = self.shortage_cost + self.holding_cost
        return self.shortage_cost / denom if denom > 0 else 0.5


@dataclass
class InventoryState:
    on_hand: float
    pipeline: list[float]

    def inventory_position(self) -> float:
        return self.on_hand + sum(self.pipeline)

    def copy(self) -> InventoryState:
        return InventoryState(self.on_hand, list(self.pipeline))


@dataclass
class WeekResult:
    week_index: int
    demand: float
    sales: float
    leftover: float
    shortage: float
    holding_eur: float
    shortage_eur: float
    total_eur: float
    order_placed: float
    arriving: float


@dataclass
class WindowResult:
    weeks: list[WeekResult]
    total_eur: float
    scored_eur: float
    weeks_1_2_eur: float


def week_cost(
    leftover: float, shortage: float, rules: VN2Rules
) -> tuple[float, float, float]:
    h = rules.holding_cost * leftover
    s = rules.shortage_cost * shortage
    return h, s, h + s


def apply_order_to_pipeline(pipeline: list[float], order: float, lead: int) -> None:
    if lead <= 0:
        return
    while len(pipeline) < lead:
        pipeline.append(0.0)
    pipeline[-1] += order


def step_one_week(
    state: InventoryState,
    demand: float,
    order: float,
    rules: VN2Rules,
) -> tuple[InventoryState, WeekResult]:
    """One competition week: receive, sell (lost sales), place order at start."""
    lead = rules.lead_time_weeks
    pipeline = list(state.pipeline) + [0.0] * max(0, lead - len(state.pipeline))
    pipeline = pipeline[:lead]

    arriving = pipeline[0] if lead > 0 else 0.0
    pipeline = pipeline[1:] + [0.0]
    apply_order_to_pipeline(pipeline, order, lead)

    start = state.on_hand + arriving
    demand = max(0.0, float(demand))
    sales = min(start, demand)
    leftover = max(0.0, start - sales)
    shortage = max(0.0, demand - sales)
    h, s, tot = week_cost(leftover, shortage, rules)

    new_state = InventoryState(leftover, pipeline)
    wr = WeekResult(
        week_index=0,
        demand=demand,
        sales=sales,
        leftover=leftover,
        shortage=shortage,
        holding_eur=h,
        shortage_eur=s,
        total_eur=tot,
        order_placed=order,
        arriving=arriving,
    )
    return new_state, wr


def simulate_sku_window(
    demand: np.ndarray | list[float],
    *,
    on_hand: float,
    in_transit_w1: float = 0.0,
    in_transit_w2: float = 0.0,
    order_fn: OrderFn | None = None,
    fixed_orders: list[float] | None = None,
    rules: VN2Rules | None = None,
    context: dict | None = None,
) -> WindowResult:
    """
    Simulate one SKU for ``window_weeks`` (default 8).

    ``order_fn(round, state, ctx)`` is called for rounds 1..n_order_rounds (1-indexed).
    """
    rules = rules or VN2Rules()
    demand = [max(0.0, float(d)) for d in demand]
    if len(demand) < rules.window_weeks:
        raise ValueError(f"need {rules.window_weeks} demand values, got {len(demand)}")
    demand = demand[: rules.window_weeks]

    lead = rules.lead_time_weeks
    pipeline = [0.0] * lead
    if lead > 0:
        pipeline[0] = float(in_transit_w1)
    if lead > 1:
        pipeline[1] = float(in_transit_w2)

    state = InventoryState(float(on_hand), pipeline)
    ctx = context or {}
    weeks: list[WeekResult] = []

    for w in range(rules.window_weeks):
        round_idx = w + 1
        if fixed_orders is not None and w < len(fixed_orders):
            order = max(0.0, float(fixed_orders[w]))
        elif round_idx <= rules.n_order_rounds and order_fn is not None:
            order = max(0, int(round(order_fn(round_idx, state.copy(), ctx))))
        else:
            order = 0.0

        state, wr = step_one_week(state, demand[w], order, rules)
        wr.week_index = round_idx
        weeks.append(wr)

    total = sum(w.total_eur for w in weeks)
    scored = sum(
        w.total_eur for w in weeks if w.week_index >= rules.score_from_week
    )
    w12 = sum(w.total_eur for w in weeks if w.week_index <= 2)
    return WindowResult(weeks=weeks, total_eur=total, scored_eur=scored, weeks_1_2_eur=w12)


def constant_order_policy(qty: float) -> OrderFn:
    def _fn(_round: int, _state: InventoryState, _ctx: dict) -> int:
        return int(max(0, round(qty)))

    return _fn


def simulate_panel_window(
    panel: pd.DataFrame,
    demand_col: str = "sales",
    *,
    on_hand_col: str = "end_inventory",
    transit_w1_col: str = "in_transit_1w",
    transit_w2_col: str = "in_transit_2w",
    order_fn: OrderFn | None = None,
    fixed_orders_by_sku: dict[tuple[int, int], list[float]] | None = None,
    rules: VN2Rules | None = None,
    skip_censored: bool = True,
    instock_col: str | None = "in_stock",
) -> dict[str, float]:
    """Sum euro costs over all SKUs in a aligned 8-week panel slice."""
    rules = rules or VN2Rules()
    total = scored = w12 = 0.0
    n = 0
    for (_store, _product), grp in panel.groupby(["store", "product"], observed=True):
        grp = grp.sort_values("date")
        if len(grp) < rules.window_weeks:
            continue
        if skip_censored and instock_col and instock_col in grp.columns:
            if not grp[instock_col].fillna(True).all():
                continue
        row0 = grp.iloc[0]
        demand = grp[demand_col].fillna(0.0).tolist()[: rules.window_weeks]
        key = (int(_store), int(_product))
        fixed = (fixed_orders_by_sku or {}).get(key)
        res = simulate_sku_window(
            demand,
            on_hand=float(row0.get(on_hand_col, row0.get("on_hand", 0)) or 0),
            in_transit_w1=float(row0.get(transit_w1_col, 0) or 0),
            in_transit_w2=float(row0.get(transit_w2_col, 0) or 0),
            order_fn=order_fn,
            fixed_orders=fixed,
            rules=rules,
        )
        total += res.total_eur
        scored += res.scored_eur
        w12 += res.weeks_1_2_eur
        n += 1
    return {
        "total_eur": total,
        "scored_eur": scored,
        "weeks_1_2_eur": w12,
        "n_skus": float(n),
    }


def load_competition_weeks(config: dict) -> pd.DataFrame:
    """Long panel for official weeks 1–8 (2024-04-15 … 2024-06-03)."""
    from src.forecasting.preprocessing import load_panel_from_config

    return load_panel_from_config(
        config,
        start=pd.Timestamp("2024-04-15"),
        end=pd.Timestamp("2024-06-03"),
    )


def attach_official_initial_state(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Merge week-0 ``End Inventory`` and in-transit onto the first row of each SKU."""
    from src.forecasting.preprocessing import load_initial_state

    init = load_initial_state(config["data"]["initial_state_file"])
    init = init.rename(columns={"Store": "store", "Product": "product"})
    keys = ["store", "product"]
    cols = {
        "end_inventory": "end_inventory",
        "in_transit_1w": "in_transit_1w",
        "in_transit_2w": "in_transit_2w",
    }
    slim = init[keys + list(cols.keys())]
    out = panel.merge(slim, on=keys, how="left", suffixes=("_drop", ""))
    for c in cols:
        drop = f"{c}_drop"
        if drop in out.columns:
            out[c] = out[c].fillna(out[drop])
            out = out.drop(columns=[drop])
    return out


def official_vn2_score(
    panel_with_init: pd.DataFrame,
    order_fn: OrderFn,
    rules: VN2Rules | None = None,
) -> float:
    """Total € for weeks 3–8 over all SKUs from the official start state."""
    agg = simulate_panel_window(
        panel_with_init,
        order_fn=order_fn,
        rules=rules,
        on_hand_col="end_inventory",
    )
    return float(agg["scored_eur"])


@dataclass
class VN2Compliance:
    ok: bool
    messages: list[str]


BURN_IN_WEEKS = 4


def burn_in_initial_state(
    demand: list[float],
    *,
    on_hand: float = 0.0,
    in_transit_w1: float = 0.0,
    in_transit_w2: float = 0.0,
    order_fn: OrderFn,
    rules: VN2Rules | None = None,
    context: dict | None = None,
) -> tuple[float, float, float]:
    """Run ``BURN_IN_WEEKS`` before a window; return (on_hand, w1, w2) pipeline."""
    rules = rules or VN2Rules()
    lead = rules.lead_time_weeks
    pipeline = [0.0] * lead
    if lead > 0:
        pipeline[0] = float(in_transit_w1)
    if lead > 1:
        pipeline[1] = float(in_transit_w2)
    state = InventoryState(float(on_hand), pipeline)
    ctx = context or {}
    for i, d in enumerate(demand[:BURN_IN_WEEKS]):
        order = max(0, int(round(order_fn(i + 1, state.copy(), ctx))))
        state, _ = step_one_week(state, d, order, rules)
    w1 = state.pipeline[0] if len(state.pipeline) > 0 else 0.0
    w2 = state.pipeline[1] if len(state.pipeline) > 1 else 0.0
    return state.on_hand, w1, w2


def enumerate_window_starts(
    weeks: pd.DatetimeIndex | list[pd.Timestamp],
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    window_weeks: int = 8,
) -> list[pd.Timestamp]:
    """Start dates of rolling ``window_weeks`` windows inside [block_start, block_end]."""
    weeks = pd.DatetimeIndex(weeks)
    block_start, block_end = pd.Timestamp(block_start), pd.Timestamp(block_end)
    starts: list[pd.Timestamp] = []
    i0 = weeks.searchsorted(block_start)
    while i0 < len(weeks):
        start = weeks[i0]
        end = start + pd.Timedelta(weeks=window_weeks - 1)
        if end > block_end:
            break
        if end <= weeks[-1]:
            starts.append(pd.Timestamp(start))
        i0 += 1
    return starts


def check_vn2_compliance(config: dict, resolved_periods: dict | None = None) -> VN2Compliance:
    """Green badge checks from the plan (config + optional resolved period dates)."""
    msgs: list[str] = []
    ok = True
    rules = VN2Rules.from_config(config)
    if rules.lead_time_weeks != 2:
        ok = False
        msgs.append("lead_time_weeks must be 2 for VN2 comparability")
    if rules.n_order_rounds != 6 or rules.window_weeks != 8:
        ok = False
        msgs.append("window must be 8 weeks with 6 order rounds")
    if resolved_periods:
        test_start = pd.Timestamp(resolved_periods.get("rl_test_start"))
        test_end = pd.Timestamp(resolved_periods.get("rl_test_end"))
        official_start = pd.Timestamp("2024-04-15")
        official_end = pd.Timestamp("2024-06-03")
        if test_start > official_start or test_end < official_end:
            ok = False
            msgs.append("test period must cover the full official window")
    if ok and not msgs:
        msgs.append("VN2-comparable settings")
    return VN2Compliance(ok=ok, messages=msgs)
