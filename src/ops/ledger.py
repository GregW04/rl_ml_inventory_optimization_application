"""
Inventory ledger of the ops loop: one *trajectory* per policy, all facing the same weekly demand.

Demand is exogenous and unmet demand is lost, so any policy's trajectory can be simulated
exactly, not estimated: the live system and its shadows (e.g. the anchor on the previous
forecaster) are compared on the very same SKU-weeks.

State at the end of week D, one row per SKU:

* ``on_hand``   — stock left at the end of D
* ``transit_1`` — arrives at the start of D+1; ``transit_2`` arrives at the start of D+2
* ``order``     — placed at the start of D+1 (decided in cycle D); it arrives at D+3

Realising a week follows ``vn2.step_one_week`` exactly (tests/test_ops_units.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.preprocessing import load_initial_state
from src.simulation.anchor import LostSalesAnchor
from src.simulation.vn2 import VN2Rules

KEYS = ["store", "product"]
STATE_COLUMNS = [*KEYS, "on_hand", "transit_1", "transit_2", "order"]
COST_COLUMNS = [*KEYS, "demand", "censored", "sales", "shortage", "leftover", "holding_eur", "shortage_eur", "cost_eur"]


def initial_state_proxy(panel: pd.DataFrame, as_of: pd.Timestamp, weeks: int = 4) -> pd.DataFrame:
    """
    Steady-state start of a replay: stock on hand and both pipeline slots at the SKU's mean weekly
    sales over the ``weeks`` up to ``as_of`` (in-stock weeks), rounded. Every strategy of a replay
    starts from the same state, so comparisons don't depend on it.
    """
    recent = panel[(panel["date"] > pd.Timestamp(as_of) - pd.Timedelta(weeks=weeks)) & (panel["date"] <= as_of)]
    level = recent.groupby(KEYS)["sales"].mean().fillna(0.0).round()
    state = level.rename("on_hand").reset_index()
    state["transit_1"] = state["on_hand"]
    state["transit_2"] = state["on_hand"]
    state["order"] = np.nan
    return state[STATE_COLUMNS]


def initial_state_vn2(path: str) -> pd.DataFrame:
    """The official VN2 start (end of 2024-04-08): end inventory and the goods in transit."""
    s = load_initial_state(path).rename(columns={"end_inventory": "on_hand", "in_transit_1w": "transit_1",
                                                 "in_transit_2w": "transit_2"})
    s["order"] = np.nan
    return s[STATE_COLUMNS].astype({c: float for c in STATE_COLUMNS[2:]})


def realise(state: pd.DataFrame, demand: pd.DataFrame, rules: VN2Rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Play one week: goods in transit_1 arrive, demand is served from stock (lost sales), the order
    placed this week joins the pipeline. ``demand``: store, product, demand, censored.
    Returns the new state (order to be decided) and the week's costs per SKU.
    """
    if state["order"].isna().any():
        raise ValueError("Every SKU needs this week's order before the week is realised.")
    df = state.merge(demand, on=KEYS, how="left", validate="one_to_one")
    if df["demand"].isna().any():
        raise ValueError("Demand is missing for some SKUs.")
    start = df["on_hand"] + df["transit_1"]
    sales = np.minimum(start, df["demand"])
    df["sales"], df["leftover"], df["shortage"] = sales, start - sales, df["demand"] - sales
    df["holding_eur"] = rules.holding_cost * df["leftover"]
    df["shortage_eur"] = rules.shortage_cost * df["shortage"]
    df["cost_eur"] = df["holding_eur"] + df["shortage_eur"]
    new = pd.DataFrame({
        "store": df["store"], "product": df["product"], "on_hand": df["leftover"],
        "transit_1": df["transit_2"], "transit_2": df["order"], "order": np.nan,
    })
    return new[STATE_COLUMNS], df[COST_COLUMNS]


def anchor_orders(state: pd.DataFrame, forecasts: pd.DataFrame, anchor: LostSalesAnchor) -> pd.Series:
    """
    The anchor's order for the decision week: project stock through the arrivals of ``transit_1``
    (this week) and ``transit_2`` (next week) with the h0/h1 forecasts, then order up to the
    delivery-week target. A missing forecast is an error here, never an order of 0.
    """
    needed = sorted(anchor.spec.inputs())
    df = state.merge(forecasts[[*KEYS, *needed]], on=KEYS, how="left", validate="one_to_one")
    if df[needed].isna().to_numpy().any():
        raise ValueError(f"Forecasts {needed} are missing for {int(df[needed].isna().any(axis=1).sum())} SKUs.")
    orders = [
        anchor.decide(r.on_hand, r.transit_1, r.transit_2, {m: getattr(r, m) for m in needed}).order
        for r in df.itertuples(index=False)
    ]
    return pd.Series(orders, index=state.index, dtype=float)


def anchor_replay(state: pd.DataFrame, forecasts: dict[pd.Timestamp, pd.DataFrame],
                  demand: dict[pd.Timestamp, pd.DataFrame], anchor: LostSalesAnchor, rules: VN2Rules) -> pd.DataFrame:
    """
    The anchor's trajectory from ``state`` (the end of the first cycle week; its order is decided
    here): in every cycle week W of ``forecasts`` it orders with the forecasts issued in W, then
    week W+1 is realised with ``demand[W+1]``. Returns the realised costs with their week.
    """
    state, costs = state.copy(), []
    for week in sorted(forecasts):
        nxt = week + pd.Timedelta(weeks=1)
        if nxt not in demand:
            break
        state["order"] = anchor_orders(state, forecasts[week], anchor)
        state, c = realise(state, demand[nxt], rules)
        costs.append(c.assign(week=nxt))
    return pd.concat(costs, ignore_index=True)
