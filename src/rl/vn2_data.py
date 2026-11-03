"""Build VN2-shaped RL episodes and forecast lookups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.features.demand_profile import add_demand_profile_features, demand_class_name
from src.simulation.anchor import AnchorSpec, LostSalesAnchor
from src.simulation.vn2 import (
    BURN_IN_WEEKS,
    VN2Rules,
    burn_in_initial_state,
    enumerate_window_starts,
)

_PANEL_COLS = ["store", "product", "date", "sales", "in_stock"]


def build_forecast_lookup(
    forecasts: pd.DataFrame,
    measure_ids: list[str],
) -> dict[tuple, dict[str, float]]:
    """(store, product, date) → {measure id: forecast} for the run's measures."""
    if forecasts.empty:
        return {}
    cols = [c for c in measure_ids if c in forecasts.columns]
    fc = forecasts[["store", "product", "date", *cols]].copy()
    fc["date"] = pd.to_datetime(fc["date"])
    values = fc[cols].astype(float).fillna(0.0).to_numpy()
    lookup: dict[tuple, dict[str, float]] = {}
    for (store, product, date), row in zip(
        fc[["store", "product", "date"]].itertuples(index=False, name=None), values
    ):
        lookup[(int(store), int(product), pd.Timestamp(date))] = dict(zip(cols, row.tolist()))
    return lookup


def compute_demand_scales(
    panel: pd.DataFrame,
    cap_end: pd.Timestamp | None = None,
) -> dict[tuple[int, int], float]:
    slim = panel[_PANEL_COLS].copy()
    slim["date"] = pd.to_datetime(slim["date"])
    if cap_end is not None:
        slim = slim[slim["date"] <= pd.Timestamp(cap_end)]
    scales: dict[tuple[int, int], float] = {}
    for (store, product), grp in slim.groupby(["store", "product"], observed=True):
        peak = float(grp["sales"].max() or 0.0)
        scales[(int(store), int(product))] = max(1.0, peak)
    return scales


def attach_demand_class(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add each row's demand pattern as of the week before it: ``dp_sbc_class`` (0-3, -1 when
    the history is too short) and ``demand_class`` (Smooth, Erratic, Intermittent, Lumpy or
    Unclassified). Uses the demand-profile cutoffs from the config; out-of-stock weeks are
    skipped, so the class is point-in-time and never sees the row's own sales.
    """
    prof = dict(config.get("forecasting", {}).get("features", {}).get("demand_profile") or {})
    prof["enabled"] = True
    base = panel[["store", "product", "date", "sales"]]
    dp = add_demand_profile_features(base, config={"forecasting": {"features": {"demand_profile": prof}}})
    out = panel.drop(columns=[c for c in ("dp_sbc_class", "demand_class") if c in panel.columns]).merge(
        dp[["store", "product", "date", "dp_sbc_class"]], on=["store", "product", "date"], how="left"
    )
    out["dp_sbc_class"] = out["dp_sbc_class"].fillna(-1).astype(int)
    out["demand_class"] = out["dp_sbc_class"].map(demand_class_name)
    return out


def context_features(panel: pd.DataFrame, week: pd.Timestamp, config: dict) -> pd.DataFrame:
    """
    ``zero_share``, ``pos_cv`` and ``dp_sbc_class`` of every SKU for the row dated ``week``, from the
    sales before it: what an episode row of that week carries, for a week whose sales are unknown
    (the decision week of a served policy). One row per SKU.
    """
    week = pd.Timestamp(week)
    hist = panel.loc[panel["date"] < week, _PANEL_COLS]
    future = hist.drop_duplicates(["store", "product"]).assign(date=week, sales=np.nan, in_stock=True)
    full = attach_demand_class(pd.concat([hist, future], ignore_index=True), config)
    rows = []
    for _, grp in full.groupby(["store", "product"], observed=True):
        rows.append(_attach_shape_features(grp).iloc[-1])
    out = pd.DataFrame(rows)[["store", "product", "zero_share", "pos_cv", "dp_sbc_class"]]
    return out.astype({"store": "int64", "product": "int64", "dp_sbc_class": "int64"}).reset_index(drop=True)


def _attach_shape_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("date").reset_index(drop=True)
    n = len(grp)
    zero_share = np.zeros(n)
    pos_cv = np.zeros(n)
    n_past = n_zero = 0
    pos_vals: list[float] = []
    for i in range(n):
        if n_past > 0:
            zero_share[i] = n_zero / n_past
            if len(pos_vals) >= 2:
                mu = float(np.mean(pos_vals))
                if mu > 0:
                    pos_cv[i] = min(5.0, float(np.std(pos_vals)) / mu)
        sales = float(grp.loc[i, "sales"]) if pd.notna(grp.loc[i, "sales"]) else 0.0
        n_past += 1
        if sales <= 0:
            n_zero += 1
        else:
            pos_vals.append(sales)
    out = grp.copy()
    out["zero_share"] = zero_share
    out["pos_cv"] = pos_cv
    return out


def build_vn2_episodes(
    panel: pd.DataFrame,
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    forecast_lookup: dict[tuple, dict[str, float]],
    config: dict,
    burn_in_anchor: AnchorSpec,
    *,
    initial_state: pd.DataFrame | None = None,
    official_window_start: pd.Timestamp | None = None,
) -> dict[tuple[int, int, pd.Timestamp], pd.DataFrame]:
    """
    Rolling 8-week episodes keyed by (store, product, window_start).

    Each window starts from the official state (official window) or from a short
    burn-in run of ``burn_in_anchor``. Pass the forecasting run's default anchor, so
    every policy compared on a run starts from identical states.
    """
    rules = VN2Rules.from_config(config)
    slim = panel.copy()
    if "in_stock" not in slim.columns:
        slim["in_stock"] = True
    slim["date"] = pd.to_datetime(slim["date"])
    slim = attach_demand_class(slim, config)
    weeks = np.array(sorted(slim["date"].unique()))
    starts = enumerate_window_starts(weeks, block_start, block_end, rules.window_weeks)
    anchor = LostSalesAnchor(burn_in_anchor, rules)
    init = None
    if initial_state is not None:
        init = initial_state.rename(
            columns={"Store": "store", "Product": "product"}
            if "Store" in initial_state.columns
            else {}
        )

    def anchor_order_fn(round_i: int, state, ctx: dict) -> int:
        fc = ctx.get("fc", {})
        return anchor.order_quantity(
            state.on_hand,
            state.pipeline[0] if state.pipeline else 0.0,
            state.pipeline[1] if len(state.pipeline) > 1 else 0.0,
            fc,
        )

    # Shape features look backward only, so one pass per SKU serves every window.
    series = {
        key: _attach_shape_features(grp)
        for key, grp in slim.groupby(["store", "product"], observed=True)
    }
    episodes: dict[tuple[int, int, pd.Timestamp], pd.DataFrame] = {}
    for wstart in starts:
        wend = wstart + pd.Timedelta(weeks=rules.window_weeks - 1)
        use_official = (
            official_window_start is not None
            and pd.Timestamp(wstart) == pd.Timestamp(official_window_start)
        )
        for (store, product), grp in series.items():
            win = grp[(grp["date"] >= wstart) & (grp["date"] <= wend)].copy()
            if len(win) != rules.window_weeks:
                continue
            if not win["in_stock"].fillna(True).all():
                continue
            hist = grp[grp["date"] < wstart].tail(BURN_IN_WEEKS + rules.lead_time_weeks)
            if use_official and init is not None:
                row_i = init[(init["store"] == store) & (init["product"] == product)]
                if row_i.empty:
                    continue
                r = row_i.iloc[0]
                oh = float(r.get("end_inventory", 0) or 0)
                t1 = float(r.get("in_transit_1w", 0) or 0)
                t2 = float(r.get("in_transit_2w", 0) or 0)
            else:
                burn_hist = hist.tail(BURN_IN_WEEKS)
                if len(burn_hist) < BURN_IN_WEEKS:
                    continue
                burn_demands = burn_hist["sales"].fillna(0.0).tolist()
                burn_dates = [pd.Timestamp(d) for d in burn_hist["date"].tolist()]
                st, pr = int(store), int(product)

                # Called right away by burn_in_initial_state; the defaults bind this window's values.
                def _burn_order(r: int, s, _c: dict, burn_dates=burn_dates, st=st, pr=pr) -> int:
                    d = burn_dates[min(r - 1, len(burn_dates) - 1)]
                    fc = forecast_lookup.get((st, pr, d), {})
                    return anchor_order_fn(r, s, {"fc": fc})

                oh, t1, t2 = burn_in_initial_state(
                    burn_demands,
                    on_hand=float(hist["sales"].fillna(0).mean() or 1.0),
                    order_fn=_burn_order,
                    rules=rules,
                )
            win = win.reset_index(drop=True)
            win.loc[win.index[0], "end_inventory"] = oh
            win.loc[win.index[0], "on_hand"] = oh
            win.loc[win.index[0], "in_transit_1w"] = t1
            win.loc[win.index[0], "in_transit_2w"] = t2
            episodes[(int(store), int(product), pd.Timestamp(wstart))] = win
    return episodes
