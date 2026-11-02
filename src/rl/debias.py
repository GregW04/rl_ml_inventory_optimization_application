"""
Training forecasts with each forecaster fit's average error removed.

A fit can run too low for a whole growth year and too high in the following decline, and
an agent trained on the first learns to order more everywhere, then pays for it once the
error flips sign. With ``training_forecasts`` set to ``unbiased`` (per fit) or
``unbiased_by_class`` (per fit and demand class), every training forecast of a cell is
multiplied by

    f = (Σ actual + k) ÷ (Σ forecast + k)

over the cell's mean forecasts at h0 and h1 (k = :data:`SHRINK_UNITS`; cells with fewer
than :data:`MIN_CELL_WEEKS` SKU-weeks use their fit's factor), so the forecasts the agent
trains on are right on average. Validation and test keep the real forecasts. Patterns
that exist even with unbiased forecasts, such as less safety stock for slow movers, stay
learnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.drift import run_fits
from src.forecasting.measures import Measure

SHRINK_UNITS = 50.0
MIN_CELL_WEEKS = 500
FACTOR_BOUNDS = (0.5, 2.0)
ALL = "all"
_WEEK = pd.Timedelta(weeks=1)


def _level_measures(measures: list[Measure]) -> list[Measure]:
    return [m for m in measures if m.kind == "mean" and m.horizon in (0, 1)]


def training_rows(
    lookup: dict,
    panel: pd.DataFrame,
    resolved: dict,
    measures: list[Measure],
    classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Training-block forecast keys with their fit, class and, per h0/h1 mean, forecast and actual."""
    fits = [f for f in run_fits(resolved) if f["block"] == "rl_train"]
    cols = ["store", "product", "date", "fit", "demand_class"]
    if not fits:
        return pd.DataFrame(columns=cols)
    start, end = fits[0]["start"], fits[-1]["end"]
    keys = [k for k in lookup if start <= k[2] <= end]
    rows = pd.DataFrame(keys, columns=["store", "product", "date"])
    rows["fit"] = None
    for f in fits:
        rows.loc[(rows["date"] >= f["start"]) & (rows["date"] <= f["end"]), "fit"] = f["label"]
    actual = panel[["store", "product", "date", "sales"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.astype({"store": "int64", "product": "int64"}).set_index(["store", "product", "date"])["sales"]
    for m in _level_measures(measures):
        rows[f"fc_{m.id}"] = [lookup[k].get(m.id, np.nan) for k in keys]
        target = rows["date"] + m.horizon * _WEEK
        y = actual.reindex(pd.MultiIndex.from_arrays([rows["store"], rows["product"], target])).to_numpy(dtype=float)
        rows[f"y_{m.id}"] = np.where(target <= end, y, np.nan)
    if classes is not None:
        cls = classes[["store", "product", "date", "demand_class"]].copy()
        cls["date"] = pd.to_datetime(cls["date"])
        rows = rows.merge(cls.astype({"store": "int64", "product": "int64"}), on=["store", "product", "date"], how="left")
        rows["demand_class"] = rows["demand_class"].fillna("Unclassified")
    else:
        rows["demand_class"] = ALL
    return rows


def _totals(g: pd.DataFrame, measures: list[Measure]) -> tuple[float, float, int]:
    actual = forecast = 0.0
    n = 0
    for m in _level_measures(measures):
        y, f = g.get(f"y_{m.id}"), g.get(f"fc_{m.id}")
        if y is None or f is None:
            continue
        ok = y.notna() & f.notna()
        actual += float(y[ok].sum())
        forecast += float(f[ok].sum())
        n += int(ok.sum())
    return actual, forecast, n


def _factor(actual: float, forecast: float) -> float:
    return float(np.clip((actual + SHRINK_UNITS) / (forecast + SHRINK_UNITS), *FACTOR_BOUNDS))


def training_bias_factors(rows: pd.DataFrame, measures: list[Measure], by_class: bool) -> pd.DataFrame:
    """Correction factor per training fit (and demand class), with the totals behind it."""
    out = []
    for fit, g in rows.groupby("fit", sort=False):
        a, f, n = _totals(g, measures)
        fit_factor = _factor(a, f)
        cells = g.groupby("demand_class", sort=True) if by_class else [(ALL, g)]
        for cls, gc in cells:
            ac, fc, nc = _totals(gc, measures)
            small = by_class and nc < MIN_CELL_WEEKS
            out.append({
                "fit": fit, "demand_class": cls, "sku_weeks": nc, "actual": ac, "forecast": fc,
                "error_pct": 100.0 * (ac - fc) / ac if ac > 0 else np.nan,
                "factor": fit_factor if small else _factor(ac, fc),
                "uses_fit_factor": bool(small),
            })
    return pd.DataFrame(out)


def debiased_lookup(lookup: dict, rows: pd.DataFrame, factors: pd.DataFrame) -> dict:
    """Copy of ``lookup`` with every training forecast multiplied by its cell's factor."""
    fmap = {(r.fit, r.demand_class): float(r.factor) for r in factors.itertuples(index=False)}
    out = dict(lookup)
    for store, product, date, fit, cls in rows[["store", "product", "date", "fit", "demand_class"]].itertuples(
        index=False, name=None
    ):
        key = (store, product, date)
        f = fmap.get((fit, cls), 1.0)
        out[key] = {m: v * f for m, v in lookup[key].items()}
    return out


def training_lookup(data, config: dict, by_class: bool) -> tuple[dict, pd.DataFrame]:
    """(corrected forecast lookup for training, factor table) for a :class:`src.rl.train.RunData`."""
    if not _level_measures(data.measures):
        raise ValueError("Unbiased training forecasts need a mean forecast at h0 or h1.")
    classes = None
    if by_class:
        from src.rl.vn2_data import attach_demand_class

        classes = attach_demand_class(data.panel[["store", "product", "date", "sales"]].copy(), config)
    rows = training_rows(data.lookup, data.panel, data.setup["resolved"], data.measures, classes)
    factors = training_bias_factors(rows, data.measures, by_class)
    return debiased_lookup(data.lookup, rows, factors), factors
