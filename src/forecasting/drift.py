"""
Forecast drift diagnostics for a forecasting run's out-of-sample RL-block forecasts.

The forecaster's level can run too low in a growth year and too high in a decline year,
and a learned policy can pick that up as a fixed offset. These helpers make it visible:

- :func:`fit_bias`: per forecaster fit (optionally per demand class), the forecast error of
  every mean measure, the coverage of every quantile measure, and total sales against the
  same weeks a year earlier.
- :func:`weekly_bias`: the weekly forecast error of one measure across all SKUs, with
  Trigg's smoothed tracking signal, which flags weeks where recent errors mostly point
  the same way.

Forecast error (bias) is (actual − forecast) ÷ actual, so positive means the forecasts were
too low. Only targets inside the forecast's own block count (a validation forecast never
reads test weeks), out-of-stock weeks (NaN sales) are skipped, and the test block is left
out unless ``include_test`` is set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.measures import Measure
from src.forecasting.periods import BLOCK_NAMES, fit_lane

# Trigg (1964): T = smoothed error ÷ smoothed absolute error, in [-1, 1]. With smoothing
# 0.2, |T| stays below 0.74 in about 95% of weeks when errors are unbiased and independent
# (checked by simulation); a persistent shift of one standard deviation of the weekly
# error is flagged after about 6 weeks, half of one after about 12.
TRIGG_ALPHA = 0.2
TRIGG_LIMIT = 0.74
TRIGG_WARMUP = 4
_WEEK = pd.Timedelta(weeks=1)


def run_fits(resolved: dict, include_test: bool = False) -> list[dict]:
    """Block fits of a saved ``resolved`` record; records saved before refits get one per block."""
    fits = [dict(f) for f in (resolved.get("block_fits") or [])]
    if not fits:
        for block in BLOCK_NAMES:
            start, end = resolved.get(f"{block}_start"), resolved.get(f"{block}_end")
            if start and end:
                fits.append({"block": block, "index": 1, "start": start, "end": end,
                             "fit_end": resolved.get(f"fit_end_{block}")})
    out = []
    for f in fits:
        block = f["block"]
        if block == "rl_test" and not include_test:
            continue
        n = sum(1 for g in fits if g["block"] == block)
        out.append({
            "block": block,
            "label": fit_lane(block, int(f.get("index", 1)), n).removeprefix("Fit · "),
            "start": pd.Timestamp(f["start"]),
            "end": pd.Timestamp(f["end"]),
            "fit_end": pd.Timestamp(f["fit_end"]) if f.get("fit_end") else pd.NaT,
            "block_end": pd.Timestamp(resolved.get(f"{block}_end") or f["end"]),
        })
    return out


def h0_level_measure(measures: list[Measure]) -> Measure | None:
    """The h0 demand level: its mean, else the h0 quantile nearest 0.5."""
    at0 = [m for m in measures if m.horizon == 0]
    means = [m for m in at0 if m.kind == "mean"]
    if means:
        return means[0]
    quantiles = [m for m in at0 if m.kind == "quantile"]
    return min(quantiles, key=lambda m: abs(m.tau - 0.5)) if quantiles else None


def scored_rows(
    forecasts: pd.DataFrame,
    sales: pd.DataFrame,
    resolved: dict,
    measures: list[Measure],
    include_test: bool = False,
) -> pd.DataFrame:
    """Forecast rows with their fit and the actual demand each horizon targets (``actual_h{h}``)."""
    fc = forecasts.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    actual = sales[["store", "product", "date", "sales"]].copy()
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.astype({"store": "int64", "product": "int64"}).set_index(["store", "product", "date"])["sales"]
    horizons = sorted({m.horizon for m in measures})
    parts = []
    for f in run_fits(resolved, include_test):
        rows = fc[(fc["date"] >= f["start"]) & (fc["date"] <= f["end"])].copy()
        if rows.empty:
            continue
        rows["block"], rows["fit"] = f["block"], f["label"]
        for h in horizons:
            target = rows["date"] + h * _WEEK
            key = pd.MultiIndex.from_arrays(
                [rows["store"].astype("int64"), rows["product"].astype("int64"), target]
            )
            y = actual.reindex(key).to_numpy(dtype=float)
            rows[f"actual_h{h}"] = np.where(target <= f["block_end"], y, np.nan)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _scores(g: pd.DataFrame, measures: list[Measure]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in measures:
        if m.id not in g.columns:
            continue
        y, f = g[f"actual_h{m.horizon}"], g[m.id]
        ok = y.notna() & f.notna()
        if not ok.any():
            continue
        if m.kind == "mean":
            total = float(y[ok].sum())
            out[f"bias {m.id} (%)"] = 100.0 * (total - float(f[ok].sum())) / total if total > 0 else np.nan
        else:
            out[f"coverage {m.id}"] = float((y[ok] <= f[ok]).mean())
    return out


def fit_bias(
    forecasts: pd.DataFrame,
    sales: pd.DataFrame,
    resolved: dict,
    measures: list[Measure],
    include_test: bool = False,
    classes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    One row per forecaster fit: forecast error of each mean measure, coverage of each
    quantile measure, and sales against the same weeks a year earlier. With ``classes``
    (store, product, date, demand_class), one row per fit and demand class instead.
    """
    rows = scored_rows(forecasts, sales, resolved, measures, include_test)
    if rows.empty:
        return pd.DataFrame()
    keys = ["fit"]
    if classes is not None:
        cls = classes[["store", "product", "date", "demand_class"]].copy()
        cls["date"] = pd.to_datetime(cls["date"])
        rows = rows.merge(cls.astype({"store": "int64", "product": "int64"}),
                          on=["store", "product", "date"], how="left")
        keys.append("demand_class")
    totals = sales.assign(date=pd.to_datetime(sales["date"])).groupby("date")["sales"].sum()
    out = []
    for f in run_fits(resolved, include_test):
        sub = rows[rows["fit"] == f["label"]]
        if sub.empty:
            continue
        for key, g in sub.groupby(keys, sort=True):
            rec = {
                "block": BLOCK_NAMES.get(f["block"], f["block"]),
                "fit": f["label"],
                "data through": f["fit_end"].date() if pd.notna(f["fit_end"]) else None,
                "forecast weeks": f"{f['start']:%Y-%m-%d} – {f['end']:%Y-%m-%d}",
            }
            if classes is not None:
                rec["demand class"] = key[1]
            rec.update(_scores(g, measures))
            if classes is None:
                now = totals[(totals.index >= f["start"]) & (totals.index <= f["end"])].sum()
                before = totals[(totals.index >= f["start"] - 52 * _WEEK) & (totals.index <= f["end"] - 52 * _WEEK)].sum()
                rec["sales vs a year earlier (%)"] = 100.0 * (now / before - 1.0) if before > 0 else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def block_bias(
    forecasts: pd.DataFrame,
    sales: pd.DataFrame,
    resolved: dict,
    measure: Measure,
    include_test: bool = False,
) -> dict[str, float]:
    """Forecast error (%) of one mean measure per block, keyed by block display name."""
    rows = scored_rows(forecasts, sales, resolved, [measure], include_test)
    out: dict[str, float] = {}
    if rows.empty:
        return out
    for block, g in rows.groupby("block", sort=False):
        y, f = g[f"actual_h{measure.horizon}"], g[measure.id]
        ok = y.notna() & f.notna()
        total = float(y[ok].sum())
        if total > 0:
            out[BLOCK_NAMES.get(block, block)] = 100.0 * (total - float(f[ok].sum())) / total
    return out


def trigg_signal(errors: np.ndarray, alpha: float = TRIGG_ALPHA) -> np.ndarray:
    """Trigg's smoothed tracking signal: smoothed error ÷ smoothed absolute error."""
    errors = np.asarray(errors, dtype=float)
    out = np.zeros(len(errors))
    if not len(errors):
        return out
    se, sae = 0.0, abs(errors[0])
    for i, e in enumerate(errors):
        se = alpha * e + (1.0 - alpha) * se
        sae = alpha * abs(e) + (1.0 - alpha) * sae
        out[i] = se / sae if sae > 0 else 0.0
    return out


def weekly_bias(
    forecasts: pd.DataFrame,
    sales: pd.DataFrame,
    resolved: dict,
    measure: Measure,
    include_test: bool = False,
) -> pd.DataFrame:
    """
    Weekly forecast error (%) of ``measure`` summed over all SKUs, Trigg's tracking signal
    and an alarm flag (|signal| above :data:`TRIGG_LIMIT` after a short warm-up).

    Week ``t``'s value uses errors up to week ``t`` only, so an alarm marks the week it
    would have fired.
    """
    cols = ["date", "block", "fit", "bias_pct", "tracking_signal", "alarm"]
    rows = scored_rows(forecasts, sales, resolved, [measure], include_test)
    if rows.empty or measure.id not in rows.columns:
        return pd.DataFrame(columns=cols)
    y, f = rows[f"actual_h{measure.horizon}"], rows[measure.id]
    ok = y.notna() & f.notna()
    g = (
        rows[ok].assign(err=(y - f)[ok], act=y[ok])
        .groupby(["date", "block", "fit"], sort=True)[["err", "act"]].sum()
        .reset_index()
        .sort_values("date")
    )
    g["bias_pct"] = 100.0 * g["err"] / g["act"].where(g["act"] > 0)
    g["tracking_signal"] = trigg_signal(g["bias_pct"].fillna(0.0).to_numpy())
    g["alarm"] = (g["tracking_signal"].abs() > TRIGG_LIMIT) & (np.arange(len(g)) >= TRIGG_WARMUP)
    g["block"] = g["block"].map(lambda b: BLOCK_NAMES.get(b, b))
    return g[cols].reset_index(drop=True)
