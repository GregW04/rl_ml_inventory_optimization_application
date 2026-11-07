"""
Drift alerts of a forecasting run: the production loop's bias alarm applied to the run's fits.

The detector is the ops loop's (``src/ops/monitor.py``, thresholds from the frozen
``configs/ops.yaml``). For each demand class, Trigg's tracking signal on the weekly relative
error of the h0 level, Σ(sales − forecast) / Σ forecast over in-stock SKU-weeks, restarted at
every refit. It counts from a fit's ``warmup_weeks``-th week, and a class alerts once the signal
has been past the limit for ``persistence_weeks`` weeks in a row. Unlike ops there is no
cost-share filter: a forecasting run has no inventory costs. Unclassified SKUs are left out.

Alerts never change anything: refits follow the fixed ``refit_every_weeks`` schedule. A fit
shorter than ``warmup_weeks + persistence_weeks − 1`` weeks ends before it can alert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.drift import run_fits
from src.ops import monitor
from src.ops.ledger import KEYS
from src.ops.settings import Rules, load_ops_config

WEEK = pd.Timedelta(weeks=1)
COLUMNS = ["block", "fit", "fit_start", "week", "week_of_fit", "demand_class", "error_pct",
           "tracking_signal", "streak", "alert", "severe"]


def default_rules() -> Rules:
    """The frozen production rules: research alerts use the same thresholds."""
    return load_ops_config().rules


def describe(rules: Rules) -> str:
    b = rules.bias
    return (f"Trigg {b.alpha} per demand class, restarted at each refit, from week {b.warmup_weeks}; "
            f"alert past ±{b.limit} for {b.persistence_weeks} weeks, severe past ±{b.severe_limit}")


def earliest_alert_week(rules: Rules) -> int:
    """The first week of a fit in which an alert can fire."""
    return rules.bias.warmup_weeks + rules.bias.persistence_weeks - 1


def class_actuals(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Weekly in-stock sales (NaN when out of stock) with each SKU-week's demand class."""
    sales = panel[[*KEYS, "date", "sales"]].assign(date=lambda d: pd.to_datetime(d["date"]))
    classes = monitor.demand_classes(sales, config)
    return sales.merge(classes, on=[*KEYS, "date"])


def fit_alerts(forecasts: pd.DataFrame, actuals: pd.DataFrame, resolved: dict, level: str,
               rules: Rules) -> pd.DataFrame:
    """
    One row per fit × week × demand class: the weekly error (%), the tracking signal, the weeks in
    a row past the limit, and whether that is an alert. Every block is included; callers hide the
    test block until it has been scored.
    """
    if level not in forecasts.columns:
        return pd.DataFrame(columns=COLUMNS)
    fc = forecasts.assign(date=pd.to_datetime(forecasts["date"]))
    act = actuals[actuals["demand_class"] != "Unclassified"]
    parts = []
    for f in run_fits(resolved, include_test=True):
        rows = fc[(fc["date"] >= f["start"]) & (fc["date"] <= f["end"])]
        if rows.empty:
            continue
        errors = monitor.weekly_errors(rows, act, column=level)
        for cls in errors.columns:
            e = errors[cls].dropna()
            if e.empty:
                continue
            s = monitor.bias_series(e.to_numpy(), rules)
            parts.append(pd.DataFrame({
                "block": f["block"], "fit": f["label"], "fit_start": f["start"], "week": e.index,
                "week_of_fit": ((e.index - f["start"]) // WEEK + 1).astype(int), "demand_class": str(cls),
                "error_pct": 100.0 * e.to_numpy(), "tracking_signal": s["ts"].to_numpy(),
                "streak": s["streak"].to_numpy(), "alert": s["alert"].to_numpy(), "severe": s["severe"].to_numpy(),
            }))
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def first_alert(forecasts: pd.DataFrame, actuals: pd.DataFrame, level: str, rules: Rules) -> tuple[pd.Timestamp | None, list[str]]:
    """
    The first week in which any demand class alerts on a fit's forecasts (errors counted from the first
    forecast week), and the classes alerting that week; ``(None, [])`` when none does. Week ``t`` uses
    errors up to ``t`` only, so it is the week the alarm would have fired.
    """
    if forecasts.empty or level not in forecasts.columns:
        return None, []
    errors = monitor.weekly_errors(forecasts.assign(date=pd.to_datetime(forecasts["date"])),
                                   actuals[actuals["demand_class"] != "Unclassified"], column=level)
    best: tuple[pd.Timestamp, list[str]] | None = None
    for cls in errors.columns:
        e = errors[cls].dropna()
        hits = np.flatnonzero(monitor.bias_series(e.to_numpy(), rules)["alert"].to_numpy()) if len(e) else []
        if len(hits):
            week = e.index[hits[0]]
            if best is None or week < best[0]:
                best = (week, [str(cls)])
            elif week == best[0]:
                best[1].append(str(cls))
    return (best[0], sorted(best[1])) if best else (None, [])


def alert_weeks(alerts: pd.DataFrame, block: str) -> int:
    """Weeks of ``block`` in which at least one class alerted."""
    return int(alerts.loc[(alerts["block"] == block) & alerts["alert"], "week"].nunique())


def first_alert_marks(alerts: pd.DataFrame) -> pd.DataFrame:
    """
    Where alerts start: one row per fit and week, with the classes that alerted for the first time in that fit
    that week. A class's later alert weeks in the same fit are left out, so a chart gets one mark per alert.
    """
    cols = ["block", "fit", "week", "classes"]
    if alerts.empty or not alerts["alert"].any():
        return pd.DataFrame(columns=cols)
    hits = alerts[alerts["alert"]].sort_values("week", kind="stable")
    first = hits.groupby(["block", "fit", "demand_class"], sort=False, as_index=False)["week"].min()
    marks = (first.groupby(["block", "fit", "week"], sort=False)["demand_class"]
             .agg(lambda c: ", ".join(sorted(c))).rename("classes").reset_index())
    return marks[cols].sort_values("week", kind="stable").reset_index(drop=True)


def alert_summary(alerts: pd.DataFrame) -> pd.DataFrame:
    """Per block × class: alert weeks, the first alert, and the largest |tracking signal|."""
    if alerts.empty:
        return pd.DataFrame(columns=["block", "demand_class", "alert_weeks", "first_alert", "max_abs_signal"])
    g = alerts.groupby(["block", "demand_class"], sort=False)
    return pd.DataFrame({
        "alert_weeks": g["alert"].sum().astype(int),
        "first_alert": g.apply(lambda x: x.loc[x["alert"], "week"].min(), include_groups=False),
        "max_abs_signal": g["tracking_signal"].apply(lambda x: x.abs().max()),
    }).reset_index()
