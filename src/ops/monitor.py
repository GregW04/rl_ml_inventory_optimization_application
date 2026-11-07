"""
Weekly monitors of the ops loop: the numbers the decision engine reads.

Triggers, on the current champion's forecasts (a forecast is scored once its week has sold):

* **bias**: Trigg's tracking signal per demand class on the weekly relative error of
  ``mean_h0``, Σ(sales − forecast) / Σ forecast, restarted at the champion's promotion (a new
  version starts from scratch). It counts once ``warmup_weeks`` errors exist.
* **coverage**: share of SKU-weeks with sales ≤ ``q83_h2`` over a trailing window, per class.

Business: while a new champion is on probation, the realised € of the live trajectory against
the shadow running the previous forecaster, on the weeks where the two can differ (from three
weeks after promotion: orders take two weeks to arrive), with a bootstrap CI over SKUs.

Diagnostics, logged and never a trigger: out-of-stock share, zero-sales share and sales level
against the same weeks last year, a seasonally matched PSI, demand-class churn since the
champion's fit, and each class's seasonally adjusted 13-week level change (MSTL). Out-of-stock
(censored) SKU-weeks are left out of every error and coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.drift import trigg_signal
from src.forecasting.features.demand_profile import add_demand_profile_features, demand_class_name
from src.ops.ledger import KEYS
from src.ops.settings import Rules

LEAD_WEEKS = 2  # an order placed at the start of week k arrives at the start of week k+2
YEAR_WEEKS = 52
TREND_WEEKS = 13
COST_SHARE_WEEKS = 13
PSI_EDGES = [-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, np.inf]
WEEK = pd.Timedelta(weeks=1)


def demand_classes(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Demand class of every SKU-week, from the trailing window before that week (point in time)."""
    prof_cfg = dict(config.get("forecasting", {}).get("features", {}).get("demand_profile") or {})
    cfg = {"forecasting": {"features": {"demand_profile": {**prof_cfg, "enabled": True}}}}
    prof = add_demand_profile_features(panel[[*KEYS, "date", "sales"]].copy(), config=cfg)
    return prof[[*KEYS, "date"]].assign(demand_class=[demand_class_name(c) for c in prof["dp_sbc_class"]])


def _streak(flags: list[bool]) -> int:
    """Consecutive True values at the end of ``flags``."""
    n = 0
    for f in reversed(flags):
        if not f:
            break
        n += 1
    return n


def _scored(forecasts: pd.DataFrame, actuals: pd.DataFrame, column: str, horizon: int) -> pd.DataFrame:
    """Forecasts of ``column`` joined to the in-stock sales of the week they forecast."""
    f = forecasts[[*KEYS, "date", column]].assign(week=forecasts["date"] + horizon * WEEK).drop(columns="date")
    out = f.merge(actuals.rename(columns={"date": "week"}), on=[*KEYS, "week"])
    return out[out["sales"].notna()]


def weekly_errors(forecasts: pd.DataFrame, actuals: pd.DataFrame, column: str = "mean_h0") -> pd.DataFrame:
    """Week × class relative error of an h0 ``column``: Σ(sales − forecast) / Σ forecast over in-stock SKU-weeks."""
    scored = _scored(forecasts, actuals, column, 0)
    weekly = scored.groupby(["week", "demand_class"])[["sales", column]].sum()
    weekly = weekly[weekly[column] > 0]
    return ((weekly["sales"] - weekly[column]) / weekly[column]).unstack("demand_class").sort_index()


def weekly_hits(forecasts: pd.DataFrame, actuals: pd.DataFrame, column: str = "q83_h2") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Week × class count of in-stock SKU-weeks with sales ≤ ``column``, and of all scored ones."""
    scored = _scored(forecasts, actuals, column, 2)
    scored = scored.assign(hit=(scored["sales"] <= scored[column]).astype(float))
    weekly = scored.groupby(["week", "demand_class"])["hit"].agg(["sum", "count"])
    return weekly["sum"].unstack("demand_class").sort_index(), weekly["count"].unstack("demand_class").sort_index()


def _bias_arrays(errors: np.ndarray, rules: Rules) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per week: the tracking signal (NaN before the warm-up ends), weeks in a row past the limit, severity."""
    ts = trigg_signal(errors, rules.bias.alpha)
    counted = np.arange(len(ts)) >= rules.bias.warmup_weeks - 1
    streak = np.zeros(len(ts), dtype=int)
    run = 0
    for i, (c, v) in enumerate(zip(counted, ts)):
        run = run + 1 if c and abs(v) > rules.bias.limit else 0
        streak[i] = run
    return np.where(counted, ts, np.nan), streak, counted & (np.abs(ts) > rules.bias.severe_limit)


def bias_series(errors: np.ndarray, rules: Rules) -> pd.DataFrame:
    """
    One class's bias monitor week by week, as :func:`bias_state` would report it after each week:
    ``ts``, ``streak``, ``alert`` (past the limit for ``persistence_weeks`` in a row; the drift
    alarm before any cost-share or cooldown rule) and ``severe``.
    """
    ts, streak, severe = _bias_arrays(errors, rules)
    return pd.DataFrame({"ts": ts, "streak": streak, "alert": streak >= rules.bias.persistence_weeks, "severe": severe})


def bias_state(errors: np.ndarray, rules: Rules) -> dict:
    """One class's tracking signal since promotion: latest value, weeks in a row past the limit, severity."""
    ts, streak, severe = _bias_arrays(errors, rules)
    last = len(ts) - 1
    return {
        "weeks": int(len(errors)),
        "ts": float(ts[last]) if last >= 0 and not np.isnan(ts[last]) else None,
        "streak": int(streak[last]) if last >= 0 else 0,
        "severe": bool(last >= 0 and severe[last]),
    }


def coverage_state(hits: np.ndarray, counts: np.ndarray, rules: Rules) -> dict:
    """One class's trailing-window coverage since promotion and weeks in a row out of band."""
    lo, hi = rules.coverage.band
    w = rules.coverage.window_weeks
    cov = [hits[i - w + 1:i + 1].sum() / counts[i - w + 1:i + 1].sum() for i in range(w - 1, len(hits))
           if counts[i - w + 1:i + 1].sum() > 0]
    return {"weeks": int(len(hits)), "value": float(cov[-1]) if cov else None,
            "streak": _streak([not lo <= v <= hi for v in cov])}


def bias_signals(forecasts: pd.DataFrame, actuals: pd.DataFrame, rules: Rules) -> dict:
    """Per class: the latest tracking signal, how many weeks in a row it is past the limit."""
    errors = weekly_errors(forecasts, actuals)
    return {str(cls): bias_state(errors[cls].dropna().to_numpy(), rules) for cls in errors.columns}


def coverage_signals(forecasts: pd.DataFrame, actuals: pd.DataFrame, rules: Rules, column: str = "q83_h2") -> dict:
    """Per class: trailing-window coverage of ``column`` and how many weeks in a row it is out of band."""
    if column not in forecasts.columns:
        return {}
    hits, counts = weekly_hits(forecasts, actuals, column)
    out = {}
    for cls in hits.columns:
        h = hits[cls].dropna()  # the weeks the class was scored
        out[str(cls)] = coverage_state(h.to_numpy(), counts.loc[h.index, cls].to_numpy(), rules)
    return out


def cost_share(costs: pd.DataFrame) -> dict:
    """Share of the realised cost per demand class over the trailing weeks (censored weeks left out)."""
    recent = costs[(~costs["censored"]) & (costs["week"] > costs["week"].max() - COST_SHARE_WEEKS * WEEK)]
    by_class = recent.groupby("demand_class")["cost_eur"].sum()
    total = float(by_class.sum())
    return {str(k): float(v / total) for k, v in by_class.items()} if total > 0 else {}


def paired_delta(live: pd.DataFrame, shadow: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp, rules: Rules) -> dict:
    """Realised € of the live trajectory against a shadow on weeks ``first``…``last``, with a bootstrap CI over SKUs."""
    cols = [*KEYS, "week", "cost_eur", "censored"]
    both = live[cols].merge(shadow[cols], on=[*KEYS, "week"], suffixes=("_live", "_shadow"))
    both = both[(both["week"] >= first) & (both["week"] <= last) & ~both["censored_live"]]
    n_weeks = int(both["week"].nunique())
    if not n_weeks:
        return {"evidence_weeks": 0}
    per_sku = both.groupby(KEYS)[["cost_eur_live", "cost_eur_shadow"]].sum()
    diff, base = (per_sku["cost_eur_live"] - per_sku["cost_eur_shadow"]).to_numpy(), per_sku["cost_eur_shadow"].to_numpy()
    rng = np.random.default_rng(rules.seed)
    idx = rng.integers(0, len(diff), size=(rules.n_boot, len(diff)))
    boot = 100.0 * diff[idx].sum(axis=1) / np.maximum(base[idx].sum(axis=1), 1e-9)
    return {
        "evidence_weeks": n_weeks,
        "delta_eur": float(diff.sum()),
        "delta_pct": float(100.0 * diff.sum() / max(base.sum(), 1e-9)),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
    }


def probation_signal(live: pd.DataFrame, shadow: pd.DataFrame, promoted: pd.Timestamp, rules: Rules) -> dict:
    """A new forecaster against the previous one's shadow, on the weeks its orders can affect."""
    first = promoted + (LEAD_WEEKS + 1) * WEEK
    return paired_delta(live, shadow, first, first + (rules.probation.evidence_weeks - 1) * WEEK, rules)


def policy_signal(live: pd.DataFrame, anchor: pd.DataFrame, promoted: pd.Timestamp, as_of: pd.Timestamp, rules: Rules) -> dict:
    """An RL policy against the anchor's shadow over the latest evidence weeks since its orders took effect."""
    first = max(promoted + (LEAD_WEEKS + 1) * WEEK, as_of - (rules.policy.evidence_weeks - 1) * WEEK)
    return paired_delta(live, anchor, first, as_of, rules)


def gate_churn(classes: pd.DataFrame, gate: list[str], fit_as_of: pd.Timestamp, as_of: pd.Timestamp) -> float:
    """Share of SKUs that moved into or out of the policy's gate since the policy was trained."""
    a = classes[classes["date"] == as_of].set_index(KEYS)["demand_class"].isin(gate)
    b = classes[classes["date"] == fit_as_of].set_index(KEYS)["demand_class"].isin(gate)
    both = a.to_frame("now").join(b.rename("fit"), how="inner")
    return float((both["now"] != both["fit"]).mean()) if len(both) else 0.0


def _psi(a: np.ndarray, b: np.ndarray) -> float:
    pa = (np.histogram(a, PSI_EDGES)[0] + 0.5) / (len(a) + 0.5 * (len(PSI_EDGES) - 1))
    pb = (np.histogram(b, PSI_EDGES)[0] + 0.5) / (len(b) + 0.5 * (len(PSI_EDGES) - 1))
    return float(((pa - pb) * np.log(pa / pb)).sum())


def diagnostics(panel: pd.DataFrame, classes: pd.DataFrame, as_of: pd.Timestamp, fit_as_of: pd.Timestamp | None) -> dict:
    """Input-side drift: logged with every cycle, never a trigger on its own."""
    now = panel[panel["date"] == as_of]
    ly = as_of - 52 * WEEK
    recent = panel[(panel["date"] > as_of - 4 * WEEK) & (panel["date"] <= as_of)]
    last_year = panel[(panel["date"] > ly - 4 * WEEK) & (panel["date"] <= ly)]
    out = {"out_of_stock_share": float(1.0 - now["in_stock"].astype(bool).mean())}
    if len(last_year):
        out["level_vs_last_year_pct"] = float(100.0 * (recent["sales"].sum() / max(last_year["sales"].sum(), 1e-9) - 1.0))
        out["psi_vs_last_year"] = _psi(recent["sales"].dropna().to_numpy(), last_year["sales"].dropna().to_numpy())
        ly_week = panel[panel["date"] == ly]["sales"].dropna()
        out["zero_share_change"] = float((now["sales"].dropna() == 0).mean() - (ly_week == 0).mean())
    if fit_as_of is not None:
        a = classes[classes["date"] == as_of].set_index(KEYS)["demand_class"]
        b = classes[classes["date"] == fit_as_of].set_index(KEYS)["demand_class"]
        both = a.to_frame("now").join(b.rename("fit"), how="inner")
        out["class_churn_since_fit"] = float((both["now"] != both["fit"]).mean()) if len(both) else 0.0
    out.update(trend_changes(panel, classes, as_of))
    return out


def trend_changes(panel: pd.DataFrame, classes: pd.DataFrame, as_of: pd.Timestamp, weeks: int = TREND_WEEKS) -> dict:
    """
    Per class, how much the seasonally adjusted sales level moved over the last ``weeks`` (%): the
    weekly sales of the SKUs in the class now, minus the seasonal component of an MSTL
    decomposition (yearly period, robust), averaged over 4 weeks, now against ``weeks`` earlier.
    A model-free view of level shifts next to the forecast-bias monitor; it needs two full years.
    (The decomposition's own trend is smoothed over more than a year and lags a recent shift.)
    """
    from statsmodels.tsa.seasonal import MSTL  # only here: keeps every other step's start-up light

    current = classes[classes["date"] == as_of][[*KEYS, "demand_class"]]
    hist = panel[panel["date"] <= as_of][[*KEYS, "date", "sales"]].merge(current, on=KEYS)
    out = {}
    for cls, g in hist.groupby("demand_class"):
        series = g.groupby("date")["sales"].sum().sort_index().to_numpy()
        if len(series) < 2 * YEAR_WEEKS + weeks + 4:
            continue
        adjusted = series - MSTL(series, periods=YEAR_WEEKS, stl_kwargs={"robust": True}).fit().seasonal
        now, then = adjusted[-4:].mean(), adjusted[-weeks - 4:-weeks].mean()
        if then > 0:
            out[f"level_change_{weeks}w_pct_{str(cls).lower()}"] = float(100.0 * (now / then - 1.0))
    return out
