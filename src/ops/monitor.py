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
against the same weeks last year, a seasonally matched PSI, and demand-class churn since the
champion's fit. Out-of-stock (censored) SKU-weeks are left out of every error and coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.drift import trigg_signal
from src.forecasting.features.demand_profile import add_demand_profile_features, demand_class_name
from src.ops.ledger import KEYS
from src.ops.settings import Rules

LEAD_WEEKS = 2  # an order placed at the start of week k arrives at the start of week k+2
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


def bias_signals(forecasts: pd.DataFrame, actuals: pd.DataFrame, rules: Rules) -> dict:
    """Per class: the latest tracking signal, how many weeks in a row it is past the limit, errors."""
    scored = _scored(forecasts, actuals, "mean_h0", 0)
    out = {}
    for cls, g in scored.groupby("demand_class"):
        weekly = g.groupby("week")[["sales", "mean_h0"]].sum().sort_index()
        weekly = weekly[weekly["mean_h0"] > 0]
        errors = ((weekly["sales"] - weekly["mean_h0"]) / weekly["mean_h0"]).to_numpy()
        ts = trigg_signal(errors, rules.bias.alpha)
        counted = ts[rules.bias.warmup_weeks - 1:] if len(ts) >= rules.bias.warmup_weeks else np.array([])
        out[cls] = {
            "weeks": int(len(errors)),
            "ts": float(counted[-1]) if len(counted) else None,
            "streak": _streak([abs(v) > rules.bias.limit for v in counted]),
            "severe": bool(len(counted) and abs(counted[-1]) > rules.bias.severe_limit),
        }
    return out


def coverage_signals(forecasts: pd.DataFrame, actuals: pd.DataFrame, rules: Rules, column: str = "q83_h2") -> dict:
    """Per class: trailing-window coverage of ``column`` and how many weeks in a row it is out of band."""
    if column not in forecasts.columns:
        return {}
    scored = _scored(forecasts, actuals, column, 2)
    scored = scored.assign(hit=(scored["sales"] <= scored[column]).astype(float))
    lo, hi = rules.coverage.band
    w = rules.coverage.window_weeks
    out = {}
    for cls, g in scored.groupby("demand_class"):
        weekly = g.groupby("week")["hit"].agg(["sum", "count"]).sort_index()
        hits, n = weekly["sum"].rolling(w).sum(), weekly["count"].rolling(w).sum()
        cov = (hits / n).dropna()
        out[cls] = {
            "weeks": int(len(weekly)),
            "value": float(cov.iloc[-1]) if len(cov) else None,
            "streak": _streak([not lo <= v <= hi for v in cov]),
        }
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
    return out
