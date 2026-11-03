"""
Calibration of the drift monitors on a shadow replay, before the rules are frozen.

The decision rule is replayed week by week on weekly per-class series, by the loop's own code
(``monitor.bias_state`` / ``coverage_state`` and ``decide``), so a calibrated rate is the rate of
the rule that runs. A refit restarts the signals as in the loop: the bias signal from the next
week's ``mean_h0``, coverage three weeks later (the first week the new ``q83_h2`` covers).

* **Null paths** (no drift): the replay's own series with the slow drift taken out (each class's
  centred rolling mean), resampled in week blocks jointly across classes and both monitors, so
  weekly shocks shared by classes and short-run autocorrelation survive.
* **False refits per 26 weeks**: the drift-triggered refits the rule makes on null paths.
* **First alarm**: when the rule would first refit on the replay's real series (detection delay).
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.ops import monitor, steps
from src.ops.decide import decide
from src.ops.monitor import LEAD_WEEKS, bias_state, coverage_state
from src.ops.settings import Rules, cycle_dir, env_dir, env_rules, env_settings, weeks
from src.utils import tracking


def simulate_refits(errors: pd.DataFrame, hits: pd.DataFrame, counts: pd.DataFrame, share: dict, rules: Rules) -> list[int]:
    """
    Week positions where ``decide`` would refit, on week × class series aligned by scored week
    (position 0 = the bootstrap model's first forecast week).
    """
    weeks = errors.index.union(hits.index)
    e_classes, h_classes = [str(c) for c in errors.columns], [str(c) for c in hits.columns]
    E = errors.reindex(weeks).to_numpy(float)
    H, N = hits.reindex(weeks).to_numpy(float), counts.reindex(weeks).to_numpy(float)
    refits: list[int] = []
    first_error, first_hit = 0, 0
    for t in range(len(weeks)):
        bias, coverage = {}, {}
        for j, c in enumerate(e_classes):
            x = E[first_error:t + 1, j]
            bias[c] = bias_state(x[~np.isnan(x)], rules)
        for j, c in enumerate(h_classes):
            h, n = H[first_hit:t + 1, j], N[first_hit:t + 1, j]
            coverage[c] = coverage_state(h[~np.isnan(h)], n[~np.isnan(h)], rules)
        signals = {"champion": {"age_weeks": t - first_error + 1}, "cost_share": share, "bias": bias, "coverage": coverage}
        if decide(signals, rules).action == "refit":
            refits.append(t)
            first_error, first_hit = t + 1, t + 1 + LEAD_WEEKS
    return refits


def detrend(errors: pd.DataFrame, hits: pd.DataFrame, counts: pd.DataFrame, trend_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Errors around 0 and hit rates around each class's mean, with the centred rolling trend removed."""
    def trend(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(trend_weeks, center=True, min_periods=max(2, trend_weeks // 2)).mean()

    rate = hits / counts
    return errors - trend(errors), rate - trend(rate) + rate.mean()


def null_paths(errors: pd.DataFrame, hits: pd.DataFrame, counts: pd.DataFrame, n_paths: int, length: int,
               block_weeks: int, trend_weeks: int, seed: int):
    """Yield (errors, hits, counts) of ``n_paths`` detrended paths resampled in blocks of ``block_weeks``."""
    e0, r0 = detrend(errors, hits, counts, trend_weeks)
    common = e0.index.intersection(r0.index)  # coverage is scored two weeks after the errors start
    e0, r0, n0 = e0.loc[common], r0.loc[common], counts.loc[common]
    valid = np.flatnonzero(e0.notna().all(axis=1).to_numpy() & r0.notna().all(axis=1).to_numpy())
    e0, r0, n0 = e0.iloc[valid], r0.iloc[valid], n0.iloc[valid]
    rng = np.random.default_rng(seed)
    n_blocks = -(-length // block_weeks)
    for _ in range(n_paths):
        starts = rng.integers(0, len(e0) - block_weeks + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_weeks) for s in starts])[:length]
        e, n = e0.iloc[idx].reset_index(drop=True), n0.iloc[idx].reset_index(drop=True)
        yield e, (r0.iloc[idx].reset_index(drop=True).clip(0, 1) * n), n


def false_refits_per_26_weeks(errors: pd.DataFrame, hits: pd.DataFrame, counts: pd.DataFrame, share: dict,
                              rules: Rules, n_paths: int = 200, length: int = 78, block_weeks: int = 4,
                              trend_weeks: int = 13, seed: int = 0) -> float:
    """Mean drift-triggered refits per 26 weeks on null paths (``rules.strategy`` should be ``drift``)."""
    paths = null_paths(errors, hits, counts, n_paths, length, block_weeks, trend_weeks, seed)
    return float(np.mean([len(simulate_refits(e, h, n, share, rules)) for e, h, n in paths]) * 26 / length)


def env_costs(config: dict, env: str) -> pd.DataFrame:
    """The realised costs of an environment's live trajectory, every week it has scored."""
    parts = [pd.read_parquet(p) for p in sorted((env_dir(config, env) / "cycles").glob("*/costs_live.parquet"))]
    return pd.concat(parts, ignore_index=True)


def compare_strategies(costs: dict[str, pd.DataFrame], reference: str, block_weeks: int = 4, n_boot: int = 2000,
                       seed: int = 0) -> pd.DataFrame:
    """
    € per SKU-week of each environment on the in-stock SKU-weeks every environment scored, and its
    difference to ``reference`` (%), with a bootstrap CI over blocks of ``block_weeks`` weeks.
    """
    keys = ["store", "product", "week"]
    common = None
    for c in costs.values():
        k = c.loc[~c["censored"], keys]
        common = k if common is None else common.merge(k, on=keys)
    weekly = pd.DataFrame({env: c.merge(common, on=keys).groupby("week")["cost_eur"].sum() for env, c in costs.items()})
    n_sku_weeks = len(common)
    blocks = np.arange(len(weekly)) // block_weeks
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, blocks.max() + 1, size=(n_boot, blocks.max() + 1))
    rows = []
    for env in weekly.columns:
        diff, base = weekly[env] - weekly[reference], weekly[reference]
        d_b = pd.Series(diff.to_numpy()).groupby(blocks).sum().to_numpy()
        b_b = pd.Series(base.to_numpy()).groupby(blocks).sum().to_numpy()
        boot = 100.0 * d_b[picks].sum(axis=1) / b_b[picks].sum(axis=1)
        rows.append({"env": env, "eur_per_sku_week": float(weekly[env].sum() / n_sku_weeks),
                     "vs_reference_pct": float(100.0 * diff.sum() / base.sum()),
                     "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows).set_index("env")


def calibration_inputs(base_config: dict, ops, env: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Weekly per-class errors, coverage hits and counts of an environment's as-issued champion
    forecasts, as the monitor step computes them, and each class's share of the realised cost.
    """
    s = env_settings(ops, env)
    start = pd.Timestamp(s.start)
    c = steps.Cycle(base_config, ops, env, pd.Timestamp(s.end))
    panel = steps.panel(c)
    classes = monitor.demand_classes(panel, c.config)
    actuals = panel[["store", "product", "date", "sales"]].merge(classes, on=["store", "product", "date"])
    forecasts = steps.read_ledger(c, "forecasts_champion.parquet", start)
    errors = monitor.weekly_errors(forecasts, actuals)
    hits, counts = monitor.weekly_hits(forecasts, actuals)
    costs = steps.read_ledger(c, "costs_live.parquet", start).merge(
        classes.rename(columns={"date": "week"}), on=["store", "product", "week"])
    by_class = costs[~costs["censored"]].groupby("demand_class")["cost_eur"].sum()
    return errors, hits, counts, {str(k): float(v / by_class.sum()) for k, v in by_class.items()}


def _with(rules: Rules, **changes) -> Rules:
    """``rules`` with nested fields changed, e.g. ``bias={"limit": 0.8}``, validated again."""
    data = rules.model_dump()
    for section, values in changes.items():
        data[section] = {**data[section], **values} if isinstance(values, dict) else values
    return Rules.model_validate(data)


def coverage_profile(hits: pd.DataFrame, counts: pd.DataFrame, window_weeks: int, trend_weeks: int = 13) -> pd.DataFrame:
    """Per class: mean coverage, the spread of the trailing-window coverage around its trend, and the design effect."""
    rows = []
    for cls in hits.columns:
        h, n = hits[cls].dropna(), counts[cls].dropna()
        w_cov = h.rolling(window_weeks).sum() / n.rolling(window_weeks).sum()
        resid = (w_cov - w_cov.rolling(trend_weeks, center=True, min_periods=max(2, trend_weeks // 2)).mean()).dropna()
        p, n_window = float(h.sum() / n.sum()), float(n.mean() * window_weeks)
        binom = np.sqrt(p * (1 - p) / n_window)
        rows.append({"class": str(cls), "coverage": p, "window_sd": float(resid.std()), "binomial_sd": float(binom),
                     "design_effect": float((resid.std() / binom) ** 2), "sku_weeks_per_window": n_window})
    return pd.DataFrame(rows).set_index("class")


def threshold_grid(errors: pd.DataFrame, hits: pd.DataFrame, counts: pd.DataFrame, share: dict, base: Rules,
                   limits=(0.6, 0.65, 0.7, 0.74, 0.8, 0.85, 0.9), persistences=(1, 2, 3),
                   half_widths=(0.03, 0.04, 0.05, 0.06, 0.08), centres=(0.8333,), **null) -> dict[str, pd.DataFrame]:
    """
    False refits per 26 weeks on null paths, and the first refit on the real series, for each bias
    (limit, persistence) with coverage off, and each coverage band (centre ± half width) with bias off.
    """
    # Only classes carrying enough of the cost can trigger; the others would only thin the null paths.
    big = [c for c in errors.columns if share.get(str(c), 0.0) >= base.bias.min_cost_share]
    errors, hits, counts = errors[big], hits[big], counts[big]
    weeks = errors.index.union(hits.index)
    off_cov = {"band": [0.0, 1.0]}
    off_bias = {"limit": 1.01, "severe_limit": 1.01}
    bias_rows, cov_rows = [], []
    for limit in limits:
        for persistence in persistences:
            r = _with(base, strategy="drift", coverage=off_cov, bias={"limit": limit, "persistence_weeks": persistence})
            first = simulate_refits(errors, hits, counts, share, r)
            bias_rows.append({"limit": limit, "persistence": persistence,
                              "false_refits_per_26w": false_refits_per_26_weeks(errors, hits, counts, share, r, **null),
                              "first_refit": str(weeks[first[0]].date()) if first else None})
    for centre in centres:
        for half in half_widths:
            r = _with(base, strategy="drift", bias=off_bias, coverage={"band": [round(centre - half, 4), round(centre + half, 4)]})
            first = simulate_refits(errors, hits, counts, share, r)
            cov_rows.append({"centre": round(centre, 4), "band": f"{centre - half:.3f}–{centre + half:.3f}",
                             "false_refits_per_26w": false_refits_per_26_weeks(errors, hits, counts, share, r, **null),
                             "first_refit": str(weeks[first[0]].date()) if first else None})
    return {"bias": pd.DataFrame(bias_rows), "coverage": pd.DataFrame(cov_rows)}


def calibration_run(base_config: dict, ops, env: str, n_paths: int = 200, seed: int = 0) -> dict[str, pd.DataFrame]:
    """The calibration tables of an environment's replay, saved under ``ops/<env>/calibration/`` and logged to MLflow."""
    errors, hits, counts, share = calibration_inputs(base_config, ops, env)
    rules = env_rules(ops, env)
    observed = float(hits.sum().sum() / counts.sum().sum())  # the recipe's own coverage, pooled
    tables = {"coverage_profile": coverage_profile(hits, counts, rules.coverage.window_weeks).reset_index(),
              **threshold_grid(errors, hits, counts, share, rules, centres=(0.8333, round(observed, 4)),
                               n_paths=n_paths, seed=seed),
              "cost_share": pd.DataFrame([share])}
    folder = env_dir(base_config, env) / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=False)
    errors.to_csv(folder / "weekly_errors.csv")
    (hits / counts).to_csv(folder / "weekly_coverage.csv")
    tracking.init_tracking(base_config)
    with mlflow.start_run(run_name=f"calibration-{env}", experiment_id=tracking.get_or_create_experiment("vn2-ops-calibration"),
                          tags={"stage": "calibration", "env": env}):
        tracking.log_provenance(base_config)
        mlflow.log_params({"env": env, "n_paths": n_paths, "seed": seed, "weeks": len(errors)})
        mlflow.log_artifacts(str(folder), artifact_path="calibration")
    return tables


def strategy_report(base_config: dict, ops, envs: list[str], reference: str) -> pd.DataFrame:
    """The strategies side by side: paired € against ``reference``, refits and their compute, logged to MLflow."""
    table = compare_strategies({e: env_costs(base_config, e) for e in envs}, reference)
    tracking.init_tracking(base_config)
    client = MlflowClient()
    for env in envs:
        s = env_settings(ops, env)
        actions = [json.loads((cycle_dir(base_config, env, w) / "decision.json").read_text())["action"]
                   for w in weeks(s.start, s.end) if (cycle_dir(base_config, env, w) / "decision.json").exists()]
        exp = client.get_experiment_by_name(f"vn2-ops-{env}")
        fits = [r for r in client.search_runs([exp.experiment_id], max_results=5000)  # IN isn't allowed on tags
                if r.data.tags.get("stage") in ("bootstrap", "refit")]
        table.loc[env, "refits"] = actions.count("refit")
        table.loc[env, "rollbacks"] = actions.count("rollback")
        # The fit itself (runs before 2026-11-03 didn't record it: NaN).
        seconds = [r.data.metrics.get("fit_seconds", np.nan) for r in fits]
        table.loc[env, "fit_minutes"] = float(np.sum(seconds)) / 60 if fits else 0.0
    folder = env_dir(base_config, reference) / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(folder / "strategies.csv", index=False)
    with mlflow.start_run(run_name="strategies", experiment_id=tracking.get_or_create_experiment("vn2-ops-calibration"),
                          tags={"stage": "strategies", "reference": reference}):
        tracking.log_provenance(base_config)
        mlflow.log_params({"envs": ",".join(envs), "reference": reference})
        mlflow.log_artifact(str(folder / "strategies.csv"), artifact_path="calibration")
    return table
