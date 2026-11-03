"""
The ops test: the official VN2 score of a live environment, scored once.

The live environment plays the real weekly drops from the official initial state under the frozen
rules. Its score is the competition's: the realised € of weeks ``score_from_week`` … ``window_weeks``
of the window that starts the week after the initial state, summed over SKUs. The anchor's exact
shadow, started from the same initial state, is scored on the very same SKU-weeks, so the
difference is paired (bootstrap CI over SKUs). As with the research test lock
(``src/rl/evaluate.py``), the test is scored once: scoring again needs ``override_lock`` and the
run is tagged ``post_hoc``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.ops import monitor, steps
from src.ops.ledger import KEYS
from src.ops.settings import OpsConfig, cycle_dir, env_settings, weeks
from src.rl.data import official_window_start
from src.utils import tracking

WEEK = pd.Timedelta(weeks=1)


class LockedTestError(RuntimeError):
    pass


def scored_weeks(config: dict, rules) -> list[pd.Timestamp]:
    """The weeks the competition scores: ``score_from_week`` … ``window_weeks`` of the official window."""
    start = official_window_start(config)
    return [start + (k - 1) * WEEK for k in range(rules.score_from_week, rules.window_weeks + 1)]


def score_live(base_config: dict, ops: OpsConfig, env: str = "live", override_lock: bool = False,
               n_boot: int = 2000, seed: int = 0) -> dict:
    """Score the live run against its anchor shadow and log the one ``ops-test`` run."""
    s = env_settings(ops, env)
    if s.kind != "live":
        raise ValueError(f"{env} is a {s.kind} environment; the ops test scores a live one.")
    missing = [w for w in weeks(s.start, s.end) if not (cycle_dir(base_config, env, w) / "done").exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} weeks of {env} haven't run yet (the first is {missing[0].date()}).")
    c = steps.Cycle(base_config, ops, env, pd.Timestamp(s.end))
    tracking.init_tracking(c.config)
    client, exp = MlflowClient(), steps.experiment_id(c)
    previous = client.search_runs([exp], "tags.ops_test = 'true'")
    if previous and not override_lock:
        raise LockedTestError(f"{env} was already scored (run {previous[0].info.run_id[:8]}). Pass override_lock "
                              "to score again; the new score is tagged post_hoc.")

    scored = scored_weeks(c.config, c.vn2)
    since = pd.Timestamp(s.start)
    live = steps.read_ledger(c, "costs_live.parquet", since)
    anchor = steps.read_ledger(c, "costs_anchor.parquet", since)
    # Without an RL policy at the start there's no shadow: the live trajectory is the anchor's.
    shadow = not anchor.empty
    # The six order rounds are decided in the cycles start … start + 5 weeks.
    rounds = weeks(s.start, pd.Timestamp(s.start) + (c.vn2.n_order_rounds - 1) * WEEK)
    units = {t: sum(pd.read_parquet(cycle_dir(base_config, env, w) / f"state_{t}.parquet")["order"].sum() for w in rounds)
             for t in (("live", "anchor") if shadow else ("live",))}
    classes = monitor.demand_classes(steps.panel(c), c.config)
    per_sku, metrics = paired_score(live, anchor if shadow else live, scored,
                                    classes[classes["date"] == scored[0]].set_index(KEYS)["demand_class"], n_boot, seed)
    metrics["units_vs_anchor_pct"] = float(100.0 * (units["live"] / units["anchor"] - 1.0)) if shadow else 0.0
    tags = {"stage": "ops-test", "ops_test": "true", "post_hoc": str(bool(previous)).lower(), "env": env,
            "scored_weeks": f"{scored[0].date()}..{scored[-1].date()}", "rules_version": c.rules.version(),
            "anchor_shadow": str(shadow).lower()}
    with mlflow.start_run(run_name="ops-test", experiment_id=exp, tags=tags) as run:
        tracking.log_provenance(c.config)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_sku.csv"
            per_sku.reset_index().to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="ops_test")
    return {"run_id": run.info.run_id, **tags, **metrics}


def paired_score(live: pd.DataFrame, anchor: pd.DataFrame, scored: list[pd.Timestamp], demand_class: pd.Series,
                 n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Per-SKU € on the scored weeks of both trajectories, and the competition score of each with the paired gap."""
    live, anchor = (x[x["week"].isin(scored)] for x in (live, anchor))
    per_sku = (live.groupby(KEYS)["cost_eur"].sum().rename("live").to_frame()
               .join(anchor.groupby(KEYS)["cost_eur"].sum().rename("anchor"), how="inner")
               .join(demand_class.rename("demand_class")))
    diff, base = (per_sku["live"] - per_sku["anchor"]).to_numpy(), per_sku["anchor"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n_boot, len(diff)))
    boot = 100.0 * diff[idx].sum(axis=1) / base[idx].sum(axis=1)
    by_class = per_sku.groupby("demand_class")[["live", "anchor"]].sum()
    n = len(per_sku)
    return per_sku, {
        "live_vn2_score_eur": float(per_sku["live"].sum()),
        "anchor_vn2_score_eur": float(per_sku["anchor"].sum()),
        "live_eur_per_sku": float(per_sku["live"].sum() / n),
        "anchor_eur_per_sku": float(per_sku["anchor"].sum() / n),
        "live_vs_anchor_pct": float(100.0 * diff.sum() / base.sum()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "n_skus": float(n),
        **{f"live_vs_anchor_pct_{str(k).lower()}": float(100.0 * (r["live"] / r["anchor"] - 1.0))
           for k, r in by_class.iterrows() if r["anchor"] > 0},
    }
