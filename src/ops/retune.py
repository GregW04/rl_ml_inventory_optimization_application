"""
An operator's retune (``python -m src.ops.cli challenge``): a new recipe of ``configs/ops.yaml``
(tuned with ``cli tune``) challenges an environment's champion on its latest weeks. The loop only
refits its recipe; a new recipe has to win first.

**Holdout.** The challenger is fit as a refit would have been ``holdout_weeks`` ago (cycle A), then
issues the forecasts of every cycle since from that week's data, as the champion did (its
as-issued forecasts are in the ledger). Both drive the anchor from the live stock of cycle A, so
the policy is held fixed and only the forecasts differ. Their realised € is paired by SKU on the
weeks where the orders can differ (from A+3: lead time 2), with a bootstrap CI.

**Outcome.** A challenger that wins by ``margin_pct`` with the CI below 0 becomes a request
(``ops/<env>/pending_recipe.json``). The next cycle decides **retune** and its refit step fits the
recipe on all the data, sanity-checks and promotes it, with the usual probation against the old
champion. Every challenge is a run (``stage=challenge``) in the environment's experiment.
"""

from __future__ import annotations

import json

import mlflow
import pandas as pd

from src.forecasting.measures import infer_measures
from src.forecasting.model import build_feature_matrix
from src.forecasting.recipe import Recipe, fit_recipe, forecast
from src.ops import ledger, monitor
from src.ops.settings import OpsConfig, cycle_dir, dump, env_dir, env_settings, recipe_config, weeks
from src.simulation.anchor import LostSalesAnchor, resolve_anchor
from src.utils import tracking

WEEK = pd.Timedelta(weeks=1)
PENDING = "pending_recipe.json"


def challenge(base_config: dict, ops: OpsConfig, env: str, recipe_name: str, holdout_weeks: int = 8,
              margin_pct: float = 0.5) -> dict:
    """Score recipe ``recipe_name`` against ``env``'s champion on its last ``holdout_weeks``; request it if it wins."""
    from src.ops import steps

    s = env_settings(ops, env)
    done = [w for w in weeks(s.start, s.end) if (cycle_dir(base_config, env, w) / "done").exists()]
    if len(done) < holdout_weeks + 1:
        raise RuntimeError(f"{env} has {len(done)} finished weeks; a {holdout_weeks}-week holdout needs {holdout_weeks + 1}.")
    last = done[-1]
    first = last - holdout_weeks * WEEK  # cycle A: the challenger is fit as A's refit would have been
    c = steps.Cycle(base_config, ops, env, last)
    panel = steps.panel(c)

    cfg = recipe_config(base_config, ops, recipe_name)
    recipe = Recipe.from_config(cfg, ops.recipes[recipe_name].model_params)
    cycles = weeks(first, last - WEEK)
    champion = {w: pd.read_parquet(cycle_dir(base_config, env, w) / "forecasts_champion.parquet") for w in cycles}
    served = infer_measures(list(champion[first].columns), c.vn2)
    lacking = sorted({m.id for m in served} - {m.id for m in recipe.measure_list()})
    if lacking:
        raise ValueError(f"Recipe {recipe_name} must forecast every measure the champion does (the anchor and "
                         f"the policy read them); it lacks {lacking}.")
    used = recipe.apply(cfg)
    model = fit_recipe(build_feature_matrix(panel, used), recipe, cfg, first)
    challenger = {w: forecast(model, panel, w, used) for w in cycles}
    spec, _ = resolve_anchor(served, c.vn2)
    anchor = LostSalesAnchor(spec, c.vn2)
    demand = {}
    for w in weeks(first + WEEK, last):
        week = panel[panel["date"] == w]
        demand[w] = pd.DataFrame({"store": week["store"].to_numpy(), "product": week["product"].to_numpy(),
                                  "demand": week["sales"].fillna(0.0).to_numpy(),
                                  "censored": ~week["in_stock"].fillna(True).astype(bool).to_numpy()})
    start = pd.read_parquet(cycle_dir(base_config, env, first) / "state_live.parquet")
    old = ledger.anchor_replay(start, champion, demand, anchor, c.vn2)
    new = ledger.anchor_replay(start, challenger, demand, anchor, c.vn2)
    result = monitor.paired_delta(new, old, first + (monitor.LEAD_WEEKS + 1) * WEEK, last, c.rules)
    passed = bool(result.get("evidence_weeks") and result["delta_pct"] <= -margin_pct and result["ci_hi"] < 0)

    tracking.init_tracking(c.config)
    day = str(last.date())
    with mlflow.start_run(run_name=f"challenge-{recipe_name}-{day}", experiment_id=steps.experiment_id(c),
                          tags={"stage": "challenge", "env": env, "as_of": day, "recipe": recipe_name,
                                "recipe_key": recipe.key(), "validation_status": "passed" if passed else "rejected"}) as run:
        tracking.log_provenance(cfg)
        mlflow.log_params({"holdout_weeks": holdout_weeks, "margin_pct": margin_pct, "fit_as_of": str(first.date())})
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, (int, float))})
        mlflow.log_dict(recipe.to_dict(), "recipe.json")
    out = {"recipe": recipe_name, "as_of": day, "passed": passed, "run_id": run.info.run_id, **result}
    if passed:
        (env_dir(base_config, env) / PENDING).write_text(dump({**out, "recipe_dict": recipe.to_dict()}), encoding="utf-8")
    return out


def pending(base_config: dict, env: str) -> dict | None:
    """The recipe an operator's challenge won and the next cycle should fit, if any."""
    path = env_dir(base_config, env) / PENDING
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
