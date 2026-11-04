"""
The weekly cycle of an ops environment, one function per step (``python -m src.ops.cli step``):

    ingest → score → monitor → decide → refit → forecast → retrain → order → report

Cycle ``as_of = D`` (the drop's last week) realises week D on every trajectory with the order
decided in cycle D−1, reads the monitors, decides, maybe refits / promotes / rolls back the
forecaster, issues the forecasts for decision week D+1 from ``@champion`` (and from ``@previous``
while a new forecaster is on probation), maybe trains / promotes / falls back the policy, and
decides the orders for D+1 with the registered policy (``src/ops/rl.py``).

Trajectories: ``live`` (champion forecaster, champion policy), ``previous`` (the previous
forecaster under the same policy, while a new forecaster is on probation) and ``anchor`` (the
champion forecaster under the anchor, while an RL policy serves).

Each step reads and writes the cycle folder ``ops/<env>/cycles/<D>/``; re-running a step
overwrites its outputs. MLflow: one run per cycle, a child of the environment's timeline run,
holds the cycle's tags, metrics, input data and files; a refit is a child run of its cycle that
logs and registers the model. Nothing here imports torch: the forecaster's LightGBM and torch
can't share a process, so a retrain that trains runs in its own process (``run_cycle``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.entities import DatasetInput, InputTag, Metric
from mlflow.tracking import MlflowClient

from src.forecasting.measures import HORIZONS, infer_measures
from src.forecasting.model import build_feature_matrix
from src.forecasting.preprocessing import build_target_columns, load_master, load_sales_panel
from src.forecasting.recipe import PANEL_COLUMNS, Recipe, fit_recipe, forecast, panel_digest
from src.ops import contract, drops, ledger, monitor, registry, retune, rl
from src.ops.decide import Decision, decide, decide_policy, drift_alert
from src.ops.settings import (
    Env,
    OpsConfig,
    Rules,
    cycle_dir,
    dump,
    env_config,
    env_dir,
    env_model_params,
    env_rules,
    env_settings,
    weeks,
)
from src.rl.vn2_data import context_features
from src.serving.forecaster import log_forecaster
from src.serving.policy import log_policy
from src.simulation.anchor import LostSalesAnchor, resolve_anchor
from src.simulation.vn2 import VN2Rules
from src.utils import tracking

STEPS = ("ingest", "score", "monitor", "decide", "refit", "forecast", "retrain", "order", "report")
# trajectory → (forecaster alias, policy): the registered policy, or the anchor
TRAJECTORIES = {"live": (registry.CHAMPION, "policy"), "previous": (registry.PREVIOUS, "policy"),
                "anchor": (registry.CHAMPION, "anchor")}
ROOT = Path(__file__).resolve().parents[2]
WEEK = pd.Timedelta(weeks=1)
KEYS = ledger.KEYS


@dataclass
class Cycle:
    base_config: dict
    ops: OpsConfig
    env: str
    as_of: pd.Timestamp

    def __post_init__(self):
        # One unit however the week arrives (a CLI string is seconds, a date range nanoseconds), so a
        # step run by the orchestrator writes the same bytes as the in-process loop.
        self.as_of = pd.Timestamp(self.as_of).as_unit("ns")

    @cached_property
    def config(self) -> dict:
        return env_config(self.base_config, self.ops, self.env)

    @cached_property
    def settings(self) -> Env:
        return env_settings(self.ops, self.env)

    @cached_property
    def rules(self) -> Rules:
        return env_rules(self.ops, self.env)

    @cached_property
    def vn2(self) -> VN2Rules:
        return VN2Rules.from_config(self.config)

    @property
    def day(self) -> str:
        return str(self.as_of.date())

    @property
    def first(self) -> bool:
        return self.as_of == pd.Timestamp(self.settings.start)

    @property
    def model_name(self) -> str:
        return registry.model_name(self.env, "forecaster")

    @property
    def policy_name(self) -> str:
        return registry.model_name(self.env, "policy")

    @property
    def first_forecast_week(self) -> pd.Timestamp:
        """The first decision week of the environment (forecast by the bootstrap cycle)."""
        return pd.Timestamp(self.settings.start) + WEEK

    def path(self, name: str, previous: bool = False) -> Path:
        return cycle_dir(self.base_config, self.env, self.as_of - WEEK if previous else self.as_of) / name

    def has(self, name: str, previous: bool = False) -> bool:
        return self.path(name, previous).exists()

    def read(self, name: str, previous: bool = False):
        p = self.path(name, previous)
        return pd.read_parquet(p) if p.suffix == ".parquet" else json.loads(p.read_text(encoding="utf-8"))

    def write(self, name: str, obj) -> None:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(p, index=False)
        else:
            p.write_text(dump(obj), encoding="utf-8")


# ---------------------------------------------------------------------------
# Data, models, MLflow runs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_panel(sales: str, instock: str, master: str) -> pd.DataFrame:
    panel = load_sales_panel(sales, instock, master)
    panel["date"] = pd.to_datetime(panel["date"])
    return build_target_columns(panel, horizons=list(HORIZONS))


def panel(c: Cycle) -> pd.DataFrame:
    """The long panel of the cycle's drop (shared within a process: don't modify it)."""
    sales, instock = drops.drop_paths(c.base_config, c.env, c.as_of)
    return _load_panel(str(sales), str(instock), c.config["data"]["master_file"])


_models: dict[str, mlflow.pyfunc.PyFuncModel] = {}


def load_model(c: Cycle, version) -> mlflow.pyfunc.PyFuncModel:
    """A registered version, downloaded once into the artifact cache under its (immutable) model id."""
    model_id = version.source.rsplit("/", 1)[-1]
    if model_id not in _models:
        local = Path(c.base_config["_project_root"]) / c.base_config["mlflow"]["cache_dir"] / "models" / model_id
        if not (local / "MLmodel").exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".download-", dir=local.parent))
            try:
                fetched = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=str(tmp))
                if not local.exists():
                    os.replace(fetched, local)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        _models[model_id] = mlflow.pyfunc.load_model(str(local))
    return _models[model_id]


def experiment_id(c: Cycle) -> str:
    tracking.init_tracking(c.config)
    return tracking.get_or_create_experiment(f"vn2-ops-{c.env}")


def _find_run(c: Cycle, **tags: str) -> str | None:
    query = " and ".join(f"tags.{k} = '{v}'" for k, v in tags.items())
    runs = MlflowClient().search_runs([experiment_id(c)], query, max_results=1)
    return runs[0].info.run_id if runs else None


def timeline_run(c: Cycle) -> str:
    """The environment's parent run: cycle runs hang under it; its metrics are indexed by week."""
    run_id = _find_run(c, stage="timeline")
    if run_id is None:
        client = MlflowClient()
        run_id = client.create_run(experiment_id(c), run_name=f"{c.env}-timeline",
                                   tags={"stage": "timeline", "env": c.env}).info.run_id
        client.set_terminated(run_id, "FINISHED")  # a container; metrics are appended to it every week
    return run_id


def cycle_run(c: Cycle, create: bool = False) -> str:
    run_id = _find_run(c, stage="cycle", as_of=c.day)
    if run_id is None and create:
        run_id = MlflowClient().create_run(
            experiment_id(c), run_name=f"cycle-{c.day}",
            tags={"stage": "cycle", "env": c.env, "as_of": c.day, "mlflow.parentRunId": timeline_run(c)},
        ).info.run_id
    if run_id is None:
        raise RuntimeError(f"No cycle run for {c.day}: run the ingest step first.")
    return run_id


def champion_info(c: Cycle) -> dict | None:
    mv = registry.resolve(c.model_name, registry.CHAMPION)
    if mv is None:
        return None
    fit = pd.Timestamp(mv.tags["as_of"])
    promoted = pd.Timestamp(mv.tags.get("promoted_as_of", mv.tags["as_of"]))
    return {"version": str(mv.version), "fit_as_of": str(fit.date()), "promoted_as_of": str(promoted.date()),
            "age_weeks": int((c.as_of - fit) / WEEK), "weeks_since_promotion": int((c.as_of - promoted) / WEEK)}


def policy_info(c: Cycle) -> dict | None:
    mv = registry.resolve(c.policy_name, registry.CHAMPION)
    if mv is None:
        return None
    attempts = [pd.Timestamp(v.tags["as_of"]) for v in MlflowClient().search_model_versions(f"name='{c.policy_name}'")
                if v.tags.get("kind") == "ppo"]
    promoted = pd.Timestamp(mv.tags.get("promoted_as_of", mv.tags["as_of"]))
    return {"version": str(mv.version), "kind": mv.tags["kind"], "fit_as_of": mv.tags["as_of"],
            "promoted_as_of": str(promoted.date()), "weeks_since_promotion": int((c.as_of - promoted) / WEEK),
            "gate": json.loads(mv.tags.get("gate") or "null"),
            "weeks_since_rl_attempt": int((c.as_of - max(attempts)) / WEEK) if attempts else None}


@lru_cache(maxsize=4)
def _context(sales: str, instock: str, master: str, week: str, features: str) -> pd.DataFrame:
    return context_features(_load_panel(sales, instock, master), pd.Timestamp(week), json.loads(features))


def context(c: Cycle) -> pd.DataFrame:
    """``zero_share``, ``pos_cv`` and the demand-class code of every SKU for decision week D+1."""
    sales, instock = drops.drop_paths(c.base_config, c.env, c.as_of)
    feats = json.dumps({"forecasting": {"features": c.config["forecasting"]["features"]}})
    return _context(str(sales), str(instock), c.config["data"]["master_file"], str((c.as_of + WEEK).date()), feats)


def fit_dates(c: Cycle) -> dict[str, str]:
    """Forecaster version → the as-of week it was fit on."""
    return {str(v.version): v.tags["as_of"] for v in MlflowClient().search_model_versions(f"name='{c.model_name}'")}


def on_probation(c: Cycle) -> bool:
    """A new champion stays on probation until its evidence weeks are realised (and while ``previous`` exists)."""
    info = champion_info(c)
    if info is None or registry.resolve(c.model_name, registry.PREVIOUS) is None:
        return False
    end = pd.Timestamp(info["promoted_as_of"]) + (monitor.LEAD_WEEKS + c.rules.probation.evidence_weeks) * WEEK
    return c.as_of <= end


def read_ledger(c: Cycle, name: str, since: pd.Timestamp) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for w in weeks(min(since, c.as_of), c.as_of)
             if (p := cycle_dir(c.base_config, c.env, w) / name).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sanity(new: pd.DataFrame, old: pd.DataFrame | None, rules: Rules) -> list[str]:
    ids = [m for m in new.columns if m not in (*KEYS, "date", "version")]
    problems = []
    if new[ids].isna().to_numpy().any():
        problems.append("missing forecasts")
    if (new[ids].to_numpy() < 0).any():
        problems.append("negative forecasts")
    if {"q50_h2", "q83_h2"} <= set(ids) and (new["q50_h2"] > new["q83_h2"] + 1e-9).any():
        problems.append("q50_h2 above q83_h2")
    for m in [m for m in ids if m in old.columns] if old is not None else []:
        change = 100.0 * (new[m].sum() / max(old[m].sum(), 1e-9) - 1.0)
        if abs(change) > rules.sanity.max_total_change_pct:
            problems.append(f"{m} total {change:+.1f}% vs the champion")
    return problems


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_ingest(c: Cycle) -> None:
    """Check the week's drop against the data contract; open the cycle run and log the drop."""
    if c.settings.kind == "live":
        _, dirty = tracking.git_state()
        problems = [] if c.ops.frozen and c.ops.frozen_rules == c.rules.version() else \
            [f"its rules ({c.rules.version()}) aren't the frozen ones ({c.ops.frozen_rules})"]
        if dirty:
            problems.append("the code has uncommitted changes")
        if problems:
            raise RuntimeError(f"The live environment refuses to run: {'; '.join(problems)}.")
    # The registry holds the environment's state as of its latest week: replaying an earlier week
    # on top of it would mix the two (re-running the latest week is fine).
    later = [w for w in weeks(c.as_of + WEEK, c.settings.end) if (cycle_dir(c.base_config, c.env, w) / "done").exists()]
    if later:
        raise RuntimeError(f"{c.env} has already run {later[-1].date()}: weeks run in order, so replaying {c.day} "
                           "needs a fresh environment.")
    sales_path, instock_path = drops.drop_paths(c.base_config, c.env, c.as_of)
    if not sales_path.exists():
        raise FileNotFoundError(f"No drop for {c.day} yet ({sales_path}).")
    if not c.first and not c.has("state_live.parquet", previous=True):
        raise RuntimeError(f"Cycle {(c.as_of - WEEK).date()} hasn't run: cycles run in order.")
    prev_path = drops.drop_paths(c.base_config, c.env, c.as_of - WEEK)[0]
    report = contract.check_drop(
        pd.read_csv(sales_path), pd.read_csv(instock_path), load_master(c.config["data"]["master_file"]),
        c.as_of, pd.read_csv(prev_path) if prev_path.exists() else None,
    )
    c.write("contract.json", report.to_dict())
    run_id, client = cycle_run(c, create=True), MlflowClient()
    client.set_tag(run_id, "rules_version", c.rules.version())
    client.set_tag(run_id, "strategy", c.rules.strategy)
    tracking.log_provenance(c.config, run_id=run_id)
    if not report.ok:
        client.log_text(run_id, dump(report.to_dict()), "cycle/contract.json")
        client.set_terminated(run_id, "FAILED")
        raise contract.ContractError("; ".join(report.errors))
    data = mlflow.data.from_pandas(panel(c)[PANEL_COLUMNS], source=str(sales_path), name=f"vn2-sales-{c.day}")
    client.log_inputs(run_id, datasets=[DatasetInput(data._to_mlflow_entity(), [InputTag("mlflow.data.context", "weekly_drop")])])


def step_score(c: Cycle) -> None:
    """Realise week D on every trajectory with the orders decided last week; start them in week one."""
    p = panel(c)
    if c.first:
        state = (ledger.initial_state_vn2(c.config["data"]["initial_state_file"]) if c.settings.kind == "live"
                 else ledger.initial_state_proxy(p, c.as_of))
        c.write("state_live.parquet", state)
        if (policy_info(c) or {}).get("kind") == "ppo":  # a carried-over RL policy: its anchor shadow starts with it
            c.write("state_anchor.parquet", state)
        return
    week = p[p["date"] == c.as_of]
    demand = pd.DataFrame({"store": week["store"].to_numpy(), "product": week["product"].to_numpy(),
                           "demand": week["sales"].fillna(0.0).to_numpy(),
                           "censored": ~week["in_stock"].fillna(True).astype(bool).to_numpy()})
    policy = policy_info(c)
    for traj in TRAJECTORIES:
        if not c.has(f"state_{traj}.parquet", previous=True):
            continue
        if (traj == "previous" and not on_probation(c)) or (traj == "anchor" and (policy or {}).get("kind") != "ppo"):
            continue
        state, costs = ledger.realise(c.read(f"state_{traj}.parquet", previous=True), demand, c.vn2)
        c.write(f"state_{traj}.parquet", state)
        c.write(f"costs_{traj}.parquet", costs.assign(week=c.as_of))


def step_monitor(c: Cycle) -> None:
    """The signals the decision reads (see src/ops/monitor.py)."""
    p = panel(c)
    classes = monitor.demand_classes(p, c.config)
    actuals = p[[*KEYS, "date", "sales"]].merge(classes, on=[*KEYS, "date"])
    info = champion_info(c)
    signals: dict = {"as_of": c.day, "champion": info}
    if info is not None:
        promoted = pd.Timestamp(info["promoted_as_of"])
        fc = read_ledger(c, "forecasts_champion.parquet", promoted)
        fc = fc[fc["version"] == info["version"]] if len(fc) else fc
        if len(fc):
            signals["bias"] = monitor.bias_signals(fc, actuals, c.rules)
            signals["coverage"] = monitor.coverage_signals(fc, actuals, c.rules)
        costs = read_ledger(c, "costs_live.parquet", c.as_of - monitor.COST_SHARE_WEEKS * WEEK)
        if len(costs):
            costs = costs.merge(classes.rename(columns={"date": "week"}), on=[*KEYS, "week"])
            signals["cost_share"] = monitor.cost_share(costs)
        if c.has("costs_previous.parquet"):
            signals["probation"] = monitor.probation_signal(
                read_ledger(c, "costs_live.parquet", promoted), read_ledger(c, "costs_previous.parquet", promoted), promoted, c.rules)
    signals["diagnostics"] = monitor.diagnostics(p, classes, c.as_of, pd.Timestamp(info["fit_as_of"]) if info else None)
    if (request := retune.pending(c.base_config, c.env)) is not None:
        signals["pending_recipe"] = {k: v for k, v in request.items() if k != "recipe_dict"}
    policy = signals["policy"] = policy_info(c)
    signals["rl_eligible"] = rl.eligible(c.as_of, c.first_forecast_week, c.rules.policy)
    if policy and policy["kind"] == "ppo":
        promoted = pd.Timestamp(policy["promoted_as_of"])
        if c.has("costs_anchor.parquet"):
            signals["policy_vs_anchor"] = monitor.policy_signal(
                read_ledger(c, "costs_live.parquet", promoted), read_ledger(c, "costs_anchor.parquet", promoted), promoted,
                c.as_of, c.rules)
        if policy["gate"]:
            signals["gate_churn"] = monitor.gate_churn(classes, policy["gate"], pd.Timestamp(policy["fit_as_of"]), c.as_of)
    c.write("signals.json", signals)


def step_decide(c: Cycle) -> None:
    signals = c.read("signals.json")
    c.write("decision.json", {**decide(signals, c.rules).to_dict(), "policy": decide_policy(signals, c.rules).to_dict(),
                              "drift_alarm": drift_alert(signals, c.rules)})


def step_refit(c: Cycle) -> None:
    """Act on the decision: bootstrap, refit or retune (then sanity-check and promote), or roll back."""
    raw = c.read("decision.json")
    d, client, name = Decision(raw["action"], raw["reasons"]), MlflowClient(), c.model_name
    if d.action == "rollback":
        rejected = registry.resolve(name, registry.CHAMPION)
        restored = registry.rollback(name)
        client.set_model_version_tag(name, str(rejected.version), "validation_status", "rejected")
        client.set_model_version_tag(name, str(rejected.version), "rejected_reason", d.reasons[0][:500])
        client.set_model_version_tag(name, restored, "promoted_as_of", c.day)
        c.path("state_previous.parquet").unlink(missing_ok=True)
        return
    if d.action not in ("bootstrap", "refit", "retune"):
        return
    p = panel(c)
    history = p[PANEL_COLUMNS]
    champion = registry.resolve(name, registry.CHAMPION)
    request = retune.pending(c.base_config, c.env) if d.action == "retune" else None
    if request is not None:  # the recipe an operator's challenge won
        recipe = Recipe.from_dict(request["recipe_dict"])
    elif champion is None:
        recipe = Recipe.from_config(c.config, env_model_params(c.ops, c.env))
    else:
        recipe = Recipe.from_dict(load_model(c, champion).unwrap_python_model().meta["recipe"])
    used = recipe.apply(c.config)
    started = time.time()
    model = fit_recipe(build_feature_matrix(p, used), recipe, c.config, c.as_of)
    fit_seconds = time.time() - started
    digest = panel_digest(p)
    # The experiment is explicit: without it the run (and its model files) would go to the store's
    # Default experiment, whose artifacts live in ./mlruns/0.
    with mlflow.start_run(run_name=f"{d.action}-{c.day}", experiment_id=experiment_id(c), parent_run_id=cycle_run(c),
                          tags={"stage": d.action, "env": c.env, "as_of": c.day}):
        tracking.log_provenance(c.config)
        mlflow.log_metric("fit_seconds", fit_seconds)
        source = str(drops.drop_paths(c.base_config, c.env, c.as_of)[0])
        mlflow.log_input(mlflow.data.from_pandas(history, source=source, name=f"vn2-sales-{c.day}"), context="training")
        info = log_forecaster(model, recipe, c.config, history, load_master(c.config["data"]["master_file"]),
                              c.as_of, digest)
    tags = {"trigger": d.action, "as_of": c.day, "recipe_key": recipe.key(), "data_digest": digest,
            "validation_status": "pending", "reasons": "; ".join(d.reasons)[:500]}
    if request is not None:
        tags.update(recipe=request["recipe"], challenge_run_id=request["run_id"],
                    holdout_delta_pct=f"{request['delta_pct']:+.3f}")
        (env_dir(c.base_config, c.env) / retune.PENDING).unlink()  # acted on, whatever the sanity check says
    version = str(registry.register(info.model_uri, name, tags).version)
    old = None if champion is None else load_model(c, champion).predict(history, params={"as_of": c.day})
    problems = _sanity(forecast(model, p, c.as_of, used), old, c.rules)
    if problems:
        client.set_model_version_tag(name, version, "validation_status", "rejected")
        client.set_model_version_tag(name, version, "rejected_reason", "; ".join(problems)[:500])
        return
    client.set_model_version_tag(name, version, "validation_status", "passed")
    client.set_model_version_tag(name, version, "promoted_as_of", c.day)
    registry.promote(name, version)
    if champion is not None:  # the previous forecaster keeps running in shadow from today's state
        c.write("state_previous.parquet", c.read("state_live.parquet"))


def step_forecast(c: Cycle) -> None:
    """Forecasts for decision week D+1 from each forecaster that drives a trajectory."""
    history = panel(c)[PANEL_COLUMNS]
    aliases = {alias for traj, (alias, _) in TRAJECTORIES.items() if c.has(f"state_{traj}.parquet")}
    for alias in sorted(aliases):
        mv = registry.resolve(c.model_name, alias)
        if mv is not None:
            fc = load_model(c, mv).predict(history, params={"as_of": c.day})
            c.write(f"forecasts_{alias}.parquet", fc.assign(version=str(mv.version)))


def step_retrain(c: Cycle) -> None:
    """The policy layer: register the anchor (bootstrap), fall back to it, or train an RL challenger."""
    d, name, client = c.read("decision.json")["policy"], c.policy_name, MlflowClient()
    fc = c.read("forecasts_champion.parquet")
    spec, _ = resolve_anchor(infer_measures(list(fc.columns), c.vn2), c.vn2)
    example = rl.decision_frame(c.read("state_live.parquet"), fc, context(c)).head(3)
    if d["action"] == "bootstrap":
        with mlflow.start_run(run_name=f"policy-bootstrap-{c.day}", experiment_id=experiment_id(c),
                              parent_run_id=cycle_run(c), tags={"stage": "policy-bootstrap", "env": c.env, "as_of": c.day}):
            info = log_policy("anchor", spec, c.vn2, example)
        v = registry.register(info.model_uri, name, {"kind": "anchor", "trigger": "bootstrap", "as_of": c.day,
                                                     "validation_status": "passed", "promoted_as_of": c.day})
        registry.promote(name, str(v.version))
    elif d["action"] == "fallback":
        rl_version = registry.resolve(name, registry.CHAMPION)
        anchors = [v for v in client.search_model_versions(f"name='{name}'") if v.tags.get("kind") == "anchor"]
        anchor_version = str(min(int(v.version) for v in anchors))
        client.set_model_version_tag(name, str(rl_version.version), "validation_status", "rejected")
        client.set_model_version_tag(name, str(rl_version.version), "rejected_reason", d["reasons"][0][:500])
        client.set_model_version_tag(name, anchor_version, "promoted_as_of", c.day)
        registry.promote(name, anchor_version)
        c.path("state_anchor.parquet").unlink(missing_ok=True)
    elif d["action"] == "retrain":
        _train_policy(c, example)


def _train_policy(c: Cycle, example: pd.DataFrame) -> None:
    """Train PPO on the forecast ledger (torch: runs in its own process), check it, register, promote."""
    from src.rl.data import block_episodes
    from src.rl.policy import TorchPolicy
    from src.rl.train import train_agent
    from src.serving.policy import actor_arrays

    pr = c.rules.policy
    fc = read_ledger(c, "forecasts_champion.parquet", c.first_forecast_week - WEEK)
    data = rl.ledger_run_data(panel(c), fc, c.as_of, pr, c.vn2, fit_dates(c))
    spec = rl.ops_spec(c.config, pr, data, c.env)
    train_eps, val_eps = block_episodes(c.config, data, "rl_train"), block_episodes(c.config, data, "rl_val")
    exp, parent = experiment_id(c), cycle_run(c)
    results = {s: train_agent(spec.with_seed(s), c.config, data, train_eps, val_eps, experiment_id=exp,
                              parent_run_id=parent, tags={"stage": "rl-train", "env": c.env, "as_of": c.day})
               for s in pr.seeds}
    val = {s: pd.read_parquet(Path(r["run_dir"]) / "val_episodes.parquet") for s, r in results.items()}
    report = rl.challenger_report(val, pr)
    seed = int(c.config.get("project", {}).get("seed", 42))
    seed = seed if seed in results else pr.seeds[0]
    policy = TorchPolicy.load(Path(results[seed]["run_dir"]) / "model.pt")
    tags = {"kind": "ppo", "trigger": "retrain", "as_of": c.day, "recipe_key": spec.config_key(),
            "gate": json.dumps(list(report["gate"]) if report["gate"] else None),
            "val_vs_anchor_pct": f"{report['mean_val_pct']:+.3f}", "seeds_beating_anchor": f"{report['share_beating']:.2f}",
            "validated_with_forecaster": champion_info(c)["version"], "training_runs": ",".join(r["rl_run_id"] for r in results.values()),
            "validation_status": "passed" if report["passed"] else "rejected",
            "rejected_reason": "; ".join(report["failed"])[:500]}
    with mlflow.start_run(run_name=f"policy-{c.day}", experiment_id=exp, parent_run_id=parent,
                          tags={"stage": "policy", "env": c.env, "as_of": c.day}):
        mlflow.log_metrics({"val_vs_anchor_pct": report["mean_val_pct"], "seeds_beating_anchor": report["share_beating"],
                            **{f"val_vs_anchor_pct_s{s}": v for s, v in report["val_pct"].items()}})
        info = log_policy("ppo", spec.anchor, c.vn2, example, spec=spec.with_seed(seed), actor=actor_arrays(policy),
                          scales=data.scales, gate=report["gate"], params={"recipe_key": spec.config_key()})
    v = str(registry.register(info.model_uri, c.policy_name, tags).version)
    if report["passed"]:
        promote_policy(c, v)


def promote_policy(c: Cycle, version: str) -> None:
    """
    Serve an RL policy version. The anchor's shadow restarts from today's state at every promotion,
    so the fallback check judges this policy alone, not the stock an earlier one left behind.
    """
    MlflowClient().set_model_version_tag(c.policy_name, version, "promoted_as_of", c.day)
    registry.promote(c.policy_name, version)
    c.write("state_anchor.parquet", c.read("state_live.parquet"))


def step_order(c: Cycle) -> None:
    """Orders for decision week D+1: the registered policy on live and previous, the anchor on its shadow."""
    ctx = context(c)
    policy = load_model(c, registry.resolve(c.policy_name, registry.CHAMPION))
    for traj, (alias, kind) in TRAJECTORIES.items():
        if not (c.has(f"state_{traj}.parquet") and c.has(f"forecasts_{alias}.parquet")):
            continue
        fc = c.read(f"forecasts_{alias}.parquet")
        state = c.read(f"state_{traj}.parquet")
        if kind == "anchor":
            spec, _ = resolve_anchor(infer_measures(list(fc.columns), c.vn2), c.vn2)
            state["order"] = ledger.anchor_orders(state, fc, LostSalesAnchor(spec, c.vn2))
        else:
            out = policy.predict(rl.decision_frame(state, fc, ctx))
            state = state.drop(columns="order").merge(out[[*KEYS, "order"]], on=KEYS, validate="one_to_one")
            if traj == "live":
                out.assign(decision_week=str((c.as_of + WEEK).date())).to_csv(c.path("orders.csv"), index=False)
        c.write(f"state_{traj}.parquet", state[ledger.STATE_COLUMNS])


def step_report(c: Cycle) -> None:
    """Metrics and files of the cycle to its run (and to the timeline, by week); close the run."""
    run_id, client = cycle_run(c), MlflowClient()
    metrics: dict[str, float] = {}
    for traj in TRAJECTORIES:
        if c.has(f"costs_{traj}.parquet"):
            k = c.read(f"costs_{traj}.parquet")
            for col in ("cost_eur", "holding_eur", "shortage_eur"):
                metrics[f"{traj}_{col}_per_sku"] = float(k[col].mean())
            metrics[f"{traj}_censored_share"] = float(k["censored"].mean())
    metrics["live_units_ordered"] = float(c.read("state_live.parquet")["order"].sum())
    orders = pd.read_csv(c.path("orders.csv"))
    metrics["rl_share_of_skus"] = float(orders["rl"].mean())
    signals, decision = c.read("signals.json"), c.read("decision.json")
    if signals.get("champion"):
        metrics["champion_age_weeks"] = float(signals["champion"]["age_weeks"])
    for cls, b in signals.get("bias", {}).items():
        if b["ts"] is not None:
            metrics[f"bias_ts_{cls}"] = b["ts"]
    for cls, v in signals.get("coverage", {}).items():
        if v["value"] is not None:
            metrics[f"coverage_q83_{cls}"] = v["value"]
    metrics["drift_alarm"] = float(decision["drift_alarm"]["active"])
    if (p := signals.get("probation")) and p.get("evidence_weeks"):
        metrics["probation_delta_pct"] = p["delta_pct"]
    for k, v in signals.get("diagnostics", {}).items():
        metrics[f"diag_{k}"] = float(v)
    now, step = int(time.time() * 1000), int((c.as_of - pd.Timestamp(c.settings.start)) / WEEK)
    client.log_batch(run_id, metrics=[Metric(k, v, now, 0) for k, v in metrics.items()])
    client.log_batch(timeline_run(c), metrics=[Metric(k, v, now, step) for k, v in metrics.items()])
    champion, previous = (registry.resolve(c.model_name, a) for a in (registry.CHAMPION, registry.PREVIOUS))
    policy = registry.resolve(c.policy_name, registry.CHAMPION)
    tags = {"decision": decision["action"], "decision_reasons": "; ".join(decision["reasons"])[:500],
            "policy_decision": decision["policy"]["action"],
            "policy_reasons": "; ".join(decision["policy"]["reasons"])[:500],
            "drift_alarm": str(decision["drift_alarm"]["active"]).lower(),
            "drift_alarm_reasons": "; ".join(decision["drift_alarm"]["reasons"])[:500],
            "policy_version": "" if policy is None else str(policy.version),
            "policy_kind": "" if policy is None else policy.tags.get("kind", ""),
            "champion_version": "" if champion is None else str(champion.version),
            "previous_version": "" if previous is None else str(previous.version)}
    for k, v in tags.items():
        client.set_tag(run_id, k, v)
    client.log_artifacts(run_id, str(cycle_dir(c.base_config, c.env, c.as_of)), artifact_path="cycle")
    client.set_terminated(run_id, "FINISHED")
    c.path("done").write_text(c.day, encoding="utf-8")


STEP_FUNCS = {name: globals()[f"step_{name}"] for name in STEPS}


def run_step(c: Cycle, name: str) -> None:
    """One step, against the configured store (each step may run in its own process)."""
    tracking.init_tracking(c.config)
    STEP_FUNCS[name](c)


def spawn_step(c: Cycle, name: str) -> None:
    """Run one step in its own process (training imports torch, which can't share a process with LightGBM)."""
    ops_path = env_dir(c.base_config, c.env) / "ops_config.yaml"
    ops_path.write_text(yaml.safe_dump(c.ops.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "src.ops.cli", "--ops-config", str(ops_path), "--project-root",
                    str(c.base_config["_project_root"]), "step", name, "--env", c.env, "--as-of", c.day],
                   cwd=ROOT, check=True)


def run_cycle(c: Cycle, force: bool = False) -> str:
    """Every step of one week, in order; a finished week is skipped unless ``force``."""
    if c.has("done") and not force:
        return "skipped"
    for name in STEPS:
        if name == "retrain" and c.read("decision.json")["policy"]["action"] == "retrain":
            spawn_step(c, name)
        else:
            run_step(c, name)
    return "done"


def run_env(base_config: dict, ops: OpsConfig, env: str, start=None, end=None, force: bool = False) -> list[str]:
    """Run the weeks of an environment in order (making sim drops as needed)."""
    s = env_settings(ops, env)
    todo = weeks(start or s.start, end or s.end)
    if s.kind == "sim":
        missing = [w for w in todo if not drops.drop_paths(base_config, env, w)[0].exists()]
        if missing:
            drops.make(base_config, env, missing)
    return [run_cycle(Cycle(base_config, ops, env, w), force) for w in todo]


def carry_over(base_config: dict, ops: OpsConfig, source: str, target: str) -> dict[str, list[str]]:
    """
    Start environment ``target`` from ``source``'s champions (the live run from the final shadow
    replay): its forecaster, keeping its fit date so the age clock carries on, and its policy with
    the anchor version ahead of it, the fallback target. The versions serve the same logged models.
    """
    tracking.init_tracking(env_config(base_config, ops, target))
    client, start = MlflowClient(), str(env_settings(ops, target).start)
    for layer in registry.LAYERS:
        if client.search_model_versions(f"name='{registry.model_name(target, layer)}'"):
            raise RuntimeError(f"{registry.model_name(target, layer)} already has versions: carry-over starts an environment.")
    out: dict[str, list[str]] = {}
    for layer in registry.LAYERS:
        src_name, dst_name = registry.model_name(source, layer), registry.model_name(target, layer)
        champion = registry.resolve(src_name, registry.CHAMPION)
        if champion is None:
            raise RuntimeError(f"{src_name} has no champion to carry over.")
        chain = [champion]
        if layer == "policy" and champion.tags.get("kind") != "anchor":
            anchors = [v for v in client.search_model_versions(f"name='{src_name}'") if v.tags.get("kind") == "anchor"]
            chain = [min(anchors, key=lambda v: int(v.version)), champion]
        out[layer] = []
        for v in chain:
            new = registry.copy_version(v, dst_name, {"trigger": "carry-over", "carried_from": f"{src_name} v{v.version}",
                                                      "promoted_as_of": start})
            registry.promote(dst_name, str(new.version))
            out[layer].append(f"{dst_name} v{new.version} ← {src_name} v{v.version}")
    return out
