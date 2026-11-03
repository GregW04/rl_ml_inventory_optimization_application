"""
The weekly cycle of an ops environment, one function per step (``python -m src.ops.cli step``):

    ingest → score → monitor → decide → refit → forecast → order → report

Cycle ``as_of = D`` (the drop's last week) realises week D on every trajectory with the order
decided in cycle D−1, reads the monitors, decides, maybe refits / promotes / rolls back, then
issues the forecasts for decision week D+1 from ``@champion`` (and from ``@previous`` while a new
champion is on probation) and decides the orders for D+1.

Each step reads and writes the cycle folder ``ops/<env>/cycles/<D>/``; re-running a step
overwrites its outputs. MLflow: one run per cycle, a child of the environment's timeline run,
holds the cycle's tags, metrics, input data and files; a refit is a child run of its cycle that
logs and registers the model. Nothing here imports torch: the forecaster's LightGBM and torch
can't share a process.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import DatasetInput, InputTag, Metric
from mlflow.tracking import MlflowClient

from src.forecasting.measures import HORIZONS, infer_measures
from src.forecasting.model import build_feature_matrix
from src.forecasting.preprocessing import build_target_columns, load_master, load_sales_panel
from src.forecasting.recipe import PANEL_COLUMNS, Recipe, fit_recipe, forecast, panel_digest
from src.ops import contract, drops, ledger, monitor, registry
from src.ops.decide import Decision, decide
from src.ops.settings import Env, OpsConfig, Rules, cycle_dir, dump, env_config, env_rules, env_settings, weeks
from src.serving.forecaster import log_forecaster
from src.simulation.anchor import LostSalesAnchor, resolve_anchor
from src.simulation.vn2 import VN2Rules
from src.utils import tracking

STEPS = ("ingest", "score", "monitor", "decide", "refit", "forecast", "order", "report")
TRAJECTORIES = {"live": registry.CHAMPION, "previous": registry.PREVIOUS}  # trajectory → forecaster alias
WEEK = pd.Timedelta(weeks=1)
KEYS = ledger.KEYS


@dataclass
class Cycle:
    base_config: dict
    ops: OpsConfig
    env: str
    as_of: pd.Timestamp

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


def on_probation(c: Cycle) -> bool:
    """A new champion stays on probation until its evidence weeks are realised (and while ``previous`` exists)."""
    info = champion_info(c)
    if info is None or registry.resolve(c.model_name, registry.PREVIOUS) is None:
        return False
    end = pd.Timestamp(info["promoted_as_of"]) + (monitor.LEAD_WEEKS + c.rules.probation.evidence_weeks) * WEEK
    return c.as_of <= end


def _ledger(c: Cycle, name: str, since: pd.Timestamp) -> pd.DataFrame:
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
    for m in ids if old is not None else []:
        change = 100.0 * (new[m].sum() / max(old[m].sum(), 1e-9) - 1.0)
        if abs(change) > rules.sanity.max_total_change_pct:
            problems.append(f"{m} total {change:+.1f}% vs the champion")
    return problems


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_ingest(c: Cycle) -> None:
    """Check the week's drop against the data contract; open the cycle run and log the drop."""
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
        return
    week = p[p["date"] == c.as_of]
    demand = pd.DataFrame({"store": week["store"].to_numpy(), "product": week["product"].to_numpy(),
                           "demand": week["sales"].fillna(0.0).to_numpy(),
                           "censored": ~week["in_stock"].fillna(True).astype(bool).to_numpy()})
    for traj in TRAJECTORIES:
        if not c.has(f"state_{traj}.parquet", previous=True) or (traj == "previous" and not on_probation(c)):
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
        fc = _ledger(c, "forecasts_champion.parquet", promoted)
        fc = fc[fc["version"] == info["version"]] if len(fc) else fc
        if len(fc):
            signals["bias"] = monitor.bias_signals(fc, actuals, c.rules)
            signals["coverage"] = monitor.coverage_signals(fc, actuals, c.rules)
        costs = _ledger(c, "costs_live.parquet", c.as_of - monitor.COST_SHARE_WEEKS * WEEK)
        if len(costs):
            costs = costs.merge(classes.rename(columns={"date": "week"}), on=[*KEYS, "week"])
            signals["cost_share"] = monitor.cost_share(costs)
        if c.has("costs_previous.parquet"):
            signals["probation"] = monitor.probation_signal(
                _ledger(c, "costs_live.parquet", promoted), _ledger(c, "costs_previous.parquet", promoted), promoted, c.rules)
    signals["diagnostics"] = monitor.diagnostics(p, classes, c.as_of, pd.Timestamp(info["fit_as_of"]) if info else None)
    c.write("signals.json", signals)


def step_decide(c: Cycle) -> None:
    c.write("decision.json", decide(c.read("signals.json"), c.rules).to_dict())


def step_refit(c: Cycle) -> None:
    """Act on the decision: bootstrap or refit (then sanity-check and promote), or roll back."""
    d, client, name = Decision(**c.read("decision.json")), MlflowClient(), c.model_name
    if d.action == "rollback":
        rejected = registry.resolve(name, registry.CHAMPION)
        restored = registry.rollback(name)
        client.set_model_version_tag(name, str(rejected.version), "validation_status", "rejected")
        client.set_model_version_tag(name, str(rejected.version), "rejected_reason", d.reasons[0][:500])
        client.set_model_version_tag(name, restored, "promoted_as_of", c.day)
        c.path("state_previous.parquet").unlink(missing_ok=True)
        return
    if d.action not in ("bootstrap", "refit"):
        return
    p = panel(c)
    history = p[PANEL_COLUMNS]
    champion = registry.resolve(name, registry.CHAMPION)
    if champion is None:
        recipe = Recipe.from_config(c.config, c.settings.model_params)
    else:
        recipe = Recipe.from_dict(load_model(c, champion).unwrap_python_model().meta["recipe"])
    used = recipe.apply(c.config)
    model = fit_recipe(build_feature_matrix(p, used), recipe, c.config, c.as_of)
    digest = panel_digest(p)
    # The experiment is explicit: without it the run (and its model files) would go to the store's
    # Default experiment, whose artifacts live in ./mlruns/0.
    with mlflow.start_run(run_name=f"{d.action}-{c.day}", experiment_id=experiment_id(c), parent_run_id=cycle_run(c),
                          tags={"stage": d.action, "env": c.env, "as_of": c.day}):
        tracking.log_provenance(c.config)
        source = str(drops.drop_paths(c.base_config, c.env, c.as_of)[0])
        mlflow.log_input(mlflow.data.from_pandas(history, source=source, name=f"vn2-sales-{c.day}"), context="training")
        info = log_forecaster(model, recipe, c.config, history, load_master(c.config["data"]["master_file"]),
                              c.as_of, digest)
    version = str(registry.register(info.model_uri, name, {
        "trigger": d.action, "as_of": c.day, "recipe_key": recipe.key(), "data_digest": digest,
        "validation_status": "pending", "reasons": "; ".join(d.reasons)[:500],
    }).version)
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
    for traj, alias in TRAJECTORIES.items():
        mv = registry.resolve(c.model_name, alias)
        if mv is None or not c.has(f"state_{traj}.parquet"):
            continue
        fc = load_model(c, mv).predict(history, params={"as_of": c.day})
        c.write(f"forecasts_{alias}.parquet", fc.assign(version=str(mv.version)))


def step_order(c: Cycle) -> None:
    """The anchor's orders for decision week D+1 on every trajectory; the live ones go to orders.csv."""
    for traj, alias in TRAJECTORIES.items():
        if not (c.has(f"state_{traj}.parquet") and c.has(f"forecasts_{alias}.parquet")):
            continue
        fc = c.read(f"forecasts_{alias}.parquet")
        spec, _ = resolve_anchor(infer_measures(list(fc.columns), c.vn2), c.vn2)
        state = c.read(f"state_{traj}.parquet")
        state["order"] = ledger.anchor_orders(state, fc, LostSalesAnchor(spec, c.vn2))
        c.write(f"state_{traj}.parquet", state)
    live = c.read("state_live.parquet").assign(decision_week=str((c.as_of + WEEK).date()))
    live[[*KEYS, "decision_week", "order"]].to_csv(c.path("orders.csv"), index=False)


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
    signals, decision = c.read("signals.json"), c.read("decision.json")
    if signals.get("champion"):
        metrics["champion_age_weeks"] = float(signals["champion"]["age_weeks"])
    for cls, b in signals.get("bias", {}).items():
        if b["ts"] is not None:
            metrics[f"bias_ts_{cls}"] = b["ts"]
    for cls, v in signals.get("coverage", {}).items():
        if v["value"] is not None:
            metrics[f"coverage_q83_{cls}"] = v["value"]
    if (p := signals.get("probation")) and p.get("evidence_weeks"):
        metrics["probation_delta_pct"] = p["delta_pct"]
    for k, v in signals.get("diagnostics", {}).items():
        metrics[f"diag_{k}"] = float(v)
    now, step = int(time.time() * 1000), int((c.as_of - pd.Timestamp(c.settings.start)) / WEEK)
    client.log_batch(run_id, metrics=[Metric(k, v, now, 0) for k, v in metrics.items()])
    client.log_batch(timeline_run(c), metrics=[Metric(k, v, now, step) for k, v in metrics.items()])
    champion, previous = (registry.resolve(c.model_name, a) for a in (registry.CHAMPION, registry.PREVIOUS))
    tags = {"decision": decision["action"], "decision_reasons": "; ".join(decision["reasons"])[:500],
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


def run_cycle(c: Cycle, force: bool = False) -> str:
    """Every step of one week, in order; a finished week is skipped unless ``force``."""
    if c.has("done") and not force:
        return "skipped"
    for name in STEPS:
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
