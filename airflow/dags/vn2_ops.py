"""
One Airflow DAG per ops environment in ``configs/ops.yaml`` (``vn2_<env>``): the weekly cycle of
``src/ops`` (docs/ops.md), one task per step.

Thin on purpose. Every task runs the project's CLI with the project's venv in its own process,
so LightGBM and torch never share one. The decisions stay in ``src/ops/decide.py``: the DAG only
routes on the ``decision.json`` the decide step wrote, so a refit or retrain that isn't needed
shows as skipped. This file runs in Airflow's venv: stdlib and PyYAML only, no project imports.

- **Schedule:** a run per Monday from the environment's start to its end, the logical date being
  the as-of week. ``catchup`` replays a sim environment, one week at a time (``max_active_runs=1``),
  and a failed week stops the next (``depends_on_past``).
- **Data:** a sim run cuts its own drop from the Week0 files. A live run waits for the drop that
  ``python -m src.ops.cli drops release`` publishes.
- **Resources:** refits and retrains share the 1-slot ``heavy`` pool (``make airflow-init``).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, task
from airflow.timetables.trigger import CronTriggerTimetable

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
OPS_CONFIG = ROOT / "configs" / "ops.yaml"
# The cycle's steps in order (src/ops/steps.py STEPS); refit and retrain run only when decided.
STEPS = ("ingest", "score", "monitor", "decide", "refit", "forecast", "retrain", "order", "report")
HEAVY = {"refit": timedelta(minutes=45), "retrain": timedelta(minutes=60)}
ENV = {"MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"), "PYTHONUNBUFFERED": "1"}


def _day(value) -> datetime:
    return datetime.combine(date.fromisoformat(str(value)), datetime.min.time(), tzinfo=UTC)


def _cli(task_id: str, args: str, **kwargs) -> BashOperator:
    return BashOperator(task_id=task_id, bash_command=f"{PYTHON} -m src.ops.cli {args}", cwd=str(ROOT), env=ENV,
                        append_env=True, **kwargs)


def _decision(env: str, ds: str) -> dict:
    return json.loads((ROOT / "ops" / env / "cycles" / ds / "decision.json").read_text(encoding="utf-8"))


def build_dag(env: str, settings: dict) -> DAG:
    with DAG(
        dag_id=f"vn2_{env}",
        description=f"Weekly ops cycle of the {settings['kind']} environment {env}",
        schedule=CronTriggerTimetable("0 0 * * 1", timezone="UTC"),
        start_date=_day(settings["start"]),
        end_date=_day(settings["end"]),
        catchup=True,
        max_active_runs=1,
        default_args={"depends_on_past": True, "retries": 0, "execution_timeout": timedelta(minutes=20)},
        tags=["vn2", settings["kind"]],
        doc_md=__doc__,
    ) as dag:
        if settings["kind"] == "live":
            @task.sensor(task_id="wait_for_drop", poke_interval=60, timeout=14 * 24 * 3600, mode="reschedule")
            def wait_for_drop(ds=None) -> bool:
                drops = ROOT / "ops" / env / "drops"
                return all((drops / f"{name}_{ds}.csv").exists() for name in ("sales", "instock"))

            first = wait_for_drop()
        else:
            first = _cli("drop", f"drops make --env {env} --start {{{{ ds }}}} --end {{{{ ds }}}}")

        steps = {
            name: _cli(name, f"step {name} --env {env} --as-of {{{{ ds }}}}",
                       **({"pool": "heavy", "execution_timeout": HEAVY[name]} if name in HEAVY else {}),
                       # After a skipped refit / retrain, the next step still runs.
                       **({"trigger_rule": "none_failed_min_one_success"} if name in ("forecast", "order") else {}))
            for name in STEPS
        }

        @task.branch(task_id="route_forecaster")
        def route_forecaster(ds=None) -> str:
            return "forecast" if _decision(env, ds)["action"] == "keep" else "refit"

        @task.branch(task_id="route_policy")
        def route_policy(ds=None) -> str:
            return "order" if _decision(env, ds)["policy"]["action"] == "keep" else "retrain"

        to_refit, to_retrain = route_forecaster(), route_policy()
        first >> steps["ingest"] >> steps["score"] >> steps["monitor"] >> steps["decide"] >> to_refit
        to_refit >> [steps["refit"], steps["forecast"]]
        steps["refit"] >> steps["forecast"] >> to_retrain
        to_retrain >> [steps["retrain"], steps["order"]]
        steps["retrain"] >> steps["order"] >> steps["report"]
    return dag


for _env, _settings in (yaml.safe_load(OPS_CONFIG.read_text(encoding="utf-8")).get("envs") or {}).items():
    globals()[f"vn2_{_env}"] = build_dag(_env, _settings)
