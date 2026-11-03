"""The DAG factory: every ops environment gets a DAG with the weekly cycle's shape and settings."""

from __future__ import annotations

import ast
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from airflow.dag_processing.dagbag import DagBag

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "airflow" / "dags"


def _project_steps() -> tuple[str, ...]:
    """STEPS of src/ops/steps.py, read without importing the project (it isn't installed here)."""
    tree = ast.parse((ROOT / "src" / "ops" / "steps.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "STEPS")
    return tuple(ast.literal_eval(node.value))


@pytest.fixture(scope="module")
def factory():
    spec = importlib.util.spec_from_file_location("vn2_ops", DAGS / "vn2_ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_environment_gets_a_dag_without_import_errors():
    bag = DagBag(dag_folder=str(DAGS))  # examples are off (conftest)
    assert not bag.import_errors
    envs = yaml.safe_load((ROOT / "configs" / "ops.yaml").read_text())["envs"]
    assert set(bag.dag_ids) == {f"vn2_{e}" for e in envs}


def test_the_dag_runs_the_project_steps_in_order(factory):
    assert factory.STEPS == _project_steps()


@pytest.mark.parametrize("kind", ["sim", "live"])
def test_cycle_shape(factory, kind):
    dag = factory.build_dag("t", {"kind": kind, "start": "2023-10-02", "end": "2023-12-11"})
    first = "wait_for_drop" if kind == "live" else "drop"
    up = {t: dag.task_dict[t].upstream_task_ids for t in dag.task_dict}
    assert up[first] == set() and up["ingest"] == {first}
    assert up["route_forecaster"] == {"decide"} and up["refit"] == {"route_forecaster"}
    assert up["forecast"] == {"route_forecaster", "refit"} and up["route_policy"] == {"forecast"}
    assert up["retrain"] == {"route_policy"} and up["order"] == {"route_policy", "retrain"} and up["report"] == {"order"}
    # A skipped refit or retrain doesn't stop the week; only they take the heavy slot.
    assert {dag.task_dict[t].trigger_rule for t in ("forecast", "order")} == {"none_failed_min_one_success"}
    assert {t for t, op in dag.task_dict.items() if op.pool == "heavy"} == {"refit", "retrain"}
    assert dag.max_active_runs == 1 and dag.catchup
    assert dag.start_date == datetime(2023, 10, 2, tzinfo=UTC) and dag.end_date == datetime(2023, 12, 11, tzinfo=UTC)
    assert all(dag.task_dict[t].depends_on_past for t in dag.task_dict)
    assert "step refit --env t --as-of {{ ds }}" in dag.task_dict["refit"].bash_command


def test_routes_follow_the_decision_file(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "ROOT", tmp_path)
    folder = tmp_path / "ops" / "t" / "cycles" / "2023-10-09"
    folder.mkdir(parents=True)
    dag = factory.build_dag("t", {"kind": "sim", "start": "2023-10-02", "end": "2023-12-11"})
    route_fc, route_policy = (dag.task_dict[t].python_callable for t in ("route_forecaster", "route_policy"))
    for action, policy, want in [("keep", "keep", ("forecast", "order")), ("refit", "retrain", ("refit", "retrain")),
                                 ("rollback", "fallback", ("refit", "retrain")), ("bootstrap", "bootstrap", ("refit", "retrain"))]:
        (folder / "decision.json").write_text(f'{{"action": "{action}", "policy": {{"action": "{policy}"}}}}')
        assert (route_fc(ds="2023-10-09"), route_policy(ds="2023-10-09")) == want
