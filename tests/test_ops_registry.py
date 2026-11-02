"""Alias semantics of the ops registry: promote keeps a rollback target, rollback restores it."""

from __future__ import annotations

import mlflow
import pytest

from src.ops import registry
from src.ops.registry import CHALLENGER, CHAMPION, PREVIOUS


class _Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def _new_version(name: str) -> str:
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(name="m", python_model=_Echo())
    return str(registry.register(info.model_uri, name, {"trigger": "test"}).version)


def _alias(name: str, alias: str) -> str | None:
    mv = registry.resolve(name, alias)
    return None if mv is None else str(mv.version)


def test_promote_keeps_the_old_champion_as_previous_and_rollback_restores_it():
    name = registry.model_name("test", "forecaster")
    v1, v2 = _new_version(name), _new_version(name)
    assert _alias(name, CHAMPION) is None

    registry.promote(name, v1)
    assert (_alias(name, CHAMPION), _alias(name, PREVIOUS)) == (v1, None)

    registry.set_challenger(name, v2)
    registry.promote(name, v2)
    assert (_alias(name, CHAMPION), _alias(name, PREVIOUS), _alias(name, CHALLENGER)) == (v2, v1, None)
    registry.promote(name, v2)  # promoting the champion again changes nothing
    assert _alias(name, PREVIOUS) == v1

    assert registry.rollback(name) == v1
    assert (_alias(name, CHAMPION), _alias(name, PREVIOUS)) == (v1, None)
    with pytest.raises(RuntimeError):
        registry.rollback(name)
    assert registry.resolve(name, CHAMPION).tags["trigger"] == "test"


def test_model_names_follow_env_and_layer():
    assert registry.model_name("live", "policy") == "live-vn2-policy"
    with pytest.raises(ValueError):
        registry.model_name("live", "orders")
