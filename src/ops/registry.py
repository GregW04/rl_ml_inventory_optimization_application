"""
Model Registry conventions of the ops loop.

One registered model per layer and environment (``<env>-vn2-forecaster``, ``<env>-vn2-policy``)
and three aliases with fixed meanings:

* ``champion``   — serves orders. Consumers resolve ``models:/<name>@champion`` at runtime, so
  moving this alias *is* the deployment.
* ``challenger`` — a candidate under validation.
* ``previous``   — the champion before the last promotion: the rollback target.

Model version tags carry the audit trail (trigger, recipe key, data digest, validation status).
Versions are handled as strings here (MLflow returns them as ints).
"""

from __future__ import annotations

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
LAYERS = ("forecaster", "policy")


def model_name(env: str, layer: str) -> str:
    """Registered model of a layer in an environment (``live`` or ``sim-<name>``)."""
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer '{layer}'; choose from {LAYERS}.")
    return f"{env}-vn2-{layer}"


def register(model_uri: str, name: str, tags: dict | None = None) -> ModelVersion:
    """A new version of ``name`` from a logged model (``models:/<model_id>``), tagged."""
    return mlflow.register_model(model_uri, name, tags={k: str(v) for k, v in (tags or {}).items()})


def resolve(name: str, alias: str) -> ModelVersion | None:
    """The version behind ``alias``, or None when the alias (or the model) doesn't exist."""
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def set_challenger(name: str, version: str | int) -> None:
    MlflowClient().set_registered_model_alias(name, CHALLENGER, str(version))


def promote(name: str, version: str | int) -> None:
    """
    Make ``version`` the champion. The old champion becomes ``previous`` first, so a failure in
    between never leaves the model without a champion; a promoted challenger loses that alias.
    """
    client, version = MlflowClient(), str(version)
    old = resolve(name, CHAMPION)
    if old is not None and str(old.version) == version:
        return
    if old is not None:
        client.set_registered_model_alias(name, PREVIOUS, str(old.version))
    client.set_registered_model_alias(name, CHAMPION, version)
    challenger = resolve(name, CHALLENGER)
    if challenger is not None and str(challenger.version) == version:
        client.delete_registered_model_alias(name, CHALLENGER)


def rollback(name: str) -> str:
    """Champion ← previous; the rolled-back version keeps no alias. Returns the restored version."""
    client = MlflowClient()
    previous = resolve(name, PREVIOUS)
    if previous is None:
        raise RuntimeError(f"{name} has no previous champion to roll back to.")
    client.set_registered_model_alias(name, CHAMPION, str(previous.version))
    client.delete_registered_model_alias(name, PREVIOUS)
    return str(previous.version)
