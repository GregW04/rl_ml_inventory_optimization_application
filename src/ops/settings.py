"""
Settings of the ops loop (``configs/ops.yaml``): environments, rules and thresholds, validated
strictly so a mistyped threshold fails instead of silently falling back to a default.

An environment (``dev``, ``sim-<name>``, ``live``) has its own registered models, MLflow
experiment and local state under ``ops/<env>/``; its rules are the shared ``rules`` with the
environment's overrides on top.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from src.utils.config import get_project_root

STRATEGIES = ("never", "fixed", "drift", "drift_backstop")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bias(_Strict):
    alpha: float = 0.2
    limit: float = 0.74
    severe_limit: float = 0.9
    warmup_weeks: int = 4
    persistence_weeks: int = 2
    min_cost_share: float = 0.15


class Coverage(_Strict):
    # False: computed and logged every week, but never a reason to refit (a diagnostic).
    trigger: bool = True
    window_weeks: int = 4
    band: tuple[float, float] = (0.79, 0.88)
    persistence_weeks: int = 2


class Probation(_Strict):
    evidence_weeks: int = 4
    min_evidence_weeks: int = 2
    rollback_margin_pct: float = 1.0


class Sanity(_Strict):
    max_total_change_pct: float = 30.0


class PolicyRules(_Strict):
    """The RL layer: when to train a PPO challenger, how it must beat the anchor, when to fall back."""

    enabled: bool = True
    agent: Literal["ppo", "grpo"] = "ppo"
    method: Literal["residual", "raw"] = "residual"
    training_forecasts: Literal["as_is", "unbiased", "unbiased_by_class"] = "unbiased_by_class"
    # Hyperparameters on top of config rl.agents.<agent> (the research champion's, by default).
    agent_params: dict = {}
    seeds: list[int] = [42, 43, 44]
    train_weeks: int = 26
    val_weeks: int = 8
    retrain_every_weeks: int = 13
    gate_churn_limit: float = 0.10
    # Promotion on the validation weeks (the gate is picked there, as in research).
    promote_margin_pct: float = 0.5
    units_band_pct: float = 2.0
    min_seed_share: float = 0.67
    # Fallback to the anchor: the live RL trajectory against the anchor's shadow.
    fallback_margin_pct: float = 0.5
    evidence_weeks: int = 4
    min_evidence_weeks: int = 2


class Rules(_Strict):
    strategy: Literal["never", "fixed", "drift", "drift_backstop"] = "drift_backstop"
    max_age_weeks: int = 13
    cooldown_weeks: int = 4
    bias: Bias = Bias()
    coverage: Coverage = Coverage()
    probation: Probation = Probation()
    sanity: Sanity = Sanity()
    policy: PolicyRules = PolicyRules()
    n_boot: int = 1000
    seed: int = 42

    def version(self) -> str:
        """Short hash of the rules: tagged on every cycle, frozen before the live run."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class RecipeSpec(_Strict):
    """A forecaster recipe environments share: overrides of config["forecasting"] and hyperparameters."""

    forecasting: dict = {}
    # Per ensemble member, or the backend's own; None → the backends' defaults.
    model_params: dict | None = None


class Env(_Strict):
    kind: Literal["sim", "live"]
    start: date
    end: date
    # A recipe from `recipes`; this environment's own forecasting / model_params override it.
    recipe: str | None = None
    forecasting: dict = {}
    model_params: dict | None = None
    rules: dict = {}


class OpsConfig(_Strict):
    # Set after the shadow replay: `frozen: true` and the version of the rules it froze. A live
    # environment runs only these rules (and only from a clean tree).
    frozen: bool = False
    frozen_rules: str | None = None
    rules: Rules = Rules()
    recipes: dict[str, RecipeSpec] = {}
    envs: dict[str, Env] = {}

    @model_validator(mode="after")
    def _known_recipes(self) -> OpsConfig:
        unknown = {e: s.recipe for e, s in self.envs.items() if s.recipe and s.recipe not in self.recipes}
        if unknown:
            raise ValueError(f"Environments use undefined recipes: {unknown}")
        return self


def load_ops_config(path: str | Path | None = None) -> OpsConfig:
    path = Path(path) if path else get_project_root() / "configs" / "ops.yaml"
    with open(path, encoding="utf-8") as f:
        return OpsConfig.model_validate(yaml.safe_load(f))


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def env_settings(ops: OpsConfig, env: str) -> Env:
    if env not in ops.envs:
        raise KeyError(f"Unknown ops environment '{env}'; configs/ops.yaml defines {sorted(ops.envs)}.")
    return ops.envs[env]


def env_rules(ops: OpsConfig, env: str) -> Rules:
    """The shared rules with the environment's overrides on top."""
    return Rules.model_validate(_merge(ops.rules.model_dump(), env_settings(ops, env).rules))


def _recipe(ops: OpsConfig, env: str) -> RecipeSpec:
    s = env_settings(ops, env)
    return ops.recipes[s.recipe] if s.recipe else RecipeSpec()


def env_config(config: dict, ops: OpsConfig, env: str) -> dict:
    """The project config with the environment's recipe and forecasting overrides applied."""
    cfg = copy.deepcopy(config)
    cfg["forecasting"] = _merge(_merge(cfg["forecasting"], _recipe(ops, env).forecasting), env_settings(ops, env).forecasting)
    return cfg


def recipe_config(config: dict, ops: OpsConfig, name: str) -> dict:
    """The project config with recipe ``name``'s forecasting overrides applied."""
    if name not in ops.recipes:
        raise KeyError(f"Unknown recipe '{name}'; configs/ops.yaml defines {sorted(ops.recipes)}.")
    cfg = copy.deepcopy(config)
    cfg["forecasting"] = _merge(cfg["forecasting"], ops.recipes[name].forecasting)
    return cfg


def env_model_params(ops: OpsConfig, env: str) -> dict | None:
    """The hyperparameters a bootstrap fits with: the environment's own, else its recipe's."""
    own = env_settings(ops, env).model_params
    return own if own is not None else _recipe(ops, env).model_params


def env_dir(config: dict, env: str) -> Path:
    """Local state of an environment: drops, per-cycle ledgers and decisions."""
    return Path(config["_project_root"]) / "ops" / env


def cycle_dir(config: dict, env: str, as_of: pd.Timestamp) -> Path:
    return env_dir(config, env) / "cycles" / str(pd.Timestamp(as_of).date())


def weeks(start: date | str, end: date | str) -> list[pd.Timestamp]:
    """The as-of weeks (Mondays) from ``start`` to ``end`` inclusive."""
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="7D"))


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
