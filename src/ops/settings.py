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
from pydantic import BaseModel, ConfigDict

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
    window_weeks: int = 4
    band: tuple[float, float] = (0.79, 0.88)
    persistence_weeks: int = 2


class Probation(_Strict):
    evidence_weeks: int = 4
    min_evidence_weeks: int = 2
    rollback_margin_pct: float = 1.0


class Sanity(_Strict):
    max_total_change_pct: float = 30.0


class Rules(_Strict):
    strategy: Literal["never", "fixed", "drift", "drift_backstop"] = "drift_backstop"
    max_age_weeks: int = 13
    cooldown_weeks: int = 4
    bias: Bias = Bias()
    coverage: Coverage = Coverage()
    probation: Probation = Probation()
    sanity: Sanity = Sanity()
    n_boot: int = 1000
    seed: int = 42

    def version(self) -> str:
        """Short hash of the rules: tagged on every cycle, frozen before the live run."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class Env(_Strict):
    kind: Literal["sim", "live"]
    start: date
    end: date
    # Overrides of config["forecasting"] and the recipe's hyperparameters for this environment.
    forecasting: dict = {}
    model_params: dict | None = None
    rules: dict = {}


class OpsConfig(_Strict):
    frozen: bool = False
    rules: Rules = Rules()
    envs: dict[str, Env] = {}


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


def env_config(config: dict, ops: OpsConfig, env: str) -> dict:
    """The project config with the environment's forecasting overrides applied."""
    cfg = copy.deepcopy(config)
    cfg["forecasting"] = _merge(cfg["forecasting"], env_settings(ops, env).forecasting)
    return cfg


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
