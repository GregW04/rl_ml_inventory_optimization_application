"""
Policy spec: everything needed to rebuild an RL policy's environment, saved with each run.

A spec fixes the forecasting run, the method (``residual`` on the anchor or ``raw`` from
scratch), the agent, the ordered state features, the anchor inputs and the action
parameters. Evaluation rebuilds the environment from the saved spec, never from the
current config, so a saved agent always sees the observation layout it was trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from src.simulation.anchor import AnchorSpec

METHODS = ("residual", "raw")
# DQN trains on Stable-Baselines3 (the baseline); PPO and GRPO train on the batched
# replay in src.rl.batch_env (src.rl.ppo, src.rl.grpo).
AGENTS = ("dqn", "ppo", "grpo")
TORCH_AGENTS = ("ppo", "grpo")
AGENT_LABELS = {
    "dqn": "DQN · Stable-Baselines3 baseline",
    "ppo": "PPO · batched replay, reward paired with the anchor",
    "grpo": "GRPO · groups of rollouts per window",
}
STEPS_PER_EPISODE = 8  # one VN2 window: 6 order rounds + 2 delivery weeks

INVENTORY_FEATURES: tuple[str, ...] = ("on_hand", "in_transit_w1", "in_transit_w2")
ANCHOR_FEATURES: tuple[str, ...] = ("anchor_order", "projected_stock", "projected_lost")
CONTEXT_FEATURES: tuple[str, ...] = ("forecast_error", "round_frac", "zero_share", "pos_cv")
# One-hot demand pattern of the week (Smooth, Erratic, Intermittent, Lumpy); all zero when
# the SKU has too little history to classify.
CLASS_FEATURES: tuple[str, ...] = ("class_smooth", "class_erratic", "class_intermittent", "class_lumpy")
# Features that are not in units of demand (not divided by the SKU's demand scale).
UNITLESS_FEATURES = frozenset({"round_frac", "zero_share", "pos_cv", *CLASS_FEATURES})
SIGNED_FEATURES = frozenset({"forecast_error"})
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "Inventory": INVENTORY_FEATURES,
    "Anchor-derived": ANCHOR_FEATURES,
    "Context": CONTEXT_FEATURES,
    "Demand class": CLASS_FEATURES,
}


def is_torch_agent(agent: str) -> bool:
    return agent in TORCH_AGENTS


def windows_per_iteration(agent: str, params: dict) -> int:
    """Rollouts per PPO / GRPO iteration (GRPO replays each sampled window ``group_size`` times)."""
    group = int(params.get("group_size", 1) or 1) if agent == "grpo" else 1
    return int(params.get("batch_windows", 0) or 0) * group


def torch_env_steps(agent: str, params: dict) -> int:
    """Environment steps of a PPO / GRPO run, comparable with DQN's ``total_timesteps``."""
    return int(params.get("iterations", 0) or 0) * windows_per_iteration(agent, params) * STEPS_PER_EPISODE


def default_state(method: str, measure_ids: list[str]) -> list[str]:
    """Residual: everything (today's layout); raw: inventory, forecasts and the round."""
    if method == "residual":
        return [*INVENTORY_FEATURES, *measure_ids, *ANCHOR_FEATURES, *CONTEXT_FEATURES]
    return [*INVENTORY_FEATURES, *measure_ids, "round_frac"]


@dataclass
class PolicySpec:
    forecast_run_id: str
    method: str
    agent: str
    state: list[str]
    anchor: AnchorSpec
    floor_frac: float = 0.25
    order_cap_mult: float = 1.0
    buckets: list[float] | None = None
    seed: int = 42
    total_timesteps: int = 100_000
    # Agent hyperparameters (``rl.agents.<agent>`` at training time), so the spec and
    # its config_key capture everything that shapes the policy.
    agent_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["anchor"] = self.anchor.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PolicySpec:
        d = dict(d)
        d["anchor"] = AnchorSpec.from_dict(d["anchor"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PolicySpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def config_key(self) -> str:
        """Hash of everything but the seed: runs that differ only by seed share it."""
        d = self.to_dict()
        d.pop("seed", None)
        d.pop("order_cap_mult" if self.method == "residual" else "floor_frac", None)
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def with_seed(self, seed: int) -> PolicySpec:
        return replace(self, seed=int(seed))

    def label(self) -> str:
        return f"{self.method} · {self.agent.upper()} · {len(self.state)} features"


def validate_spec(spec: PolicySpec, measure_ids: list[str]) -> list[str]:
    """Problems that block training with ``spec`` on a run with ``measure_ids``."""
    errors: list[str] = []
    if spec.method not in METHODS:
        errors.append(f"Unknown method '{spec.method}'.")
    if spec.agent not in AGENTS:
        errors.append(f"Unknown agent '{spec.agent}'.")
    known = (
        set(INVENTORY_FEATURES) | set(ANCHOR_FEATURES) | set(CONTEXT_FEATURES) | set(CLASS_FEATURES)
        | set(measure_ids)
    )
    unknown = [f for f in spec.state if f not in known]
    if unknown:
        errors.append(f"State features not available in this forecasting run: {unknown}.")
    missing_inv = [f for f in INVENTORY_FEATURES if f not in spec.state]
    if missing_inv:
        errors.append(f"The state must include the inventory pipeline {list(INVENTORY_FEATURES)}.")
    if len(set(spec.state)) != len(spec.state):
        errors.append("State features must be unique.")
    missing_anchor = sorted(spec.anchor.inputs() - set(measure_ids))
    if missing_anchor:
        errors.append(f"Anchor inputs not in this forecasting run: {missing_anchor}.")
    if spec.agent == "dqn" and not spec.buckets:
        errors.append("DQN needs a list of action buckets.")
    if spec.agent in TORCH_AGENTS:
        # Missing keys fall back to the trainer's defaults; values that are present must be valid.
        p = spec.agent_params
        for key in ("iterations", "batch_windows", "epochs", "minibatch_size"):
            if key not in p:
                continue
            try:
                ok = int(p[key]) >= 1
            except (TypeError, ValueError):
                ok = False
            if not ok:
                errors.append(f"{spec.agent.upper()} needs a positive whole `{key}`.")
        if spec.agent == "grpo" and "group_size" in p and int(p["group_size"] or 0) < 2:
            errors.append("GRPO needs a group of at least 2 rollouts per window.")
    if spec.method == "raw" and spec.order_cap_mult <= 0:
        errors.append("order_cap_mult must be positive.")
    if spec.method == "residual" and spec.floor_frac <= 0:
        errors.append("floor_frac must be positive.")
    return errors


def build_spec(
    config: dict,
    forecast_run_id: str,
    method: str,
    agent: str,
    state: list[str],
    anchor: AnchorSpec,
    *,
    total_timesteps: int | None = None,
    iterations: int | None = None,
    floor_frac: float | None = None,
    order_cap_mult: float | None = None,
    seed: int | None = None,
) -> PolicySpec:
    """Spec with action parameters and agent hyperparameters filled from the config.

    DQN's budget is ``total_timesteps``. PPO and GRPO budget in ``iterations`` (each one
    replays a batch of windows); their ``total_timesteps`` is derived so budgets compare.
    """
    mcfg = config["rl"]["methods"][method]
    params = dict(config["rl"]["agents"][agent])
    if agent in TORCH_AGENTS:
        params.pop("total_timesteps", None)
        if iterations is not None:
            params["iterations"] = int(iterations)
        steps = torch_env_steps(agent, params)
    else:
        steps = int(total_timesteps or params.pop("total_timesteps", 100_000))
        params.pop("total_timesteps", None)
    return PolicySpec(
        forecast_run_id=forecast_run_id,
        method=method,
        agent=agent,
        state=list(state),
        anchor=anchor,
        floor_frac=float(floor_frac if floor_frac is not None else config["rl"]["methods"]["residual"]["floor_frac"]),
        order_cap_mult=float(
            order_cap_mult if order_cap_mult is not None else config["rl"]["methods"]["raw"]["order_cap_mult"]
        ),
        buckets=[float(b) for b in mcfg["dqn_buckets"]] if agent == "dqn" else None,
        seed=int(seed if seed is not None else config.get("project", {}).get("seed", 42)),
        total_timesteps=steps,
        agent_params=params,
    )
