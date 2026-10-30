"""PPO and GRPO on the batched replay: start at the anchor, save/load, learn a biased anchor away."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.rl.batch_env import build_episode_arrays
from src.rl.grpo import train_grpo
from src.rl.policy import Actor, TorchPolicy
from src.rl.ppo import train_ppo
from src.rl.rollout import ZeroResidual, rollout
from src.rl.spec import PolicySpec, default_state, torch_env_steps, validate_spec
from src.simulation.anchor import AnchorSpec
from src.simulation.vn2 import VN2Rules
from tests.test_rl_batch_env import _synthetic

RULES = VN2Rules()
MEASURES = ["mean_h0", "mean_h1", "q50_h2", "q83_h2"]
ANCHOR = AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 0.8333)
SMALL = {"iterations": 25, "batch_windows": 64, "epochs": 4, "minibatch_size": 256, "learning_rate": 3e-3,
         "group_size": 8, "hidden": [32, 32]}


def _spec(agent: str, method: str = "residual") -> PolicySpec:
    return PolicySpec("run", method, agent, default_state(method, MEASURES), ANCHOR, agent_params=dict(SMALL))


def _biased(n: int = 160):
    """Synthetic windows whose delivery-week quantile is three times too high: the anchor overstocks."""
    episodes, lookup, scales = _synthetic(n=n, seed=11)
    for f in lookup.values():
        f["q83_h2"] *= 3.0
    return episodes, lookup, scales


def test_untrained_residual_policy_is_the_anchor():
    episodes, lookup, scales = _synthetic(seed=4)
    spec = _spec("ppo")
    policy = TorchPolicy(Actor(len(spec.state)), {"obs_dim": len(spec.state), "hidden": [64, 64]})
    a, _ = rollout(policy, spec, episodes, lookup, scales, RULES)
    b, _ = rollout(ZeroResidual(), spec, episodes, lookup, scales, RULES)
    assert a["window_cost_eur"] == pytest.approx(b["window_cost_eur"], abs=1e-12)


def test_save_load_round_trip(tmp_path):
    actor = Actor(7, (16, 16))
    with torch.no_grad():
        for prm in actor.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    policy = TorchPolicy(actor, {"obs_dim": 7, "hidden": [16, 16], "agent": "grpo"})
    policy.save(tmp_path / "model.pt")
    loaded = TorchPolicy.load(tmp_path / "model.pt")
    obs = np.random.default_rng(0).uniform(0, 2, (20, 7)).astype(np.float32)
    for o in obs:
        assert loaded.predict(o)[0] == pytest.approx(policy.predict(o)[0])
        assert -1.0 <= float(loaded.predict(o)[0][0]) <= 1.0


@pytest.mark.parametrize("agent", ["ppo", "grpo"])
def test_learns_to_correct_a_biased_anchor(agent):
    episodes, lookup, scales = _biased()
    spec = _spec(agent)
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    curve: list[tuple[int, float]] = []
    trainer = train_ppo if agent == "ppo" else train_grpo
    policy = trainer(spec, arrays, RULES, seed=0, eval_idx=torch.arange(len(arrays)),
                     on_eval=lambda w, pct: curve.append((w, pct)))
    assert curve[0][1] == pytest.approx(0.0, abs=1e-9)  # starts at the anchor
    assert curve[-1][1] < -5.0, curve                    # learns to order less
    # the trained policy behaves the same through the canonical VN2Env rollout
    a, _ = rollout(policy, spec, episodes, lookup, scales, RULES)
    b, _ = rollout(ZeroResidual(), spec, episodes, lookup, scales, RULES)
    assert a["window_cost_eur"] < b["window_cost_eur"]


@pytest.mark.parametrize("agent", ["ppo", "grpo"])
def test_raw_method_trains(agent):
    episodes, lookup, scales = _synthetic(n=80, seed=6)
    spec = replace(_spec(agent, "raw"), order_cap_mult=1.0, agent_params={**SMALL, "iterations": 5})
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    trainer = train_ppo if agent == "ppo" else train_grpo
    policy = trainer(spec, arrays, RULES, seed=0)
    kpis, _ = rollout(policy, spec, episodes, lookup, scales, RULES)
    assert np.isfinite(kpis["window_cost_eur"])


def test_spec_checks_and_budget():
    spec = _spec("grpo")
    assert validate_spec(spec, MEASURES) == []
    assert torch_env_steps("grpo", spec.agent_params) == 25 * 64 * 8 * 8
    bad = replace(spec, agent_params={**SMALL, "group_size": 1})
    assert any("group" in e for e in validate_spec(bad, MEASURES))
    bad = replace(_spec("ppo"), agent_params={**SMALL, "iterations": 0})
    assert any("iterations" in e for e in validate_spec(bad, MEASURES))
