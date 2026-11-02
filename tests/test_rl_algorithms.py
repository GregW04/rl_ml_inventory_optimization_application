"""PPO and GRPO on the batched replay: start at the anchor, save/load, learn a biased anchor away."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.rl.batch_env import build_episode_arrays
from src.rl.batch_rollout import EarlyStop
from src.rl.grpo import train_grpo
from src.rl.policy import Actor, TorchPolicy
from src.rl.ppo import train_ppo
from src.rl.rollout import ZeroResidual, rollout
from src.rl.spec import EARLY_STOP_KEYS, PolicySpec, build_spec, default_state, torch_env_steps, validate_spec
from src.simulation.anchor import AnchorSpec
from src.simulation.vn2 import VN2Rules
from src.utils.config import load_config
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
    curve: list[tuple[int, float, float]] = []
    trainer = train_ppo if agent == "ppo" else train_grpo
    policy = trainer(spec, arrays, RULES, seed=0, eval_idx=torch.arange(len(arrays)),
                     on_eval=lambda w, pct, units: curve.append((w, pct, units)))
    assert curve[0][1] == pytest.approx(0.0, abs=1e-9)  # starts at the anchor
    assert curve[0][2] == pytest.approx(0.0, abs=1e-9)  # ... ordering the anchor's units
    assert curve[-1][1] < -5.0, curve                    # learns to order less
    assert curve[-1][2] < 0.0, curve
    # the trained policy behaves the same through the canonical VN2Env rollout
    a, _ = rollout(policy, spec, episodes, lookup, scales, RULES)
    b, _ = rollout(ZeroResidual(), spec, episodes, lookup, scales, RULES)
    assert a["window_cost_eur"] < b["window_cost_eur"]


@pytest.mark.parametrize("agent", ["ppo", "grpo"])
def test_logs_the_same_training_health_metrics(agent):
    episodes, lookup, scales = _synthetic(n=80, seed=3)
    spec = replace(_spec(agent), agent_params={**SMALL, "iterations": 3})
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    stats: list[dict] = []
    trainer = train_ppo if agent == "ppo" else train_grpo
    trainer(spec, arrays, RULES, seed=0, on_iteration=lambda it, s: stats.append(s))
    shared = {"train_vs_anchor_pct", "train_units_vs_anchor_pct", "policy_std", "approx_kl", "kl_to_start",
              "clip_frac", "grad_norm"}
    own = {"explained_variance"} if agent == "ppo" else {"group_spread_eur", "zero_spread_share"}
    assert len(stats) == 3
    for s in stats:
        assert set(s) == shared | own and all(np.isfinite(v) for v in s.values()), s
        assert s["approx_kl"] >= 0 and s["kl_to_start"] >= -1e-9 and 0 <= s["clip_frac"] <= 1 and s["grad_norm"] > 0
        if agent == "ppo":
            assert s["explained_variance"] <= 1
        else:
            assert s["group_spread_eur"] >= 0 and 0 <= s["zero_spread_share"] <= 1
    assert stats[-1]["kl_to_start"] > 0  # training moved the policy away from where it started


@pytest.mark.parametrize("agent", ["ppo", "grpo"])
def test_raw_method_trains(agent):
    episodes, lookup, scales = _synthetic(n=80, seed=6)
    spec = replace(_spec(agent, "raw"), order_cap_mult=1.0, agent_params={**SMALL, "iterations": 5})
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    trainer = train_ppo if agent == "ppo" else train_grpo
    policy = trainer(spec, arrays, RULES, seed=0)
    kpis, _ = rollout(policy, spec, episodes, lookup, scales, RULES)
    assert np.isfinite(kpis["window_cost_eur"])


def test_early_stop_rule_on_a_logged_curve():
    # A real PPO run's training sample, scored every 200 iterations: no gain of 0.2 pts after iteration 600.
    stop = EarlyStop(patience=400, min_delta=0.2)
    curve = [(0, 0.0), (200, -2.57), (400, -2.65), (600, -2.85), (800, -3.02), (1000, -2.89), (1200, -3.20)]
    assert [it for it, v in curve if stop.should_stop(it, v)] == [1000]
    with pytest.raises(ValueError):
        EarlyStop(patience=0, min_delta=0.2)


@pytest.mark.parametrize("agent", ["ppo", "grpo"])
def test_trainers_stop_early_on_the_training_sample(agent):
    episodes, lookup, scales = _synthetic(n=80, seed=3)
    # No evaluation beats the best by 100 pts, so training stops once 2 iterations pass without one.
    params = {**SMALL, "iterations": 10, "early_stop": True, "early_stop_patience": 2,
              "early_stop_min_delta": 100.0, "early_stop_every": 1}
    spec = replace(_spec(agent), agent_params=params)
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    trainer = train_ppo if agent == "ppo" else train_grpo
    scored: list[int] = []
    policy = trainer(spec, arrays, RULES, seed=0, eval_idx=torch.arange(len(arrays)),
                     on_eval=lambda w, pct, units: scored.append(w))
    assert policy.meta["iterations_run"] == 2 and len(scored) == 3  # scored at iterations 0, 1 and 2
    full = replace(spec, agent_params={**SMALL, "iterations": 3})
    assert trainer(full, arrays, RULES, seed=0).meta["iterations_run"] == 3
    with pytest.raises(ValueError, match="eval_idx"):
        trainer(spec, arrays, RULES, seed=0)


def test_build_spec_carries_early_stopping_only_when_on():
    config, state = load_config(), default_state("residual", MEASURES)
    off = build_spec(config, "run", "residual", "ppo", state, ANCHOR, iterations=900)
    assert not set(EARLY_STOP_KEYS) & set(off.agent_params)  # keys of earlier runs stay the same
    on = build_spec(config, "run", "residual", "ppo", state, ANCHOR, iterations=900,
                    early_stop={"early_stop": True, "early_stop_patience": 300})
    assert on.agent_params["early_stop_patience"] == 300 and "early_stop_every" in on.agent_params
    assert on.config_key() != off.config_key() and validate_spec(on, MEASURES) == []
    bad = replace(on, agent_params={**on.agent_params, "early_stop_every": 0})
    assert any("early_stop_every" in e for e in validate_spec(bad, MEASURES))


def test_spec_checks_and_budget():
    spec = _spec("grpo")
    assert validate_spec(spec, MEASURES) == []
    assert torch_env_steps("grpo", spec.agent_params) == 25 * 64 * 8 * 8
    bad = replace(spec, agent_params={**SMALL, "group_size": 1})
    assert any("group" in e for e in validate_spec(bad, MEASURES))
    bad = replace(_spec("ppo"), agent_params={**SMALL, "iterations": 0})
    assert any("iterations" in e for e in validate_spec(bad, MEASURES))
