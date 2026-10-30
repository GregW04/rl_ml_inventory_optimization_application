"""VN2Env: observation layout from the spec, residual and raw action mappings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.rl.rollout import ZeroResidual, anchor_baseline, rollout
from src.rl.spec import PolicySpec, default_state, validate_spec
from src.simulation.anchor import AnchorSpec
from src.simulation.vn2 import VN2Rules
from src.simulation.vn2_env import VN2Env

RULES = VN2Rules()
ANCHOR = AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 0.8333)
MEASURES = ["mean_h0", "mean_h1", "q83_h2"]
DATES = pd.date_range("2024-01-01", periods=8, freq="W-MON")


def _episode():
    return pd.DataFrame({
        "store": 1, "product": 2, "date": DATES, "sales": [5, 3, 0, 8, 4, 6, 2, 7],
        "in_stock": True, "zero_share": 0.1, "pos_cv": 0.5,
        "end_inventory": [4.0] + [np.nan] * 7, "in_transit_1w": [2.0] + [np.nan] * 7,
        "in_transit_2w": [1.0] + [np.nan] * 7,
    })


LOOKUP = {(1, 2, d): {"mean_h0": 5.0, "mean_h1": 4.0, "q83_h2": 7.0} for d in DATES}


def _spec(**kw):
    base = PolicySpec("run", "residual", "ppo", default_state("residual", MEASURES), ANCHOR)
    return replace(base, **kw)


def test_observation_follows_spec_state():
    spec = _spec(state=["on_hand", "in_transit_w1", "in_transit_w2", "q83_h2", "round_frac", "forecast_error"])
    env = VN2Env(spec, _episode(), LOOKUP, demand_scale=10.0, rules=RULES)
    obs, _ = env.reset()
    assert obs.shape == (6,)
    assert obs[:4] == pytest.approx([0.4, 0.2, 0.1, 0.7])
    assert obs[4] == pytest.approx(1 / 6)
    assert env.observation_space.low[5] < 0  # forecast_error is signed
    obs, *_ = env.step(np.array([0.0]))
    assert obs[5] == pytest.approx((5 - 5.0) / 10.0)


def test_zero_residual_is_the_anchor():
    env = VN2Env(_spec(), _episode(), LOOKUP, demand_scale=10.0, rules=RULES)
    env.reset()
    for _ in range(RULES.n_order_rounds):
        assert env.resolve_order(np.array([0.0])) == env._decision.order
        env.step(np.array([0.0]))


def test_raw_action_mapping():
    spec = _spec(method="raw", order_cap_mult=2.0, state=default_state("raw", MEASURES))
    env = VN2Env(spec, _episode(), LOOKUP, demand_scale=10.0, rules=RULES)
    env.reset()
    assert env.resolve_order(np.array([-1.0])) == 0.0
    assert env.resolve_order(np.array([1.0])) == 20.0
    assert env.resolve_order(np.array([0.0])) == 10.0
    dqn = replace(spec, agent="dqn", buckets=[0.0, 0.25, 1.0])
    env = VN2Env(dqn, _episode(), LOOKUP, demand_scale=10.0, rules=RULES)
    env.reset()
    assert env.action_space.n == 3
    assert env.resolve_order(1) == 5.0


def test_anchor_baseline_matches_for_raw_specs():
    raw = _spec(method="raw", agent="dqn", buckets=[0.0, 1.0], state=default_state("raw", MEASURES))
    base_spec, policy = anchor_baseline(raw)
    eps = {(1, 2, DATES[0]): _episode()}
    a, _ = rollout(policy, base_spec, eps, LOOKUP, {(1, 2): 10.0}, RULES)
    b, _ = rollout(ZeroResidual(), _spec(), eps, LOOKUP, {(1, 2): 10.0}, RULES)
    assert a["window_cost_eur"] == pytest.approx(b["window_cost_eur"])


def test_spec_validation_and_config_key():
    spec = _spec()
    assert validate_spec(spec, MEASURES) == []
    assert any("not available" in e for e in validate_spec(replace(spec, state=spec.state + ["q90_h2"]), MEASURES))
    assert any("inventory pipeline" in e for e in validate_spec(replace(spec, state=["q83_h2"]), MEASURES))
    assert spec.config_key() == spec.with_seed(7).config_key()
    assert spec.config_key() != replace(spec, agent="dqn", buckets=[0.0]).config_key()
    assert PolicySpec.from_dict(spec.to_dict()) == spec
