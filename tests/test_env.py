"""VN2Env: observation layout from the spec, residual and raw action mappings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.rl.explain import drivers, explain_order
from src.rl.rollout import ZeroResidual, anchor_baseline, rollout, vs_anchor
from src.rl.selection import class_gate, gate_episodes
from src.rl.spec import UNITLESS_FEATURES, PolicySpec, default_state, validate_spec
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


class _OnHandPolicy:
    """Correction of 0.1 × on_hand (observation units): only the first input matters."""

    def predict(self, obs, deterministic: bool = True):
        return np.array([0.1 * obs[0]], dtype=np.float32), None


def test_explain_order_splits_the_order_and_finds_the_driver():
    env = VN2Env(_spec(), _episode(), LOOKUP, demand_scale=10.0, rules=RULES)
    obs, _ = env.reset()
    zero = explain_order(env, ZeroResidual(), obs)
    assert zero["anchor_order"] == max(0, round(zero["target"] - zero["projected_stock"]))
    assert zero["rl_order"] == zero["anchor_order"] and zero["adjustment"] == 0
    assert (drivers(env, ZeroResidual(), obs)["effect_units"] == 0).all()

    policy = _OnHandPolicy()
    e = explain_order(env, policy, obs)
    assert e["rl_order"] == env.resolve_order(policy.predict(obs)[0])
    assert e["adjustment"] == e["rl_order"] - e["anchor_order"]
    effects = drivers(env, policy, obs)
    assert not set(effects["input"]) & UNITLESS_FEATURES  # round, pattern and class inputs are left out
    assert effects.iloc[0]["input"] == "on_hand"
    # +0.1 in on_hand moves f by 0.01, i.e. 0.01 of the residual step, in units.
    assert effects.iloc[0]["effect_units"] == pytest.approx(0.01 * e["step"], rel=1e-4)
    assert (effects.iloc[1:]["effect_units"] == 0).all()


def test_vs_anchor_by_group():
    eps = pd.DataFrame({"g": ["a", "a", "b"], "cost_eur": [1.0, 3.0, 2.0], "anchor_cost_eur": [2.0, 2.0, 4.0],
                        "units": [5.0, 5.0, 2.0], "anchor_units": [4.0, 4.0, 4.0]})
    out = vs_anchor(eps, "g").set_index("g")
    assert out.loc["a", "windows"] == 2 and out.loc["a", "cost_vs_anchor_pct"] == pytest.approx(0.0)
    assert out.loc["b", "cost_vs_anchor_pct"] == pytest.approx(-50.0) and out.loc["b", "delta_eur"] == pytest.approx(-2.0)
    assert out.loc["a", "units_vs_anchor_pct"] == pytest.approx(25.0)
    assert vs_anchor(eps.drop(columns=["units", "anchor_units"]), "g")["units_vs_anchor_pct"].isna().all()


class _Constant:
    """Policy that always takes the same action."""

    def __init__(self, action):
        self.action = np.asarray(action)

    def predict(self, obs, deterministic: bool = True):
        return self.action, None


def test_rollout_gate_orders_the_anchor_outside_the_gate():
    eps = {(1, 2, DATES[0]): _episode().assign(demand_class="Smooth")}
    data = (eps, LOOKUP, {(1, 2): 10.0}, RULES)
    anchor = rollout(ZeroResidual(), _spec(), *data)[0]["window_cost_eur"]
    more = _Constant([0.5])
    ungated = rollout(more, _spec(), *data)[0]["window_cost_eur"]
    assert ungated != pytest.approx(anchor)
    assert rollout(more, _spec(), *data, gate=("Smooth",))[0]["window_cost_eur"] == pytest.approx(ungated)
    assert rollout(more, _spec(), *data, gate=("Lumpy",))[0]["window_cost_eur"] == pytest.approx(anchor)
    # A raw policy outside the gate orders the anchor too, not its own zero action.
    raw = _spec(method="raw", state=default_state("raw", MEASURES))
    assert rollout(_Constant(1), raw, *data, gate=())[0]["window_cost_eur"] == pytest.approx(anchor)


def test_class_gate_keeps_the_classes_won_over_seeds():
    def eps(smooth_cost, lumpy_cost):
        return pd.DataFrame({
            "demand_class": ["Smooth", "Smooth", "Lumpy"], "cost_eur": [smooth_cost, 2.0, lumpy_cost],
            "anchor_cost_eur": [2.0, 2.0, 4.0], "units": [5.0, 4.0, 1.0], "anchor_units": [4.0, 4.0, 2.0],
            "holding_eur": 1.0, "shortage_eur": 1.0,
        })

    # Smooth: −1 and +2 over the two seeds (lost overall); Lumpy: −1 and −2 (won).
    seeds = [eps(1.0, 3.0), eps(4.0, 2.0)]
    assert class_gate(seeds) == ("Lumpy",)
    assert class_gate([s.drop(columns="demand_class") for s in seeds]) is None
    gated = gate_episodes(seeds[0], ("Lumpy",))
    assert gated["cost_eur"].tolist() == [2.0, 2.0, 3.0] and gated["units"].tolist() == [4.0, 4.0, 1.0]
    assert "holding_eur" not in gated


def test_anchor_baseline_matches_for_raw_specs():
    raw = _spec(method="raw", state=default_state("raw", MEASURES))
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
    assert spec.config_key() != replace(spec, agent="grpo").config_key()
    assert PolicySpec.from_dict(spec.to_dict()) == spec
    # Specs of the retired DQN still load with their old key (leaderboards keep them) but can't train.
    old = replace(spec, agent="dqn", buckets=[0.0, 1.0])
    assert PolicySpec.from_dict(old.to_dict()).config_key() == old.config_key() != spec.config_key()
    assert any("retired" in e for e in validate_spec(old, MEASURES))
