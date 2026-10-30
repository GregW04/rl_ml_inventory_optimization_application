"""BatchVN2Env is an exact twin of VN2Env: same observations, orders and euro costs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from src.rl.batch_env import BatchVN2Env, build_episode_arrays
from src.rl.rollout import ZeroResidual, rollout
from src.rl.spec import (
    ANCHOR_FEATURES,
    CLASS_FEATURES,
    CONTEXT_FEATURES,
    INVENTORY_FEATURES,
    PolicySpec,
)
from src.simulation.anchor import AnchorSpec
from src.simulation.vn2 import VN2Rules
from src.simulation.vn2_env import VN2Env
from src.utils.config import get_project_root, load_config

RULES = VN2Rules()
MEASURES = ["mean_h0", "mean_h1", "q50_h2", "q83_h2"]
ANCHOR = AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 0.8333)
STATE = [*INVENTORY_FEATURES, *MEASURES, *ANCHOR_FEATURES, *CONTEXT_FEATURES, *CLASS_FEATURES]


def _synthetic(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    episodes, lookup, scales = {}, {}, {}
    for i in range(n):
        store, product = i % 5, i
        dates = pd.date_range("2023-01-02", periods=8, freq="W-MON") + pd.Timedelta(weeks=i % 3)
        level = rng.gamma(1.5, 4.0)
        sales = rng.poisson(level, 8).astype(float)
        if i % 7 == 0:
            sales[3] = np.nan  # missing sales count as zero demand, as in the env
        ep = pd.DataFrame({
            "store": store, "product": product, "date": dates, "sales": sales, "in_stock": True,
            "zero_share": rng.uniform(0, 1, 8), "pos_cv": rng.uniform(0, 7, 8),
            "dp_sbc_class": rng.integers(-1, 4, 8), "demand_class": "Smooth",
        })
        ep["end_inventory"] = [float(rng.integers(0, 30))] + [np.nan] * 7
        ep["in_transit_1w"] = [float(rng.integers(0, 10))] + [np.nan] * 7
        ep["in_transit_2w"] = [float(rng.integers(0, 10))] + [np.nan] * 7
        episodes[(store, product, dates[0])] = ep
        scales[(store, product)] = float(rng.uniform(0.5, 40))
        for d in dates:
            m0, m1 = rng.gamma(1.5, level / 1.5 + 0.1), rng.gamma(1.5, level / 1.5 + 0.1)
            q50 = rng.gamma(1.5, level / 1.5 + 0.1)
            lookup[(store, product, d)] = {"mean_h0": m0, "mean_h1": m1, "q50_h2": q50,
                                           "q83_h2": q50 * rng.uniform(1.0, 2.0)}
    return episodes, lookup, scales


def _spec(method: str) -> PolicySpec:
    spec = PolicySpec("run", method, "ppo", list(STATE), ANCHOR, floor_frac=0.25, order_cap_mult=1.5)
    return spec


def _run_vn2env(spec, episodes, lookup, scales, actions):
    obs_all, costs = [], []
    for i, (key, ep) in enumerate(episodes.items()):
        env = VN2Env(spec, ep, lookup, scales.get(key[:2], 1.0), RULES)
        obs, _ = env.reset()
        seq, info = [], {}
        for t in range(RULES.window_weeks):
            seq.append(obs)
            obs, _r, done, _tr, info = env.step(np.array([actions[i, t]], dtype=np.float32))
        obs_all.append(np.stack(seq))
        costs.append(info["scored_cost_eur"])
    return np.stack(obs_all), np.array(costs)


def _run_batch(spec, episodes, lookup, scales, actions):
    arrays = build_episode_arrays(episodes, lookup, spec, scales, RULES)
    env = BatchVN2Env(spec, arrays, RULES)
    obs = env.reset(torch.arange(len(arrays)))
    seq, cost, shadow = [], torch.zeros(len(arrays), dtype=torch.float64), torch.zeros(len(arrays), dtype=torch.float64)
    for t in range(RULES.window_weeks):
        seq.append(obs.numpy())
        a = torch.as_tensor(actions[:, t]) if t < RULES.n_order_rounds else None
        obs, c, sh = env.step(a)
        cost += c
        shadow += sh
    return np.stack(seq, axis=1), cost.numpy(), shadow.numpy()


@pytest.mark.parametrize("method", ["residual", "raw"])
def test_observations_and_costs_match_vn2env(method):
    episodes, lookup, scales = _synthetic()
    spec = _spec(method)
    actions = np.random.default_rng(1).uniform(-1.2, 1.2, (len(episodes), 8)).astype(np.float32)
    obs_a, cost_a = _run_vn2env(spec, episodes, lookup, scales, actions)
    obs_b, cost_b, _ = _run_batch(spec, episodes, lookup, scales, actions)
    assert np.array_equal(obs_a, obs_b)
    assert cost_b == pytest.approx(cost_a, abs=1e-9)


def test_shadow_anchor_equals_the_anchor_rollout():
    episodes, lookup, scales = _synthetic(seed=3)
    spec = _spec("residual")
    zeros = np.zeros((len(episodes), 8), dtype=np.float32)
    _, cost_b, shadow = _run_batch(spec, episodes, lookup, scales, zeros)
    _, per_ep = rollout(ZeroResidual(), spec, episodes, lookup, scales, RULES)
    assert shadow == pytest.approx(per_ep["cost_eur"].to_numpy(), abs=1e-9)
    assert cost_b == pytest.approx(shadow, abs=1e-12)  # residual 0 is the anchor


def test_poisson_target_anchor():
    episodes, lookup, scales = _synthetic(seed=5)
    for f in lookup.values():
        f["mean_h2"] = f["q50_h2"]
    spec = replace(_spec("residual"), anchor=AnchorSpec("mean_h0", "mean_h1", "poisson", ("mean_h2",), 0.8333))
    zeros = np.zeros((len(episodes), 8), dtype=np.float32)
    _, _, shadow = _run_batch(spec, episodes, lookup, scales, zeros)
    _, per_ep = rollout(ZeroResidual(), spec, episodes, lookup, scales, RULES)
    assert shadow == pytest.approx(per_ep["cost_eur"].to_numpy(), abs=1e-9)


@pytest.mark.skipif(
    not (get_project_root() / "artifacts" / "forecasts" / "forecast_setup.json").exists(),
    reason="needs a forecasting run on disk",
)
def test_real_validation_windows_match():
    from src.rl.spec import default_state
    from src.rl.train import block_episodes, load_run_data

    config = load_config()
    data = load_run_data(config, None)
    eps = block_episodes(config, data, "rl_val")
    eps = {k: eps[k] for k in list(eps)[:300]}
    spec = PolicySpec(str(data.run_id), "residual", "ppo",
                      default_state("residual", data.measure_ids) + list(CLASS_FEATURES), data.default_anchor)
    _, per_ep = rollout(ZeroResidual(), spec, eps, data.lookup, data.scales, data.rules)
    zeros = np.zeros((len(eps), 8), dtype=np.float32)
    _, _, shadow = _run_batch(spec, eps, data.lookup, data.scales, zeros)
    assert shadow == pytest.approx(per_ep["cost_eur"].to_numpy(), abs=1e-9)
    actions = np.random.default_rng(2).uniform(-1, 1, (len(eps), 8)).astype(np.float32)
    obs_a, cost_a = _run_vn2env(spec, eps, data.lookup, data.scales, actions)
    obs_b, cost_b, _ = _run_batch(spec, eps, data.lookup, data.scales, actions)
    assert np.array_equal(obs_a, obs_b)
    assert cost_b == pytest.approx(cost_a, abs=1e-9)
