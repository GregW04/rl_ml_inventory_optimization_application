"""Anchor input resolution and parity with the original lost-sales anchor."""

from __future__ import annotations

import numpy as np

from src.forecasting.measures import Measure
from src.simulation.anchor import AnchorSpec, LostSalesAnchor, resolve_anchor
from src.simulation.vn2 import VN2Rules

RULES = VN2Rules()
BASE = [Measure("mean", 0), Measure("mean", 1)]


def test_quantile_at_critical_ratio_wins():
    spec, warn = resolve_anchor(BASE + [Measure("quantile", 2, 0.8333), Measure("mean", 2)], RULES)
    assert (spec.target, spec.target_ids, warn) == ("quantile", ("q83_h2",), [])


def test_poisson_then_nearest_quantile():
    spec, _ = resolve_anchor(BASE + [Measure("mean", 2)], RULES)
    assert spec.target == "poisson" and spec.target_ids == ("mean_h2",)
    spec, warn = resolve_anchor(BASE + [Measure("quantile", 2, 0.75), Measure("quantile", 2, 0.95)], RULES)
    assert spec.target_ids == ("q75_h2",) and warn


def test_levels_fall_back_to_the_median():
    ms = [Measure("quantile", 0, 0.5), Measure("quantile", 1, 0.9), Measure("quantile", 1, 0.4),
          Measure("quantile", 2, 0.8333)]
    spec, _ = resolve_anchor(ms, RULES)
    assert (spec.h0, spec.h1) == ("q50_h0", "q40_h1")


def test_missing_level_has_no_anchor():
    spec, warn = resolve_anchor([Measure("mean", 1), Measure("mean", 2)], RULES)
    assert spec is None and warn


def _old_anchor(on_hand, arrive_k, arrive_k1, fc):
    """The pre-measures anchor (mean_h0, mean_h1, q83_h2), copied verbatim in logic."""
    mean_h0, mean_h1, q83 = max(0.0, fc["mean_h0"]), max(0.0, fc["mean_h1"]), max(0.0, fc["q83_h2"])
    start_k = on_hand + arrive_k
    sales_k = min(start_k, mean_h0)
    end_k = max(0.0, start_k - sales_k)
    start_k1 = end_k + arrive_k1
    sales_k1 = min(start_k1, mean_h1)
    end_k1 = max(0.0, start_k1 - sales_k1)
    return max(0, int(round(q83 - end_k1)))


def test_parity_with_original_anchor():
    anchor = LostSalesAnchor(AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 0.8333), RULES)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        fc = {"mean_h0": rng.gamma(1.5, 3), "mean_h1": rng.gamma(1.5, 3), "q83_h2": rng.gamma(2, 4)}
        state = rng.integers(0, 15, size=3).astype(float)
        assert anchor.order_quantity(*state, fc) == _old_anchor(*state, fc)


def test_poisson_target():
    poisson = LostSalesAnchor(AnchorSpec("mean_h0", "mean_h1", "poisson", ("mean_h2",), 0.8333), RULES)
    assert poisson.target_level({"mean_h2": 4.0}) == 6.0  # Poisson(4): P(Y ≤ 5) = 0.785, P(Y ≤ 6) = 0.889
