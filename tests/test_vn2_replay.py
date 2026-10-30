"""VN2 replay parity checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.vn2 import (
    VN2Rules,
    constant_order_policy,
    simulate_sku_window,
)
from src.utils.config import load_config


def test_organizer_single_product_example():
    """Worked example from the competition materials (12.8 EUR)."""
    rules = VN2Rules()
    demand = [5, 8, 4, 10, 11, 10, 7, 5]
    orders = [23, 4, 5, 1, 6, 15]
    res = simulate_sku_window(
        demand,
        on_hand=7,
        in_transit_w1=0,
        in_transit_w2=5,
        fixed_orders=orders + [0, 0],
        rules=rules,
    )
    assert abs(res.scored_eur - 12.8) < 0.05


def test_weeks_1_2_constant_cost():
    """Weeks 1–2 total 913.8 EUR for any policy (full panel)."""
    config = load_config()
    from src.simulation.vn2 import attach_official_initial_state, load_competition_weeks

    panel = attach_official_initial_state(load_competition_weeks(config), config)
    panel = panel[panel["date"] >= "2024-04-15"].copy()
    panel = panel.sort_values(["store", "product", "date"])

    from src.simulation.vn2 import simulate_panel_window

    for label, fn in [
        ("zero", constant_order_policy(0)),
        ("random", lambda r, s, c: int(np.random.randint(0, 20))),
    ]:
        agg = simulate_panel_window(panel, order_fn=fn)
        assert abs(agg["weeks_1_2_eur"] - 913.8) < 2.0, label
