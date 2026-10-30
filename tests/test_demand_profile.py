"""Demand-profile features are point-in-time, skip masked weeks, and classify demand pattern / volume tier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.features.demand_profile import add_demand_profile_features

CFG = {"forecasting": {"features": {"demand_profile": {"enabled": True, "window_weeks": 52, "min_periods": 13}}}}
DP_COLS = ["dp_adi", "dp_cv2", "dp_zero_share", "dp_level", "dp_volume_pct", "dp_weeks_since_sale",
           "dp_sbc_class", "dp_volume_tier"]


def _panel(series: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for product, sales in series.items():
        for t, y in enumerate(sales):
            rows.append({"store": 0, "product": product,
                         "date": pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=t), "sales": y})
    return pd.DataFrame(rows)


def _last(out: pd.DataFrame, product: int) -> pd.Series:
    return out[out["product"] == product].sort_values("date").iloc[-1]


def test_features_use_only_earlier_weeks():
    rng = np.random.default_rng(0)
    base = {p: rng.poisson(1.0 + 3 * p, size=80).astype(float) for p in range(4)}
    base[1][rng.choice(80, 10, replace=False)] = np.nan
    cut = pd.Timestamp("2022-01-03") + pd.Timedelta(weeks=50)
    before = add_demand_profile_features(_panel({p: list(s) for p, s in base.items()}), config=CFG)
    changed = {p: s.copy() for p, s in base.items()}
    for s in changed.values():
        s[50:] = rng.poisson(40.0, size=30)
    after = add_demand_profile_features(_panel({p: list(s) for p, s in changed.items()}), config=CFG)
    keep = before["date"] <= cut
    pd.testing.assert_frame_equal(before.loc[keep, DP_COLS], after.loc[keep, DP_COLS])
    assert not before.loc[~keep, "dp_level"].equals(after.loc[~keep, "dp_level"])


def test_sbc_classes():
    n = 60
    out = add_demand_profile_features(_panel({
        0: [5.0] * n,                                                      # smooth
        1: [1.0 if t % 2 else 20.0 for t in range(n)],                     # erratic
        2: [3.0 if t % 3 == 0 else 0.0 for t in range(n)],                 # intermittent
        3: [(1.0 if t % 2 else 20.0) if t % 3 == 0 else 0.0 for t in range(n)],  # lumpy
    }), config=CFG)
    assert [int(_last(out, p)["dp_sbc_class"]) for p in range(4)] == [0, 1, 2, 3]
    assert _last(out, 2)["dp_adi"] == pytest.approx(3.0, abs=0.1)
    assert _last(out, 0)["dp_cv2"] == pytest.approx(0.0)


def test_masked_weeks_are_not_zero_demand():
    out = add_demand_profile_features(_panel({0: [np.nan if t % 2 else 5.0 for t in range(60)]}), config=CFG)
    row = _last(out, 0)
    assert row["dp_adi"] == pytest.approx(1.0)
    assert row["dp_zero_share"] == pytest.approx(0.0)
    assert row["dp_level"] == pytest.approx(5.0)


def test_short_history_is_unclassified():
    out = add_demand_profile_features(_panel({0: [2.0] * 30}), config=CFG).sort_values("date")
    early = out.iloc[:13]
    assert (early["dp_sbc_class"] == -1).all() and (early["dp_volume_tier"] == -1).all()
    assert early["dp_level"].isna().all()
    assert out.iloc[13]["dp_sbc_class"] == 0


def test_volume_tiers_by_share_of_the_date_total():
    out = add_demand_profile_features(_panel({0: [80.0] * 20, 1: [15.0] * 20, 2: [5.0] * 20}), config=CFG)
    assert [int(_last(out, p)["dp_volume_tier"]) for p in range(3)] == [0, 1, 2]
    assert _last(out, 0)["dp_volume_pct"] == pytest.approx(1.0)


def test_weeks_since_last_sale():
    out = add_demand_profile_features(_panel({0: [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]}), config=CFG).sort_values("date")
    weeks = out["dp_weeks_since_sale"].tolist()
    assert np.isnan(weeks[2]) and weeks[3:] == [1.0, 2.0, 3.0]
