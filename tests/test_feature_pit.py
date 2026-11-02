"""
Point-in-time invariance of the feature matrix.

Training rows are built from a long panel; at serving time the panel ends at the decision
week. A feature whose value for week t depends on where the panel stops is train/serve skew:
the model learned one value and is served another. So for every feature family, week t's
features must be identical whether the panel ends at t or runs on for months.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.forecasting.model import build_feature_matrix, get_feature_columns
from src.forecasting.preprocessing import load_panel_from_config
from src.utils.config import load_config

# Weeks just before an event: Easter 2023 (week of Apr 3), the summer peak (Jul 10), the
# November promo (Nov 13) and Christmas Eve (Dec 18); plus a week right after Christmas.
CUTS = ["2023-03-20", "2023-06-26", "2023-11-06", "2023-12-11", "2024-01-01"]


@pytest.fixture(scope="module")
def setup():
    cfg = copy.deepcopy(load_config())
    cfg["forecasting"]["features"]["demand_profile"]["enabled"] = True  # every family
    panel = load_panel_from_config(cfg)
    skus = panel[["store", "product"]].drop_duplicates().head(40)
    panel = panel.merge(skus, on=["store", "product"])
    return cfg, panel, build_feature_matrix(panel, cfg)


def _rows(features: pd.DataFrame, week: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    rows = features[features["date"] == week].sort_values(["store", "product"])[cols].reset_index(drop=True)
    return rows.apply(lambda c: c.astype(object) if isinstance(c.dtype, pd.CategoricalDtype) else c)


@pytest.mark.parametrize("cut", CUTS)
def test_features_do_not_depend_on_where_the_panel_ends(setup, cut):
    cfg, panel, full = setup
    week = pd.Timestamp(cut)
    short = build_feature_matrix(panel[panel["date"] <= week], cfg)
    cols = get_feature_columns(full)
    pd.testing.assert_frame_equal(_rows(short, week, cols), _rows(full, week, cols), check_dtype=False)
