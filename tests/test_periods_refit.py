"""Refit cadence: block fits, embargo per fit, test-block rule, timeline lanes, forecast age."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.periods import (
    block_fit_counts,
    decision_ages,
    mean_decision_age,
    periods_to_dict,
    resolve_periods,
    timeline,
)

WEEKS = pd.date_range("2021-04-12", "2024-06-03", freq="W-MON").values


def _periods(refit: int, train: int = 52, val: int = 8, test: int = 8):
    cfg = {
        "periods": {"rl_train_weeks": train, "rl_val_weeks": val, "rl_test_weeks": test,
                    "refit_every_weeks": refit},
        "forecasting": {"validation": {"n_folds": 3, "fold_size_weeks": 13, "min_train_size_weeks": 52}},
        "vn2": {},
        "data": {"initial_state_date": "2024-04-08"},
    }
    return resolve_periods(cfg, WEEKS)


def _by_block(p, block):
    return [f for f in p.block_fits if f.block == block]


def test_thirteen_week_refits_cover_each_block_once():
    p = _periods(13)
    assert not p.errors
    assert [f.weeks for f in _by_block(p, "rl_train")] == [13, 13, 13, 13]
    assert len(_by_block(p, "rl_val")) == 1 and len(_by_block(p, "rl_test")) == 1
    for block in ("rl_train", "rl_val", "rl_test"):
        fits = _by_block(p, block)
        assert fits[0].start == getattr(p, f"{block}_start") and fits[-1].end == getattr(p, f"{block}_end")
        for a, b in zip(fits, fits[1:]):
            assert b.start - a.end == pd.Timedelta(weeks=1)  # contiguous, no overlap
    # first fit of each block is the fit the block used before refits existed
    assert _by_block(p, "rl_train")[0].fit_end == p.fit_end_rl_train
    assert _by_block(p, "rl_val")[0].fit_end == p.fit_end_rl_val
    assert _by_block(p, "rl_test")[0].fit_end == p.fit_end_rl_test


def test_each_fit_stops_three_weeks_before_its_forecasts():
    p = _periods(13)
    for f in p.block_fits:
        # rows up to fit_end; labels reach fit_end + 2 (the horizon), the week before start
        assert f.start - f.fit_end == pd.Timedelta(weeks=p.gap_weeks + 1)


def test_zero_means_one_fit_per_block():
    p = _periods(0)
    assert [(f.block, f.index) for f in p.block_fits] == [("rl_train", 1), ("rl_val", 1), ("rl_test", 1)]


def test_remainder_chunk_and_validation_refits_but_test_keeps_one_fit():
    p = _periods(10, val=16)
    assert [f.weeks for f in _by_block(p, "rl_train")] == [10, 10, 10, 10, 10, 2]
    assert [f.weeks for f in _by_block(p, "rl_val")] == [10, 6]
    assert len(_by_block(_periods(1), "rl_test")) == 1


def test_negative_cadence_is_an_error():
    assert any("Refit cadence" in e for e in _periods(-3).errors)


def test_many_fits_warn():
    assert any("forecaster fits" in w for w in _periods(2).warnings)


def test_timeline_has_one_lane_per_fit_with_a_two_week_gap():
    seg = timeline(_periods(13))
    lanes = [lane for lane in seg["lane"].unique() if lane.startswith("Fit · ")]
    assert lanes == ["Fit · training 1/4", "Fit · training 2/4", "Fit · training 3/4", "Fit · training 4/4",
                     "Fit · validation", "Fit · test"]
    for lane in lanes:
        parts = seg[seg["lane"] == lane].set_index("part")
        assert set(parts.index) == {"Fit", "Label gap", "Forecasts written"}
        assert int(parts.loc["Label gap", "weeks"]) == 2


def test_forecast_age_matches_the_analysis():
    today = mean_decision_age(decision_ages(_periods(0)))
    refit = mean_decision_age(decision_ages(_periods(13)))
    assert today["rl_train"] == pytest.approx(27.5, abs=0.05)
    assert refit["rl_train"] == pytest.approx(8.9, abs=0.05)
    assert today["rl_val"] == pytest.approx(5.5) and today["rl_test"] == pytest.approx(5.5)
    ages = decision_ages(_periods(13))
    train = ages[ages["block"] == "rl_train"]
    assert train["age_weeks"].min() == 3 and train["age_weeks"].max() == 15


def test_saved_record_and_fallback_counts():
    rec = periods_to_dict(_periods(13))
    assert rec["refit_every_weeks"] == 13 and len(rec["block_fits"]) == 6
    assert block_fit_counts(rec) == {"rl_train": 4, "rl_val": 1, "rl_test": 1}
    assert block_fit_counts({"refit_per_block": True}) == {"rl_train": 1, "rl_val": 1, "rl_test": 1}
