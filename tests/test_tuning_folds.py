"""The folds a tuning scores: the run's own CV settings, shared by the first tuning and every retune."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.tuning import expected_tuning_folds, tuning_folds

WEEKS = pd.date_range("2023-01-02", periods=60, freq="W-MON")


def _config(n_folds, fold_weeks, min_train, n_cv_folds):
    return {"forecasting": {"validation": {"n_folds": n_folds, "fold_size_weeks": fold_weeks, "min_train_size_weeks": min_train},
                            "tuning": {"n_cv_folds": n_cv_folds}}}


def test_the_folds_follow_the_runs_settings_and_end_at_the_last_week():
    folds = tuning_folds(pd.DataFrame({"date": WEEKS}), _config(3, 5, 10, 3))
    assert [len(f.val_dates) for f in folds] == [5, 5, 5]
    assert folds[-1].val_end == WEEKS[-1] and folds[1].val_end == WEEKS[-6]
    assert (folds[0].val_start - folds[0].train_end).days // 7 == 3  # label gap of 2 weeks, then the fold


def test_the_search_scores_the_newest_n_cv_folds_of_the_runs_folds():
    config = _config(4, 5, 10, 2)
    folds = tuning_folds(pd.DataFrame({"date": WEEKS}), config)
    assert len(folds) == 2 and folds[-1].val_end == WEEKS[-1] and expected_tuning_folds(config) == 2
    assert expected_tuning_folds(_config(3, 5, 10, 5)) == 3  # never more than the run has


def test_a_frame_that_holds_no_fold_is_an_error():
    with pytest.raises(RuntimeError):
        tuning_folds(pd.DataFrame({"date": WEEKS[:20]}), _config(5, 13, 52, 5))
