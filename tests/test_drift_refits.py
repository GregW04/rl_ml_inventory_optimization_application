"""Drift-triggered refits: the walk over a block, the age backstop, the retune escalation, the caps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.drift_alerts import first_alert
from src.forecasting.drift_refits import Retune, WalkState, walk_block
from src.ops.settings import Rules

WEEK = pd.Timedelta(weeks=1)
START, END = pd.Timestamp("2023-08-21"), pd.Timestamp("2024-02-12")  # a 26-week training block
ALL_WEEKS = pd.date_range("2023-01-02", "2024-03-04", freq="W-MON")
ONSET = pd.Timestamp("2023-10-09")
RULES = Rules()
LEVEL = "mean_h0"


def _actuals() -> pd.DataFrame:
    """Four SKUs selling 10 a week: stores 0 and 1 Smooth, 2 and 3 Lumpy, 4 Unclassified."""
    cls = {0: "Smooth", 1: "Smooth", 2: "Lumpy", 3: "Lumpy", 4: "Unclassified"}
    return pd.DataFrame([{"store": s, "product": 1, "date": d, "sales": 10.0, "demand_class": c}
                         for s, c in cls.items() for d in ALL_WEEKS])


ACTUALS = _actuals()


def stub_fit_predict(onset: pd.Timestamp, cure_after_weeks: float, calls: list | None = None):
    """
    Forecasts of 10 a week, 40% too low from ``onset`` until the model is cured: by a fit whose data end
    ``cure_after_weeks`` after the onset, or by adopted settings flagged ``cured``.
    """
    def fit_predict(fit_end, start, end, params):
        if calls is not None:
            calls.append((fit_end, start, end))
        cured = bool((params or {}).get("cured")) or fit_end >= onset + cure_after_weeks * WEEK
        rows = []
        for store in range(5):
            for d in pd.date_range(start, end, freq="W-MON"):
                rows.append({"store": store, "product": 1, "date": d, LEVEL: 10.0 * (0.6 if d >= onset and not cured else 1.0)})
        return pd.DataFrame(rows)
    return fit_predict


def alert_at(chunk: pd.DataFrame):
    return first_alert(chunk, ACTUALS, LEVEL, RULES)


def _shape(fits):
    return [(str(f.start.date()), str(f.end.date()), f.trigger, f.classes, f.retuned) for f in fits]


def test_first_alert_names_the_week_and_the_classes_and_skips_unclassified():
    fc = stub_fit_predict(ONSET, 999)(pd.Timestamp("2023-08-01"), START, END, None)
    week, classes = first_alert(fc, ACTUALS, LEVEL, RULES)
    # Zero error for seven weeks, then +67% a week: the signal is past the limit from the onset week and
    # the second week in a row is the alert.
    assert week == ONSET + WEEK and classes == ["Lumpy", "Smooth"]
    assert first_alert(stub_fit_predict(END + 400 * WEEK, 999)(pd.Timestamp("2023-08-01"), START, END, None),
                       ACTUALS, LEVEL, RULES) == (None, [])
    assert first_alert(fc.iloc[:0], ACTUALS, LEVEL, RULES) == (None, [])


def test_an_alert_adds_a_refit_from_the_next_week_that_cures_it():
    calls: list = []
    fits, parts = walk_block("rl_train", START, END, 2, 13, stub_fit_predict(ONSET, -1, calls), alert_at, WalkState())
    assert _shape(fits) == [
        ("2023-08-21", "2023-10-16", "start", "", False),
        ("2023-10-23", "2024-01-15", "drift", "Lumpy,Smooth", False),  # 13 weeks: the age backstop restarts at the refit
        ("2024-01-22", "2024-02-12", "schedule", "", False),
    ]
    # Every fit stops 3 weeks before the first week it forecasts, like the fixed schedule.
    assert all(f.start - f.fit_end == pd.Timedelta(weeks=3) for f in fits)
    # The kept forecasts tile the block once: nothing after a cut survives.
    kept = pd.concat(parts)
    assert kept["date"].nunique() == 26 and not kept.duplicated(["store", "date"]).any()
    assert list(kept["fit_end"].unique()) == [f.fit_end for f in fits]
    # The cured fit forecasts the demand: no error left after the refit.
    assert (kept[kept["date"] > ONSET + WEEK][LEVEL] == 10.0).all()
    assert calls[0] == (START - 3 * WEEK, START, END)  # the first fit predicts to the end of the block, then is cut


def test_without_an_alert_the_walk_is_the_fixed_schedule():
    never = stub_fit_predict(END + 400 * WEEK, 999)
    fits, _ = walk_block("rl_train", START, END, 2, 13, never, alert_at, WalkState())
    assert _shape(fits) == [("2023-08-21", "2023-11-13", "start", "", False), ("2023-11-20", "2024-02-12", "schedule", "", False)]
    one, _ = walk_block("rl_val", START, END, 2, 0, never, alert_at, WalkState())  # cadence 0: one fit
    assert _shape(one) == [("2023-08-21", "2024-02-12", "start", "", False)]


def test_with_cadence_zero_only_alerts_refit():
    fits, _ = walk_block("rl_train", START, END, 2, 0, stub_fit_predict(ONSET, -1), alert_at, WalkState())
    assert _shape(fits) == [("2023-08-21", "2023-10-16", "start", "", False), ("2023-10-23", "2024-02-12", "drift", "Lumpy,Smooth", False)]


def test_an_alert_that_returns_after_a_refit_retunes_once_and_adopts_a_win():
    asked: list = []

    def retune(week, fit_end, classes, params):
        asked.append((week, fit_end, classes))
        return Retune(week, "rl_train", classes, 1.00, 0.95, 5.0, True), {"cured": True}

    state = WalkState()
    # The refit does not cure the drift (cure_after 999 weeks); only the adopted settings do.
    fits, _ = walk_block("rl_train", START, END, 2, 13, stub_fit_predict(ONSET, 999), alert_at, state, retune=retune)
    assert _shape(fits) == [
        ("2023-08-21", "2023-10-16", "start", "", False),
        ("2023-10-23", "2023-12-11", "drift", "Lumpy,Smooth", False),   # the same classes alert again
        ("2023-12-18", "2024-02-12", "drift", "Lumpy,Smooth", True),    # refit with the adopted settings
    ]
    # Asked at the alert week, with the data the refit will have: rows to 2023-11-27, whose labels reach that week.
    assert asked == [(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Lumpy", "Smooth"])]
    assert [r.adopted for r in state.retunes] == [True] and state.params == {"cured": True}


def test_a_lost_challenge_keeps_the_settings_and_the_cap_stops_a_second_retune():
    calls = []

    def retune(week, fit_end, classes, params):
        calls.append(week)
        return Retune(week, "rl_train", classes, 1.00, 1.00, 0.0, False), params

    state = WalkState(params={"depth": 6})
    fits, _ = walk_block("rl_train", START, END, 2, 13, stub_fit_predict(ONSET, 999), alert_at, state, retune=retune, max_retunes=1)
    assert len(calls) == 1 and [r.adopted for r in state.retunes] == [False] and state.params == {"depth": 6}
    assert [f.trigger for f in fits] == ["start", "drift", "drift", "drift"] and not any(f.retuned for f in fits)


def test_a_return_needs_the_same_class_and_the_memory_spans_blocks():
    state = WalkState()
    asked = []

    def retune(week, fit_end, classes, params):
        asked.append(week)
        return Retune(week, "rl_val", classes, 1.0, 1.0, 0.0, False), params

    # The first alert of a run is not a return, and the block's end leaves its classes in the memory.
    walk_block("rl_train", START, pd.Timestamp("2023-11-13"), 2, 0, stub_fit_predict(ONSET, 999), alert_at, state, retune=retune)
    assert state.last_alert_classes == {"Lumpy", "Smooth"} and not asked
    # A different class alerted last: the next block's first alert (Lumpy, Smooth) is not a return, its second is.
    state.last_alert_classes = {"Erratic"}
    walk_block("rl_val", START, END, 2, 0, stub_fit_predict(ONSET, 999), alert_at, state, retune=retune)
    assert asked == [pd.Timestamp("2023-12-11")]
    # The same classes alerted last: the first alert of the next block is already a return.
    state2, asked[:] = WalkState(last_alert_classes={"Smooth"}), []
    walk_block("rl_val", START, END, 2, 0, stub_fit_predict(ONSET, 999), alert_at, state2, retune=retune)
    assert asked[0] == ONSET + WEEK


def test_the_fit_cap_stops_alert_driven_cuts():
    fits, _ = walk_block("rl_train", START, END, 2, 0, stub_fit_predict(ONSET, 999), alert_at, WalkState(), max_fits=2)
    assert len(fits) == 2 and fits[-1].end == END


# ---------------------------------------------------------------------------
# The pipeline wiring (stub forecasters: nothing is trained)
# ---------------------------------------------------------------------------

def test_the_retune_challenge_adopts_only_a_clear_holdout_win(monkeypatch):
    from src.forecasting import pipeline
    from src.forecasting.measures import Measure, primary_metric_name

    seen = {"train_end": [], "val": [], "tuned_to": []}

    class Stub:
        calibration: dict = {}

        def __init__(self, params):
            self.params = params

        def fit(self, train, feature_cols=None):
            seen["train_end"].append(train["date"].max())

        def predict(self, val, feature_cols=None):
            seen["val"].append((val["date"].min(), val["date"].max()))
            return pd.DataFrame({"params": [self.params]})

    primary = Measure("quantile", 2, 5 / 6)
    metric = primary_metric_name(primary)
    scores = {"cur": 1.0, "new": 0.97}
    monkeypatch.setattr(pipeline, "make_forecaster", lambda config, name, params, measures: Stub(params))
    monkeypatch.setattr(pipeline, "_tune_all", lambda df, config, model_name: seen["tuned_to"].append(df["date"].max()) or "new")
    monkeypatch.setattr(pipeline, "evaluate_measures", lambda scored, ms: {metric: scores[scored["params"].iloc[0]]})
    retune = pipeline._retune_challenge(pd.DataFrame({"date": ALL_WEEKS}), run_config(), "catboost", [primary], primary, [], {}, 2)
    outcome, params = retune(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Smooth"], "cur")
    assert outcome.adopted and params == "new" and round(outcome.gain_pct, 1) == 3.0
    # The holdout is the last 8 weeks up to the new fit's data; training stops gap + 1 = 3 weeks before it.
    assert seen["val"][0] == (pd.Timestamp("2023-10-09"), pd.Timestamp("2023-11-27"))
    assert seen["train_end"][0] == pd.Timestamp("2023-09-18")
    # The tuning sees the training rows and nothing later, so none of its CV folds touches the holdout.
    assert seen["tuned_to"][0] == pd.Timestamp("2023-09-18") < pd.Timestamp("2023-10-09")
    scores["new"] = 0.995  # a 0.5% gain is under the 1% minimum: the current settings stay
    outcome, params = retune(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Smooth"], "cur")
    assert not outcome.adopted and params == "cur"


def run_config(n_folds=2, fold_weeks=4, min_train=8, n_cv_folds=2, **monitoring) -> dict:
    """A run's config as far as the retune reads it: its CV settings, its tuning settings, the monitoring keys."""
    return {"monitoring": {"retune_min_gain_pct": 1.0, **monitoring},
            "forecasting": {"validation": {"n_folds": n_folds, "fold_size_weeks": fold_weeks, "min_train_size_weeks": min_train},
                            "tuning": {"n_cv_folds": n_cv_folds}}}


def _retune_with_stubs(monkeypatch, config):
    from src.forecasting import pipeline
    from src.forecasting.measures import Measure, primary_metric_name

    tuned = []

    class Stub:
        calibration: dict = {}

        def __init__(self, params):
            self.params = params

        def fit(self, train, feature_cols=None):
            pass

        def predict(self, val, feature_cols=None):
            return pd.DataFrame({"params": [self.params]})

    primary = Measure("quantile", 2, 5 / 6)
    scores = {"cur": 1.0, "new": 0.9}
    monkeypatch.setattr(pipeline, "make_forecaster", lambda config, name, params, measures: Stub(params))
    monkeypatch.setattr(pipeline, "_tune_all", lambda df, cfg, model_name: tuned.append((df["date"].max(), cfg)) or "new")
    monkeypatch.setattr(pipeline, "evaluate_measures",
                        lambda scored, ms: {primary_metric_name(primary): scores[scored["params"].iloc[0]]})
    retune = pipeline._retune_challenge(pd.DataFrame({"date": ALL_WEEKS}), config, "catboost", [primary], primary, [], {}, 2)
    return retune, tuned


def test_the_retune_tunes_with_the_runs_own_cv_settings_and_records_them(monkeypatch):
    config = run_config(n_folds=3, fold_weeks=5, n_cv_folds=3)
    retune, tuned = _retune_with_stubs(monkeypatch, config)
    outcome, _ = retune(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Smooth"], "cur")
    assert tuned[0][1] is config  # the run's config itself reaches the tuning: its folds, its search settings
    assert (outcome.folds, outcome.fold_weeks) == (3, 5) and not outcome.skipped
    assert (outcome.tuned_to, outcome.holdout_start, outcome.holdout_end) == ("2023-09-18", "2023-10-09", "2023-11-27")
    assert outcome.to_dict()["folds"] == 3


def test_the_holdout_length_is_a_setting(monkeypatch):
    retune, tuned = _retune_with_stubs(monkeypatch, run_config(retune_holdout_weeks=13))
    outcome, _ = retune(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Smooth"], "cur")
    assert (outcome.holdout_start, outcome.holdout_end, outcome.tuned_to) == ("2023-09-04", "2023-11-27", "2023-08-14")
    assert tuned[0][0] == pd.Timestamp("2023-08-14")


def test_a_retune_that_cannot_use_the_runs_folds_is_skipped_not_run_on_fewer(monkeypatch):
    # 37 weeks lie before the holdout: 5 folds of 13 weeks (and 52 weeks of training) do not fit.
    retune, tuned = _retune_with_stubs(monkeypatch, run_config(n_folds=5, fold_weeks=13, min_train=52, n_cv_folds=5))
    outcome, params = retune(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"), ["Smooth"], "cur")
    assert outcome.skipped and not outcome.adopted and params == "cur" and not tuned
    assert outcome.folds == 0 and "of the run's 5 CV folds" in outcome.note
    assert outcome.current_score is None and outcome.to_dict()["skipped"] is True


def test_a_skipped_retune_does_not_use_up_the_cap():
    asked = []

    def retune(week, fit_end, classes, params):
        asked.append(week)
        skipped = len(asked) == 1
        return Retune(week, "", classes, None if skipped else 1.0, None if skipped else 1.0, None if skipped else 0.0,
                      False, skipped=skipped), params

    state = WalkState()
    walk_block("rl_train", START, END, 2, 13, stub_fit_predict(ONSET, 999), alert_at, state, retune=retune, max_retunes=1)
    assert len(asked) == 2 and [r.skipped for r in state.retunes] == [True, False]  # the second try ran: the first did not count


def _pipeline_fixture(monkeypatch, drift_refits: bool, tuning: bool = False):
    from src.forecasting import drift_alerts, pipeline
    from src.forecasting.measures import Measure
    from src.forecasting.periods import resolve_periods

    weeks = pd.date_range("2021-04-12", "2024-06-03", freq="W-MON")
    cfg = {"periods": {"rl_train_weeks": 26, "rl_val_weeks": 8, "rl_test_weeks": 8, "refit_every_weeks": 13},
           "forecasting": {"validation": {"n_folds": 3, "fold_size_weeks": 13, "min_train_size_weeks": 52},
                           "tuning": {"enabled": tuning}},
           "monitoring": {"drift_refits": drift_refits, "retune_on_return": True, "max_retunes": 1},
           "vn2": {}, "data": {"initial_state_date": "2024-04-08"}}
    periods = resolve_periods(cfg, weeks.values)
    panel = pd.DataFrame([{"store": s, "product": 1, "date": d, "sales": 10.0} for s in range(4) for d in weeks])
    df = panel.assign(target_h0=10.0, target_h1=10.0, target_h2=10.0, f1=1.0)
    fits_made = []

    class Stub:
        calibration: dict = {}

        def __init__(self, params):
            self.params, self.fit_end = params, None

        def fit(self, train, feature_cols=None):
            self.fit_end = train["date"].max()
            fits_made.append(self.fit_end)

        def predict(self, rows, feature_cols=None):
            # 40% too low from the onset until a fit has seen the new level or adopted settings.
            cured = bool((self.params or {}).get("cured")) or self.fit_end >= ONSET
            return rows.assign(mean_h0=np.where((rows["date"] >= ONSET) & (not cured), 6.0, 10.0), q83_h2=12.0)

    monkeypatch.setattr(pipeline, "make_forecaster", lambda config, name, params, measures: Stub(params))
    monkeypatch.setattr(pipeline, "get_feature_columns", lambda frame: ["f1"])
    monkeypatch.setattr(drift_alerts, "class_actuals", lambda p, c: p.assign(demand_class="Smooth"))
    measures = [Measure("mean", 0), Measure("quantile", 2, 5 / 6)]

    def run():
        return pipeline._fit_rl_block_forecasts(df, cfg, periods, measures, "catboost", None, {}, panel=panel, primary=measures[1])
    return run, periods, fits_made


def test_pipeline_fixed_schedule_is_unchanged_when_drift_refits_are_off(monkeypatch):
    run, periods, _ = _pipeline_fixture(monkeypatch, drift_refits=False)
    combined, _, forecaster, fits, retunes = run()
    assert fits == periods.block_fits and retunes == []
    assert combined["date"].nunique() == 42 and forecaster.fit_end == pd.Timestamp("2024-03-25")  # the test fit is the last


def test_pipeline_alert_adds_refits_in_train_and_validation_only_and_keeps_one_test_fit(monkeypatch):
    run, periods, fits_made = _pipeline_fixture(monkeypatch, drift_refits=True)
    combined, metrics, forecaster, fits, retunes = run()
    assert [(f.block, f.trigger) for f in fits] == [("rl_train", "start"), ("rl_train", "drift"), ("rl_train", "drift"),
                                                    ("rl_val", "start"), ("rl_test", "start")]
    assert [str(f.start.date()) for f in fits if f.block == "rl_train"] == ["2023-08-21", "2023-10-23", "2023-12-18"]
    assert len([f for f in fits if f.block == "rl_test"]) == 1 and retunes == []  # tuning is off: no retune
    assert combined["date"].nunique() == 42 and forecaster.fit_end == pd.Timestamp("2024-03-25")
    assert not combined.duplicated(["store", "date"]).any()
    assert len(fits_made) == 5  # three in training, one each in validation and test: a cut only discards forecasts
    assert metrics["oos_rl_train_bias_mean_h0"] != 0.0  # the biased first weeks are scored as they were forecast


def test_pipeline_adopted_retune_reaches_the_next_fit(monkeypatch):
    from src.forecasting import pipeline

    run, _, _ = _pipeline_fixture(monkeypatch, drift_refits=True, tuning=True)
    asked = []

    def retune(alert_week, fit_end, classes, params):
        asked.append((alert_week, fit_end))
        return Retune(alert_week, "", classes, 1.0, 0.9, 10.0, True), {"cured": True}

    monkeypatch.setattr(pipeline, "_retune_challenge", lambda *a, **k: retune)
    _, _, _, fits, retunes = run()
    assert asked == [(pd.Timestamp("2023-12-11"), pd.Timestamp("2023-11-27"))]
    assert [r.block for r in retunes] == ["rl_train"] and fits[2].retuned and fits[2].trigger == "drift"
