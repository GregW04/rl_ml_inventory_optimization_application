"""
The Streamlit app renders against an empty tracking store (the test's own, see conftest.py).

AppTest starts at ``app/Home.py`` and switches pages from there: a page opened on its own
fails at its ``st.page_link`` calls, which need the multipage context.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
HOME = str(Path(__file__).resolve().parents[1] / "app" / "Home.py")


def _home() -> AppTest:
    at = AppTest.from_file(HOME, default_timeout=180)
    at.run()
    assert not at.exception
    return at


def test_home_has_a_production_section():
    at = _home()
    assert "Production loop" in [s.value for s in at.subheader]
    assert any("hasn't been scored yet" in m.value for m in at.markdown)  # empty store: no live run


def _toggle(at, label):
    return next(t for t in at.toggle if t.label == label)


def test_run_forecasting_shows_the_refit_schedule_and_the_two_drift_toggles():
    at = _home()
    at.switch_page("pages/1_Run_Forecasting.py").run()
    assert not at.exception
    monitoring, refits = _toggle(at, "Drift monitoring (alerts)"), _toggle(at, "Drift-triggered refits")
    assert monitoring.value is True and not monitoring.disabled and refits.value is False  # configs/config.yaml
    assert any(c.value.startswith("**Refits:** 2023-08-21 training 1/2") for c in at.caption)
    assert not any("before an alert can fire" in w.value for w in at.warning) and not at.checkbox

    # Drift refits need the alerts: monitoring is on and locked, and the retune option appears (it needs tuning).
    refits.set_value(True).run()
    assert not at.exception
    monitoring = _toggle(at, "Drift monitoring (alerts)")
    assert monitoring.value is True and monitoring.disabled
    retune = next(c for c in at.checkbox if c.label.startswith("Retune if the same class alerts again"))
    assert retune.disabled  # hyperparameter tuning is off
    assert any(c.value.startswith("**Scheduled refits** (alerts add more") for c in at.caption)
    _toggle(at, "Hyperparameter tuning").set_value(True).run()
    assert not next(c for c in at.checkbox if c.label.startswith("Retune if")).disabled

    refits = _toggle(at, "Drift-triggered refits")
    refits.set_value(False).run()
    assert not at.checkbox and not _toggle(at, "Drift monitoring (alerts)").disabled

    cadence = next(n for n in at.number_input if n.label.startswith("Refit the forecaster"))
    cadence.set_value(3).run()
    assert any("before an alert can fire" in w.value for w in at.warning)


def test_forecasts_page_without_runs():
    at = _home()
    at.switch_page("pages/4_Forecasts.py").run()
    assert not at.exception
    assert any("No forecasting runs yet" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# The run timeline: folds, fits, alerts, refits, retunes (pure helpers, no running app)
# ---------------------------------------------------------------------------

def _timeline_inputs():
    import pandas as pd

    refits = pd.DataFrame({
        "date": pd.to_datetime(["2023-08-21", "2023-10-23"]), "block": ["training", "training"],
        "label": ["training 1/2", "training 2/2"], "data_through": pd.to_datetime(["2023-07-31", "2023-10-02"]),
        "trigger": ["start", "drift"], "classes": ["", "Lumpy"], "retuned": [False, True]})
    alerts = pd.DataFrame({"block": ["rl_train"], "fit": ["training 1/2"], "week": pd.to_datetime(["2023-10-16"]), "classes": ["Lumpy"]})
    retunes = pd.DataFrame([
        {"week": "2023-10-16", "block": "rl_train", "classes": ["Lumpy"], "current_score": 1.0, "new_score": 0.9, "gain_pct": 10.0,
         "adopted": True, "tuned_to": "2023-07-24", "holdout_start": "2023-08-14", "holdout_end": "2023-10-02", "folds": 5,
         "fold_weeks": 13, "skipped": False, "note": ""},
        {"week": "2023-12-11", "block": "rl_train", "classes": ["Smooth"], "current_score": None, "new_score": None,
         "gain_pct": None, "adopted": False, "tuned_to": "", "holdout_start": "2023-09-25", "holdout_end": "2023-12-04",
         "folds": 4, "fold_weeks": 13, "skipped": True, "note": "only 4 of the run's 5 CV folds fit in the rows before the holdout"}])
    return refits, alerts, retunes


def test_event_marks_order_retunes_under_refits_under_alerts_and_date_a_retune_at_its_refit():
    import pandas as pd

    from app import lib

    marks = lib.event_marks(*_timeline_inputs())
    assert marks["event"].tolist() == ["retune: adopted", "retune: skipped", "refit: block start", "refit: after alert", "alert"]
    by_event = marks.set_index("event")["date"]
    assert by_event["retune: adopted"] == pd.Timestamp("2023-10-23")  # the alert week plus one: the refit that followed
    assert by_event["alert"] == pd.Timestamp("2023-10-16") and by_event["retune: skipped"] == pd.Timestamp("2023-12-18")
    assert "skipped, only 4 of the run's 5 CV folds" in marks.set_index("event").loc["retune: skipped", "detail"]
    assert "alert: Lumpy" in marks.set_index("event").loc["refit: after alert", "detail"]
    assert lib.event_marks().empty and lib.event_marks(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()).empty


def test_a_retune_lane_shows_the_tuning_rows_the_label_gap_and_the_holdout_the_tuning_never_saw():
    from app import lib

    _, _, retunes = _timeline_inputs()
    seg = lib.retune_segments(retunes, {"data_start": "2021-04-12"})  # the skipped retune tuned nothing: no lane
    assert seg["lane"].unique().tolist() == ["Retune · 2023-10-16"]
    rows = {r.part: (str(r.start.date()), str(r.end.date()), r.weeks) for r in seg.itertuples()}
    assert rows["Tuning rows"][:2] == ("2021-04-12", "2023-07-24")
    assert rows["Label gap"] == ("2023-07-31", "2023-08-07", 2)
    assert rows["Holdout"] == ("2023-08-14", "2023-10-02", 8)
    assert lib.retune_segments(retunes.drop(columns=["tuned_to"]), {"data_start": "2021-04-12"}).empty  # saved before the rows were recorded


def test_the_retune_table_names_the_cv_settings_and_the_outcome():
    from app import lib

    _, _, retunes = _timeline_inputs()
    table = lib.retune_table(retunes).set_index("Alert week")
    assert table.loc["2023-10-16", "CV folds tuned on"] == "5 × 13 wk" and table.loc["2023-10-16", "Outcome"] == "new settings used"
    assert table.loc["2023-10-16", "Holdout"] == "2023-08-14 … 2023-10-02"
    assert table.loc["2023-12-11", "Outcome"].startswith("skipped: only 4 of the run's 5 CV folds")
    old = lib.retune_table(retunes[["week", "block", "classes", "current_score", "new_score", "gain_pct", "adopted"]].iloc[:1])
    assert old.loc[0, "CV folds tuned on"] == "—" and old.loc[0, "Outcome"] == "new settings used"  # records from before the CV details


def test_the_timeline_chart_draws_the_events_in_one_legend_and_zooms_to_the_scored_windows():
    import pandas as pd

    from app import lib

    refits, alerts, retunes = _timeline_inputs()
    segments = pd.DataFrame([
        {"lane": "CV fold 0", "part": "Validation", "start": "2023-02-06", "end": "2023-05-01", "weeks": 13},
        {"lane": "Fit · training 1/2", "part": "Fit", "start": "2021-04-12", "end": "2023-07-31", "weeks": 120},
        {"lane": "Fit · training 1/2", "part": "Forecasts written", "start": "2023-08-21", "end": "2023-10-16", "weeks": 9}])
    spec = lib.timeline_chart(segments, refits, alerts, retunes, zoom=True).to_dict()
    bars, rules = spec["layer"]
    assert bars["encoding"]["x"]["scale"]["domain"][0] == "2023-01-09T00:00:00"  # four weeks before the first validation window
    scale = rules["encoding"]["color"]["scale"]
    assert scale["domain"] == ["retune: adopted", "retune: skipped", "refit: block start", "refit: after alert", "alert"]
    assert rules["encoding"]["strokeDash"]["scale"]["domain"] == scale["domain"] == rules["encoding"]["strokeWidth"]["scale"]["domain"]
    unzoomed = lib.timeline_chart(segments, refits).to_dict()["layer"][0]["encoding"]["x"]
    assert "scale" not in unzoomed
    assert lib.timeline_chart(segments).to_dict()["encoding"]["x"]["field"] == "start"  # no events: a plain Gantt

