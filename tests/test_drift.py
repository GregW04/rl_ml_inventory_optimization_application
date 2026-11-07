"""Forecast drift diagnostics: per-fit bias, block-end rule, hidden test block, tracking signal, alerts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.drift import (
    TRIGG_LIMIT,
    block_bias,
    fit_bias,
    run_fits,
    trigg_signal,
    weekly_bias,
)
from src.forecasting.measures import Measure

DATES = pd.date_range("2022-01-03", periods=90, freq="W-MON")
TRAIN = (DATES[60], DATES[73])   # two 7-week fits
VAL = (DATES[74], DATES[81])
TEST = (DATES[82], DATES[89])
MEAN0, MEAN1, Q83 = Measure("mean", 0), Measure("mean", 1), Measure("quantile", 2, 5 / 6)


def _resolved(with_fits: bool = True) -> dict:
    d = {f"{b}_{k}": str(v.date()) for b, (s, e) in (("rl_train", TRAIN), ("rl_val", VAL), ("rl_test", TEST))
         for k, v in (("start", s), ("end", e))}
    d.update({"fit_end_rl_train": str(DATES[57].date()), "fit_end_rl_val": str(DATES[71].date()),
              "fit_end_rl_test": str(DATES[79].date())})
    if with_fits:
        d["block_fits"] = [
            {"block": "rl_train", "index": 1, "start": str(DATES[60].date()), "end": str(DATES[66].date()),
             "fit_end": str(DATES[57].date())},
            {"block": "rl_train", "index": 2, "start": str(DATES[67].date()), "end": str(DATES[73].date()),
             "fit_end": str(DATES[64].date())},
            {"block": "rl_val", "index": 1, "start": d["rl_val_start"], "end": d["rl_val_end"],
             "fit_end": d["fit_end_rl_val"]},
            {"block": "rl_test", "index": 1, "start": d["rl_test_start"], "end": d["rl_test_end"],
             "fit_end": d["fit_end_rl_test"]},
        ]
    return d


def _data(factor: dict[str, float], test_sales: float = 10.0):
    """Two SKUs selling 10 a week (test weeks: ``test_sales``); forecasts = 10 × factor per block."""
    sales, fc = [], []
    for store in (0, 1):
        for d in DATES:
            block = ("rl_test" if d >= TEST[0] else "rl_val" if d >= VAL[0] else
                     "rl_train" if d >= TRAIN[0] else None)
            y = test_sales if block == "rl_test" else 10.0
            sales.append({"store": store, "product": 7, "date": d, "sales": y})
            if block:
                f = 10.0 * factor[block]
                fc.append({"store": store, "product": 7, "date": d, "sales": y,
                           "mean_h0": f, "mean_h1": f, "q83_h2": 12.0})
    return pd.DataFrame(fc), pd.DataFrame(sales)


def test_bias_sign_and_size_per_fit():
    fc, sales = _data({"rl_train": 0.9, "rl_val": 1.1, "rl_test": 1.0})
    t = fit_bias(fc, sales, _resolved(), [MEAN0, Q83])
    assert list(t["fit"]) == ["training 1/2", "training 2/2", "validation"]
    assert t["bias mean_h0 (%)"].tolist() == pytest.approx([10.0, 10.0, -10.0])   # too low, then too high
    assert t["coverage q83_h2"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert t["sales vs a year earlier (%)"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_test_block_hidden_unless_asked():
    fc, sales = _data({"rl_train": 1.0, "rl_val": 1.0, "rl_test": 1.0})
    assert "test" not in set(fit_bias(fc, sales, _resolved(), [MEAN0])["fit"])
    assert "test" in set(fit_bias(fc, sales, _resolved(), [MEAN0], include_test=True)["fit"])
    assert all(f["block"] != "rl_test" for f in run_fits(_resolved()))
    assert "test" not in block_bias(fc, sales, _resolved(), MEAN0)


def test_validation_targets_never_read_test_weeks():
    # Huge test-week sales would make the validation h1 bias hugely positive if they leaked in.
    fc, sales = _data({"rl_train": 1.0, "rl_val": 1.0, "rl_test": 1.0}, test_sales=1000.0)
    t = fit_bias(fc, sales, _resolved(), [MEAN1]).set_index("fit")
    assert t.loc["validation", "bias mean_h1 (%)"] == pytest.approx(0.0)


def test_records_without_fits_get_one_row_per_block():
    fc, sales = _data({"rl_train": 0.8, "rl_val": 1.0, "rl_test": 1.0})
    t = fit_bias(fc, sales, _resolved(with_fits=False), [MEAN0])
    assert list(t["fit"]) == ["training", "validation"]
    assert t["bias mean_h0 (%)"].tolist() == pytest.approx([20.0, 0.0])


def test_per_class_rows():
    fc, sales = _data({"rl_train": 0.9, "rl_val": 1.1, "rl_test": 1.0})
    classes = sales[["store", "product", "date"]].assign(
        demand_class=np.where(sales["store"] == 0, "Smooth", "Lumpy"))
    t = fit_bias(fc, sales, _resolved(), [MEAN0], classes=classes)
    assert set(t["demand class"]) == {"Smooth", "Lumpy"}
    assert len(t) == 6


def test_tracking_signal_flags_persistent_bias_not_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 10.0, 400)
    assert np.mean(np.abs(trigg_signal(noise)[4:]) > TRIGG_LIMIT) < 0.1
    shifted = rng.normal(0.0, 10.0, 30)
    shifted[10:] += 15.0                     # 1.5 standard deviations from week 10
    t = trigg_signal(shifted)
    assert np.any(t[10:] > TRIGG_LIMIT)


def test_weekly_bias():
    fc, sales = _data({"rl_train": 0.8, "rl_val": 0.8, "rl_test": 1.0})
    w = weekly_bias(fc, sales, _resolved(), MEAN0)
    assert w["bias_pct"].round(6).eq(20.0).all()
    assert set(w["block"]) == {"training", "validation"}
    assert list(w.columns) == ["date", "block", "fit", "bias_pct"]


# ---------------------------------------------------------------------------
# Drift alerts: the production bias alarm on a run's fits
# ---------------------------------------------------------------------------

def _class_actuals(sales: pd.DataFrame) -> pd.DataFrame:
    """Store 0 Smooth, store 1 Lumpy (the same volume: no size filter), plus an Unclassified SKU."""
    extra = sales[sales["store"] == 0].assign(store=2)
    act = pd.concat([sales, extra], ignore_index=True)
    return act.assign(demand_class=act["store"].map({0: "Smooth", 1: "Lumpy", 2: "Unclassified"}))


def test_drift_alerts_restart_at_each_refit():
    from src.forecasting.drift_alerts import alert_weeks, earliest_alert_week, fit_alerts
    from src.ops.settings import Rules

    # 20% too low everywhere but the test block: (10 − 8) / 8 = +25% a week. From a restart the
    # signal is 1 − 0.8^k: past 0.74 from a fit's week 7, so an alert (2 weeks in a row) needs week 8.
    fc, sales = _data({"rl_train": 0.8, "rl_val": 0.8, "rl_test": 1.0})
    fc = pd.concat([fc, fc[fc["store"] == 0].assign(store=2)], ignore_index=True)
    rules = Rules()
    a = fit_alerts(fc, _class_actuals(sales), _resolved(), "mean_h0", rules)
    assert set(a["demand_class"]) == {"Smooth", "Lumpy"}  # Unclassified is left out
    assert a["error_pct"].round(6).isin([25.0, 0.0]).all()
    # Each 7-week training fit restarts before it can alert; a pooled signal would have alerted.
    assert alert_weeks(a, "rl_train") == 0 and a.loc[a["block"] == "rl_train", "streak"].max() == 1
    hits = a[a["alert"]]
    assert set(hits["block"]) == {"rl_val"} and set(hits["week_of_fit"]) == {8}
    assert set(hits["demand_class"]) == {"Smooth", "Lumpy"}
    assert hits["week_of_fit"].min() >= earliest_alert_week(rules)
    assert alert_weeks(a, "rl_val") == 1 and alert_weeks(a, "rl_test") == 0
    # The warm-up: no signal before a fit's 4th week.
    assert a.loc[a["week_of_fit"] < rules.bias.warmup_weeks, "tracking_signal"].isna().all()


def test_drift_alert_summary():
    from src.forecasting.drift_alerts import alert_summary, fit_alerts
    from src.ops.settings import Rules

    fc, sales = _data({"rl_train": 0.8, "rl_val": 0.8, "rl_test": 1.0})
    s = alert_summary(fit_alerts(fc, _class_actuals(sales), _resolved(), "mean_h0", Rules())).set_index(["block", "demand_class"])
    assert s.loc[("rl_val", "Smooth"), "alert_weeks"] == 1
    assert s.loc[("rl_val", "Smooth"), "first_alert"] == VAL[1]
    assert pd.isna(s.loc[("rl_train", "Smooth"), "first_alert"])


def test_first_alert_marks_keep_each_classs_first_alert_in_each_fit():
    from src.forecasting.drift_alerts import first_alert_marks

    alerts = pd.DataFrame({
        "block": "rl_train", "fit": ["training 1/2"] * 4 + ["training 2/2"] * 3,
        "week": pd.to_datetime(["2023-10-02", "2023-10-09", "2023-10-09", "2023-10-16", "2023-11-06", "2023-11-13", "2023-11-20"]),
        "demand_class": ["Lumpy", "Lumpy", "Smooth", "Lumpy", "Smooth", "Smooth", "Lumpy"],
        "alert": [True, True, True, True, True, True, False]})
    marks = first_alert_marks(alerts)
    # Lumpy first alerts on 10-02 and Smooth on 10-09 in fit 1; their later alert weeks are not marked; fit 2: Smooth only.
    assert marks[["fit", "week", "classes"]].values.tolist() == [
        ["training 1/2", pd.Timestamp("2023-10-02"), "Lumpy"], ["training 1/2", pd.Timestamp("2023-10-09"), "Smooth"],
        ["training 2/2", pd.Timestamp("2023-11-06"), "Smooth"]]
    both = first_alert_marks(alerts.assign(week=pd.Timestamp("2023-10-02"), fit="training 1/2").iloc[:3])
    assert both["classes"].tolist() == ["Lumpy, Smooth"]  # classes alerting first in the same week share a mark
    assert first_alert_marks(alerts.assign(alert=False)).empty and first_alert_marks(alerts.iloc[:0]).empty

