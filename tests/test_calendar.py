"""Holiday anchors match the event weeks seen in the VN2 data (2021–2023)."""

from __future__ import annotations

import pandas as pd

from src.forecasting.features.calendar import EVENTS, add_calendar_features


def _flags(event: str) -> list[str]:
    weeks = pd.date_range("2021-10-04", "2024-01-29", freq="W-MON")
    df = pd.DataFrame({"date": weeks, "store": 0, "product": 0})
    cfg = {"forecasting": {"features": {"calendar": {"enabled": True}, "holidays": {"events": list(EVENTS)}}}}
    out = add_calendar_features(df, config=cfg)
    return [str(d.date()) for d in out.loc[out[f"{event}_is_holiday"] == 1, "date"]]


def test_november_promo_is_second_monday_week():
    assert _flags("november_promo") == ["2021-11-08", "2022-11-14", "2023-11-13"]


def test_christmas_is_the_week_of_christmas_eve():
    # 2023: Dec 24 is a Sunday, so the peak week starts Dec 18, not Dec 25.
    assert _flags("christmas") == ["2021-12-20", "2022-12-19", "2023-12-18"]
