"""Date and global holiday features for weekly demand forecasting."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _black_friday(year: int) -> date:
    """US Black Friday: Friday after 4th Thursday of November."""
    nov1 = date(year, 11, 1)
    # weekday: Mon=0 ... Sun=6; Thanksgiving = 4th Thursday
    first_thu_offset = (3 - nov1.weekday()) % 7
    thanksgiving = nov1.replace(day=1 + first_thu_offset + 21)
    from datetime import timedelta

    return thanksgiving + timedelta(days=1)


def _holiday_dates_for_year(year: int) -> dict[str, date]:
    """Global-ish retail holidays mapped to a single anchor date per year."""
    return {
        "new_year": date(year, 1, 1),
        "easter": _easter_sunday(year),
        "halloween": date(year, 10, 31),
        "black_friday": _black_friday(year),
        "christmas": date(year, 12, 25),
        "summer_peak": date(year, 7, 15),
    }


def _week_start(d: date) -> pd.Timestamp:
    """ISO week start (Monday) for a calendar date."""
    ts = pd.Timestamp(d)
    return ts - pd.Timedelta(days=ts.weekday())


def _build_holiday_calendar(
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
    events: list[str],
) -> pd.DataFrame:
    """One row per (week_start, event) with holiday week flag."""
    years = range(min_date.year - 1, max_date.year + 2)
    rows = []
    for year in years:
        holidays = _holiday_dates_for_year(year)
        for event in events:
            if event not in holidays:
                continue
            ws = _week_start(holidays[event])
            rows.append({"date": ws, "event": event, "is_holiday_week": 1})

    if not rows:
        return pd.DataFrame(columns=["date", "event", "is_holiday_week"])

    cal = pd.DataFrame(rows).drop_duplicates(subset=["date", "event"])
    return cal


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Add date features and global holiday proximity features.

    Expects `date` as weekly period start (Monday).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    feat_cfg = (config or {}).get("forecasting", {}).get("features", {})
    cal_cfg = feat_cfg.get("calendar", {})
    hol_cfg = feat_cfg.get("holidays", {})

    if cal_cfg.get("enabled", True):
        dt = out[date_col]
        out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        out["month"] = dt.dt.month
        out["quarter"] = dt.dt.quarter
        out["year"] = dt.dt.year
        out["week_of_month"] = ((dt.dt.day - 1) // 7 + 1).astype(int)
        out["is_year_start"] = (out["week_of_year"] <= 2).astype(int)
        out["is_year_end"] = (out["week_of_year"] >= 52).astype(int)

    if cal_cfg.get("enabled", True) and cal_cfg.get("cyclical_encoding", True):
        # Harmonic 1 keeps the original column names. Higher harmonics can
        # represent a sharp seasonal peak that a single sine cannot.
        week = out["week_of_year"].to_numpy(dtype=float)
        n_harmonics = int(cal_cfg.get("fourier_harmonics", 3))
        for k_harm in range(1, max(n_harmonics, 1) + 1):
            suffix = "" if k_harm == 1 else f"_{k_harm}"
            angle = 2 * np.pi * k_harm * week / 52
            out[f"week_of_year_sin{suffix}"] = np.sin(angle)
            out[f"week_of_year_cos{suffix}"] = np.cos(angle)
        month = out["month"].to_numpy(dtype=float)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)

    if not hol_cfg.get("enabled", True):
        return out

    events = hol_cfg.get(
        "events",
        ["new_year", "easter", "halloween", "black_friday", "christmas", "summer_peak"],
    )
    weeks_until_max = hol_cfg.get("weeks_until_max", 4)
    weeks_since_max = hol_cfg.get("weeks_since_max", 2)

    min_d, max_d = out[date_col].min(), out[date_col].max()
    hol_cal = _build_holiday_calendar(min_d, max_d, events)

    unique_dates = pd.DataFrame({date_col: sorted(out[date_col].unique())})

    for event in events:
        event_weeks = hol_cal.loc[hol_cal["event"] == event, ["date", "is_holiday_week"]]
        event_weeks = event_weeks.rename(columns={"date": date_col})

        merged = unique_dates.merge(event_weeks, on=date_col, how="left")
        merged["is_holiday_week"] = merged["is_holiday_week"].fillna(0).astype(int)

        # weeks until next holiday week
        hol_idx = merged.index[merged["is_holiday_week"] == 1].tolist()
        until = np.full(len(merged), weeks_until_max + 1)
        for i in range(len(merged)):
            future = [h for h in hol_idx if h >= i]
            if future:
                until[i] = min(future[0] - i, weeks_until_max + 1)
        merged[f"{event}_weeks_until"] = np.clip(until, 0, weeks_until_max)
        merged[f"{event}_is_holiday"] = merged["is_holiday_week"]

        # weeks since last holiday week
        since = np.full(len(merged), weeks_since_max + 1)
        for i in range(len(merged)):
            past = [h for h in hol_idx if h <= i]
            if past:
                since[i] = min(i - past[-1], weeks_since_max + 1)
        merged[f"{event}_weeks_since"] = np.clip(since, 0, weeks_since_max)

        drop_cols = ["is_holiday_week"]
        feat_cols = [c for c in merged.columns if c not in [date_col] + drop_cols]
        out = out.merge(merged[[date_col] + feat_cols], on=date_col, how="left")

    return out
