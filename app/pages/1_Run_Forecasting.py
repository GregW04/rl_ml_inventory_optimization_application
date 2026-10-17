"""Set periods and features, preview how the data will be split, launch forecasting."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import pandas as pd
import streamlit as st

from app import lib
from src.forecasting.periods import (
    current_setup_hash,
    load_setup,
    lookback_breakdown,
    panel_weeks,
    periods_to_dict,
    timeline,
)

st.set_page_config(page_title="Run Forecasting", page_icon="🚀", layout="wide")
st.title("Run Forecasting")
st.caption(
    "Set periods and features, preview how the data will be used, then launch a forecasting "
    "run. Train RL on a finished run from **Run RL**."
)

base_cfg = lib.get_config()
latest_setup = load_setup(base_cfg)


def _form_defaults(record: dict | None) -> dict:
    """Form values from a saved setup record, falling back to config.yaml."""
    fc = base_cfg["forecasting"]
    record = record or {}
    return {
        "periods": record.get("periods_input") or base_cfg.get("periods") or {},
        "forecasting": {
            "forecaster_type": fc.get("forecaster_type", "quantile"),
            "validation": dict(fc["validation"]),
            "model_selection": {"enabled": fc.get("model_selection", {}).get("enabled", False)},
            "tuning": {
                "enabled": fc.get("tuning", {}).get("enabled", False),
                "n_trials": int(fc.get("tuning", {}).get("n_trials", 50)),
            },
            "features": record.get("features") or fc.get("features", {}),
        },
    }


# Widget keys carry a form version so Reset recreates every input. The form dict is
# rewritten each run, so values survive a visit to another page.
# Rev 2 is the horizon-only embargo and seasonal lags 51–53. An open session
# still holding the previous form is moved onto the config.yaml defaults once.
_FORM_REV = 2
if "fc_form" not in st.session_state or st.session_state.get("fc_form_rev") != _FORM_REV:
    migrating = "fc_form" in st.session_state
    st.session_state.fc_form = _form_defaults(None)
    st.session_state.fc_form_v = int(st.session_state.get("fc_form_v", 0)) + int(migrating)
    st.session_state.fc_form_rev = _FORM_REV
form = st.session_state.fc_form
v = st.session_state.fc_form_v


def k(name: str) -> str:
    return f"{name}_{v}"


weeks = panel_weeks(base_cfg)
week_labels = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks]


def _week_index(value, fallback: int) -> int:
    label = str(value or "")[:10]
    return week_labels.index(label) if label in week_labels else fallback


pin = form["periods"]
fc_form = form["forecasting"]
feat = fc_form["features"]
lag_cfg = feat.get("lags", {})
cal_cfg = feat.get("calendar", {})
hol_cfg = feat.get("holidays", {})
hier_cfg = feat.get("hierarchical", {})

left, right = st.columns([1, 1.2])

with left:
    tab_p, tab_f, tab_m = st.tabs(["Periods", "Features", "Model"])

    with tab_p:
        dr = st.select_slider(
            "Data range (week starts)",
            options=week_labels,
            value=(
                week_labels[_week_index(pin.get("data_start"), 0)],
                week_labels[_week_index(pin.get("end"), len(week_labels) - 1)],
            ),
            help="Inclusive range of sales weeks loaded for features and RL.",
            key=k("data_range"),
        )
        data_start = None if dr[0] == week_labels[0] else dr[0]
        data_end = dr[1]
        c1, c2, c3 = st.columns(3)
        rl_train_w = c1.number_input(
            "RL train weeks", 4, 80, int(pin.get("rl_train_weeks", 26)), key=k("rl_train_w")
        )
        rl_val_w = c2.number_input(
            "RL validation weeks", 4, 40, int(pin.get("rl_val_weeks", 8)), key=k("rl_val_w")
        )
        rl_test_w = c3.number_input(
            "RL test weeks", 4, 40, int(pin.get("rl_test_weeks", 9)), key=k("rl_test_w")
        )
        refit_per_block = st.toggle(
            "Refit forecaster per RL block",
            value=bool(pin.get("refit_per_block", True)),
            key=k("refit"),
        )
        with st.expander("Forecast cross-validation"):
            val_cfg = fc_form["validation"]
            n_folds = st.number_input(
                "CV folds", 1, 8, int(val_cfg.get("n_folds", 5)), key=k("n_folds")
            )
            fold_size = st.number_input(
                "Fold size (weeks)", 4, 26, int(val_cfg.get("fold_size_weeks", 13)),
                key=k("fold_size"),
            )
            min_train = st.number_input(
                "Min train weeks per fold",
                12,
                104,
                int(val_cfg.get("min_train_size_weeks", 52)),
                key=k("min_train"),
            )

    with tab_f:
        cal_on = st.toggle("Calendar", value=cal_cfg.get("enabled", True), key=k("cal"))
        if cal_on:
            n_harm = int(cal_cfg.get("fourier_harmonics", 3))
            st.caption(
                "Week of year is a model category. Cyclical encoding adds Fourier "
                f"harmonics 1–{n_harm} (sine and cosine) so a sharp seasonal peak "
                "is not forced onto a single sine wave."
            )
        hol_on = st.toggle("Holidays", value=hol_cfg.get("enabled", True), key=k("hol"))
        lags_on = st.toggle("Lags", value=lag_cfg.get("enabled", True), key=k("lags"))
        roll_on = st.toggle(
            "Rolling stats", value=bool(lag_cfg.get("rolling_windows")), key=k("roll")
        )
        exp_on = st.toggle(
            "Expanding stats", value=bool(lag_cfg.get("expanding_aggs")), key=k("exp")
        )
        mom_on = st.toggle("Momentum", value=lag_cfg.get("momentum", True), key=k("mom"))
        hier_on = st.toggle("Hierarchy", value=hier_cfg.get("enabled", True), key=k("hier"))

        lag_options = [1, 2, 4, 8, 13, 26, 51, 52, 53]
        lag_weeks = st.multiselect(
            "Lag weeks",
            lag_options,
            default=[x for x in (lag_cfg.get("lags") or lag_options) if x in lag_options],
            disabled=not lags_on,
            key=k("lag_weeks"),
            help=(
                "51–53 are the same weeks last year. They look backward only and "
                "do not widen the embargo or drop validation folds."
            ),
        )
        roll_options = [4, 8, 13, 26, 52]
        roll_windows = st.multiselect(
            "Rolling windows",
            roll_options,
            default=[w for w in (lag_cfg.get("rolling_windows") or roll_options) if w in roll_options],
            disabled=not roll_on,
            key=k("roll_windows"),
        )
        roll_aggs = st.multiselect(
            "Rolling aggregations",
            ["mean", "std", "min", "max", "median"],
            default=lag_cfg.get("rolling_aggs") or ["mean", "std", "min", "max", "median"],
            disabled=not roll_on,
            key=k("roll_aggs"),
        )
        exp_aggs = st.multiselect(
            "Expanding aggregations",
            ["mean", "std", "min", "max"],
            default=lag_cfg.get("expanding_aggs") or ["mean", "std", "min", "max"],
            disabled=not exp_on,
            key=k("exp_aggs"),
        )
        all_events = ["new_year", "easter", "halloween", "black_friday", "christmas", "summer_peak"]
        events = st.multiselect(
            "Holiday events",
            all_events,
            default=hol_cfg.get("events") or all_events,
            disabled=not hol_on,
            key=k("events"),
        )
        group_lags = st.multiselect(
            "Group lag weeks",
            [1, 2, 4, 8, 26, 51, 52, 53],
            default=hier_cfg.get("group_lags") or [1, 2, 4, 8, 26, 52],
            disabled=not hier_on,
            key=k("group_lags"),
        )
        group_roll = st.multiselect(
            "Group rolling windows",
            [4, 13, 52],
            default=hier_cfg.get("group_rolling_windows") or [4, 13, 52],
            disabled=not hier_on,
            key=k("group_roll"),
        )

    with tab_m:
        forecaster_type = st.selectbox(
            "Forecaster type",
            ["quantile", "mean"],
            index=0 if fc_form.get("forecaster_type", "quantile") == "quantile" else 1,
            key=k("forecaster_type"),
        )
        if forecaster_type == "mean":
            st.warning("RL expects quantile forecasts (Q10/Q50/Q90). Prefer quantile.")
        enable_selection = st.toggle(
            "Model selection",
            value=bool(fc_form["model_selection"].get("enabled", False)),
            key=k("selection"),
        )
        enable_tuning = st.toggle(
            "Hyperparameter tuning",
            value=bool(fc_form["tuning"].get("enabled", False)),
            key=k("tuning"),
        )
        n_trials = st.slider(
            "Optuna trials (if tuning)",
            5,
            100,
            int(fc_form["tuning"].get("n_trials", 50)),
            step=5,
            key=k("n_trials"),
        )

overrides = {
    "periods": {
        "data_start": data_start,
        "end": data_end,
        "rl_train_weeks": int(rl_train_w),
        "rl_val_weeks": int(rl_val_w),
        "rl_test_weeks": int(rl_test_w),
        "refit_per_block": refit_per_block,
    },
    "forecasting": {
        "forecaster_type": forecaster_type,
        "validation": {
            "n_folds": int(n_folds),
            "fold_size_weeks": int(fold_size),
            "min_train_size_weeks": int(min_train),
        },
        "model_selection": {"enabled": enable_selection},
        "tuning": {"enabled": enable_tuning, "n_trials": int(n_trials)},
        "features": {
            "calendar": {
                "enabled": cal_on,
                "cyclical_encoding": True,
                "fourier_harmonics": int(cal_cfg.get("fourier_harmonics", 3)),
            },
            "holidays": {
                "enabled": hol_on,
                "events": events if hol_on else [],
            },
            "lags": {
                "enabled": lags_on,
                "momentum": mom_on and lags_on,
                "lags": lag_weeks if lags_on else [],
                "rolling_windows": roll_windows if roll_on else [],
                "rolling_aggs": roll_aggs if roll_on else [],
                "expanding_aggs": exp_aggs if exp_on else [],
                "expanding_min_periods": 4,
            },
            "hierarchical": {
                "enabled": hier_on,
                "group_levels": hier_cfg.get("group_levels")
                or base_cfg["forecasting"]["features"]["hierarchical"]["group_levels"],
                "group_lags": group_lags if hier_on else [],
                "group_rolling_windows": group_roll if hier_on else [],
                "add_share_within_group": hier_cfg.get("add_share_within_group", True),
            },
        },
    },
}
st.session_state.fc_form = overrides

merged = lib.merged_config(overrides)
periods = lib.resolve_from_config_dict(merged)

with right:
    st.subheader("How data will be used")
    parts = lookback_breakdown(merged)
    detail = ", ".join(f"{name} {n}w" for name, n in parts.items()) or "no lag or window features"
    st.caption(
        f"Embargo before each scored block: **{periods.gap_weeks}** weeks, the forecast "
        f"horizon. Labels reach that far forward, so those weeks stay out of the fit. "
        f"Lags and rolling windows ({detail}) use earlier sales only, so a 52-week lag "
        "does not shorten training or drop CV folds."
    )
    lib.render_split_view(timeline(periods), periods_to_dict(periods))
    for err in periods.errors:
        st.error(err)
    for warn in periods.warnings:
        st.warning(warn)
    if latest_setup and latest_setup.get("config_hash") == current_setup_hash(merged):
        st.success(
            f"The latest forecasting run (`{(latest_setup.get('run_id') or '')[:8]}`) "
            "used exactly these settings."
        )

st.divider()
fc_job = lib.job_state("forecast")
fc_running = bool(fc_job and fc_job["running"])
b1, b2, _ = st.columns([1, 1, 2])
run_fc = b1.button(
    "Run forecasting",
    type="primary",
    disabled=bool(periods.errors) or fc_running,
    width="stretch",
)
if b2.button("Reset to config.yaml defaults", width="stretch"):
    st.session_state.fc_form = _form_defaults(None)
    st.session_state.fc_form_v = v + 1
    st.rerun()

if run_fc:
    cfg_path = lib.build_override_config(overrides)
    lib.start_job(
        "forecast",
        "Forecasting",
        [["-m", "src.forecasting.pipeline", "--config", cfg_path, "--force"]],
    )
    st.rerun()

if not fc_running and lib.forecasts_in_progress():
    st.info("A forecasting run started outside this page is still running.")
job = lib.job_panel("forecast")
if job and job["exit_code"] == 0:
    st.page_link("pages/2_Run_RL.py", label="Train RL on this run", icon="➡️")
