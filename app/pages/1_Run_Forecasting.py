"""Set periods, features and forecast measures, preview the split, launch forecasting."""

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
from src.forecasting.features.calendar import EVENTS
from src.forecasting.forecaster import MEAN_OBJECTIVES
from src.forecasting.measures import (
    CRITICAL_RATIO,
    HORIZONS,
    default_primary,
    parse_measures,
    validate_measures,
)
from src.forecasting.periods import (
    current_setup_hash,
    load_setup,
    lookback_breakdown,
    panel_weeks,
    periods_to_dict,
    timeline,
)
from src.simulation.vn2 import VN2Rules

st.set_page_config(page_title="Run Forecasting", page_icon="🚀", layout="wide")
st.title("Run Forecasting")
st.caption(
    "Set periods, features and the measures to forecast per horizon, preview how the data "
    "will be used, then launch a forecasting run. Train RL on a finished run from **Run RL**."
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
            "measures": list(fc.get("measures") or []),
            "primary_measure": fc.get("primary_measure"),
            "calibrate_quantiles": bool(fc.get("calibrate_quantiles", False)),
            "model": {
                "name": fc.get("model", {}).get("name", "lightgbm"),
                "mean_objective": fc.get("model", {}).get("mean_objective", "poisson"),
                "tweedie_variance_power": float(fc.get("model", {}).get("tweedie_variance_power", 1.5)),
            },
            "validation": dict(fc["validation"]),
            "model_selection": {
                "enabled": fc.get("model_selection", {}).get("enabled", False),
                "candidates": list(fc.get("model_selection", {}).get("candidates", BACKENDS)),
            },
            "ensemble": {
                "enabled": bool((fc.get("ensemble") or {}).get("enabled", False)),
                "members": list((fc.get("ensemble") or {}).get("members") or ["catboost", "lightgbm"]),
            },
            "tuning": {
                "enabled": fc.get("tuning", {}).get("enabled", False),
                "n_trials": int(fc.get("tuning", {}).get("n_trials", 50)),
            },
            "features": record.get("features") or fc.get("features", {}),
        },
    }


BACKENDS = ["lightgbm", "xgboost", "catboost"]
RULES = VN2Rules.from_config(base_cfg)
# Measure choices per horizon: label → (kind, tau). The critical ratio stays a token so
# the saved config reads as the cost-optimal level, not a bare number.
PRESETS: dict[str, tuple[str, object]] = {
    "mean": ("mean", None),
    "q50 (median)": ("quantile", 0.5),
    f"q83 (critical ratio {RULES.critical_ratio:.1%})": ("quantile", CRITICAL_RATIO),
    **{f"q{p}": ("quantile", p / 100) for p in (10, 25, 75, 90, 95, 97, 99)},
}


def _preset_label(entry: dict) -> str | None:
    """Preset label of a config measure entry, or None when it is a custom level."""
    kind, tau = entry.get("kind"), entry.get("tau")
    if kind == "mean":
        return kind
    if isinstance(tau, str) or (tau is not None and abs(float(tau) - RULES.critical_ratio) < 0.005):
        return next(k for k, v in PRESETS.items() if v[1] == CRITICAL_RATIO)
    for label, (k, t) in PRESETS.items():
        if k == "quantile" and isinstance(t, float) and abs(t - float(tau)) < 1e-9:
            return label
    return None


# Widget keys carry a form version so Reset recreates every input. The form dict is
# rewritten each run, so values survive a visit to another page.
# Rev 3 adds forecast measures and the backend choice. Rev 4 adds the demand-profile
# features and the mean objective, and drops measure kinds no longer offered. Rev 5
# replaces the refit-per-block switch with a refit cadence and adds the ensemble. An open
# session still holding a previous form is moved onto the config.yaml defaults once.
_FORM_REV = 5
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
prof_cfg = feat.get("demand_profile", {})

left, right = st.columns([1, 1.2])

with left:
    tab_p, tab_f, tab_q, tab_m = st.tabs(["Periods", "Features", "Measures", "Model"])

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
            "RL train weeks", 12, 80, int(pin.get("rl_train_weeks", 26)), key=k("rl_train_w")
        )
        rl_val_w = c2.number_input(
            "RL validation weeks", 8, 40, int(pin.get("rl_val_weeks", 8)), key=k("rl_val_w")
        )
        rl_test_w = c3.number_input(
            "RL test weeks", 8, 40, int(pin.get("rl_test_weeks", 8)), key=k("rl_test_w")
        )
        refit_every = st.number_input(
            "Refit the forecaster every n weeks (0 = once per block)",
            0,
            52,
            int(pin.get("refit_every_weeks", 13) or 0),
            key=k("refit_every"),
            help=(
                "Applies inside the RL training and validation blocks. Each fit only uses "
                "labels from before the weeks it forecasts, so shorter cadences keep the "
                "training forecasts as fresh as the ones the agent is judged on."
            ),
        )
        st.caption("The test block always keeps one fit made before its first week (VN2 rule).")
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
        prof_on = st.toggle(
            "Demand profile",
            value=bool(prof_cfg.get("enabled", False)),
            key=k("profile"),
            help=(
                "Each SKU's profile from the trailing window ending last week: intermittency "
                "(ADI, CV² of non-zero sales), level, volume percentile, weeks since the last "
                "sale, its demand pattern (smooth, erratic, intermittent or lumpy) and its "
                "volume tier."
            ),
        )

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
        all_events = list(EVENTS)
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
        prof_window = st.select_slider(
            "Demand profile window (weeks)",
            [13, 26, 52],
            value=int(prof_cfg.get("window_weeks", 52)),
            disabled=not prof_on,
            key=k("prof_window"),
        )

    with tab_q:
        st.caption(
            "Pick what to forecast for each week of the order: h0 = this week, h1 = next "
            "week, h2 = delivery week. Each horizon needs a mean or a quantile so the anchor "
            "can project stock."
        )
        measure_entries: list[dict] = []
        saved = fc_form.get("measures") or []
        for h in HORIZONS:
            at_h = [e for e in saved if int(e.get("horizon", -1)) == h]
            chosen = [lbl for e in at_h if (lbl := _preset_label(e))]
            custom = [e for e in at_h if _preset_label(e) is None]
            c1, c2 = st.columns([3, 1])
            picks = c1.multiselect(
                f"h{h}", list(PRESETS), default=[c for c in PRESETS if c in chosen], key=k(f"measures_h{h}")
            )
            extra = c2.text_input(
                f"Other quantiles h{h} (%)",
                ", ".join(f"{float(e['tau']) * 100:g}" for e in custom),
                key=k(f"custom_h{h}"),
                help="Comma-separated levels in percent, e.g. 60, 70",
            )
            for lbl in picks:
                kind, tau = PRESETS[lbl]
                measure_entries.append({"kind": kind, "horizon": h, **({"tau": tau} if tau is not None else {})})
            for part in [x.strip() for x in extra.split(",") if x.strip()]:
                try:
                    measure_entries.append({"kind": "quantile", "horizon": h, "tau": float(part) / 100})
                except ValueError:
                    st.error(f"`{part}` is not a number (h{h} quantile levels).")
        parsed = parse_measures({"forecasting": {"measures": measure_entries}, "vn2": base_cfg.get("vn2", {})})
        levels = [m.id for m in parsed]
        fallback = default_primary(parsed, RULES)
        saved_primary = fc_form.get("primary_measure")
        primary = st.selectbox(
            "Primary measure (drives model selection and tuning)",
            levels or ["—"],
            index=(levels.index(saved_primary) if saved_primary in levels
                   else levels.index(fallback.id) if fallback and fallback.id in levels else 0),
            disabled=not levels,
            key=k("primary"),
            help="Pinball loss for a quantile, RMSE for a mean (their proper scores).",
        )
        calibrate = st.toggle(
            "Calibrate quantiles (split-conformal, on out-of-fold data)",
            value=bool(fc_form.get("calibrate_quantiles", False)),
            key=k("calibrate"),
            help="Shifts each quantile so its coverage on the CV region matches its level.",
        )
        measure_errors = validate_measures(parsed, primary if levels else None)
        st.caption("Forecast columns: " + (", ".join(f"`{m.id}`" for m in parsed) or "none"))
        for err in measure_errors:
            st.error(err)

    with tab_m:
        ens_cfg = fc_form.get("ensemble") or {}
        enable_ensemble = st.toggle(
            "Ensemble",
            value=bool(ens_cfg.get("enabled", False)),
            key=k("ensemble"),
            help=(
                "Fit every chosen backend at each fit and average their forecasts. "
                "Model selection is off while this is on; tuning tunes each member."
            ),
        )
        members = st.multiselect(
            "Ensemble members",
            BACKENDS,
            default=[m for m in (ens_cfg.get("members") or ["catboost", "lightgbm"]) if m in BACKENDS],
            disabled=not enable_ensemble,
            key=k("members"),
        )
        if enable_ensemble and len(members) < 2:
            st.error("Pick at least two ensemble members.")
        sel_cfg = fc_form["model_selection"]
        enable_selection = st.toggle(
            "Model selection",
            value=bool(sel_cfg.get("enabled", False)) and not enable_ensemble,
            key=k("selection"),
            disabled=enable_ensemble,
            help="Score candidate backends with default parameters on the last CV fold.",
        ) and not enable_ensemble
        candidates = st.multiselect(
            "Candidates",
            BACKENDS,
            default=[c for c in sel_cfg.get("candidates", BACKENDS) if c in BACKENDS],
            disabled=not enable_selection,
            key=k("candidates"),
        )
        backend = st.selectbox(
            "Backend",
            BACKENDS,
            index=BACKENDS.index((fc_form.get("model") or {}).get("name", "lightgbm")),
            disabled=enable_selection or enable_ensemble,
            key=k("backend"),
            help="Used when model selection and the ensemble are off.",
        )
        model_form = fc_form.get("model") or {}
        mean_objective = st.selectbox(
            "Mean loss",
            MEAN_OBJECTIVES,
            index=MEAN_OBJECTIVES.index(model_form.get("mean_objective", "poisson")),
            key=k("mean_objective"),
            help=(
                "Loss of the mean boosters (mean_h*). Tweedie's variance grows as mean^p, "
                "which suits zero-heavy, overdispersed demand; quantiles are unaffected."
            ),
        )
        tweedie_power = st.slider(
            "Tweedie variance power p",
            1.1,
            1.9,
            float(model_form.get("tweedie_variance_power", 1.5)),
            step=0.1,
            disabled=mean_objective != "tweedie",
            key=k("tweedie_power"),
            help="Closer to 1 behaves like Poisson; closer to 2 like Gamma (heavier tails).",
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
        "refit_every_weeks": int(refit_every),
    },
    "forecasting": {
        "measures": measure_entries,
        "primary_measure": primary if levels else None,
        "calibrate_quantiles": calibrate,
        "model": {
            "name": backend,
            "mean_objective": mean_objective,
            "tweedie_variance_power": round(float(tweedie_power), 2),
        },
        "validation": {
            "n_folds": int(n_folds),
            "fold_size_weeks": int(fold_size),
            "min_train_size_weeks": int(min_train),
        },
        "model_selection": {"enabled": enable_selection, "candidates": candidates or BACKENDS},
        "ensemble": {"enabled": enable_ensemble, "members": members or ["catboost", "lightgbm"]},
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
            "demand_profile": {
                "enabled": prof_on,
                "window_weeks": int(prof_window),
            },
        },
    },
}
st.session_state.fc_form = overrides

N_MEMBERS = len(members) if enable_ensemble else 1
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
    if not periods.errors:
        lib.render_refit_preview(periods, n_members=N_MEMBERS)
    lib.render_split_view(timeline(periods), periods_to_dict(periods))
    if not periods.errors:
        chart = lib.age_chart(periods)
        if chart is not None:
            st.markdown("**Forecaster age at each order decision**")
            st.altair_chart(chart, width="stretch")
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
    disabled=bool(periods.errors) or bool(measure_errors) or fc_running
    or (enable_ensemble and len(members) < 2),
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
