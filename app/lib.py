"""
Shared glue for the Streamlit app.

Domain logic stays in ``src/``; this module only adds Streamlit-specific concerns:
locating the project root, caching config / MLflow queries, building override configs
for triggered runs, running pipelines as background jobs, and the shared split view.

``VN2_CONFIG`` (an environment variable) points the app at another base config, e.g. a
scratch config whose tracking store and artifact folders are isolated for smoke tests.

All MLflow access goes through :mod:`src.utils.tracking` — the app never imports ``mlflow``.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml


def _find_root() -> Path:
    """Walk up until we find the repo root (contains both ``src/`` and ``app/``)."""
    for p in Path(__file__).resolve().parents:
        if (p / "src").is_dir() and (p / "app").is_dir():
            return p
    return Path(__file__).resolve().parents[1]


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = Path(os.environ.get("VN2_CONFIG") or ROOT / "configs" / "config.yaml")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://127.0.0.1:8080")

from src.forecasting.measures import Measure, run_measures  # noqa: E402
from src.forecasting.periods import (  # noqa: E402
    BLOCK_NAMES,
    block_fit_counts,
    decision_ages,
    fetch_run_file,
    load_run_setup,
    mean_decision_age,
    panel_weeks,
    refit_events,
    resolve_periods,
)
from src.simulation.vn2 import check_vn2_compliance  # noqa: E402
from src.utils import tracking  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Config + tracking
# ---------------------------------------------------------------------------

@st.cache_resource
def get_config() -> dict:
    """Load and cache the project config once per session."""
    return load_config(str(CONFIG_PATH))


def init_tracking() -> str:
    """Point MLflow at the canonical (absolute) tracking store; return the URI."""
    return tracking.init_tracking(get_config())


def experiment_names() -> dict[str, str]:
    """Map section → experiment name from config."""
    cfg = get_config()
    return {
        stage: tracking.experiment_name(cfg, stage) for stage in ("forecasting", "rl")
    }


@st.cache_data(ttl=15)
def cached_runs(experiment_name: str):
    """Cached run search (short TTL so new runs appear quickly)."""
    init_tracking()
    return tracking.search_runs(experiment_name)


def merged_config(updates: dict) -> dict:
    """Base config + nested updates (not written to disk)."""
    base = load_config(str(CONFIG_PATH))
    _deep_update(base, updates)
    return base


def resolve_from_config_dict(cfg: dict):
    weeks = panel_weeks(cfg)
    return resolve_periods(cfg, weeks)


def vn2_compliance_badge(cfg: dict | None = None) -> tuple[bool, list[str]]:
    """Green/red VN2 comparability from config and resolved periods."""
    cfg = cfg or get_config()
    periods = resolve_from_config_dict(cfg)
    resolved = {
        "rl_test_start": str(periods.rl_test_start.date()),
        "rl_test_end": str(periods.rl_test_end.date()),
    }
    comp = check_vn2_compliance(cfg, resolved)
    return comp.ok, comp.messages


# ---------------------------------------------------------------------------
# Forecasting runs (as seen by the RL page)
# ---------------------------------------------------------------------------

def forecast_runs() -> pd.DataFrame:
    """Finished forecasting runs that saved a period setup, newest first."""
    runs = cached_runs(experiment_names()["forecasting"])
    if runs.empty or "params.rl_train_start" not in runs:
        return pd.DataFrame()
    return runs[(runs["status"] == "FINISHED") & runs["params.rl_train_start"].notna()]


def forecasts_in_progress() -> bool:
    """True while an app forecasting job or a recent RUNNING forecasting run exists."""
    job = job_state("forecast")
    if job and job["running"]:
        return True
    runs = cached_runs(experiment_names()["forecasting"])
    if runs.empty:
        return False
    recent = runs["start_time"] > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    return bool(((runs["status"] == "RUNNING") & recent).any())


@st.cache_data(ttl=60, show_spinner=False)
def run_setup(run_id: str) -> dict | None:
    """Period setup record saved with a forecasting run."""
    init_tracking()
    return load_run_setup(get_config(), run_id)


@st.cache_data(ttl=60, show_spinner=False)
def run_has_forecasts(run_id: str) -> bool:
    """Whether the run's RL-block forecasts can be loaded (on disk or from MLflow)."""
    init_tracking()
    return fetch_run_file(get_config(), run_id, "train_forecasts.parquet") is not None


@st.cache_data(ttl=60, show_spinner=False)
def measures_of_run(run_id: str) -> list[dict]:
    """Measures of a forecasting run (manifest, or inferred for runs saved before manifests)."""
    init_tracking()
    return [m.to_dict() for m in run_measures(get_config(), run_id)]


def as_measures(rows: list[dict]) -> list[Measure]:
    return [Measure.from_dict(d) for d in rows]


@st.cache_data(ttl=600, show_spinner="Loading sales history…")
def sales_history() -> pd.DataFrame:
    """Weekly sales of every SKU (out-of-stock weeks are NaN), for drift diagnostics."""
    from src.forecasting.preprocessing import load_panel_from_config

    return load_panel_from_config(get_config())[["store", "product", "date", "sales"]]


@st.cache_data(ttl=600, show_spinner="Classifying demand patterns…")
def demand_classes() -> pd.DataFrame:
    """Demand pattern of every SKU-week (Smooth, Erratic, Intermittent, Lumpy, Unclassified)."""
    from src.rl.vn2_data import attach_demand_class

    return attach_demand_class(sales_history().copy(), get_config())[["store", "product", "date", "demand_class"]]


@st.cache_data(ttl=60, show_spinner="Computing forecast drift…")
def run_drift(run_id: str, include_test: bool) -> dict:
    """Forecast drift of a run: per-fit table, per-class table, weekly errors; {} without forecasts."""
    from src.forecasting import drift

    init_tracking()
    record = run_setup(run_id)
    path = fetch_run_file(get_config(), run_id, "train_forecasts.parquet")
    if record is None or path is None:
        return {}
    forecasts = pd.read_parquet(path)
    measures = as_measures(measures_of_run(run_id))
    resolved = record.get("resolved", {})
    sales = sales_history()
    level = drift.h0_level_measure(measures)
    return {
        "level": level.id if level else None,
        "fits": drift.fit_bias(forecasts, sales, resolved, measures, include_test),
        "by_class": (drift.fit_bias(forecasts, sales, resolved, [level], include_test, classes=demand_classes())
                     if level else pd.DataFrame()),
        "weekly": drift.weekly_bias(forecasts, sales, resolved, level, include_test) if level else pd.DataFrame(),
        "blocks": drift.block_bias(forecasts, sales, resolved, level, include_test) if level and level.kind == "mean" else {},
    }


@st.cache_data(ttl=60, show_spinner=False)
def run_block_bias(run_id: str, include_test: bool) -> tuple[str | None, dict[str, float]]:
    """(h0 mean id, forecast error % per block) of a run; (None, {}) when it cannot be computed."""
    from src.forecasting import drift

    init_tracking()
    record = run_setup(run_id)
    path = fetch_run_file(get_config(), run_id, "train_forecasts.parquet")
    level = drift.h0_level_measure(as_measures(measures_of_run(run_id)))
    if record is None or path is None or level is None or level.kind != "mean":
        return None, {}
    blocks = drift.block_bias(pd.read_parquet(path), sales_history(), record.get("resolved", {}), level, include_test)
    return level.id, blocks


@st.cache_data(ttl=600, show_spinner="Classifying demand patterns…")
def alert_actuals() -> pd.DataFrame:
    """In-stock weekly sales with each SKU-week's demand class, as the drift alerts score them."""
    from src.forecasting.drift_alerts import class_actuals

    return class_actuals(sales_history(), get_config())


@st.cache_data(ttl=60, show_spinner="Loading drift alerts…")
def run_alerts(run_id: str, include_test: bool) -> tuple[pd.DataFrame, bool]:
    """
    (alerts, logged): the drift alerts the run logged, or, for runs that didn't log them, the same
    detector computed now on the run's forecasts. The test block is dropped unless ``include_test``.
    """
    from src.forecasting import drift_alerts
    from src.forecasting.drift import h0_level_measure

    init_tracking()
    path = fetch_run_file(get_config(), run_id, "drift_alerts.parquet")
    if path is not None:
        alerts, logged = pd.read_parquet(path), True
    else:
        record = run_setup(run_id)
        fc_path = fetch_run_file(get_config(), run_id, "train_forecasts.parquet")
        level = h0_level_measure(as_measures(measures_of_run(run_id)))
        if record is None or fc_path is None or level is None:
            return pd.DataFrame(), False
        alerts = drift_alerts.fit_alerts(pd.read_parquet(fc_path), alert_actuals(), record.get("resolved", {}),
                                         level.id, drift_alerts.default_rules())
        logged = False
    if not include_test:
        alerts = alerts[alerts["block"] != "rl_test"]
    return alerts.reset_index(drop=True), logged


@st.cache_data(ttl=60, show_spinner=False)
def run_retunes(run_id: str) -> pd.DataFrame:
    """The retune challenges a run made (week, block, classes, holdout scores, adopted); empty when it made none."""
    init_tracking()
    path = fetch_run_file(get_config(), run_id, "retunes.json")
    rows = json.loads(Path(path).read_text(encoding="utf-8")) if path is not None else []
    return pd.DataFrame(rows)


def forecast_run_label(r: pd.Series) -> str:
    """Start time, short id and the run's primary CV score."""
    started = r["start_time"].to_pydatetime().astimezone()
    score = ""
    metric = r.get("params.primary_metric")
    value = r.get("metrics.cv_primary")
    if isinstance(metric, str) and pd.notna(value):
        score = f" · {metric} {value:.3f}"
    elif pd.notna(r.get("metrics.cv_pinball_q83_h2_mean")):
        score = f" · pinball_q83_h2 {r['metrics.cv_pinball_q83_h2_mean']:.3f}"
    return f"{started:%Y-%m-%d %H:%M} · {r['run_id'][:8]}{score}"


# ---------------------------------------------------------------------------
# RL runs, leaderboard and the test lock
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def rl_leaderboard(forecast_run_id: str) -> pd.DataFrame:
    """Validation leaderboard (configurations + anchor) for a forecasting run."""
    from src.rl.selection import leaderboard

    init_tracking()
    return leaderboard(get_config(), forecast_run_id)


@st.cache_data(ttl=15, show_spinner=False)
def test_scores(forecast_run_id: str) -> pd.DataFrame:
    """Finished test scorings of a forecasting run, newest first (empty = not yet scored)."""
    from src.rl.evaluate import scored_test_runs

    init_tracking()
    return scored_test_runs(get_config(), forecast_run_id)


@st.cache_data(ttl=15, show_spinner=False)
def scored_forecast_runs() -> set[str]:
    """Forecasting runs whose test block has been scored (a FINISHED ``stage=test`` RL run)."""
    init_tracking()
    runs = tracking.search_runs(
        tracking.experiment_name(get_config(), "rl"),
        filter_string="tags.stage = 'test' and attributes.status = 'FINISHED'",
    )
    if runs.empty or "tags.forecast_run_id" not in runs:
        return set()
    return set(runs["tags.forecast_run_id"].dropna())


@st.cache_data(ttl=60, show_spinner=False)
def legacy_test_count(forecast_run_id: str) -> int:
    """How often the test block was evaluated before the lock existed."""
    from src.rl.evaluate import legacy_test_evaluations

    return legacy_test_evaluations(get_config(), forecast_run_id)


def rl_train_runs(forecast_run_id: str | None = None) -> pd.DataFrame:
    """Finished training runs (``stage=train``), optionally for one forecasting run."""
    runs = cached_runs(experiment_names()["rl"])
    if runs.empty or "tags.stage" not in runs:
        return pd.DataFrame()
    keep = (runs["tags.stage"] == "train") & (runs["status"] == "FINISHED")
    if forecast_run_id:
        keep &= runs.get("tags.forecast_run_id") == forecast_run_id
    return runs[keep]


@st.cache_data(ttl=60, show_spinner=False)
def rl_history(run_id: str, keys: tuple[str, ...]) -> pd.DataFrame:
    """Logged history of ``keys`` for one RL training run, long format (key, step, value)."""
    init_tracking()
    frames = [h.assign(key=k) for k in keys if not (h := tracking.get_metric_history(run_id, k)).empty]
    if not frames:
        return pd.DataFrame(columns=["key", "step", "value"])
    return pd.concat(frames, ignore_index=True)[["key", "step", "value"]]


@st.cache_data(ttl=60, show_spinner=False)
def rl_episodes(run_id: str, name: str) -> pd.DataFrame:
    """A per-episode table saved with an RL run (``val_episodes``, ``insample_episodes``); empty if missing."""
    from src.rl.selection import policy_dir

    try:
        return pd.read_parquet(policy_dir(get_config(), run_id) / f"{name}.parquet")
    except Exception:  # noqa: BLE001 - a missing artifact only hides its charts
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Production loop (src/ops): the live VN2 run, read only
# ---------------------------------------------------------------------------

OPS_LIVE_EXPERIMENT = "vn2-ops-live"


@st.cache_data(ttl=300, show_spinner=False)
def ops_live_result() -> dict | None:
    """The live run's ops test (scored once): scores, CI, rules version and a link; None if it hasn't run."""
    uri = init_tracking()
    runs = tracking.search_runs(OPS_LIVE_EXPERIMENT, filter_string="tags.ops_test = 'true'")
    if runs.empty:
        return None
    r = runs.iloc[0]
    exps = tracking.list_experiments()
    exp_id = exps.loc[exps["name"] == OPS_LIVE_EXPERIMENT, "experiment_id"].iloc[0]
    out = {k: r.get(f"metrics.{k}") for k in ("live_vn2_score_eur", "anchor_vn2_score_eur", "live_eur_per_sku",
                                               "anchor_eur_per_sku", "live_vs_anchor_pct", "ci_lo", "ci_hi")}
    out.update({
        "run_id": r["run_id"], "scored_weeks": r.get("tags.scored_weeks"), "rules_version": r.get("tags.rules_version"),
        "post_hoc": r.get("tags.post_hoc") == "true", "n_scorings": len(runs),
        "url": f"{uri.rstrip('/')}/#/experiments/{exp_id}/runs/{r['run_id']}" if uri.startswith("http") else None,
        "airflow_url": f"{AIRFLOW_URL}/dags/vn2_live",
    })
    return out


# ---------------------------------------------------------------------------
# Split view (live on the Forecasting page, frozen on RL / Experiments)
# ---------------------------------------------------------------------------

_SEGMENT_COLORS = {
    "Embargo": "#EECA3B",
    "CV fit region": "#4C78A8",
    "Label gap": "#9D9D9D",
    "Train": "#F58518",
    "Validation (selection)": "#54A24B",
    "Test (production)": "#E45756",
    "Fit": "#72B7B2",
    "Forecasts written": "#9ECAE9",
    "Validation": "#B279A2",
    # The lane of a retune: the rows its tuning saw, and the holdout it never saw.
    "Tuning rows": "#72B7B2",
    "Holdout": "#B279A2",
    # Labels saved by runs before the rename.
    "Validation (checkpoint)": "#54A24B",
    "Test (final score)": "#E45756",
}
_BLOCK_ORDER = {name: i for i, name in enumerate(BLOCK_NAMES.values())}
_BLOCK_COLORS = {"training": "#F58518", "validation": "#54A24B", "test": "#E45756"}


def _lane_key(lane: str) -> tuple:
    """Forecaster, CV folds, block fits (training, validation, test), then RL."""
    if lane == "Features":
        return (0, 0, 0, lane)
    if lane == "Forecaster":
        return (1, 0, 0, lane)
    if lane.startswith("CV fold"):
        tail = lane.rsplit(" ", 1)[-1]
        return (2, int(tail) if tail.isdigit() else 0, 0, lane)
    if lane.startswith("Fit · "):
        rest = lane[len("Fit · "):].split(" ")
        idx = int(rest[1].split("/")[0]) if len(rest) > 1 and rest[1].split("/")[0].isdigit() else 1
        return (3, _BLOCK_ORDER.get(rest[0], 9), idx, lane)
    if lane.startswith("Retune · "):
        return (3, 5, 0, lane)
    if lane == "RL":
        return (4, 0, 0, lane)
    return (5, 0, 0, lane)


REFIT_KINDS = {"start": "block start", "schedule": "schedule", "drift": "alert"}
REFIT_COLORS = {"start": "#8a8a8a", "schedule": "#4C78A8", "drift": "#E45756"}
REFIT_EVENT = {"start": "refit: block start", "schedule": "refit: schedule", "drift": "refit: after alert"}
# The vertical marks of a run's timeline, in drawing order (retunes under refits under alerts):
# event -> (colour, dash pattern, line width).
EVENT_STYLES = {
    "retune: adopted": ("#7B3FB0", [1, 0], 6.0),
    "retune: kept settings": ("#B39DDB", [1, 0], 6.0),
    "retune: skipped": ("#BDBDBD", [1, 0], 6.0),
    "refit: block start": ("#8a8a8a", [6, 3], 1.5),
    "refit: schedule": ("#4C78A8", [6, 3], 1.5),
    "refit: after alert": ("#E45756", [6, 3], 1.5),
    "alert": ("#8B1A1A", [2, 3], 2.5),
}
WEEK = pd.Timedelta(weeks=1)


def _is_true(value) -> bool:
    """True for a real True; a missing cell (NaN, None) is not."""
    return isinstance(value, bool | np.bool_) and bool(value)


def _num(value, spec: str) -> str:
    """A number formatted with ``spec``; 'n/a' for a missing one."""
    return "n/a" if value is None or pd.isna(value) else format(value, spec)


def _retune_rows(retunes: pd.DataFrame | None) -> list[dict]:
    return [] if retunes is None or retunes.empty else retunes.to_dict("records")


def event_marks(refits: pd.DataFrame | None = None, alerts: pd.DataFrame | None = None,
                retunes: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    The vertical marks of a run's timeline: ``date``, ``event`` (a key of ``EVENT_STYLES``) and a tooltip
    ``detail``. ``alerts`` is ``drift_alerts.first_alert_marks``; a retune is drawn at the refit that
    followed its alert, the week from which its settings (or the kept ones) forecast.
    """
    rows: list[dict] = []
    if refits is not None and not refits.empty:
        for r in refits.itertuples():
            through = f", data through {r.data_through:%Y-%m-%d}" if pd.notna(r.data_through) else ""
            rows.append({
                "date": r.date, "event": REFIT_EVENT.get(r.trigger, f"refit: {r.trigger}"),
                "detail": f"{r.block} fit {r.label}{through}" + (f"; alert: {r.classes}" if r.classes else "")
                + ("; with retuned settings" if r.retuned else "")})
    if alerts is not None and not alerts.empty:
        rows += [{"date": a.week, "event": "alert", "detail": f"{a.classes} alerted in fit {a.fit}"} for a in alerts.itertuples()]
    for r in _retune_rows(retunes):
        week, classes = pd.Timestamp(r["week"]), ", ".join(r.get("classes") or [])
        if _is_true(r.get("skipped")):
            event, detail = "retune: skipped", f"alert {week:%Y-%m-%d} ({classes}): skipped, {r.get('note')}"
        else:
            adopted = _is_true(r.get("adopted"))
            event = "retune: adopted" if adopted else "retune: kept settings"
            detail = (f"alert {week:%Y-%m-%d} ({classes}): holdout score of the current settings "
                      f"{_num(r.get('current_score'), '.3f')}, of the new {_num(r.get('new_score'), '.3f')} "
                      f"({_num(r.get('gain_pct'), '+.1f')}%); "
                      + ("new settings used" if adopted else "kept the current settings"))
        rows.append({"date": week + WEEK, "event": event, "detail": detail})
    if not rows:
        return pd.DataFrame(columns=["date", "event", "detail"])
    order = {k: i for i, k in enumerate(EVENT_STYLES)}
    marks = pd.DataFrame(rows)
    return marks.sort_values("event", key=lambda e: e.map(order), kind="stable").reset_index(drop=True)


def retune_segments(retunes: pd.DataFrame | None, resolved: dict) -> pd.DataFrame:
    """
    Timeline rows for the retunes that tuned, one lane each, like a CV fold: the rows the tuning saw, the label
    gap, and the holdout the tuning never saw. Runs saved before retunes recorded their rows give none.
    """
    rows = []
    start = pd.Timestamp(resolved["data_start"]) if resolved.get("data_start") else None

    def add(lane, part, a, b):
        rows.append({"lane": lane, "part": part, "start": a, "end": b, "weeks": (b - a).days // 7 + 1})

    for r in _retune_rows(retunes):
        if _is_true(r.get("skipped")) or not r.get("tuned_to") or not r.get("holdout_start") or start is None:
            continue
        lane = f"Retune · {r['week']}"
        tuned_to, h0, h1 = pd.Timestamp(r["tuned_to"]), pd.Timestamp(r["holdout_start"]), pd.Timestamp(r["holdout_end"])
        add(lane, "Tuning rows", start, tuned_to)
        if h0 - tuned_to > WEEK:
            add(lane, "Label gap", tuned_to + WEEK, h0 - WEEK)
        add(lane, "Holdout", h0, h1)
    return pd.DataFrame(rows, columns=["lane", "part", "start", "end", "weeks"])


def retune_table(retunes: pd.DataFrame) -> pd.DataFrame:
    """The retunes of a run as a table: the CV settings used, the holdout, the scores and the outcome."""
    rows = []
    for r in _retune_rows(retunes):
        folds, weeks = int(r.get("folds") or 0), int(r.get("fold_weeks") or 0)
        skipped, adopted = _is_true(r.get("skipped")), _is_true(r.get("adopted"))
        rows.append({
            "Alert week": r["week"], "Block": BLOCK_NAMES.get(r.get("block"), r.get("block")),
            "Alerting classes": ", ".join(r.get("classes") or []),
            "CV folds tuned on": f"{folds} × {weeks} wk" if folds else "—",
            "Tuned on rows to": r.get("tuned_to") or "—",
            "Holdout": f"{r['holdout_start']} … {r['holdout_end']}" if r.get("holdout_start") else "—",
            "Current settings": r.get("current_score"), "New settings": r.get("new_score"), "Gain %": r.get("gain_pct"),
            "Outcome": f"skipped: {r.get('note')}" if skipped else ("new settings used" if adopted else "kept the current settings"),
        })
    return pd.DataFrame(rows)


# Parts that mark a window that is scored or forecast (what a zoomed timeline starts from).
_WINDOW_PARTS = ("Validation", "Holdout", "Forecasts written", "Train", "Validation (selection)", "Test (production)")


def timeline_chart(segments_df, refits: pd.DataFrame | None = None, alerts: pd.DataFrame | None = None,
                   retunes: pd.DataFrame | None = None, zoom: bool = False):
    """
    Altair Gantt of period segments (lane × part) with vertical lines at the forecaster refits and, for a
    finished run, at its alerts and retunes (``event_marks``). ``zoom`` starts the axis four weeks before the
    first scored or forecast window, so the long fit bars run off the left edge and the events get room.
    """
    import altair as alt

    if segments_df is None or segments_df.empty:
        return None
    df = segments_df.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"]) + pd.Timedelta(days=6)
    lanes = sorted(df["lane"].unique(), key=_lane_key)
    present = [p for p in _SEGMENT_COLORS if p in set(df["part"])]
    color_scale = alt.Scale(domain=present, range=[_SEGMENT_COLORS[p] for p in present])
    windows = df[df["part"].isin(_WINDOW_PARTS)]
    domain = [(windows["start"].min() - pd.Timedelta(weeks=4)).isoformat(), (df["end"].max() + WEEK).isoformat()] \
        if zoom and not windows.empty else None
    scale = alt.Scale(domain=domain) if domain else alt.Undefined
    chart = (
        alt.Chart(df)
        .mark_bar(clip=domain is not None)
        .encode(
            y=alt.Y("lane:N", title="Lane", sort=lanes, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("start:T", title="Week start", scale=scale),
            x2="end:T",
            color=alt.Color("part:N", title="Segment", scale=color_scale),
            tooltip=["lane", "part", "start", "end", "weeks"],
        )
    )
    events = event_marks(refits, alerts, retunes)
    if not events.empty:
        kinds = [k for k in EVENT_STYLES if k in set(events["event"])]

        def style(i: int):
            return alt.Scale(domain=kinds, range=[EVENT_STYLES[k][i] for k in kinds])

        rules = alt.Chart(events).mark_rule(clip=domain is not None).encode(
            x=alt.X("date:T", scale=scale),
            color=alt.Color("event:N", title="Event", scale=style(0)),
            strokeDash=alt.StrokeDash("event:N", title="Event", scale=style(1)),
            strokeWidth=alt.StrokeWidth("event:N", title="Event", scale=style(2)),
            tooltip=[alt.Tooltip("event:N", title="Event"), alt.Tooltip("date:T", title="Week"),
                     alt.Tooltip("detail:N", title="Detail")],
        )
        chart = alt.layer(chart, rules).resolve_scale(color="independent", strokeDash="independent",
                                                      strokeWidth="independent")
    return chart.properties(height=max(260, 60 + 30 * len(lanes)), width="container")


def _weeks_between(start, end) -> int | None:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return None if pd.isna(s) or pd.isna(e) else (e - s).days // 7 + 1


def _fit_ends(resolved: dict, block: str, fallback_key: str) -> str:
    """'2023-01-30' for one fit, '2023-01-30 … 2023-10-30' for several."""
    ends = [f["fit_end"] for f in (resolved.get("block_fits") or []) if f.get("block") == block]
    if not ends:
        return resolved.get(fallback_key, "—")
    return ends[0] if len(ends) == 1 else f"{ends[0]} … {ends[-1]}"


def render_split_view(segments: pd.DataFrame, resolved: dict) -> None:
    """Timeline chart plus block table, from live periods or a saved setup record."""
    refits = refit_events(resolved)
    chart = timeline_chart(segments, refits)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        if not refits.empty:
            st.caption("Dashed lines: forecaster refits, by why they were made (hover for the data each fit saw).")
    counts = block_fit_counts(resolved)
    blocks = [
        ("Forecaster CV region", "forecaster_cv", None),
        ("RL train", "rl_train", "fit_end_rl_train"),
        ("RL validation", "rl_val", "fit_end_rl_val"),
        ("RL test", "rl_test", "fit_end_rl_test"),
    ]
    rows = [
        {
            "Block": name,
            "Start": resolved.get(f"{key}_start"),
            "End": resolved.get(f"{key}_end"),
            "Weeks": _weeks_between(resolved.get(f"{key}_start"), resolved.get(f"{key}_end")),
            "Forecaster fits": str(counts.get(key, 1)) if fit else "—",
            "Forecasts from fits through": _fit_ends(resolved, key, fit) if fit else "—",
        }
        for name, key, fit in blocks
        if resolved.get(f"{key}_start")
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def refit_schedule(resolved: dict) -> str:
    """The refit dates of a schedule, e.g. ``2023-08-21 (data through 2023-07-31) · …``."""
    ev = refit_events(resolved)
    return " · ".join(
        f"{r.date:%Y-%m-%d} {r.label} (data through {r.data_through:%Y-%m-%d}"
        + (f"; alert {r.classes}" if r.trigger == "drift" else "") + (", retuned" if r.retuned else "") + ")"
        for r in ev.itertuples())


def refit_summary(resolved: dict) -> str:
    """One line on a run's refit cadence, for saved records (old ones: one fit per block)."""
    n = resolved.get("refit_every_weeks")
    counts = block_fit_counts(resolved)
    if n:
        return (
            f"forecaster refit every **{n}** weeks in training and validation "
            f"({counts['rl_train']} + {counts['rl_val']} fits; test {counts['rl_test']})"
        )
    return "one forecaster fit per block"


def render_refit_preview(periods, n_members: int = 1) -> None:
    """Fit counts and mean forecast age per block for live (unsaved) periods."""
    fits = periods.block_fits
    by_block = {b: sum(1 for f in fits if f.block == b) for b in BLOCK_NAMES}
    total = len(periods.folds) + len(fits)
    ages = decision_ages(periods)
    mean_age = mean_decision_age(ages) if not ages.empty else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Forecaster fits", total,
        help=f"{len(periods.folds)} CV folds + {len(fits)} RL-block fits"
        + (f"; each fit trains {n_members} backends (ensemble)" if n_members > 1 else ""),
    )
    c2.metric("Fits in training block", by_block["rl_train"])
    c3.metric("Forecast age, training", f"{mean_age.get('rl_train', float('nan')):.1f} wk")
    c4.metric(
        "Forecast age, val / test",
        f"{mean_age.get('rl_val', float('nan')):.1f} / {mean_age.get('rl_test', float('nan')):.1f} wk",
    )
    model_fits = total * n_members
    st.caption(
        f"{by_block['rl_train']} training + {by_block['rl_val']} validation + {by_block['rl_test']} test "
        f"block fits and {len(periods.folds)} CV folds"
        + (f", {model_fits} backend fits with the ensemble" if n_members > 1 else "")
        + ". Forecast age is the number of weeks between a fit's last data week and the "
        "Monday an order is decided, averaged over every order decision the agent sees."
    )


def age_chart(periods):
    """Altair bars: forecaster age at each decision Monday, coloured by block."""
    import altair as alt

    ages = decision_ages(periods)
    if ages.empty:
        return None
    df = ages.assign(block=ages["block"].map(BLOCK_NAMES))
    present = [b for b in _BLOCK_COLORS if b in set(df["block"])]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Decision Monday"),
            y=alt.Y("age_weeks:Q", title="Forecaster age (weeks)"),
            color=alt.Color(
                "block:N", title="Block",
                scale=alt.Scale(domain=present, range=[_BLOCK_COLORS[b] for b in present]),
            ),
            tooltip=["block", "date:T", "age_weeks", "fit", "windows"],
        )
        .properties(height=200, width="container")
    )


# ---------------------------------------------------------------------------
# Triggering runs
# ---------------------------------------------------------------------------

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_override_config(updates: dict) -> str:
    """Write a temp config = base config + nested ``updates``; return its path."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    _deep_update(base, updates)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="cfg_override_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    return path


# Jobs run detached and log to a file, so leaving the page (or restarting the app)
# never stalls a pipeline on an unread output pipe.
JOBS_DIR = ROOT / "artifacts" / "app_jobs"
_EXIT_PREFIX = "[exit code "


def start_job(kind: str, label: str, commands: list[list[str]]) -> None:
    """Run ``python <args>`` for each command in turn (stop at the first failure)."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{kind}.log"
    chain = " && ".join(shlex.join([sys.executable, *args]) for args in commands)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", f'{chain}; echo "{_EXIT_PREFIX}$?]"'],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
    meta = {"pid": proc.pid, "label": label, "started": time.time(), "log": str(log_path)}
    (JOBS_DIR / f"{kind}.json").write_text(json.dumps(meta), encoding="utf-8")


def _job_alive(pid: int) -> bool:
    # ``ps`` shows a finished-but-unreaped shell as "(sh)", and a reused pid as some
    # other command, so matching our module name avoids both false positives.
    out = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    return "-m src." in out


def _log_tail(path: Path, max_bytes: int = 64_000) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - max_bytes))
        return f.read().decode("utf-8", errors="replace").splitlines()


def job_state(kind: str) -> dict | None:
    """Latest app job of ``kind`` with ``running``, ``exit_code`` and log tail; None if none."""
    meta_path = JOBS_DIR / f"{kind}.json"
    if not meta_path.exists():
        return None
    job = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = _log_tail(Path(job["log"]))
    exit_code = None
    if lines and lines[-1].startswith(_EXIT_PREFIX):
        exit_code = int(lines.pop()[len(_EXIT_PREFIX):-1])
    job["exit_code"] = exit_code
    job["running"] = exit_code is None and _job_alive(job["pid"])
    job["log_tail"] = "\n".join(lines[-60:])
    return job


def stop_job(kind: str) -> None:
    """Interrupt the job's process group (MLflow then marks its run FAILED)."""
    job = job_state(kind)
    if job and job["running"]:
        os.killpg(job["pid"], signal.SIGINT)


def job_panel(kind: str) -> dict | None:
    """Status, Stop button and log of the latest ``kind`` job; polls while it runs."""
    job = job_state(kind)
    if job is None:
        return None

    @st.fragment(run_every=2 if job["running"] else None)
    def _panel() -> None:
        j = job_state(kind) or job
        started = time.strftime("%H:%M", time.localtime(j["started"]))
        if j["running"]:
            c1, c2 = st.columns([5, 1])
            c1.info(f"{j['label']} running since {started}.")
            if c2.button("Stop", key=f"stop_{kind}", width="stretch"):
                stop_job(kind)
        elif j["exit_code"] == 0:
            st.success(f"{j['label']} (started {started}) finished.")
        elif j["exit_code"] is None:
            st.warning(f"{j['label']} (started {started}) was stopped before finishing.")
        else:
            st.error(f"{j['label']} (started {started}) failed with exit code {j['exit_code']}.")
        with st.expander("Log", expanded=j["running"] or j["exit_code"] not in (0, None)):
            st.code(j["log_tail"] or "(no output yet)", language="text")
            st.caption(f"Full log: `{j['log']}`")
        if job["running"] and not j["running"]:
            cached_runs.clear()
            rl_leaderboard.clear()
            test_scores.clear()
            st.rerun()

    _panel()
    return job
