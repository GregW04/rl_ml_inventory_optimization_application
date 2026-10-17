"""
Shared glue for the Streamlit app.

Domain logic stays in ``src/``; this module only adds Streamlit-specific concerns:
locating the project root, caching config / MLflow queries, building override configs
for triggered runs, running pipelines as background jobs, and the shared split view.

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

from src.forecasting.periods import (  # noqa: E402
    fetch_run_file,
    load_run_setup,
    panel_weeks,
    resolve_periods,
)
from src.utils import tracking  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Config + tracking
# ---------------------------------------------------------------------------

@st.cache_resource
def get_config() -> dict:
    """Load and cache the project config once per session."""
    return load_config(str(ROOT / "configs" / "config.yaml"))


def init_tracking() -> str:
    """Point MLflow at the canonical (absolute) tracking store; return the URI."""
    return tracking.init_tracking(get_config(), "forecasting")


def experiment_names() -> dict[str, str]:
    """Map section → experiment name from config."""
    cfg = get_config()
    return {
        "forecasting": cfg["forecasting"]["mlflow"]["experiment_name"],
        "rl": cfg["rl"]["mlflow"]["experiment_name"],
    }


@st.cache_data(ttl=15)
def cached_runs(experiment_name: str):
    """Cached run search (short TTL so new runs appear quickly)."""
    init_tracking()
    return tracking.search_runs(experiment_name)


def merged_config(updates: dict) -> dict:
    """Base config + nested updates (not written to disk)."""
    base = load_config(str(ROOT / "configs" / "config.yaml"))
    _deep_update(base, updates)
    return base


def resolve_from_config_dict(cfg: dict):
    weeks = panel_weeks(cfg)
    return resolve_periods(cfg, weeks)


def forecasts_path() -> Path:
    cfg = get_config()
    return (
        ROOT
        / cfg["forecasting"]["artifacts"]["forecasts_dir"]
        / "train_forecasts.parquet"
    )


def rl_model_path(agent: str) -> Path:
    return ROOT / "artifacts" / "rl" / f"{agent}_inventory.zip"


def agent_meta(agent: str) -> dict:
    """Sidecar written by RL training: the forecasting run the saved agent was trained on."""
    path = rl_model_path(agent).with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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


# ---------------------------------------------------------------------------
# Split view (live on the Forecasting page, frozen on RL / Experiments)
# ---------------------------------------------------------------------------

_SEGMENT_COLORS = {
    "Feature warm-up": "#EECA3B",
    "Embargo": "#EECA3B",
    "CV fit region": "#4C78A8",
    "Label gap": "#9D9D9D",
    "Train": "#F58518",
    "Validation (checkpoint)": "#54A24B",
    "Test (final score)": "#E45756",
    "Fit": "#72B7B2",
    "Gap": "#BAB0AC",
    "Validation": "#B279A2",
}
_LANE_RANK = {"Features": 0, "Forecaster": 1, "RL": 3}


def timeline_chart(segments_df):
    """Altair Gantt of period segments (lane × part)."""
    import altair as alt

    if segments_df is None or segments_df.empty:
        return None
    df = segments_df.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"]) + pd.Timedelta(days=6)
    lanes = sorted(df["lane"].unique(), key=lambda lane: (_LANE_RANK.get(lane, 2), lane))
    present = [p for p in _SEGMENT_COLORS if p in set(df["part"])]
    color_scale = alt.Scale(domain=present, range=[_SEGMENT_COLORS[p] for p in present])
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("lane:N", title="Lane", sort=lanes, axis=alt.Axis(labelOverlap=False)),
            x=alt.X("start:T", title="Week start"),
            x2="end:T",
            color=alt.Color("part:N", title="Segment", scale=color_scale),
            tooltip=["lane", "part", "start", "end", "weeks"],
        )
        .properties(height=max(260, 60 + 30 * len(lanes)), width="container")
    )
    return chart


def _weeks_between(start, end) -> int | None:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return None if pd.isna(s) or pd.isna(e) else (e - s).days // 7 + 1


def render_split_view(segments: pd.DataFrame, resolved: dict) -> None:
    """Timeline chart plus block table, from live periods or a saved setup record."""
    chart = timeline_chart(segments)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
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
            "Forecasts from a fit through": resolved.get(fit, "—") if fit else "—",
        }
        for name, key, fit in blocks
        if resolved.get(f"{key}_start")
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


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
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
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
            st.rerun()

    _panel()
    return job
