"""
Shared glue for the Streamlit app.

Domain logic stays in ``src/``; this module only adds Streamlit-specific concerns:
locating the project root, caching config / MLflow queries, building override configs
for triggered runs, and streaming subprocess output into the UI.

All MLflow access goes through :mod:`src.utils.tracking` — the app never imports ``mlflow``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

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


def forecasts_path() -> Path:
    cfg = get_config()
    return (
        ROOT
        / cfg["forecasting"]["artifacts"]["forecasts_dir"]
        / "train_forecasts.parquet"
    )


def rl_model_path(agent: str) -> Path:
    return ROOT / "artifacts" / "rl" / f"{agent}_inventory.zip"


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


def stream_command(module_args: list[str]) -> Iterator[str]:
    """
    Run ``python -m <module_args>`` from the repo root and yield stdout lines live.

    Uses the same interpreter running Streamlit so the venv (and src/) is consistent.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, *module_args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.stdout.close()
    rc = proc.wait()
    yield f"\n[process exited with code {rc}]"


def run_with_live_log(module_args: list[str], height: int = 360) -> None:
    """Stream a command into a scrolling log box inside the current Streamlit context."""
    log_box = st.empty()
    lines: list[str] = []
    for line in stream_command(module_args):
        lines.append(line)
        log_box.code("\n".join(lines[-400:]), language="text")
