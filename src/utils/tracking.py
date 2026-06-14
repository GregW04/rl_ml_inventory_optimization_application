"""
Centralized MLflow utilities — the single place every entry point talks to MLflow.

Used by the forecasting pipeline, the RL trainer, the notebooks, and the Streamlit
app.  Splitting cleanly into:

* **setup / write helpers** — resolve the tracking URI, guard duplicate runs, and log
  matplotlib figures / dataframes without repeating tempfile boilerplate.
* **read / query helpers** — list experiments, search runs, pull metric history and
  artifacts.  These are what the frontend consumes so it never touches ``mlflow`` directly.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------------
# Tracking-URI resolution + setup
# ---------------------------------------------------------------------------

def resolve_tracking_uri(uri: str, root: str | Path) -> str:
    """
    Make a relative tracking URI absolute so every process writes to the *same* store
    regardless of the current working directory.

    ``sqlite:///mlruns.db`` resolved against ``/proj`` → ``sqlite:////proj/mlruns.db``.
    Absolute URIs and remote schemes (http, file://, etc.) are returned unchanged.
    """
    root = Path(root)
    if uri.startswith("sqlite:///"):
        db_path = uri[len("sqlite:///"):]
        if not Path(db_path).is_absolute():
            db_path = str((root / db_path).resolve())
        return f"sqlite:///{db_path}"
    # Plain path with no scheme → resolve relative to root.
    if "://" not in uri:
        p = Path(uri)
        if not p.is_absolute():
            p = (root / p).resolve()
        return str(p)
    return uri


def init_tracking(config: dict, section: str = "forecasting") -> str:
    """Resolve and activate the tracking URI for ``config[section]['mlflow']``."""
    root = config.get("_project_root", ".")
    uri = config[section]["mlflow"]["tracking_uri"]
    resolved = resolve_tracking_uri(uri, root)
    mlflow.set_tracking_uri(resolved)
    return resolved


def setup_mlflow(config: dict, section: str = "forecasting") -> str:
    """Activate the tracking URI and select the section's experiment; return its name."""
    init_tracking(config, section)
    experiment_name = config[section]["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    return experiment_name


# ---------------------------------------------------------------------------
# Duplicate-run guard (config hashing)
# ---------------------------------------------------------------------------

def config_hash(section_dict: dict, ignore_keys: tuple[str, ...] = ()) -> str:
    """Stable short hash of an experiment-config section (e.g. ``config['forecasting']``)."""
    relevant = {k: v for k, v in section_dict.items() if k not in ignore_keys}
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_duplicate_run(experiment_name: str, run_hash: str, force: bool = False) -> None:
    """Raise if a FINISHED run with this ``config_hash`` tag already exists."""
    if force:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.config_hash = '{run_hash}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if runs.empty:
        return

    run_id = runs.iloc[0]["run_id"]
    raise RuntimeError(
        f"A FINISHED run already exists for this config "
        f"(config_hash={run_hash}, run_id={run_id}). Pass --force to run anyway."
    )


# ---------------------------------------------------------------------------
# Write helpers (active run required)
# ---------------------------------------------------------------------------

def log_figure(fig: Any, artifact_path: str, filename: str | None = None) -> None:
    """Save a matplotlib figure to a temp PNG, log it as an artifact, and close it."""
    import matplotlib.pyplot as plt

    suffix = ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        fig.savefig(f.name, dpi=100, bbox_inches="tight")
        tmp_path = f.name
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)


def log_dataframe(df: pd.DataFrame, artifact_path: str, filename: str | None = None) -> None:
    """Write a dataframe to a temp CSV and log it as an artifact."""
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)


# ---------------------------------------------------------------------------
# Read / query helpers (consumed by the Streamlit app)
# ---------------------------------------------------------------------------

def list_experiments() -> pd.DataFrame:
    """Return all experiments as a tidy dataframe (name, id, lifecycle stage)."""
    client = MlflowClient()
    rows = [
        {
            "experiment_id": e.experiment_id,
            "name": e.name,
            "lifecycle_stage": e.lifecycle_stage,
            "artifact_location": e.artifact_location,
        }
        for e in client.search_experiments()
    ]
    return pd.DataFrame(rows)


def search_runs(
    experiment_name: str,
    max_results: int = 200,
    order_by: list[str] | None = None,
) -> pd.DataFrame:
    """Search runs in an experiment; returns an empty frame if it doesn't exist."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=order_by or ["attributes.start_time DESC"],
    )


def get_metric_history(run_id: str, key: str) -> pd.DataFrame:
    """Step-by-step history for a single metric (for re-plotting training curves)."""
    client = MlflowClient()
    history = client.get_metric_history(run_id, key)
    rows = [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def list_artifacts(run_id: str, path: str = "") -> pd.DataFrame:
    """List artifacts under ``path`` for a run (name, is_dir, size)."""
    client = MlflowClient()
    rows = [
        {"path": f.path, "is_dir": f.is_dir, "file_size": f.file_size}
        for f in client.list_artifacts(run_id, path)
    ]
    return pd.DataFrame(rows)


def download_artifact(run_id: str, artifact_path: str) -> str:
    """Download a run artifact to a local cache path and return that path."""
    return mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )


def best_run(
    experiment_name: str,
    metric: str,
    mode: str = "min",
) -> pd.Series | None:
    """Return the best run (as a row Series) for ``metric``; None if no runs."""
    runs = search_runs(experiment_name, max_results=1000)
    col = f"metrics.{metric}"
    if runs.empty or col not in runs.columns:
        return None
    runs = runs.dropna(subset=[col])
    if runs.empty:
        return None
    ascending = mode == "min"
    return runs.sort_values(col, ascending=ascending).iloc[0]
