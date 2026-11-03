"""
Centralized MLflow utilities for the forecasting pipeline, the RL trainer and evaluator, and the
Streamlit app.

* **setup** — resolve the tracking URI (``MLFLOW_TRACKING_URI`` first, e.g. the local tracking
  server), select a stage's experiment from ``config['mlflow']['experiments']``, and create new
  experiments with a pinned artifact location.
* **write helpers** — run provenance (git commit, uncommitted changes, effective config), figures
  and tables under real file names, and the forecasting duplicate-run guard.
* **read helpers** — run search, metric history, and a read-through cache of run artifacts in
  ``config['mlflow']['cache_dir']/<run_id>/``: a finished run's artifacts never change, so a
  cached file is never stale.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Code folders whose uncommitted changes make a run irreproducible from its git commit.
CODE_PATHS = ("src", "app", "configs", "airflow", "Makefile", "pyproject.toml")


# ---------------------------------------------------------------------------
# Tracking URI + experiments
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


def init_tracking(config: dict) -> str:
    """
    Resolve and activate the tracking URI: ``MLFLOW_TRACKING_URI`` when set (the local
    tracking server, ``make mlflow-server``), otherwise ``config['mlflow']['tracking_uri']``.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    resolved = resolve_tracking_uri(uri, config.get("_project_root", "."))
    mlflow.set_tracking_uri(resolved)
    return resolved


def experiment_name(config: dict, stage: str) -> str:
    """Experiment of a stage (``forecasting`` / ``rl``)."""
    return config["mlflow"]["experiments"][stage]


def experiment_artifact_location(name: str) -> str | None:
    """
    Artifact location for a *new* experiment. A tracking server decides itself (proxied
    ``mlflow-artifacts:/``); a SQLite store keeps artifacts next to its database in
    ``mlartifacts/<name>``, never in ``./mlruns`` of whatever directory the process runs in.
    """
    uri = mlflow.get_tracking_uri()
    if uri.startswith("sqlite:///"):
        db = Path(uri[len("sqlite:///"):])
        return (db.parent / "mlartifacts" / name).resolve().as_uri()
    return None


def get_or_create_experiment(name: str) -> str:
    """Id of experiment ``name``, creating it with a pinned artifact location when missing."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return mlflow.create_experiment(name, artifact_location=experiment_artifact_location(name))
    except MlflowException:  # another process created it in the meantime
        return mlflow.get_experiment_by_name(name).experiment_id


def setup_mlflow(config: dict, stage: str) -> str:
    """Activate the tracking URI and select the stage's experiment; return its name."""
    init_tracking(config)
    name = experiment_name(config, stage)
    mlflow.set_experiment(experiment_id=get_or_create_experiment(name))
    return name


# ---------------------------------------------------------------------------
# Provenance + duplicate-run guard (active run required)
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


def git_state() -> tuple[str, bool]:
    """The commit checked out and whether the code folders have uncommitted changes."""
    sha = _git("rev-parse", "HEAD").strip()
    return sha, bool(_git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS).strip())


def log_provenance(config: dict, run_id: str | None = None) -> None:
    """
    Tag a run (the active one by default) with the code it ran and log the effective config.

    MLflow records ``mlflow.source.git.commit`` by itself, but not whether the code folders had
    uncommitted changes. Tags ``git_sha`` and ``git_dirty``; for a dirty tree also logs
    ``provenance/git_status.txt`` and ``provenance/git_diff.patch`` so the run can be rebuilt.
    ``config/effective.yaml`` is the config the run actually used (including app overrides).
    """
    client = MlflowClient()
    run_id = run_id or mlflow.active_run().info.run_id
    sha = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS)
    dirty = bool(status.strip())
    client.set_tag(run_id, "git_sha", sha)
    client.set_tag(run_id, "git_dirty", str(dirty).lower())
    effective = {k: v for k, v in config.items() if not k.startswith("_")}
    client.log_text(run_id, yaml.safe_dump(effective, sort_keys=False), "config/effective.yaml")
    if dirty:
        client.log_text(run_id, status, "provenance/git_status.txt")
        client.log_text(run_id, _git("diff", "HEAD", "--", *CODE_PATHS), "provenance/git_diff.patch")


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

def log_figure(fig: Any, artifact_path: str, filename: str) -> None:
    """Log a matplotlib figure as ``<artifact_path>/<filename>`` and close it."""
    import matplotlib.pyplot as plt

    mlflow.log_figure(fig, f"{artifact_path}/{filename}", save_kwargs={"dpi": 100, "bbox_inches": "tight"})
    plt.close(fig)


def log_dataframe(df: pd.DataFrame, artifact_path: str, filename: str) -> None:
    """Log a dataframe as the CSV ``<artifact_path>/<filename>``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


# ---------------------------------------------------------------------------
# Read / query helpers
# ---------------------------------------------------------------------------

def run_cache_dir(config: dict, run_id: str) -> Path:
    """Local folder mirroring a run's artifacts: ``<mlflow.cache_dir>/<run_id>``."""
    return Path(config["_project_root"]) / config["mlflow"]["cache_dir"] / run_id


def cached_artifact(config: dict, run_id: str, artifact_path: str) -> Path | None:
    """
    Local path of a run's artifact (file or folder), downloaded once into the run's cache folder;
    None if the run has no such artifact. The pipeline and the RL trainer write into the cache
    as they log, so their own runs never download. Downloads land in a temp folder first and
    are moved into place whole, so an interrupted download never looks like a cached file.
    """
    local = run_cache_dir(config, run_id) / artifact_path
    if local.exists():
        return local
    init_tracking(config)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".download-", dir=local.parent))
    try:
        fetched = Path(download_artifact(run_id, artifact_path, dst_path=str(tmp)))
        if not local.exists():
            os.replace(fetched, local)
    except (MlflowException, OSError):
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return local if local.exists() else None


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
    max_results: int = 1000,
    order_by: list[str] | None = None,
    filter_string: str = "",
) -> pd.DataFrame:
    """Search runs in an experiment; returns an empty frame if it doesn't exist."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=order_by or ["attributes.start_time DESC"],
        filter_string=filter_string,
    )


def get_metric_history(run_id: str, key: str) -> pd.DataFrame:
    """Step-by-step history for a single metric (for re-plotting training curves)."""
    client = MlflowClient()
    history = client.get_metric_history(run_id, key)
    rows = [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]
    if not rows:
        return pd.DataFrame(columns=["step", "value", "timestamp"])
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def list_artifacts(run_id: str, path: str = "") -> pd.DataFrame:
    """List artifacts under ``path`` for a run (name, is_dir, size)."""
    client = MlflowClient()
    rows = [
        {"path": f.path, "is_dir": f.is_dir, "file_size": f.file_size}
        for f in client.list_artifacts(run_id, path)
    ]
    return pd.DataFrame(rows)


def download_artifact(run_id: str, artifact_path: str, dst_path: str | None = None) -> str:
    """Download a run artifact (to ``dst_path`` or a temp cache) and return its local path."""
    return mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
    )
