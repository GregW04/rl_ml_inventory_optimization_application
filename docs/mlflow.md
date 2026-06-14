# MLflow & experiment tracking

All MLflow access in this project goes through one module — [`src/utils/tracking.py`](../src/utils/tracking.py).
The forecasting pipeline, the RL trainer, the notebooks, and the Streamlit app all import
it; **nothing else calls `mlflow` directly**. That keeps setup, logging, and querying
consistent and gives the frontend a clean read API.

## The single store

`configs/config.yaml` defines the tracking URI for each stage:

```yaml
forecasting:
  mlflow:
    experiment_name: vn2-demand-forecasting
    tracking_uri: sqlite:///mlruns.db
    skip_duplicate_runs: true
rl:
  mlflow:
    experiment_name: vn2-inventory-rl
    tracking_uri: sqlite:///mlruns.db
```

Both stages share one SQLite backend (`mlruns.db`) with artifacts under `mlruns/`.

> **Why runs always land in the same place.** `sqlite:///mlruns.db` is a *relative* URI.
> `tracking.setup_mlflow` resolves it to an absolute path rooted at the project directory
> before calling `mlflow.set_tracking_uri`, so running from `notebooks/`, the repo root, or
> the app all write to the **same** `mlruns.db`. (A stray `notebooks/mlruns.db` from before
> this fix was removed during cleanup.)

Browse the store:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## What each run logs

### `vn2-demand-forecasting` (one run per pipeline execution)

- **Params**: `forecaster_type`, `primary_metric`, CV settings, `horizons`, `n_features`,
  `n_skus`, `n_weeks`, `champion_model`, tuned `best_*` params.
- **Metrics**: `cv_{mae,smape,mase,wql}_mean`, per-horizon `cv_*_h{0,1,2}_mean`, per-fold
  step metrics (`mae_h0`, …), full-data `train_*_h{h}`, model-selection / tuning scores.
- **Artifacts**: `feature_importance/` (CSV + chart), `cv_charts/` (CV MAE, OOF sample),
  `cv_detail/` (per-fold CSV), the saved forecaster `.joblib`, and `cv_metrics.json`.
- **Tag**: `config_hash` — a hash of the forecasting config used by the duplicate-run guard.

### `vn2-inventory-rl` (one run per agent training)

- **Params**: `agent`, cost/lead-time settings, `total_timesteps`, `n_skus`, learning rate,
  and DQN `order_buckets` / `buffer_size`.
- **Metrics**: per-episode `ep_reward_mean`, `fill_rate_mean`, `shortage_rate_mean`,
  `avg_leftover_mean` (step = episode), plus `final_*` summaries.
- **Artifacts**: `rl_charts/` (reward + fill-rate curves) and the saved agent `.zip`.

## Duplicate-run guard

When `skip_duplicate_runs: true`, the forecasting pipeline hashes the forecasting config
(excluding `mlflow`/`artifacts` paths) via `tracking.config_hash` and refuses to start if a
**FINISHED** run with the same hash already exists. Override with `--force` (CLI) or the
"Force re-run" checkbox in the app.

## `tracking.py` API

**Setup / write**
- `setup_mlflow(config, section)` — resolve URI (absolute) + select the experiment.
- `init_tracking(config, section)` — resolve + activate URI only (used by the app).
- `config_hash(section_dict, ignore_keys)` — stable short hash of a config section.
- `check_duplicate_run(experiment_name, run_hash, force)` — raise on duplicate FINISHED run.
- `log_figure(fig, artifact_path)` / `log_dataframe(df, artifact_path)` — log a matplotlib
  figure / dataframe without tempfile boilerplate.

**Read / query (used by the Streamlit app)**
- `list_experiments()` → dataframe of experiments.
- `search_runs(experiment_name, ...)` → dataframe of runs (params + metrics).
- `get_metric_history(run_id, key)` → per-step series for a metric.
- `list_artifacts(run_id, path)` / `download_artifact(run_id, artifact_path)`.
- `best_run(experiment_name, metric, mode)` → best run row by a metric.
