# MLflow & experiment tracking

Most MLflow access goes through [`src/utils/tracking.py`](../src/utils/tracking.py).
The forecasting pipeline, RL trainer, and app use it for setup, logging and querying; some modules
also call `mlflow` directly for run lifecycle.

## The store and the tracking server

One store serves every stage (`configs/config.yaml`):

```yaml
mlflow:
  tracking_uri: sqlite:///mlruns.db
  experiments:
    forecasting: vn2-demand-forecasting
    rl: vn2-inventory-rl
  cache_dir: artifacts/cache
```

`MLFLOW_TRACKING_URI` overrides `tracking_uri` (`tracking.init_tracking`). Day to day, the app and every
job talk to a **local tracking server** on top of the same SQLite file:

```bash
make mlflow-server   # .venv/bin/mlflow server on http://127.0.0.1:5050 (log: artifacts/logs/)
make app             # Streamlit on :8501 with MLFLOW_TRACKING_URI=http://127.0.0.1:5050
make backup          # consistent online copy of mlruns.db into backups/
```

- Port 5050, because macOS's AirPlay Receiver holds 5000.
- Always the venv's `mlflow` (`mlflow==3.14.0`): another MLflow version migrates the SQLite schema.
- `mlruns.db` runs in WAL mode, so readers don't block the writer. Back it up with `make backup`
  (SQLite's `.backup`), never with `cp`. A plain `sqlite3 -readonly` session needs the `mlruns.db-shm` file,
  which exists while the server runs.
- Artifact locations: experiments 1 and 2 keep their local folders `mlruns/<id>/`. Experiments created
  through the server store artifacts under `mlartifacts/` (proxied `mlflow-artifacts:/`). Created directly on
  a SQLite store, they go next to the database in `mlartifacts/<name>` (`tracking.get_or_create_experiment`),
  never into `./mlruns` of whatever folder a process runs in.

## Artifact cache

`artifacts/cache/<run_id>/` mirrors a run's artifacts: `forecasts/` for forecasting runs and `policy/` for
RL runs.

- The pipeline and the RL trainer write their files there and log them from there, so their own runs never
  download.
- Other runs are downloaded once by `tracking.cached_artifact`. A finished run's artifacts never change, so a
  cached file is never stale. Delete a run's folder to free space.
- "The latest forecasting run" is the newest finished forecasting run in MLflow
  (`periods.latest_forecast_run_id`). It is no longer a local file that the next run overwrites.

## Run provenance

Every forecasting, RL training and test run (`tracking.log_provenance`) records:

- **Tags** `git_sha` and `git_dirty` (uncommitted changes under `src/`, `app/`, `configs/`, `airflow/`,
  `Makefile` or `pyproject.toml`). MLflow's own `mlflow.source.git.commit` doesn't say whether the tree was
  dirty.
- **Artifact** `config/effective.yaml`: the config the run actually used, including the app's overrides.
- **For a dirty tree**: `provenance/git_status.txt` and `provenance/git_diff.patch`.

## What each run logs

### `vn2-demand-forecasting` (one run per pipeline execution)

- **Params**: `measures` (comma-separated ids), `primary_measure`, `primary_metric`,
  `calibrate_quantiles`, `target_transform`, `mean_objective`, `tweedie_variance_power`,
  `demand_profile`, CV settings, `n_features`, `n_skus`, `n_weeks`,
  the RL block dates, `refit_every_weeks`, `n_block_fits`, `ensemble`, `ensemble_members`,
  `champion_model` (`ensemble` when it is on) and tuned `best_*` params (`best_{member}_*` per
  ensemble member).
- **Metrics**: per measure and CV mean `cv_{metric}_{id}`. Quantiles log `pinball` and
  `coverage`, means log `rmse`, `mae` and `bias`. `cv_primary` is the
  primary metric. Per-fold step metrics use the same names. Out-of-sample scores per RL block
  are `oos_{rl_train,rl_val,rl_test}_{metric}_{id}`. The app hides `oos_rl_test_*` until the run's test
  block has been scored. Also logged: `calibration_shift_{id}` when calibration is on, plus model-selection
  and tuning scores.
- **Artifacts**:
  - `feature_importance/feature_importance.{csv,png}` (the primary measure's booster);
  - `cv_charts/cv_<metric>_per_fold.png`, `cv_charts/oof_sample.png`, `cv_detail/fold_metrics.csv`;
  - `cv_by_entity/cv_by_entity.csv`;
  - `forecaster.joblib` (the test-block fit), `cv_metrics.json`, `warnings.json` (period warnings, if any);
  - `forecasts/`, which contains:
    - `forecast_setup.json`: periods (including `refit_every_weeks` and `block_fits`), timeline, and the measure manifest (`schema: 2`, `measures`, `primary_measure`, `calibration`);
    - `train_forecasts.parquet`: out-of-sample forecasts for the RL blocks, one column per measure id, plus `fit_end` (the last data week of the fit that wrote the row);
    - `oof_forecasts.parquet`.

  RL reads `forecasts/` of the run it is given.

  Runs made before 2026-11-02 have random `tmpXXXX` file names in the chart and table folders.
- **Tags**:
  - `config_hash`: a hash of the forecasting config, used by the duplicate-run guard;
  - the provenance tags above;
  - `legacy_test_evaluations` on the three runs whose test block was evaluated before the test lock
    existed (see Legacy runs).

### `vn2-inventory-rl`

- **Training runs** (`stage=train`, one per seed):
  - **Tags:** `forecast_run_id`, `config_key` (spec hash without the seed, which groups seeds), `seed`, `method`, `agent`, `role` (`deploy` for the project seed, `seed_check` otherwise) and the provenance tags.
  - **Params:** the policy spec (`method`, `agent`, `state`, `anchor`, `floor_frac` or `order_cap_mult`, `total_timesteps`, `agent_*` hyperparameters, `training_forecasts`) and the RL block dates.
  - **Metrics:**
    - **Learning curve (step = training windows replayed; display only):** `insample_vs_anchor_pct` (cost against the anchor on 2,000 fixed training windows, mean action) and `insample_units_vs_anchor_pct` (units against the anchor on the same windows), about 10 points per run, plus `final_insample_vs_anchor_pct`.
    - **PPO / GRPO, per iteration (step = iteration):** `train_vs_anchor_pct` and `train_units_vs_anchor_pct` (the sampled batch), `policy_std`, `approx_kl` (k3 estimator of the KL between the policy before and after the update, mean over minibatches), `clip_frac` (share of decisions clipped, mean over minibatches), `grad_norm` (gradient norm before clipping, mean over minibatches) and `kl_to_start` (KL to the frozen starting policy after the update; the anchor for residual). PPO adds `explained_variance` (1 − Var(return − value) / Var(return), critic before the update). GRPO adds `group_spread_eur` (mean within-group std of rollout cost, €) and `zero_spread_share` (groups whose rollouts all cost the same). Runs trained before this was added logged `approx_kl` (PPO) and `kl_to_start` (GRPO) from the last minibatch only.
    - **Validation (one pass on the final weights):** `val_cost_eur_per_sku`, `val_holding_eur_per_sku`, `val_shortage_eur_per_sku`, `val_anchor_cost_eur_per_sku`, `val_minus_anchor_eur`, `beats_anchor`, `val_minus_anchor_pct_{class}` per demand class, and `val_units_vs_anchor_pct` (units ordered against the anchor, %) overall and per class.

      `cost` counts weeks 3–8 of each window (the VN2 score), while `holding` and `shortage` cover all 8 weeks, so they don't add up to `cost`.
  - **Artifacts:** `policy/` and `rl_charts/insample_curve.png`. A local copy of `policy/` sits in `artifacts/cache/<run_id>/policy/`.
    - `policy/` holds `model.pt`, `spec.json` and `val_episodes.parquet` (each episode's demand class and the units the policy and the anchor ordered).
    - It also holds `insample_episodes.parquet`, the same table for the final weights on the learning curve's 2,000 training windows and forecasts.
    - With unbiased training forecasts, it adds `training_bias_factors.csv`, the correction per training fit and class.
  - **DQN** (Stable-Baselines3) was retired on 2026-11-02. Its runs keep their metrics and episodes and stay on the leaderboard, but their `model.zip` can no longer be loaded.
- **Test runs** (`policy_test`, `stage=test`, at most one per forecasting run unless overridden):
  - **Tags:**
    - `forecast_run_id`;
    - `champion` (an RL run id or `anchor`);
    - `champion_config_key` (ending in `+gate` for a gated champion);
    - `champion_gate` (the demand classes a gated champion keeps the policy on, empty otherwise);
    - `post_hoc` (`true` when the lock was overridden or the forecasting run has `legacy_test_evaluations`);
    - the provenance tags.
  - **Metrics:** `test_cost_eur_per_sku`, `test_anchor_cost_eur_per_sku`, `test_minus_anchor_eur` with `test_delta_ci_lo` / `test_delta_ci_hi`, `test_minus_anchor_pct_{class}`, `test_units_vs_anchor_pct`, `test_vn2_score_eur` / `test_anchor_vn2_score_eur` (official window), and the champion's `val_mean`.
  - **Artifacts:** `test/test_episodes.parquet` and `test/test_vs_anchor.png`.

## Duplicate-run guard

When `forecasting.skip_duplicate_runs: true`, the forecasting pipeline hashes the setup (periods, the
forecasting config, the sales file and initial-state date) via `periods.current_setup_hash`, and refuses to
start if a **FINISHED** run with the same hash already exists. Override with `--force` (CLI); runs launched
from the app always pass it.

## Legacy runs (maintenance on 2026-11-02)

- **Purged from the live store:** the 77 stage-less RL runs from before the current protocol
  (`inventory_dqn`, `inventory_ppo`, `policy_evaluation`, `smoketest_ckpt`) and 27 runs that were already
  soft-deleted (`mlflow gc`, then VACUUM: 316 → 171 MB).
- **Archive:** a full copy of the store from before the purge is in `archive/mlruns-archive-2026-11-02.db`,
  and the legacy runs' artifacts are in `archive/mlruns_artifacts/`. Browse it with
  `.venv/bin/mlflow ui --backend-store-uri sqlite:///archive/mlruns-archive-2026-11-02.db --port 5001`.
- **What carried over:** the `policy_evaluation` runs had evaluated the test block of three forecasting runs
  before the test lock existed. That count lives on as the `legacy_test_evaluations` tag of those forecasting
  runs: `78a20635` (10), `c93d7b19` (2) and `1dd6e16a` (1). A later test score on them is tagged `post_hoc`.

## `tracking.py` API

**Setup**
- `init_tracking(config)`: activate `MLFLOW_TRACKING_URI`, else `mlflow.tracking_uri` (made absolute).
- `experiment_name(config, stage)`: the experiment of `forecasting` or `rl`.
- `setup_mlflow(config, stage)`: activate the URI and select the stage's experiment, creating it when missing
  (`get_or_create_experiment`, with a pinned artifact location).

**Write (active run required)**
- `log_provenance(config)`: git tags, effective config, and the diff for a dirty tree.
- `log_figure(fig, artifact_path, filename)` / `log_dataframe(df, artifact_path, filename)`: log a
  matplotlib figure / a CSV under a real file name.
- `config_hash(section_dict, ignore_keys)` / `check_duplicate_run(experiment_name, run_hash, force)`: the
  duplicate-run guard.

**Read / query**
- `cached_artifact(config, run_id, artifact_path)` / `run_cache_dir(config, run_id)`: the artifact cache.
- `list_experiments()`: a dataframe of experiments.
- `search_runs(experiment_name, ..., filter_string)`: a dataframe of runs (params + metrics); up to 1,000
  runs by default.
- `get_metric_history(run_id, key)`: the per-step series of a metric.
- `list_artifacts(run_id, path)` / `download_artifact(run_id, artifact_path, dst_path=None)`.
