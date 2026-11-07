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
- The server's job runner is off (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`). Its 7 huey workers (about
  1.2 GB) only serve GenAI features such as trace evaluation and scorers, which this project doesn't use.
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

## Model Registry and serving

The production loop ([ops.md](ops.md)) keeps two registered models per environment
(`src/ops/registry.py`), `<env>-vn2-forecaster` and `<env>-vn2-policy`. Three aliases have fixed meanings:

- `champion` serves. Consumers resolve `models:/<name>@champion` at runtime, so moving the alias *is* the
  deployment.
- `challenger` is under validation.
- `previous` is the champion before the last promotion. `promote` sets it before moving `champion`, and
  `rollback` restores it.

Model version tags hold the audit trail (trigger, recipe key, data digest, validation status).

**The forecaster as an MLflow model** (`src/serving/forecaster.py`, `log_forecaster`):

- **Logged from code.** `python_model=src/serving/forecaster_model.py`, with the project's `src/` as
  `code_paths`. Nothing is unpickled at load time except the fitted boosters.
- **Artifacts:**
  - `forecaster.joblib` (single backend or ensemble, read by `forecaster.load_forecaster`);
  - `master.csv` (the hierarchy);
  - `meta.json` (the recipe, its key, the as-of week, the data digest and the serving config).
- **Environment:** pip requirements pinned to the serving packages only, with no torch. MLflow also copies
  `uv.lock` and `pyproject.toml` into the model.
- **Contract:** the input is the weekly history (`store`, `product`, `date`, `sales`, `in_stock`) up to the
  as-of week, and the param `as_of` defaults to the latest week in the input. The output is one row per SKU for
  the decision week `as_of + 1 week`, one column per measure. The LoggedModel's params are `recipe_key`,
  `model_name`, `as_of` and `data_digest`.

```python
model = mlflow.pyfunc.load_model("models:/live-vn2-forecaster@champion")  # in a torch-free process
forecasts = model.predict(history, params={"as_of": "2024-04-01"})
```

**The policy as an MLflow model** (`src/serving/policy.py`, `log_policy`):

- **Logged from code**: `python_model=src/serving/policy_model.py`, with `src/` as `code_paths`.
- **Two kinds:**
  - `anchor`: the lost-sales anchor. Version 1 of every policy model, the fallback.
  - `ppo`: a PPO actor plus an optional gate of demand classes it acts on; the anchor orders for the rest.
- **Artifacts:**
  - `policy.json`: kind, VN2 rules, anchor spec, gate, and for PPO the policy spec;
  - for PPO, `actor.npz` (the actor's weights as numpy arrays) and `scales.csv` (the demand scale per SKU).
- **No torch at serving.** The actor runs in numpy, and the observation and order come from the same
  functions the training environment uses (`src/rl/observation.py`). The pip requirements are mlflow,
  pandas, numpy and PyYAML.
- **Contract.** The input has one row per SKU for the decision week:
  - `store`, `product`;
  - stock: `on_hand`, `transit_1` (arrives this week), `transit_2` (arrives next week);
  - the decision week's forecasts;
  - context: `zero_share`, `pos_cv`, `dp_sbc_class`.

  The output is `order`, `anchor_order`, `adjustment` (order − anchor) and `rl` (whether the learned policy
  decided). Missing forecasts raise instead of ordering 0.

```python
policy = mlflow.pyfunc.load_model("models:/live-vn2-policy@champion")
orders = policy.predict(decision_frame)   # src.ops.rl.decision_frame(state, forecasts, context)
```

**Recipe and refit** (`src/forecasting/recipe.py`):

- A recipe covers features, measures, the backend or ensemble, its hyperparameters, the objective and the
  transform. Its `key()` is the same for every refit of that recipe; refits differ by `panel_digest`.
- A refit at `as_of` trains on rows up to `as_of − 2` weeks, so their h2 label is known. That is the same
  cut the research block fits use.
- The inference frame appends the decision week to the panel.
- `tests/test_feature_pit.py` and `tests/test_recipe.py` check that a week's features never depend on where
  the panel ends, so training and serving rows are identical.

**Two fixes from 2026-11-02 that affect the research pipeline:**

- **Holiday distances** now come from the event calendar instead of the weeks present in the panel. Before
  the fix, a panel ending shortly before an event served "no event in sight" (Easter 2023: 4 instead of 2).
  On the full research panel nothing changes (0 of 98,835 rows).
- **LightGBM bagging is now active.** `subsample` was set without `subsample_freq`, so it never sampled rows.
  LightGBM results of forecasting runs made after the fix differ from earlier ones.

## What each run logs

### `vn2-demand-forecasting` (one run per pipeline execution)

- **Params**: `measures` (comma-separated ids), `primary_measure`, `primary_metric`,
  `calibrate_quantiles`, `target_transform`, `mean_objective`, `tweedie_variance_power`,
  `demand_profile`, CV settings, `n_features`, `n_skus`, `n_weeks`,
  the RL block dates, `refit_every_weeks`, `n_block_fits`, `ensemble`, `ensemble_members`,
  `champion_model` (`ensemble` when it is on) and tuned `best_*` params (`best_{member}_*` per
  ensemble member). With drift monitoring on, also `drift_rule` and `drift_level_measure`.
- **Metrics**: per measure and CV mean `cv_{metric}_{id}`. Quantiles log `pinball` and
  `coverage`, means log `rmse`, `mae` and `bias`. `cv_primary` is the
  primary metric. Per-fold step metrics use the same names. Out-of-sample scores per RL block
  are `oos_{rl_train,rl_val,rl_test}_{metric}_{id}`. The app hides `oos_rl_test_*` until the run's test
  block has been scored. Also logged: `calibration_shift_{id}` when calibration is on, plus model-selection
  and tuning scores. With drift monitoring on: `drift_alert_weeks_{rl_train,rl_val}`, the weeks in which at
  least one demand class alerted (no test-block count). With drift-triggered refits on: `n_block_fits_actual`,
  `n_drift_refits`, `n_retunes` (challenges run, skipped ones not counted) and `n_retunes_adopted`.
  - **Block-end rule (runs tagged `scores_within_block=true`):** a block's scores count only targets inside
    the block. Earlier runs scored the last one or two validation weeks of their h1/h2 measures on test-week
    sales (25% of the h2 validation targets with an 8-week block); their numbers are left as logged.
- **Artifacts**:
  - `feature_importance/feature_importance.{csv,png}` (the primary measure's booster);
  - `cv_charts/cv_<metric>_per_fold.png`, `cv_charts/oof_sample.png`, `cv_detail/fold_metrics.csv`;
  - `cv_by_entity/cv_by_entity.csv`;
  - `forecaster.joblib` (the test-block fit), `cv_metrics.json`, `warnings.json` (period warnings, if any);
  - `forecasts/`, which contains:
    - `forecast_setup.json`: periods (including `refit_every_weeks` and `block_fits`: the fits actually made, each with
      its `trigger`, `start` for the block's first fit, `schedule` for the age rule or `drift` for an alert, the
      alerting `classes` and `retuned`), timeline, and the measure manifest (`schema: 2`, `measures`,
      `primary_measure`, `calibration`);
    - `train_forecasts.parquet`: out-of-sample forecasts for the RL blocks, one column per measure id, plus `fit_end` (the last data week of the fit that wrote the row);
    - `oof_forecasts.parquet`;
    - `retunes.json` (drift-triggered refits on): the retune challenges the run made (alert week, block, alerting
      classes, holdout scores of the current and the new settings, gain %, adopted), with the CV settings they
      tuned with (`folds` × `fold_weeks`), the last row the tuning saw (`tuned_to`) and the holdout (`holdout_start`,
      `holdout_end`); `skipped` and a `note` when the run's CV folds did not all fit before the holdout (no tuning);
      `[]` when none;
    - `drift_alerts.parquet` (drift monitoring on): one row per fit × week × demand class with the weekly
      error, the tracking signal, the weeks in a row past the limit, `alert` and `severe`. It is the
      production loop's bias alarm (thresholds from `configs/ops.yaml`), restarted at each refit, with no
      cost-share filter. Every block is in the file; the app hides the test block until it has been scored.
      Alerts never refit: refits follow `refit_every_weeks`.

  RL reads `forecasts/` of the run it is given.

  Runs made before 2026-11-02 have random `tmpXXXX` file names in the chart and table folders.
- **Tags**:
  - `config_hash`: a hash of the forecasting config, used by the duplicate-run guard. The
    `monitoring.drift_alerts` switch is outside the hashed sections: it changes no forecast. Drift-triggered
    refits do change the forecasts, so `monitoring.drift_refits` and its options join the hash when it is on;
  - `drift_monitoring`, `drift_refits` (`true` or `false`) and `scores_within_block` (see Metrics);
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

### `vn2-ops-<env>` (the weekly loop, [ops.md](ops.md))

- **`<env>-timeline`** (`stage=timeline`): the container of the environment's cycles. Its metrics are the
  cycles' metrics with step = week index, so the MLflow chart view shows them as time series.
- **`cycle-<D>`** (`stage=cycle`, child of the timeline), one per week:
  - **Tags:**
    - `as_of`, `env`, `rules_version` (hash of the effective rules), `strategy`;
    - the decisions: `decision`, `decision_reasons`, `policy_decision`, `policy_reasons`;
    - the drift alert (under any strategy; it never acts under `fixed`): `drift_alarm` (`true` or `false`) and
      `drift_alarm_reasons`;
    - what served: `champion_version`, `previous_version`, `policy_version`, `policy_kind`;
    - the provenance tags.
  - **Metrics:**
    - per trajectory: `<trajectory>_{cost,holding,shortage}_eur_per_sku` and `<trajectory>_censored_share`;
    - `live_units_ordered`, `rl_share_of_skus` and `champion_age_weeks`;
    - the monitors: `bias_ts_<class>`, `coverage_q83_<class>`, `drift_alarm` (0 or 1), `probation_delta_pct`
      and `diag_*`.
  - **Input:** the week's drop (context `weekly_drop`).
  - **Artifacts:** `cycle/`, a copy of the cycle folder.
- **Children of a cycle run:**
  - `bootstrap-<D>` / `refit-<D>` / `retune-<D>`: the training data (input, context `training`), the
    forecaster and `fit_seconds` (the fit itself; recorded from 2026-11-03);
  - `policy-bootstrap-<D>`: the anchor policy;
  - on a retrain, one `rl-train` run per seed and a `policy-<D>` run:
    - the `rl-train` runs follow the training-run layout above, with `forecast_run_id=ledger:<env>`;
    - `policy-<D>` logs `val_vs_anchor_pct` (also per seed, `val_vs_anchor_pct_s<seed>`),
      `seeds_beating_anchor` and the policy model.
- **`challenge-<recipe>-<D>`** (`stage=challenge`): an operator's holdout test of a new recipe. It logs the paired €
  (`delta_pct`, `ci_lo`, `ci_hi`, `evidence_weeks`), the `validation_status` tag and `recipe.json`.

### `vn2-ops-recipes` and `vn2-ops-calibration` (setting the loop up)

- **`tune-<recipe>-<D>`** (`stage=tune`, `cli tune`):
  - params: the CV design, trials, metric and data digest;
  - metrics per backend: `tuned_<metric>_<backend>`, `trials_<backend>`, `minutes_<backend>`;
  - the `recipe_key` tag;
  - artifacts: `recipe.json` and `model_params.yaml` (pasted into `configs/ops.yaml`).
- **`calibration-<env>`** (`cli calibrate`): `calibration/` holds the threshold grids (`bias.csv`, `coverage.csv`),
  `coverage_profile.csv`, the cost shares and the weekly class series they came from.
- **`strategies`** (`cli compare`): `calibration/strategies.csv`, the paired € of each refit strategy against the
  reference with its CI, refits and rollbacks.

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
