# MLflow & experiment tracking

Most MLflow access goes through [`src/utils/tracking.py`](../src/utils/tracking.py).
The forecasting pipeline, RL trainer, and app use it for setup and querying; some modules
also call `mlflow` directly for run lifecycle. That keeps setup, logging, and querying
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

- **Params**: `measures` (comma-separated ids), `primary_measure`, `primary_metric`,
  `calibrate_quantiles`, `target_transform`, `mean_objective`, `tweedie_variance_power`,
  `demand_profile`, CV settings, `n_features`, `n_skus`, `n_weeks`,
  the RL block dates, `refit_every_weeks`, `n_block_fits`, `ensemble`, `ensemble_members`,
  `champion_model` (`ensemble` when it is on) and tuned `best_*` params (`best_{member}_*` per
  ensemble member).
- **Metrics**: per measure and CV mean `cv_{metric}_{id}`. Quantiles log `pinball` and
  `coverage`, means log `rmse`, `mae` and `bias`. `cv_primary` is the
  primary metric. Per-fold step metrics use the same names. Out-of-sample scores per RL block
  are `oos_{rl_train,rl_val,rl_test}_{metric}_{id}`. Also logged: `calibration_shift_{id}`
  when calibration is on, plus model-selection and tuning scores.
- **Artifacts**:
  - `feature_importance/` (the primary measure's booster);
  - `cv_charts/` and `cv_detail/` (per-fold CSV);
  - `cv_by_entity/`;
  - `forecaster.joblib` and `cv_metrics.json`;
  - `forecasts/`, which contains:
    - `forecast_setup.json`: periods (including `refit_every_weeks` and `block_fits`), timeline, and the measure manifest (`schema: 2`, `measures`, `primary_measure`, `calibration`);
    - `train_forecasts.parquet`: out-of-sample forecasts for the RL blocks, one column per measure id, plus `fit_end` (the last data week of the fit that wrote the row);
    - `oof_forecasts.parquet`.

  RL reads `forecasts/` of the run it is given.
- **Tag**: `config_hash` — a hash of the forecasting config used by the duplicate-run guard.

### `vn2-inventory-rl`

- **Training runs** (`stage=train`, one per seed):
  - **Tags:** `forecast_run_id`, `config_key` (spec hash without the seed, which groups seeds), `seed`, `method`, `agent`, and `role` (`deploy` for the project seed, `seed_check` otherwise).
  - **Params:** the policy spec (`method`, `agent`, `state`, `anchor`, `floor_frac` or `order_cap_mult`, `buckets`, `total_timesteps`, `agent_*` hyperparameters, `training_forecasts`) and the RL block dates.
  - **Metrics:**
    - **Learning curve (every agent; step = training windows replayed; display only):** `insample_vs_anchor_pct` (cost against the anchor on 2,000 fixed training windows, mean action) and `insample_units_vs_anchor_pct` (units against the anchor on the same windows), about 10 points per run, plus `final_insample_vs_anchor_pct`. DQN also logs `exploration_rate` (ε) at these steps.
    - **PPO / GRPO, per iteration (step = iteration):** `train_vs_anchor_pct` and `train_units_vs_anchor_pct` (the sampled batch), `policy_std`, `approx_kl` (k3 estimator of the KL between the policy before and after the update, mean over minibatches), `clip_frac` (share of decisions clipped, mean over minibatches), `grad_norm` (gradient norm before clipping, mean over minibatches) and `kl_to_start` (KL to the frozen starting policy after the update; the anchor for residual). PPO adds `explained_variance` (1 − Var(return − value) / Var(return), critic before the update). GRPO adds `group_spread_eur` (mean within-group std of rollout cost, €) and `zero_spread_share` (groups whose rollouts all cost the same). Runs trained before this was added logged `approx_kl` (PPO) and `kl_to_start` (GRPO) from the last minibatch only.
    - **Validation (one pass on the final weights):** `val_cost_eur_per_sku`, `val_holding_eur_per_sku`, `val_shortage_eur_per_sku`, `val_anchor_cost_eur_per_sku`, `val_minus_anchor_eur`, `beats_anchor`, `val_minus_anchor_pct_{class}` per demand class, and `val_units_vs_anchor_pct` (units ordered against the anchor, %) overall and per class.
  - **Artifacts:** `policy/` (DQN: `model.zip` and `vecnormalize.pkl`; PPO / GRPO: `model.pt`; all: `spec.json`, `val_episodes.parquet` with each episode's demand class and the units the policy and the anchor ordered, and `insample_episodes.parquet`, the same table for the final weights on the learning curve's 2,000 training windows and forecasts; with unbiased training forecasts also `training_bias_factors.csv`, the correction per training fit and class; also kept under `artifacts/rl/runs/<run_id>/`) and `rl_charts/`.
- **Test runs** (`policy_test`, `stage=test`, at most one per forecasting run unless overridden):
  - **Tags:** `forecast_run_id`, `champion` (an RL run id or `anchor`), `champion_config_key` (ending in `+gate` for a gated champion), `champion_gate` (the demand classes a gated champion keeps the policy on, empty otherwise), and `post_hoc` (`true` when the lock was overridden).
  - **Metrics:** `test_cost_eur_per_sku`, `test_anchor_cost_eur_per_sku`, `test_minus_anchor_eur` with `test_delta_ci_lo` / `test_delta_ci_hi`, `test_minus_anchor_pct_{class}`, `test_units_vs_anchor_pct`, `test_vn2_score_eur` / `test_anchor_vn2_score_eur` (official window), and the champion's `val_mean`.
  - **Artifacts:** `test/` (per-episode costs and a comparison chart).

## Duplicate-run guard

When `skip_duplicate_runs: true`, the forecasting pipeline hashes the forecasting config
(excluding `mlflow`/`artifacts` paths) via `tracking.config_hash` and refuses to start if a
**FINISHED** run with the same hash already exists. Override with `--force` (CLI); runs
launched from the app always pass it.

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
- `search_runs(experiment_name, ..., filter_string)` → dataframe of runs (params + metrics).
- `get_metric_history(run_id, key)` → per-step series for a metric.
- `list_artifacts(run_id, path)` / `download_artifact(run_id, artifact_path, dst_path=None)`.
