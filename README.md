# RL-ML Inventory Optimization

VN2-aligned inventory optimization ([VN2 Inventory Planning Challenge](https://supchains.com/vn2-inventory-competition/)):

1. **Forecasting** — you choose the measures per horizon (h0 = this week, h1 = next week, h2 = delivery week): means, the median or any quantile, via CatBoost/LightGBM/XGBoost or an ensemble of them, refit every n weeks inside the RL training and validation blocks
2. **Lost-sales anchor** — projects stock under official rules, then orders up to a delivery-week target built from the run's measures (by default the cost-optimal q83)
3. **RL** — PPO or GRPO either corrects the anchor (**residual**) or sets orders from scratch (**raw**), with a state you pick from the run's measures. Trained on train, selected on validation, shipped to test once; costs in **euros (€)**

Everything is wired through a **single MLflow store** and a **Streamlit app** so you can run, browse, and inspect the whole workflow from one place.

## Setup

Dependencies are declared in `pyproject.toml` and pinned exactly in `uv.lock`
(`requirements.txt` is exported from the lock for pip users).

```bash
uv sync --extra rl --extra app --extra dev   # creates .venv from uv.lock
uv run pre-commit install                    # ruff + large-file guard on every commit
uv run pytest -m "not slow"                  # what CI runs (plus lint); -m slow adds the protocol tests
```

The forecasting pipeline needs only the core dependencies; `rl` adds torch, `app` adds Streamlit.
Keep them in separate processes: LightGBM and torch segfault in one process on macOS.

## Project layout

```
configs/config.yaml      # single source of truth
src/forecasting/         # preprocessing, features, validation, model, pipeline
src/simulation/          # VN2 replay engine, lost-sales anchor
src/rl/                  # policy spec, environment and its batched twin, PPO / GRPO training, selection, test scoring
src/serving/             # forecaster and ordering policy as MLflow pyfunc models
src/ops/                 # weekly production loop: drops, data contract, ledgers, monitors, decisions, registry
src/utils/               # metrics, config, and tracking.py (MLflow setup, provenance, artifact cache)
app/                     # Streamlit frontend (Home + 5 pages)
airflow/                 # DAG factory (one DAG per ops environment), its tests and requirements
notebooks/               # EDA and pipeline demos
docs/mlflow.md           # MLflow + tracking-utility guide
docs/ops.md              # the weekly ops loop: cycle, decisions, RL challenger, Airflow
data/                    # VN2 competition CSVs
```

## Quick start

### Option A — the app (recommended)

```bash
make app        # starts the MLflow server (make mlflow-server) and Streamlit against it
```

One UI to **trigger runs**, **browse & compare** MLflow runs, view **forecasts** (actual vs
each measure per horizon, coverage, feature importance), and inspect **inventory/RL** results
(training curves, validation vs the anchor, per-SKU order recommendations).

**Run Forecasting** sets periods, features and the measures to forecast, then launches a run.
**Run RL** picks a finished forecasting run, shows its frozen split and measures, and lets you
choose the method (residual or raw), the state features and the seeds. It then shows the
validation leaderboard and scores the champion once on the test block (locked afterwards).
Runs execute in the background (logs in `artifacts/app_jobs/`), so you can switch pages while
they work. Set `VN2_CONFIG=<path>` to point the app at another base config.

### Option B — the CLI

```bash
# Demand forecasting
python -m src.forecasting.pipeline --config configs/config.yaml

# RL: a policy spec (JSON, see src/rl/spec.py; the app writes one for you) fixes the
# forecasting run, method, agent, state and anchor. One MLflow run per seed.
python -m src.rl.train --config configs/config.yaml --spec spec.json --seeds 42 43 44

# Score the validation champion once on the test block (locked afterwards)
python -m src.rl.evaluate --config configs/config.yaml --forecast-run <run_id>
```

### Option C — the weekly ops loop (production-like)

```bash
python -m src.ops.cli run --env dev     # every step of every week, in order (no orchestrator)

make airflow-install                    # once: Airflow 3.3.2 in .venv-airflow (official constraints)
make airflow-init                       # once: metadata DB and the 1-slot heavy pool
make airflow                            # http://127.0.0.1:8080, one DAG per environment; unpause to replay
```

See [docs/ops.md](docs/ops.md): data contract, monitors, the retrain / keep / rollback rules, the RL
challenger and the DAGs.

## MLflow

All runs (forecasting **and** RL) log to one store defined in `configs/config.yaml`:

```yaml
mlflow:
  tracking_uri: sqlite:///mlruns.db   # resolved to an absolute path; MLFLOW_TRACKING_URI overrides it
  experiments: {forecasting: vn2-demand-forecasting, rl: vn2-inventory-rl}
  cache_dir: artifacts/cache          # local copy of run artifacts, one folder per run id
```

Day to day, a local tracking server sits on top of that SQLite file; the app and CLI jobs reach it through
`MLFLOW_TRACKING_URI`:

```bash
make mlflow-server                                  # http://127.0.0.1:5050 (UI + API), project venv's mlflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050    # for CLI jobs in this shell (also in .env)
make backup                                         # consistent online copy of mlruns.db
```

Every run records its git commit, whether the code had uncommitted changes, and the effective config.
See [docs/mlflow.md](docs/mlflow.md) for what each run logs, the artifact cache, the legacy-run archive
and the `tracking.py` API.

## Architecture

- **Frequency**: weekly (ISO week starts, Mondays in the data)
- **OOS masking**: `InStock=False` → `sales=NaN` (censored demand)
- **Periods**: configure train / validation / test RL blocks and forecaster CV in the app. Forecasts for the RL blocks are out of sample: `periods.refit_every_weeks` (n, default 13; 0 = one fit per block) refits the forecaster every n weeks inside the training and validation blocks, each fit using rows up to three weeks before the weeks it forecasts. The test block keeps one fit made before its first week, so the VN2 window only uses labels known by week 0. The Run Forecasting page previews the fits as lanes and the forecast age at each order decision; `forecast_setup.json` saves them as `block_fits`, and the forecasts parquet records each row's `fit_end`
- **Validation**: expanding-window CV. Before each scored block the training fit stops for an embargo equal to the forecast horizon, so a training label cannot reach into the scored weeks. Backward lags (including 51–53) use earlier sales only and do not widen that embargo. Lag and rolling features use sales through *t−1*; seasonal lags use *t−51* through *t−53*.
- **Demand profile** (`forecasting.features.demand_profile`, off by default): each SKU's profile from the trailing 52 weeks ending at *t−1*: ADI and CV² of non-zero sales, zero share, level, volume percentile on the date, weeks since the last sale, its demand pattern (Syntetos–Boylan–Croston: smooth / erratic / intermittent / lumpy) and its volume tier (`dp_volume_tier`: 0 = the SKUs holding the first 80% of the date's volume, 1 = the next 15%, 2 = the rest). Out-of-stock weeks are skipped, not counted as zero demand; cutoffs are fixed, so nothing is fitted and the features are point-in-time
- **Forecast measures**: `forecasting.measures` lists what to predict per horizon. A mean is a Poisson booster on demand, or a Tweedie one with `forecasting.model.mean_objective: tweedie` (variance ∝ mean^p, `tweedie_variance_power` p in (1, 2), for zero-heavy, overdispersed demand). A quantile is a pinball booster in log1p space, and the quantiles of a horizon are sorted so they never cross. Ids double as column names and RL state features: `mean_h0`, `q50_h1`, `q83_h2`. `primary_measure` (pinball for a quantile, RMSE for a mean) drives model selection and tuning. With `calibrate_quantiles`, each quantile gets a split-conformal shift fitted on the CV region's out-of-fold data. Each run saves its measures in `forecast_setup.json`. With `forecasting.ensemble.enabled`, every fit (CV and RL blocks) trains each member backend (default CatBoost and LightGBM) and averages their forecasts; calibration is applied once after averaging, model selection is skipped and tuning tunes each member
- **Anchor**: `end_k = max(0, on_hand + arrive_k − d_h0)`, `y = max(0, end_k + arrive_k1 − d_h1)`, `order = max(0, round(target_h2 − y))`. `d_h0` and `d_h1` are each horizon's mean (else its median). The target is a quantile at the critical ratio, else the Poisson quantile of the mean
- **RL methods**: residual `order = max(0, round(anchor + f·max(anchor, floor·scale)))`, where f = 0 is the anchor; raw `order = round(f·k·scale)`, f ∈ [0, 1], where scale is the SKU's peak weekly training sales. The state is the inventory pipeline plus any of the run's measures, anchor-derived and context features, and optionally the demand class (one-hot Smooth / Erratic / Intermittent / Lumpy). The spec is saved with each run, so evaluation always rebuilds the same observation layout
- **RL agents**: **PPO** and **GRPO** train on a batched replay of the training windows (`src/rl/batch_env.py`, an exact twin of the evaluation environment). The residual policy starts exactly at the anchor; each iteration replays thousands of windows; PPO's reward is paired with a shadow anchor on the same demand; GRPO (Group Relative Policy Optimization) replays every sampled window `group_size` times and judges each rollout against its group, with a KL penalty to the starting policy. Costs stay in euros. **DQN** (Stable-Baselines3) was retired on 2026-11-02; its runs keep their metrics in MLflow
- **Training forecasts** (`rl.training_forecasts`, chosen on the Run RL page): a forecaster fit can run too low for a whole growth year and too high in the next decline, and an agent can learn that as a fixed "order more" offset. `unbiased` multiplies each training fit's forecasts by (Σ actual + k) ÷ (Σ forecast + k) of its h0/h1 means; `unbiased_by_class` does it per fit and demand class (cells under 500 SKU-weeks use the fit's factor); `as_is` keeps them. Validation and test always use the real forecasts. Every run logs units ordered against the anchor, so a remaining bet on the demand level shows on the leaderboard
- **Forecast drift** (Forecasts page): per forecaster fit, the error of each mean measure ((actual − forecast) ÷ actual, positive = too low), quantile coverage and sales against a year earlier, per demand class too, plus the weekly error with Trigg's smoothed tracking signal (smoothing 0.2, alarm beyond ±0.74, about 5% false alarms on unbiased errors). Test-block forecast diagnostics stay hidden until that run's test block is scored
- **RL protocol**: each seed is fit on the `rl_train` windows for its full budget (PPO / GRPO iterations), and its final weights are kept. A learning curve (cost against the anchor on 2,000 fixed training windows) is logged for display only. It is scored once on `rl_val` next to the anchor. The leaderboard ranks configurations by mean validation cost over seeds, with a paired bootstrap CI vs the anchor and the change per demand class; the anchor is a candidate too. A configuration that beats the anchor on some demand classes and not others also gets a gated candidate ("RL on …"): the policy on the classes it won on validation, summed over seeds, and the anchor elsewhere. That gate is fit on validation, so its validation score is optimistic and the test score is the honest check. The champion ships its project-seed weights. `rl_test` is scored once per forecasting run; a second score needs an explicit override and is tagged `post_hoc`
- **Evaluation (€)**: VN2-shaped 8-week windows, 6 orders, cost of weeks 3–8 (0.2 €/unit left, 1 € per lost sale). The official window (2024-04-15 … 2024-06-03) from the official start state gives the VN2 score; the app shows a compliance badge
- **Citation**: Vandeput, N. (2025), *VN2 Inventory Planning Challenge*, https://supchains.com/vn2-inventory-competition/

## Data

Place VN2 CSVs in `data/` (included in this repo):

- `Week0-*-Sales.csv` … `Week8-*-Sales.csv` — weekly sales panels
- `Week0-InStock.csv` — in-stock flags
- `Week0-Master.csv` — product/store hierarchy
- `Week0-*-Initial State.csv` — simulation week-0 inventory
