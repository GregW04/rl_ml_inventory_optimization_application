# RL-ML Inventory Optimization

Two-stage inventory optimization for the [VN2 Inventory Planning Challenge](https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn2-inventory-planning-challenge/notebook/37/show):

1. **Demand forecasting** — weekly LightGBM/XGBoost/CatBoost quantile models with calendar, lag, rolling, expanding, and hierarchical (cannibalization) features
2. **RL simulation** — Gymnasium environment where PPO (continuous) and DQN (discrete) agents use multi-horizon forecast uncertainty to decide order quantities

Everything is wired through a **single MLflow store** and a **Streamlit app** so you can run, browse, and inspect the whole workflow from one place.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
configs/config.yaml      # single source of truth
src/forecasting/         # preprocessing, features, validation, model, pipeline
src/simulation/          # Gymnasium env + reward
src/rl/                  # PPO / DQN agents + training
src/utils/               # metrics, config, and tracking.py (all MLflow access)
app/                     # Streamlit frontend (Home + 4 pages)
notebooks/               # EDA and pipeline demos
docs/mlflow.md           # MLflow + tracking-utility guide
data/                    # VN2 competition CSVs
```

## Quick start

### Option A — the app (recommended)

```bash
streamlit run app/Home.py
```

One UI to **trigger runs**, **browse & compare** MLflow runs, view **forecasts** (actual
vs predicted, Q10–Q90 intervals, feature importance), and inspect **inventory/RL** results
(training curves, fill-rate KPIs, per-SKU order recommendations).

### Option B — the CLI

```bash
# Demand forecasting
python -m src.forecasting.pipeline --config configs/config.yaml

# RL training
python -m src.rl.train --config configs/config.yaml --agent ppo
python -m src.rl.train --config configs/config.yaml --agent dqn
```

## MLflow

All runs (forecasting **and** RL) log to one canonical store defined in
`configs/config.yaml`:

```yaml
mlflow:
  tracking_uri: sqlite:///mlruns.db   # resolved to an ABSOLUTE path at runtime
```

The tracking URI is resolved to an absolute path rooted at the project directory by
`src/utils/tracking.py:setup_mlflow`, so **every entry point — CLI, notebook, or app —
writes to the same `mlruns.db` regardless of the working directory**. Browse it with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Experiments: `vn2-demand-forecasting` and `vn2-inventory-rl`. See
[docs/mlflow.md](docs/mlflow.md) for what each run logs and the `tracking.py` API.

## Architecture

- **Frequency**: weekly (ISO week starts, Mondays in the data)
- **OOS masking**: `InStock=False` → `sales=NaN` (censored demand)
- **Validation**: expanding windows with embargo = `max(lag_features)` weeks
- **Forecasts**: Q10/Q50/Q90 per horizon h ∈ {0, 1, 2} → μ = Q50, σ ≈ (Q90−Q10)/2.56
- **RL reward**: `R_t = −0.2·leftover_t − 1.0·shortage_t`

## Data

Place VN2 CSVs in `data/` (included in this repo):

- `Week0-*-Sales.csv` … `Week8-*-Sales.csv` — weekly sales panels
- `Week0-InStock.csv` — in-stock flags
- `Week0-Master.csv` — product/store hierarchy
- `Week0-*-Initial State.csv` — simulation week-0 inventory
