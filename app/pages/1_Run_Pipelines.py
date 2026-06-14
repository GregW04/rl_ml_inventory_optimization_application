"""Trigger the forecasting pipeline or RL training from the UI."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import streamlit as st

from app import lib

st.set_page_config(page_title="Run Pipelines", page_icon="🚀", layout="wide")
st.title("🚀 Run Pipelines")
st.caption("Launch a run as a subprocess; output streams below and logs to MLflow normally.")

cfg = lib.get_config()
fc_tab, rl_tab = st.tabs(["Demand forecasting", "RL training"])

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
with fc_tab:
    with st.form("forecast_form"):
        c1, c2, c3 = st.columns(3)
        forecaster_type = c1.selectbox(
            "Forecaster type", ["mean", "quantile"],
            index=["mean", "quantile"].index(
                cfg["forecasting"].get("forecaster_type", "mean")
            ),
            help="`quantile` adds Q10/Q50/Q90 uncertainty; `mean` is a point forecast.",
        )
        enable_selection = c2.toggle(
            "Model selection",
            value=cfg["forecasting"].get("model_selection", {}).get("enabled", False),
        )
        enable_tuning = c3.toggle(
            "Hyperparameter tuning",
            value=cfg["forecasting"].get("tuning", {}).get("enabled", False),
        )
        n_trials = st.slider(
            "Optuna trials (if tuning)", 5, 100,
            int(cfg["forecasting"].get("tuning", {}).get("n_trials", 50)),
            step=5,
        )
        force = st.checkbox("Force re-run (ignore duplicate-config guard)", value=False)
        submitted = st.form_submit_button("Run forecasting", type="primary")

    if submitted:
        overrides = {
            "forecasting": {
                "forecaster_type": forecaster_type,
                "model_selection": {"enabled": enable_selection},
                "tuning": {"enabled": enable_tuning, "n_trials": int(n_trials)},
            }
        }
        cfg_path = lib.build_override_config(overrides)
        args = ["-m", "src.forecasting.pipeline", "--config", cfg_path]
        if force:
            args.append("--force")
        st.info(f"Running: `python {' '.join(args)}`")
        with st.status("Forecasting pipeline running…", expanded=True) as status:
            lib.run_with_live_log(args)
            status.update(label="Forecasting run finished", state="complete")
        lib.cached_runs.clear()
        st.success("Done. See **Experiments** / **Forecasts** for results.")

# ---------------------------------------------------------------------------
# RL training
# ---------------------------------------------------------------------------
with rl_tab:
    with st.form("rl_form"):
        c1, c2 = st.columns(2)
        agent = c1.selectbox("Agent", ["ppo", "dqn"])
        default_steps = int(cfg["rl"]["agents"][agent].get("total_timesteps", 100_000))
        timesteps = c2.number_input(
            "Total timesteps", min_value=1_000, max_value=2_000_000,
            value=min(default_steps, 50_000), step=1_000,
            help="Lower values train faster (good for a quick smoke run).",
        )
        submitted_rl = st.form_submit_button("Run RL training", type="primary")

    if submitted_rl:
        if not lib.forecasts_path().exists():
            st.warning(
                "No forecasts found — RL uses forecast uncertainty as state. "
                "Run the forecasting pipeline first for best results."
            )
        overrides = {"rl": {"agents": {agent: {"total_timesteps": int(timesteps)}}}}
        cfg_path = lib.build_override_config(overrides)
        args = ["-m", "src.rl.train", "--config", cfg_path, "--agent", agent]
        st.info(f"Running: `python {' '.join(args)}`")
        with st.status(f"Training {agent.upper()}…", expanded=True) as status:
            lib.run_with_live_log(args)
            status.update(label=f"{agent.upper()} training finished", state="complete")
        lib.cached_runs.clear()
        st.success("Done. See **Experiments** / **Inventory & RL** for results.")
