"""RL-ML Inventory Optimization — Streamlit control panel (entry point)."""

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
from src.utils import tracking

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

st.title("📦 RL-ML Inventory Optimization")
st.caption(
    "Two-stage VN2 workflow: demand forecasting → RL ordering. "
    "Trigger runs, browse MLflow experiments, and inspect forecasts & inventory policy — "
    "all backed by a single MLflow store."
)

uri = lib.init_tracking()
exps = lib.experiment_names()

st.subheader("Tracking store")
st.code(uri, language="text")

experiments = tracking.list_experiments()
if experiments.empty:
    st.info("No MLflow experiments yet. Use **Run Pipelines** to create the first run.")
else:
    rows = []
    for _, e in experiments.iterrows():
        runs = tracking.search_runs(e["name"])
        rows.append({"experiment": e["name"], "runs": 0 if runs.empty else len(runs)})
    st.dataframe(rows, hide_index=True, use_container_width=True)

st.subheader("Latest champions")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Forecasting** · `{exps['forecasting']}`")
    best = tracking.best_run(exps["forecasting"], "cv_mae_mean", mode="min")
    if best is None:
        st.write("— no scored runs yet —")
    else:
        st.metric("Best CV MAE (mean)", f"{best['metrics.cv_mae_mean']:.4f}")
        champ = best.get("params.champion_model", "?")
        st.caption(f"champion model: `{champ}` · run `{best['run_id'][:8]}`")

with col2:
    st.markdown(f"**Inventory RL** · `{exps['rl']}`")
    best_rl = tracking.best_run(exps["rl"], "final_ep_reward_mean", mode="max")
    if best_rl is None:
        st.write("— no RL runs yet —")
    else:
        st.metric("Best episode reward (mean)", f"{best_rl['metrics.final_ep_reward_mean']:.3f}")
        fr = best_rl.get("metrics.final_fill_rate")
        agent = best_rl.get("params.agent", "?")
        st.caption(
            f"agent: `{agent}` · fill rate: "
            f"{fr:.2%}" if fr is not None else f"agent: `{agent}`"
        )

st.divider()
st.markdown(
    "Use the pages in the sidebar:\n"
    "1. **Run Pipelines** — launch forecasting / RL training with config overrides\n"
    "2. **Experiments** — browse & compare MLflow runs\n"
    "3. **Forecasts** — actual vs predicted, prediction intervals, feature importance\n"
    "4. **Inventory & RL** — training curves, fill-rate KPIs, order recommendations"
)
