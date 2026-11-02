"""RL-ML Inventory Optimization — Streamlit control panel (entry point)."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import pandas as pd
import streamlit as st

from app import lib
from src.rl.selection import champion
from src.utils import tracking

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide")

st.title("📦 RL-ML Inventory Optimization")
st.caption(
    "Two-stage VN2 workflow: forecast the measures you choose (refit every n weeks, one "
    "model or an ensemble) → lost-sales anchor → RL (PPO, GRPO or the DQN baseline; residual "
    "on the anchor, or raw) trained on train, selected on validation, shipped to test once. "
    "Costs are reported in euros (€)."
)
ok, msgs = lib.vn2_compliance_badge()
st.markdown(
    f"**VN2 comparability:** {'✅' if ok else '⚠️'} " + "; ".join(msgs)
)

uri = lib.init_tracking()
exps = lib.experiment_names()

st.subheader("Tracking store")
st.code(uri, language="text")

experiments = tracking.list_experiments()
if experiments.empty:
    st.info("No MLflow experiments yet. Use **Run Forecasting** to create the first run.")
else:
    rows = []
    for _, e in experiments.iterrows():
        runs = tracking.search_runs(e["name"])
        rows.append({"experiment": e["name"], "runs": 0 if runs.empty else len(runs)})
    st.dataframe(rows, hide_index=True, width="stretch")

st.subheader("Forecasting runs and their champions")
fc_runs = lib.forecast_runs()
if fc_runs.empty:
    st.write("— no finished forecasting runs yet —")
else:
    rows = []
    for _, r in fc_runs.head(8).iterrows():
        board = lib.rl_leaderboard(r["run_id"])
        champ = champion(board)
        tests = lib.test_scores(r["run_id"])
        rows.append({
            "forecasting run": lib.forecast_run_label(r),
            "measures": r.get("params.measures") or "(legacy)",
            "RL configurations": int(((board["kind"] == "rl") & board["gate"].isna()).sum()) if not board.empty else 0,
            "validation champion": "anchor" if champ is None or champ["kind"] == "anchor" else champ["candidate"],
            "test (production)": (
                f"{tests.iloc[0]['metrics.test_cost_eur_per_sku']:.3f} € / SKU-window"
                + (" · post-hoc" if tests.iloc[0].get("tags.post_hoc") == "true" else "")
                if not tests.empty else "not scored"
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()
st.markdown(
    "Use the pages in the sidebar:\n"
    "1. **Run Forecasting** — set periods, refit cadence, features, measures and the ensemble, preview the folds, launch\n"
    "2. **Run RL** — pick a forecasting run, choose state and method, train, compare on validation, score test once\n"
    "3. **Experiments** — browse & compare MLflow runs\n"
    "4. **Forecasts** — actual vs forecast per horizon, coverage, feature importance\n"
    "5. **Inventory & RL** — training sample → validation for one or more configurations, where a policy wins and "
    "loses, and order recommendations with the reason"
)
