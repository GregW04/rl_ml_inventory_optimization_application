"""Browse and compare MLflow runs."""

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
from src.utils import tracking

st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("🧪 Experiments")

lib.init_tracking()
exps = lib.experiment_names()

c1, c2 = st.columns([3, 1])
exp_label = c1.selectbox("Experiment", list(exps.values()))
if c2.button("🔄 Refresh"):
    lib.cached_runs.clear()

runs = lib.cached_runs(exp_label)
if runs.empty:
    st.info("No runs for this experiment yet.")
    st.stop()

# Short, readable overview table
meta_cols = [c for c in ["run_id", "tags.mlflow.runName", "status", "start_time"] if c in runs.columns]
metric_cols = sorted(c for c in runs.columns if c.startswith("metrics."))
overview = runs[meta_cols + metric_cols].copy()
if "run_id" in overview:
    overview.insert(0, "run", overview["run_id"].str[:8])
    overview = overview.drop(columns=["run_id"])
st.subheader(f"Runs ({len(runs)})")
st.dataframe(overview, hide_index=True, width="stretch")

# ---- compare selected runs ----
st.subheader("Compare runs")
labels = {
    f"{r['run_id'][:8]} · {r.get('tags.mlflow.runName', '')}": r["run_id"]
    for _, r in runs.iterrows()
}
chosen = st.multiselect("Pick runs to compare", list(labels), max_selections=5)
sel_ids = [labels[c] for c in chosen]

if sel_ids:
    sub = runs[runs["run_id"].isin(sel_ids)].copy()
    param_cols = sorted(c for c in sub.columns if c.startswith("params."))
    keep = ["run_id"] + param_cols + metric_cols
    comp = sub[keep].set_index("run_id").T
    comp.columns = [cid[:8] for cid in comp.columns]
    # drop rows that are identical across runs to highlight differences
    show_all = st.checkbox("Show identical fields too", value=False)
    if not show_all:
        comp = comp[comp.nunique(axis=1) > 1]
    st.dataframe(comp, width="stretch")

    if metric_cols:
        metric = st.selectbox("Bar-chart metric", metric_cols)
        chart_df = sub[["run_id", metric]].copy()
        chart_df["run"] = chart_df["run_id"].str[:8]
        st.bar_chart(chart_df.set_index("run")[metric])

# ---- metric history for one run ----
st.subheader("Metric history (per-step)")
hist_label = st.selectbox("Run", list(labels), key="hist_run")
hist_run = labels[hist_label]

if exps["forecasting"] == exp_label:
    with st.expander("Period timeline (saved with run)"):
        record = lib.run_setup(hist_run)
        if record is None:
            st.caption("No period setup was saved with this run.")
        else:
            lib.render_split_view(
                pd.DataFrame(record.get("timeline") or []), record.get("resolved", {})
            )
step_metrics = [c.replace("metrics.", "") for c in metric_cols]
metric_key = st.selectbox("Metric", step_metrics, key="hist_metric")
if metric_key:
    hist = tracking.get_metric_history(hist_run, metric_key)
    if hist.empty or len(hist) < 2:
        st.caption("This metric was logged once (no step history).")
    else:
        st.line_chart(hist.set_index("step")["value"])
