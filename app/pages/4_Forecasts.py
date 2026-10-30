"""Visualize a forecasting run's measures: actual vs forecast per horizon, coverage, importance."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import lib
from src.forecasting.measures import HORIZONS
from src.forecasting.periods import fetch_run_file
from src.utils import tracking

st.set_page_config(page_title="Forecasts", page_icon="📈", layout="wide")
st.title("📈 Forecasts")

lib.init_tracking()
runs = lib.forecast_runs()
if runs.empty:
    st.info("No forecasting runs yet. Run the forecasting pipeline from **Run Forecasting**.")
    st.stop()

labels = {lib.forecast_run_label(r): r["run_id"] for _, r in runs.iterrows()}
run_id = labels[st.selectbox("Forecasting run", list(labels))]
run = runs[runs["run_id"] == run_id].iloc[0]
measures = lib.as_measures(lib.measures_of_run(run_id))


@st.cache_data(ttl=60, show_spinner=False)
def _load(run_id: str) -> pd.DataFrame:
    path = fetch_run_file(lib.get_config(), run_id, "train_forecasts.parquet")
    if path is None:
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = _load(run_id)
if df.empty:
    st.error("This run's RL-block forecasts are not available (not on disk and not logged to MLflow).")
    st.stop()

c1, c2, c3 = st.columns(3)
horizon = c1.selectbox("Horizon", list(HORIZONS), format_func=lambda h: f"h{h}", index=len(HORIZONS) - 1)
store = c2.selectbox("Store", sorted(df["store"].unique()))
product = c3.selectbox("Product", sorted(df[df["store"] == store]["product"].unique()))

at_h = [m for m in measures if m.horizon == horizon and m.id in df.columns]
sku = df[(df["store"] == store) & (df["product"] == product)].sort_values("date").copy()
# A forecast made in week t for horizon h targets demand in week t + h.
sku["target_date"] = sku["date"] + pd.Timedelta(weeks=horizon)
actual = sku.set_index("date")["sales"]

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(actual.index, actual.values, label="Actual", color="steelblue")
quantiles = sorted((m for m in at_h if m.kind == "quantile"), key=lambda m: m.tau)
if len(quantiles) >= 2:
    ax.fill_between(sku["target_date"], sku[quantiles[0].id], sku[quantiles[-1].id], alpha=0.15,
                    color="orange", label=f"{quantiles[0].id} – {quantiles[-1].id}")
for m in quantiles:
    ax.plot(sku["target_date"], sku[m.id], linestyle=":", label=m.id)
for m in at_h:
    if m.kind == "mean":
        ax.plot(sku["target_date"], sku[m.id], color="darkorange", linestyle="--", label=m.id)
ax.set_title(f"store={store}, product={product} · forecasts for h{horizon}, plotted at the week they target")
ax.set_xlabel("Week")
ax.set_ylabel("Units")
ax.legend(fontsize=8)
st.pyplot(fig)

with st.expander("Forecast table"):
    st.dataframe(sku[["date", "target_date", "sales", *[m.id for m in at_h]]], hide_index=True, width="stretch")

# ---- per-measure scores (CV and out-of-sample RL blocks) ----
st.subheader("Scores per measure")
rows = []
for m in measures:
    row = {"measure": m.id}
    for prefix, name in [("cv", "CV"), ("oos_rl_train", "RL train"), ("oos_rl_val", "RL val"), ("oos_rl_test", "RL test")]:
        for metric in ("pinball", "coverage", "rmse"):
            v = run.get(f"metrics.{prefix}_{metric}_{m.id}")
            if v is not None and pd.notna(v):
                row[f"{name} {metric}"] = v
    if m.kind == "quantile":
        row["target coverage (τ)"] = m.tau
    rows.append(row)
scores = pd.DataFrame(rows)
if len(scores.columns) > 1:
    st.dataframe(scores, hide_index=True, width="stretch")
    st.caption(
        "Coverage is the share of weeks with demand ≤ the quantile; it should be close to τ. "
        "RL-block scores are out of sample (forecasts from a fit that ends before each block)."
    )
else:
    st.caption("This run logged no per-measure scores (it predates measure manifests).")

# ---- feature importance ----
st.subheader("Feature importance")
arts = tracking.list_artifacts(run_id, "feature_importance")
if arts.empty:
    st.caption("No feature-importance artifact logged for this run.")
else:
    png = arts[arts["path"].str.endswith(".png")]
    csv = arts[arts["path"].str.endswith(".csv")]
    try:
        if not png.empty:
            st.image(tracking.download_artifact(run_id, png.iloc[0]["path"]), width="stretch")
        elif not csv.empty:
            imp = pd.read_csv(tracking.download_artifact(run_id, csv.iloc[0]["path"]))
            st.bar_chart(imp.set_index("feature")["importance"])
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load artifact: {exc}")
