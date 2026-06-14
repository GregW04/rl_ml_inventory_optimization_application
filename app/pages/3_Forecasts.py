"""Visualize saved forecasts: actual vs predicted, intervals, feature importance."""

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
from src.utils import tracking

st.set_page_config(page_title="Forecasts", page_icon="📈", layout="wide")
st.title("📈 Forecasts")

lib.init_tracking()
path = lib.forecasts_path()
if not path.exists():
    st.info("No forecasts yet. Run the forecasting pipeline from **Run Pipelines**.")
    st.stop()


@st.cache_data(ttl=30)
def _load(p: str) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = _load(str(path))

c1, c2 = st.columns(2)
store = c1.selectbox("Store", sorted(df["store"].unique()))
products = sorted(df[df["store"] == store]["product"].unique())
product = c2.selectbox("Product", products)

sku = df[(df["store"] == store) & (df["product"] == product)].sort_values("date")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(sku["date"], sku["sales"], label="Actual", color="steelblue")
if "mu_h0" in sku.columns:
    ax.plot(sku["date"], sku["mu_h0"], label="Forecast μ (h=0)", color="orange", linestyle="--")
if "pred_q10_h0" in sku.columns and "pred_q90_h0" in sku.columns:
    ax.fill_between(
        sku["date"], sku["pred_q10_h0"], sku["pred_q90_h0"],
        alpha=0.2, color="orange", label="Q10–Q90",
    )
ax.set_title(f"store={store}, product={product}")
ax.set_xlabel("Date")
ax.set_ylabel("Units")
ax.legend()
st.pyplot(fig)

with st.expander("Forecast table"):
    cols = [c for c in sku.columns if c in ("date", "sales") or c.startswith(("mu_h", "sigma_h", "pred_q"))]
    st.dataframe(sku[cols], hide_index=True, use_container_width=True)

# ---- feature importance from the latest forecasting run ----
st.subheader("Feature importance (latest run)")
runs = lib.cached_runs(lib.experiment_names()["forecasting"])
if runs.empty:
    st.caption("No forecasting runs found.")
else:
    run_id = runs.iloc[0]["run_id"]
    arts = tracking.list_artifacts(run_id, "feature_importance")
    if arts.empty:
        st.caption("No feature-importance artifact logged for the latest run.")
    else:
        png = arts[arts["path"].str.endswith(".png")]
        csv = arts[arts["path"].str.endswith(".csv")]
        try:
            if not png.empty:
                local = tracking.download_artifact(run_id, png.iloc[0]["path"])
                st.image(local, use_container_width=True)
            elif not csv.empty:
                local = tracking.download_artifact(run_id, csv.iloc[0]["path"])
                imp = pd.read_csv(local)
                st.bar_chart(imp.set_index("feature")["importance"])
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Could not load artifact: {exc}")
