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

# Test-block forecast diagnostics stay hidden until the run's test block has been scored,
# so nothing about test can steer choices made before that.
test_scored = not lib.test_scores(run_id).empty

# ---- per-measure scores (CV and out-of-sample RL blocks) ----
st.subheader("Scores per measure")
blocks = [("cv", "CV"), ("oos_rl_train", "RL train"), ("oos_rl_val", "RL val")]
if test_scored:
    blocks.append(("oos_rl_test", "RL test"))
rows = []
for m in measures:
    row = {"measure": m.id}
    for prefix, name in blocks:
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
        + ("" if test_scored else " Test-block scores appear once this run's test block has been scored.")
    )
else:
    st.caption("This run logged no per-measure scores (it predates measure manifests).")

# ---- forecast drift per fit and per week ----
st.subheader("Forecast drift by fit")
drift_data = lib.run_drift(run_id, test_scored)
fits = drift_data.get("fits", pd.DataFrame())
if fits.empty:
    st.caption("Drift needs the run's saved forecasts and its period setup.")
else:
    st.dataframe(fits.round(3), hide_index=True, width="stretch")
    level = drift_data.get("level")
    weekly = drift_data.get("weekly", pd.DataFrame())
    if not weekly.empty:
        import altair as alt

        from src.forecasting.drift import TRIGG_LIMIT

        present = [b for b in lib._BLOCK_COLORS if b in set(weekly["block"])]
        color = alt.Color("block:N", title="Block", sort=present,
                          scale=alt.Scale(domain=present, range=[lib._BLOCK_COLORS[b] for b in present]))
        tooltip = [alt.Tooltip("date:T", title="Week"), "fit:N",
                   alt.Tooltip("bias_pct:Q", title="Error %", format="+.1f"),
                   alt.Tooltip("tracking_signal:Q", title="Tracking signal", format="+.2f")]
        bars = alt.Chart(weekly).mark_bar().encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("bias_pct:Q", title=f"Weekly error of {level}, %"),
            color=color, tooltip=tooltip,
        ).properties(height=180)
        signal = alt.Chart(weekly).mark_line(point=True, color="#4C78A8").encode(
            x=alt.X("date:T", title="Week"),
            y=alt.Y("tracking_signal:Q", title="Tracking signal", scale=alt.Scale(domain=[-1, 1])),
            tooltip=tooltip,
        ).properties(height=160)
        limits = alt.Chart(pd.DataFrame({"y": [-TRIGG_LIMIT, TRIGG_LIMIT]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        alarms = alt.Chart(weekly[weekly["alarm"]]).mark_point(color="#E45756", size=80, filled=True).encode(
            x="date:T", y="tracking_signal:Q", tooltip=tooltip)
        st.altair_chart(alt.vconcat(bars, signal + limits + alarms).resolve_scale(x="shared"), width="stretch")
        n_alarm = int(weekly["alarm"].sum())
        st.caption(
            "Error = (actual − forecast) ÷ actual, summed over all SKUs: positive means the forecasts were "
            "too low. The tracking signal is Trigg's smoothed error ÷ smoothed absolute error (smoothing 0.2); "
            f"beyond ±{TRIGG_LIMIT} (red points, {n_alarm} week{'s' if n_alarm != 1 else ''}) recent errors "
            "point the same way more often than noise would in about 95% of weeks. It catches a sustained "
            "shift of one weekly-error standard deviation in about 6 weeks, half of one in about 12."
        )
    by_class = drift_data.get("by_class", pd.DataFrame())
    col = f"bias {level} (%)"
    if not by_class.empty and col in by_class.columns:
        with st.expander(f"Error of {level} by demand class"):
            table = by_class.pivot_table(index="fit", columns="demand class", values=col, sort=False)
            st.dataframe(table.round(1), width="stretch")
            st.caption("Demand pattern of each SKU that week, from its trailing 52 weeks of sales.")
    st.caption(
        "One row per forecaster fit. Only targets inside the fit's own block count, and out-of-stock weeks are "
        "skipped. A learned policy trained on a block whose forecasts ran too low can learn to order more "
        "everywhere, which costs money once the error flips sign."
        + ("" if test_scored else " The test fit is hidden until this run's test block has been scored.")
    )

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
