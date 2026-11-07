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
from src.forecasting import drift_alerts
from src.forecasting.measures import HORIZONS
from src.forecasting.periods import BLOCK_NAMES, fetch_run_file, refit_events
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

# Test-block forecasts and diagnostics stay hidden until the run's test block has been scored,
# so nothing about test can steer choices made before that.
test_scored = not lib.test_scores(run_id).empty
resolved = (lib.run_setup(run_id) or {}).get("resolved", {})
test_start = pd.Timestamp(resolved["rl_test_start"]) if resolved.get("rl_test_start") else None
refits = refit_events(resolved)
if not test_scored and test_start is not None:
    df = df[df["date"] < test_start]
    refits = refits[refits["date"] < test_start]

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
kinds_seen: set[str] = set()
for r in refits.itertuples():
    # The first week a new fit's forecasts reach at this horizon, coloured by why the fit was made.
    kind = lib.REFIT_KINDS.get(r.trigger, r.trigger)
    ax.axvline(r.date + pd.Timedelta(weeks=horizon), color=lib.REFIT_COLORS.get(r.trigger, "gray"), linestyle="--",
               linewidth=0.9, label=None if kind in kinds_seen else f"refit: {kind}")
    kinds_seen.add(kind)
ax.set_title(f"store={store}, product={product} · forecasts for h{horizon}, plotted at the week they target")
ax.set_xlabel("Week")
ax.set_ylabel("Units")
ax.legend(fontsize=8)
st.pyplot(fig)

st.caption(
    "Dashed lines: forecaster refits, coloured by why they were made (block start, schedule, alert), at the "
    "first week each new fit forecasts at this horizon."
    + ("" if test_scored else " Test-block weeks are hidden until this run's test block has been scored.")
)

with st.expander("Forecast table"):
    st.dataframe(sku[["date", "target_date", "sales", *[m.id for m in at_h]]], hide_index=True, width="stretch")

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
        + ("" if run.get("tags.scores_within_block") == "true" else
           " This run predates the block-end rule: its h1/h2 validation scores also count the first one or "
           "two test weeks (later runs score only targets inside each block).")
    )
else:
    st.caption("This run logged no per-measure scores (it predates measure manifests).")

# ---- the run on its folds: alerts, refits, retunes ----
alerts, alerts_logged = lib.run_alerts(run_id, test_scored)
retunes = lib.run_retunes(run_id)
st.subheader("Folds, alerts, refits and retunes")
setup = lib.run_setup(run_id) or {}
lanes = [d for d in (pd.DataFrame(setup.get("timeline") or []), lib.retune_segments(retunes, resolved)) if not d.empty]
segments = pd.concat(lanes, ignore_index=True) if lanes else pd.DataFrame()
chart = lib.timeline_chart(segments, refits, drift_alerts.first_alert_marks(alerts), retunes, zoom=True)
if chart is None:
    st.caption("This run saved no timeline.")
else:
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Bars: the forecaster's CV folds, each RL-block fit (the data it saw, the label gap, the weeks it forecasts) "
        "and the RL blocks. Vertical lines: refits (dashed; red when made after an alert), alerts (dotted, the week a "
        "demand class first alerted in a fit) and retunes (thick purple: new settings used; pale: the current ones "
        "kept), drawn at the refit that followed the alert. A retune lane shows the rows its tuning saw and the "
        "holdout it never saw. Hover a line for details. The axis starts four weeks before the first CV validation "
        "window; every fit uses all the data before it."
        + ("" if test_scored else " Test-block alerts and refits are hidden until this run's test block has been scored.")
    )
if not retunes.empty:
    st.markdown("**Retunes**")
    st.dataframe(lib.retune_table(retunes).round(3), hide_index=True, width="stretch")
    st.caption(
        "An alert came back for a class that had triggered the previous alert-driven refit, so tuning ran again with "
        "this run's own CV settings (\"CV folds tuned on\": folds × weeks), on the rows before the holdout. The scores "
        "are the primary measure on the holdout, which the tuning never saw (lower is better); the new settings are used "
        "only if they gain at least the minimum. A retune is skipped when the run's CV folds do not all fit before the "
        "holdout."
    )

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

        present = [b for b in lib._BLOCK_COLORS if b in set(weekly["block"])]
        color = alt.Color("block:N", title="Block", sort=present,
                          scale=alt.Scale(domain=present, range=[lib._BLOCK_COLORS[b] for b in present]))
        bars = alt.Chart(weekly).mark_bar().encode(
            x=alt.X("date:T", title="Week"),
            y=alt.Y("bias_pct:Q", title=f"Weekly error of {level}, %"),
            color=color,
            tooltip=[alt.Tooltip("date:T", title="Week"), "fit:N",
                     alt.Tooltip("bias_pct:Q", title="Error %", format="+.1f")],
        ).properties(height=180)
        st.altair_chart(bars, width="stretch")
        st.caption(
            "Error = (actual − forecast) ÷ actual, summed over all SKUs: positive means the forecasts were "
            "too low. Whether a drift is persistent is judged per demand class below."
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

# ---- drift alerts: the production rule on this run's fits ----
st.subheader("Drift alerts")
if alerts.empty:
    st.caption("Drift alerts need the run's saved forecasts, its period setup and an h0 level measure.")
else:
    import altair as alt

    rules = drift_alerts.default_rules()
    view = alerts.assign(block=alerts["block"].map(lambda b: BLOCK_NAMES.get(b, b)))
    present = [b for b in lib._BLOCK_COLORS if b in set(view["block"])]
    color = alt.Color("block:N", title="Block", sort=present,
                      scale=alt.Scale(domain=present, range=[lib._BLOCK_COLORS[b] for b in present]))
    tooltip = [alt.Tooltip("week:T", title="Week"), "fit:N", alt.Tooltip("week_of_fit:Q", title="Week of fit"),
               alt.Tooltip("error_pct:Q", title="Error %", format="+.1f"),
               alt.Tooltip("tracking_signal:Q", title="Tracking signal", format="+.2f"), "streak:Q"]
    limits = pd.DataFrame({"y": [-rules.bias.limit, rules.bias.limit]})
    # Refit marks coloured by trigger, as literal colours (no legend per panel).
    marks = refits.assign(color=refits["trigger"].map(lib.REFIT_COLORS).fillna("#5a5a5a"),
                          kind=refits["trigger"].map(lib.REFIT_KINDS).fillna(refits["trigger"]))
    panels = []
    for cls in sorted(view["demand_class"].unique()):
        g = view[view["demand_class"] == cls]
        line = alt.Chart(g).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
            x=alt.X("week:T", title=None), y=alt.Y("tracking_signal:Q", title=cls, scale=alt.Scale(domain=[-1, 1])),
            color=color, detail="fit:N", tooltip=tooltip)
        bounds = alt.Chart(limits).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
        refit_rules = alt.Chart(marks).mark_rule(strokeDash=[6, 3]).encode(
            x="date:T", color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("label:N", title="Refit"), alt.Tooltip("kind:N", title="Why"),
                     alt.Tooltip("classes:N", title="Alerting classes"), alt.Tooltip("date:T", title="From")])
        hits = alt.Chart(g[g["alert"]]).mark_point(color="#E45756", size=70, filled=True).encode(
            x="week:T", y="tracking_signal:Q", tooltip=tooltip)
        panels.append(alt.layer(refit_rules, bounds, line, hits).resolve_scale(color="independent").properties(height=120))
    st.altair_chart(alt.vconcat(*panels).resolve_scale(x="shared"), width="stretch")

    summary = drift_alerts.alert_summary(view)
    summary = summary.rename(columns={"block": "Block", "demand_class": "Demand class", "alert_weeks": "Alert weeks",
                                      "first_alert": "First alert", "max_abs_signal": "Largest |signal|"})
    st.dataframe(summary.round(2), hide_index=True, width="stretch")
    n_weeks = int(view.loc[view["alert"], "week"].nunique())
    st.caption(
        f"The production loop's bias alarm, applied to this run's fits: {drift_alerts.describe(rules)}. "
        f"A fit can alert from its week {drift_alerts.earliest_alert_week(rules)}. Red points mark alert weeks "
        f"({n_weeks} week{'s' if n_weeks != 1 else ''} with at least one class alerting); dashed lines are refits "
        "(red: made after an alert). "
        + ("This run refit after alerts. " if (refits["trigger"] == "drift").any() else
           "Alerts never refit unless the run used drift-triggered refits: this one followed its fixed schedule. ")
        + ("Logged by the run." if alerts_logged else "Computed now: this run did not log drift alerts.")
        + ("" if test_scored else " The test block is hidden until this run's test block has been scored.")
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
