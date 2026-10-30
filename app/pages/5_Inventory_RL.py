"""Inventory / RL: one training run's curve, validation result and order recommendations."""

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

st.set_page_config(page_title="Inventory & RL", page_icon="🤖", layout="wide")
st.title("🤖 Inventory & RL")

lib.init_tracking()
cfg = lib.get_config()

runs = lib.rl_train_runs()
if runs.empty:
    st.info("No RL training runs yet. Train an agent from **Run RL**.")
    st.stop()

labels = {
    f"{r['run_id'][:8]} · {r.get('tags.method', '?')} · {str(r.get('tags.agent', '?')).upper()} · "
    f"seed {r.get('tags.seed', '?')} · forecasts {str(r.get('tags.forecast_run_id', ''))[:8]}": r["run_id"]
    for _, r in runs.iterrows()
}
sel = st.selectbox("RL training run", list(labels))
run = runs[runs["run_id"] == labels[sel]].iloc[0]


def _metric(key: str) -> str:
    v = run.get(f"metrics.{key}")
    return f"{v:.3f}" if pd.notna(v) else "—"


k1, k2, k3, k4 = st.columns(4)
k1.metric("Validation cost, agent (€ / SKU-window)", _metric("val_cost_eur_per_sku"))
k2.metric("Validation cost, anchor (€ / SKU-window)", _metric("val_anchor_cost_eur_per_sku"))
k3.metric("Δ vs anchor (€)", _metric("val_minus_anchor_eur"))
k4.metric("Config / seed", f"{run.get('tags.config_key', '—')} · {run.get('tags.seed', '—')}")
st.caption(
    f"Anchor: {run.get('params.anchor', '—')} · state ({run.get('params.n_state', '?')}): "
    f"{run.get('params.state', '—')}. Final weights, scored once on the validation windows; "
    "the champion is chosen on **Run RL**."
)

st.subheader("Training curve")
hist = tracking.get_metric_history(run["run_id"], "insample_vs_anchor_pct")
if not hist.empty and len(hist) > 1:
    st.caption(
        "Cost against the anchor on a fixed sample of 2,000 training windows (%), by training "
        "windows replayed. Below 0 beats the anchor. For display only: the final weights are "
        "always the ones kept and scored."
    )
    st.line_chart(hist.rename(columns={"step": "training windows replayed", "value": "cost vs anchor (%)"})
                  .set_index("training windows replayed")["cost vs anchor (%)"])
else:
    legacy = tracking.get_metric_history(run["run_id"], "ep_reward_mean")
    if not legacy.empty and len(legacy) > 1:
        st.caption("Episode reward (normalized −€), training windows. Older run, logged before the euro curve.")
        st.line_chart(legacy.set_index("step")["value"])
    else:
        st.caption("No training curve logged for this run.")


@st.cache_data(ttl=60, show_spinner=False)
def _val_episodes(rl_run_id: str) -> pd.DataFrame:
    from src.rl.selection import policy_dir

    try:
        return pd.read_parquet(policy_dir(cfg, rl_run_id) / "val_episodes.parquet")
    except Exception:  # noqa: BLE001 - a missing artifact only hides the table
        return pd.DataFrame()


st.subheader("Validation by demand class")
val_eps = _val_episodes(run["run_id"])
if val_eps.empty or "demand_class" not in val_eps or val_eps["demand_class"].isna().all():
    st.caption("This run's validation episodes carry no demand class (it was trained before classes existed).")
else:
    from src.forecasting.features.demand_profile import DEMAND_CLASSES, UNCLASSIFIED

    rows = []
    for cls in (*DEMAND_CLASSES, UNCLASSIFIED):
        g = val_eps[val_eps["demand_class"] == cls]
        if g.empty:
            continue
        agent_cost, anchor_cost = float(g["cost_eur"].mean()), float(g["anchor_cost_eur"].mean())
        rows.append({
            "demand class": cls,
            "SKU-windows": len(g),
            "agent € / SKU-window": round(agent_cost, 3),
            "anchor € / SKU-window": round(anchor_cost, 3),
            "Δ vs anchor": f"{100 * (g['cost_eur'].sum() / g['anchor_cost_eur'].sum() - 1):+.1f}%"
            if g["anchor_cost_eur"].sum() > 0 else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Demand pattern of each SKU at the window start, from its trailing 52 weeks of sales.")

# ---------------------------------------------------------------------------
# Order recommendations from the saved policy (spec-built environment)
# ---------------------------------------------------------------------------
st.subheader("Order recommendations")


@st.cache_resource(show_spinner="Loading the policy and its forecasting run…")
def _load(rl_run_id: str):
    from src.rl.selection import load_policy
    from src.rl.train import block_episodes, load_run_data

    spec, agent = load_policy(cfg, rl_run_id)
    data = load_run_data(cfg, spec.forecast_run_id)
    return spec, agent, data, block_episodes(cfg, data, "rl_test")


try:
    spec, agent, data, episodes = _load(run["run_id"])
except Exception as exc:  # noqa: BLE001 - show why instead of a stack trace
    st.warning(f"Could not load this run's policy: {exc}")
    st.stop()

st.caption("First order round of each SKU's first window in the test block (production).")
from src.simulation.vn2_env import VN2Env  # noqa: E402


def _recommend(key) -> dict:
    env = VN2Env(spec, episodes[key], data.lookup, data.scales.get(key[:2], 1.0), data.rules)
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    a0, a1 = env._arrivals()
    out = {"store": key[0], "product": key[1], "window_start": key[2].date(),
           "on_hand": env.inventory, "in_transit_w1": a0, "in_transit_w2": a1}
    out.update({m: data.lookup.get((key[0], key[1], key[2]), {}).get(m) for m in data.measure_ids})
    out["anchor_order"] = env._decision.order
    out["recommended_order"] = env.resolve_order(action)
    return out


first = {}
for key in sorted(episodes):
    first.setdefault(key[:2], key)
sku_keys = sorted(first)
mode = st.radio("View", ["Single SKU", "Top 25 SKUs"], horizontal=True)
if mode == "Single SKU":
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted({s for s, _ in sku_keys}))
    product = c2.selectbox("Product", sorted({p for s, p in sku_keys if s == store}))
    rec = _recommend(first[(store, product)])
    st.metric("Recommended order (units)", rec["recommended_order"], delta=rec["recommended_order"] - rec["anchor_order"])
    st.json(rec)
else:
    rows = [_recommend(first[k]) for k in sku_keys[:25]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
