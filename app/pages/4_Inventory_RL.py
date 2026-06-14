"""Inventory / RL: training KPIs, curves, and order recommendations."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import streamlit as st

from app import lib
from src.utils import tracking

st.set_page_config(page_title="Inventory & RL", page_icon="🤖", layout="wide")
st.title("🤖 Inventory & RL")

lib.init_tracking()
cfg = lib.get_config()
rl_exp = lib.experiment_names()["rl"]

# ---------------------------------------------------------------------------
# Training KPIs + curves
# ---------------------------------------------------------------------------
runs = lib.cached_runs(rl_exp)
if runs.empty:
    st.info("No RL runs yet. Train an agent from **Run Pipelines**.")
    st.stop()

labels = {
    f"{r['run_id'][:8]} · {r.get('params.agent', '?')} · {r.get('tags.mlflow.runName', '')}": r["run_id"]
    for _, r in runs.iterrows()
}
sel = st.selectbox("RL run", list(labels))
run = runs[runs["run_id"] == labels[sel]].iloc[0]

k1, k2, k3 = st.columns(3)
k1.metric("Episode reward (final)", f"{run.get('metrics.final_ep_reward_mean', float('nan')):.3f}")
fr = run.get("metrics.final_fill_rate")
k2.metric("Fill rate (final)", f"{fr:.2%}" if pd.notna(fr) else "—")
sr = run.get("metrics.final_shortage_rate")
k3.metric("Shortage rate (final)", f"{sr:.2%}" if pd.notna(sr) else "—")

st.subheader("Training curves")
for key, title in [("ep_reward_mean", "Episode reward"), ("fill_rate_mean", "Fill rate")]:
    hist = tracking.get_metric_history(run["run_id"], key)
    if not hist.empty and len(hist) > 1:
        st.caption(title)
        st.line_chart(hist.set_index("step")["value"])

# ---------------------------------------------------------------------------
# Order recommendations (reuse the real env + saved agent)
# ---------------------------------------------------------------------------
st.subheader("Order recommendations")
agent_type = st.selectbox("Agent policy", ["ppo", "dqn"])
model_path = lib.rl_model_path(agent_type)
if not model_path.exists():
    st.info(f"No saved `{agent_type}` agent at `{model_path.name}`. Train it first.")
    st.stop()


@st.cache_resource
def _load_rl_data():
    from src.rl.train import prepare_rl_data
    return prepare_rl_data(cfg)


@st.cache_resource
def _load_agent(agent: str, path: str, mtime: float):
    from stable_baselines3 import DQN, PPO
    return (DQN if agent == "dqn" else PPO).load(path)


episodes, lookup = _load_rl_data()
agent = _load_agent(agent_type, str(model_path), model_path.stat().st_mtime)
buckets = cfg["rl"]["agents"]["dqn"]["order_buckets"] if agent_type == "dqn" else None
env_cfg = cfg["rl"]["environment"]


def _recommend(store: int, product: int) -> dict:
    from src.simulation.env import InventoryEnv
    env = InventoryEnv(
        episode_data=episodes[(store, product)],
        forecast_lookup=lookup,
        holding_cost=env_cfg["holding_cost"],
        shortage_cost=env_cfg["shortage_cost"],
        lead_time_split=env_cfg["lead_time_split"],
        max_order=env_cfg["max_order"],
        discrete_buckets=buckets,
    )
    obs, _ = env.reset()
    action, _ = agent.predict(obs, deterministic=True)
    order = env._resolve_order(action)
    return {
        "on_hand": float(obs[0]),
        "in_transit_1w": float(obs[1]),
        "in_transit_2w": float(obs[2]),
        "mu_h0": float(obs[3]),
        "sigma_h0": float(obs[4]),
        "recommended_order": round(order, 2),
    }


sku_keys = sorted(episodes.keys())
mode = st.radio("View", ["Single SKU", "Top 25 SKUs"], horizontal=True)

if mode == "Single SKU":
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted({s for s, _ in sku_keys}))
    product = c2.selectbox("Product", sorted({p for s, p in sku_keys if s == store}))
    rec = _recommend(store, product)
    st.metric("Recommended order (units)", rec["recommended_order"])
    st.json(rec)
else:
    rows = []
    for store, product in sku_keys[:25]:
        r = _recommend(store, product)
        rows.append({"store": store, "product": product, **r})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
