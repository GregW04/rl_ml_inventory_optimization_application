"""Inventory / RL: training sample → validation, where a policy wins and loses, and explained orders."""

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

from app import lib, rl_charts
from src.forecasting.features.demand_profile import DEMAND_CLASSES, UNCLASSIFIED
from src.rl.explain import drivers, explain_order
from src.rl.rollout import vs_anchor
from src.rl.selection import ANCHOR
from src.rl.spec import TRAINING_FORECAST_LABELS, is_torch_agent, windows_per_iteration

st.set_page_config(page_title="Inventory & RL", page_icon="🤖", layout="wide")
st.title("🤖 Inventory & RL")
st.caption(
    "How training went from the training sample to validation, where a policy wins and loses against the "
    "anchor, and its order recommendations with the reason. Everything is read back from MLflow and the run folders."
)

lib.init_tracking()
cfg = lib.get_config()
CLASS_ORDER = [*DEMAND_CLASSES, UNCLASSIFIED]

# ---------------------------------------------------------------------------
# Forecasting run and configurations
# ---------------------------------------------------------------------------
runs = lib.rl_train_runs()
fc_runs = lib.forecast_runs()
if not runs.empty and not fc_runs.empty and "tags.config_key" in runs:
    fc_runs = fc_runs[fc_runs["run_id"].isin(set(runs["tags.forecast_run_id"].dropna()))]
if runs.empty or fc_runs.empty or "tags.config_key" not in runs:
    st.info("No RL training runs yet. Train an agent from **Run RL**.")
    st.stop()

fc_labels = {lib.forecast_run_label(r): r["run_id"] for _, r in fc_runs.iterrows()}
fc_id = fc_labels[st.selectbox("Forecasting run", list(fc_labels), help="Runs with trained RL configurations, newest first")]
board = lib.rl_leaderboard(fc_id)
configs = board[board["kind"] != ANCHOR].set_index("config_key") if not board.empty else pd.DataFrame()
fc_rl = runs[runs["tags.forecast_run_id"] == fc_id]
newest = [k for k in dict.fromkeys(fc_rl.sort_values("start_time", ascending=False)["tags.config_key"].dropna())
          if k in configs.index]
if not newest:
    st.info("No validated configurations on this forecasting run yet.")
    st.stop()


def _label(key: str) -> str:
    r = configs.loc[key]
    tf = r["training_forecasts"]
    extra = "" if tf == "as_is" else f" · {TRAINING_FORECAST_LABELS.get(tf, tf).lower()}"
    return f"{r['method']} · {str(r['agent']).upper()} · {key[:6]}{extra}"


chosen = st.multiselect(
    "Configurations", newest, default=newest[:1], format_func=_label, max_selections=4,
    help="Trained on this forecasting run, newest first. Each groups its seeds; compare up to 4.",
)
if not chosen:
    st.info("Pick at least one configuration.")
    st.stop()
seed_runs = fc_rl[fc_rl["tags.config_key"].isin(chosen)].dropna(subset=["metrics.val_cost_eur_per_sku"])
domain = [_label(k) for k in chosen]


def _pct(v, unit: str = "%") -> str:
    return "—" if pd.isna(v) else f"{v:+.1f}{unit}"


def _iterations_run(key: str) -> str:
    """Iterations each seed ran, e.g. '1,000, 1,200, 1,000 of 2,000'; '—' for runs saved before it was logged."""
    g = seed_runs[seed_runs["tags.config_key"] == key].sort_values("tags.seed")
    ran = g.get("metrics.iterations_run", pd.Series(dtype=float)).dropna()
    if ran.empty:
        return "—"
    budget = g.get("params.agent_iterations", pd.Series(dtype=object)).dropna()
    return ", ".join(f"{int(v):,}" for v in ran) + (f" of {int(float(budget.iloc[0])):,}" if not budget.empty else "")


def _focus(key: str) -> str:
    """The configuration a tab inspects through its project-seed run."""
    return chosen[0] if len(chosen) == 1 else st.selectbox("Configuration", chosen, format_func=_label, key=key)


tab_train, tab_where, tab_orders = st.tabs(["Training → validation", "Where it wins and loses", "Order recommendations"])

# ---------------------------------------------------------------------------
# 1. Training sample → validation (same charts for every agent and method)
# ---------------------------------------------------------------------------
with tab_train:
    st.dataframe(pd.DataFrame([
        {
            "configuration": _label(k),
            "seeds": configs.loc[k, "seeds"],
            "iterations run": _iterations_run(k),
            "training sample Δ": _pct(configs.loc[k, "insample_pct"]),
            "validation Δ (± seeds)": f"{_pct(configs.loc[k, 'val_pct'])} ± {configs.loc[k, 'val_pct_sd']:.1f}",
            "gap": _pct(configs.loc[k, "val_pct"] - configs.loc[k, "insample_pct"], " pts"),
            "units vs anchor (validation)": _pct(configs.loc[k, "units_vs_anchor"]),
            "seeds beating anchor": "—" if pd.isna(b := configs.loc[k, "share_beats"]) else f"{b:.0%}",
        }
        for k in chosen
    ]), hide_index=True, width="stretch")
    st.caption(
        "Δ: cost against the anchor in %, mean over seeds (below 0 is cheaper). Training sample: the final weights on "
        "a fixed sample of 2,000 training windows. Gap: validation minus training sample. Units: project-seed run. "
        "Iterations run: per seed, in seed order; fewer than the budget means early stopping ended the run."
    )

    frames = []
    for _, r in seed_runs.iterrows():
        h = lib.rl_history(r["run_id"], tuple(k for k in rl_charts.KEYS if pd.notna(r.get(f"metrics.{k}"))))
        if h.empty:
            continue
        agent = r.get("tags.agent")
        per_it = windows_per_iteration(agent, {"batch_windows": r.get("params.agent_batch_windows"),
                                               "group_size": r.get("params.agent_group_size")}) if is_torch_agent(agent) else 1
        frames.append(h.assign(
            windows=np.where(h["key"].isin(list(rl_charts.PER_ITERATION)), h["step"] * per_it, h["step"]),
            configuration=_label(r["tags.config_key"]), seed=str(r["tags.seed"]), run=r["run_id"],
        ))
    hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["key", "step", "value", "windows", "configuration", "seed", "run"])

    final_x = hist[hist["key"] == "insample_vs_anchor_pct"].groupby("run")["windows"].max()
    marks = []
    for _, r in seed_runs.iterrows():
        if r["run_id"] not in final_x:
            continue
        mark = {"configuration": _label(r["tags.config_key"]), "seed": str(r["tags.seed"]), "windows": final_x[r["run_id"]]}
        cost, anc = r.get("metrics.val_cost_eur_per_sku"), r.get("metrics.val_anchor_cost_eur_per_sku")
        if pd.notna(cost) and pd.notna(anc) and anc > 0:
            marks.append({**mark, "metric": "cost", "value": 100.0 * (cost / anc - 1.0)})
        if pd.notna(units := r.get("metrics.val_units_vs_anchor_pct")):
            marks.append({**mark, "metric": "units", "value": units})
    marks = pd.DataFrame(marks, columns=["configuration", "seed", "windows", "metric", "value"])
    curves = hist[hist["key"].isin(list(rl_charts.LEARNING))].assign(metric=lambda d: d["key"].map(rl_charts.LEARNING))
    # One x range for every chart below, so curves and panels line up.
    x_max = max((float(d["windows"].max()) for d in (hist, marks) if not d.empty), default=None)

    st.markdown("**Training sample → validation**")
    for metric, title in (("cost", "Cost vs anchor (%)"), ("units", "Units vs anchor (%)")):
        c, m = curves[curves["metric"] == metric], marks[marks["metric"] == metric]
        if not (c.empty and m.empty):
            st.altair_chart(rl_charts.learning_chart(c, m, title, domain, x_max), width="stretch")
    if missing := [lab for lab in domain if lab not in set(curves["configuration"])]:
        st.caption(f"No training-sample curve was logged for {', '.join(missing)} (trained before it existed); the "
                   "table above has its validation result.")
    st.caption(
        "Lines: the fixed training sample, scored with the policy's mean action about 10 times during training, one "
        "line per seed. ◆: validation, scored once on the final weights. 0 is the anchor. Runs trained before units "
        "were logged on the training sample show only the validation ◆ for units."
    )

    present = [k for k in rl_charts.PANELS if (hist["key"] == k).any()]
    if present:
        st.markdown("**Training health**")
        max_norms = sorted({float(v) for v in seed_runs.get("params.agent_max_grad_norm", pd.Series(dtype=object)).dropna()})
        cols = st.columns(3)
        for i, key in enumerate(present):
            title, refs, _ = rl_charts.PANELS[key]
            refs = refs + max_norms if key == "grad_norm" else refs
            cols[i % 3].altair_chart(rl_charts.panel_chart(hist[hist["key"] == key], title, refs, domain, x_max),
                                     width="stretch")
        st.caption(
            "x axis: training windows replayed, the same range in every panel. PPO replays `batch_windows` windows "
            "per iteration, GRPO `batch_windows × group_size`, so the same number of iterations ends further right "
            "for GRPO."
        )
    with st.expander("How to read these charts"):
        st.markdown("\n".join([
            "- **Training sample line:** flat at the end means training converged; still falling means the budget ran out.",
            "- **◆ against the line's end:** the gap between training and validation. A ◆ well above the line's end means "
            "the gain did not transfer (overfitting, or validation weeks behave differently from training weeks).",
            "- **Longer training:** if more iterations lower the line but not the ◆, the extra training did not help.",
            "- **Units:** a policy ordering more than the anchor is betting on the demand level; validation cost often "
            "tracks the extra units.",
            *[f"- **{title}:** {hint}" for key, (title, _, hint) in rl_charts.PANELS.items() if key in present],
            "- Faint lines are the logged values; solid lines a rolling mean over about a tenth of training.",
        ]))
    st.caption(
        f"Forecasting run `{fc_id[:8]}` · {len(seed_runs)} training runs: "
        + ", ".join(f"`{rid[:8]}`" for rid in seed_runs["run_id"])
    )
    st.download_button("Download training history (CSV)", hist.to_csv(index=False).encode(),
                       file_name=f"rl_training_history_{fc_id[:8]}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# 2. Where it wins and loses (one configuration's project-seed run)
# ---------------------------------------------------------------------------
with tab_where:
    focus = _focus("rl_focus_where")
    deploy = configs.loc[focus, "deploy_run_id"]
    val_eps = lib.rl_episodes(deploy, "val_episodes") if isinstance(deploy, str) else pd.DataFrame()
    if val_eps.empty:
        st.info("This configuration has no saved validation episodes for its project-seed run.")
    else:
        ins_eps = lib.rl_episodes(deploy, "insample_episodes")
        blocks = {"training sample": ins_eps, "validation": val_eps}
        st.caption(
            f"Project-seed run `{deploy[:8]}`. Training sample: the curve's 2,000 training windows with the final "
            f"weights. Validation: {len(val_eps)} SKU-windows."
        )
        classed = {b: e for b, e in blocks.items()
                   if not e.empty and "demand_class" in e and e["demand_class"].notna().any()}
        if classed:
            by_class = pd.concat([vs_anchor(e, "demand_class").assign(block=b) for b, e in classed.items()],
                                 ignore_index=True)
            st.markdown("**By demand class**")
            c1, c2 = st.columns(2)
            c1.altair_chart(rl_charts.class_chart(by_class, "cost_vs_anchor_pct", "Cost vs anchor (%)", CLASS_ORDER),
                            width="stretch")
            if by_class["units_vs_anchor_pct"].notna().any():
                c2.altair_chart(rl_charts.class_chart(by_class, "units_vs_anchor_pct", "Units vs anchor (%)",
                                                      CLASS_ORDER), width="stretch")
        if ins_eps.empty:
            st.caption("The training-sample side is saved for runs trained from now on; this run shows validation only.")
        else:
            by_month = pd.concat([
                vs_anchor(e, e["window_start"].dt.to_period("M").dt.to_timestamp().rename("month")).assign(block=b)
                for b, e in blocks.items()
            ], ignore_index=True)
            st.markdown("**Over calendar time**")
            c1, c2 = st.columns(2)
            c1.altair_chart(rl_charts.time_chart(by_month, "cost_vs_anchor_pct", "Cost vs anchor (%)"), width="stretch")
            if by_month["units_vs_anchor_pct"].notna().any():
                c2.altair_chart(rl_charts.time_chart(by_month, "units_vs_anchor_pct", "Units vs anchor (%)"),
                                width="stretch")
            st.caption("By the month a window starts: where in the training year the policy beat the anchor, and how "
                       "the validation window compares.")
        d = val_eps.assign(delta_eur=val_eps["cost_eur"] - val_eps["anchor_cost_eur"])
        show = [c for c in ("store", "product", "window_start", "demand_class", "cost_eur", "anchor_cost_eur",
                            "delta_eur", "units", "anchor_units") if c in d]
        worst, best = d.nlargest(10, "delta_eur")[show], d.nsmallest(10, "delta_eur")[show]
        c1, c2 = st.columns(2)
        c1.markdown("**Biggest losses on validation**")
        c1.dataframe(worst, hide_index=True, width="stretch")
        c2.markdown("**Biggest wins on validation**")
        c2.dataframe(best, hide_index=True, width="stretch")
        st.caption(
            f"Net validation Δ {d['delta_eur'].sum():+.1f} € over {len(d)} SKU-windows; the 10 biggest losses add "
            f"{worst['delta_eur'].sum():+.1f} € and the 10 biggest wins {best['delta_eur'].sum():+.1f} €."
        )

# ---------------------------------------------------------------------------
# 3. Order recommendations (project-seed weights, first order round of the test block)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the policy…")
def _policy(rl_run_id: str):
    from src.rl.selection import load_policy

    return load_policy(cfg, rl_run_id)


@st.cache_resource(show_spinner="Loading the forecasting run…", max_entries=2)
def _run_data(forecast_run_id: str):
    """Forecasts, panel and test windows, shared by every configuration of one forecasting run."""
    from src.rl.train import block_episodes, load_run_data

    data = load_run_data(cfg, forecast_run_id)
    return data, block_episodes(cfg, data, "rl_test")


def _load(rl_run_id: str):
    spec, agent = _policy(rl_run_id)
    return (spec, agent, *_run_data(spec.forecast_run_id))


def _first_round(rl_run_id: str, key):
    """The environment and observation at the first order round of ``key``'s test window."""
    from src.simulation.vn2_env import VN2Env

    spec, _, data, episodes = _load(rl_run_id)
    env = VN2Env(spec, episodes[key], data.lookup, data.scales.get(key[:2], 1.0), data.rules)
    obs, _ = env.reset()
    return env, obs


def _explain(env, agent, obs, cls, gate: tuple[str, ...] | None) -> dict:
    """``explain_order``; outside the gate's demand classes the RL order is the base-stock order."""
    e = explain_order(env, agent, obs)
    if gate is not None and cls not in gate:
        e = {**e, "rl_order": e["anchor_order"], "adjustment": 0, "gated_off": True}
    return e


@st.cache_data(show_spinner="Computing order recommendations…")
def _orders(rl_run_id: str, gate: tuple[str, ...] | None) -> pd.DataFrame:
    """First order round of each SKU's first test window: the anchor's order and the policy's."""
    _, agent, _, episodes = _load(rl_run_id)
    first: dict = {}
    for key in sorted(episodes):
        first.setdefault(key[:2], key)
    rows = []
    for key in first.values():
        env, obs = _first_round(rl_run_id, key)
        ep = episodes[key]
        cls = ep.sort_values("date")["demand_class"].iloc[0] if "demand_class" in ep else None
        e = _explain(env, agent, obs, cls, gate)
        a0, a1 = env._arrivals()
        rows.append({
            "store": key[0], "product": key[1], "window_start": key[2],
            "demand class": cls,
            "on hand": env.inventory, "in transit w1": a0, "in transit w2": a1,
            "delivery-week target": round(e["target"], 1),
            "base-stock order": e["anchor_order"], "RL order": e["rl_order"], "Δ units": e["adjustment"],
            "Δ %": round(100.0 * e["adjustment"] / e["anchor_order"], 1) if e["anchor_order"] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _reason(e: dict, spec) -> str:
    """One sentence on how the order was built."""
    a = spec.anchor
    what = f"the {a.tau:.0%} quantile forecast" if a.target == "quantile" else f"the Poisson {a.tau:.0%} quantile"
    if e["anchor_order"] == 0 and e["target"] <= e["projected_stock"]:
        text = (f"About **{e['projected_stock']:.0f}** units are expected in stock when an order placed now arrives, "
                f"at or above the delivery-week target of **{e['target']:.0f}** ({what} of that week's demand), so "
                "base-stock orders nothing.")
    else:
        text = (f"Base-stock orders up to the delivery-week target of **{e['target']:.0f}** ({what} of that week's "
                f"demand) minus the **{e['projected_stock']:.0f}** units expected in stock when the order arrives: "
                f"**{e['anchor_order']}** units.")
    if e["projected_lost"] > 0.5:
        text += f" Before then the forecast expects about {e['projected_lost']:.0f} units of lost sales."
    if e.get("gated_off"):
        return text + " This demand class is outside the validation gate, so the base-stock order stands."
    adj = e["adjustment"]
    if spec.method == "raw":
        return text + f" The RL policy sets the order itself: **{e['rl_order']:.0f}** units ({adj:+.0f} against base-stock)."
    if adj == 0:
        return text + " The RL policy keeps the base-stock order."
    pct = f", {100 * adj / e['anchor_order']:+.0f}%" if e["anchor_order"] > 0 else ""
    return text + f" The RL policy {'adds' if adj > 0 else 'removes'} **{abs(adj):.0f}** units{pct}, for **{e['rl_order']:.0f}**."


with tab_orders:
    focus = _focus("rl_focus_orders")
    deploy = configs.loc[focus, "deploy_run_id"]
    gate = configs.loc[f"{focus}+gate", "gate"] if f"{focus}+gate" in configs.index else None
    if gate is not None and not st.toggle(
        f"Validation gate: RL on {', '.join(gate)}, base-stock elsewhere", value=True,
        help="The demand classes where this configuration beat the anchor on validation (summed over seeds). It is "
             "the 'RL on …' candidate on the Run RL leaderboard.",
    ):
        gate = None
    orders = None
    if not isinstance(deploy, str):
        st.info("This configuration has no project-seed run to take orders from.")
    else:
        try:
            orders = _orders(deploy, gate)
        except Exception as exc:  # noqa: BLE001 - show why instead of a stack trace
            st.warning(f"Could not load this configuration's policy: {exc}")
    if orders is not None and orders.empty:
        st.info("The test block of this forecasting run has no windows.")
    elif orders is not None:
        spec, agent, data, _ = _load(deploy)
        val = lib.rl_episodes(deploy, "val_episodes")
        if not val.empty:
            by_sku = val.groupby(["store", "product"])[["cost_eur", "anchor_cost_eur"]].sum().round(2)
            orders = orders.join(by_sku.rename(columns={"cost_eur": "validation € RL",
                                                        "anchor_cost_eur": "validation € base-stock"}),
                                 on=["store", "product"])
        table = orders.sort_values("Δ units", key=np.abs, ascending=False, ignore_index=True)
        base_units = float(table["base-stock order"].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("SKUs", len(table))
        m2.metric("RL units vs base-stock", f"{100 * (table['RL order'].sum() / base_units - 1):+.1f}%" if base_units > 0 else "—")
        m3.metric("SKUs where RL deviates by more than 10%",
                  int((table["Δ units"].abs() > 0.1 * table["base-stock order"].clip(lower=1)).sum()))
        st.caption(
            f"Project-seed weights of `{deploy[:8]}` · first order round of each SKU's first test window "
            f"({pd.Timestamp(table['window_start'].min()):%Y-%m-%d}). Orders only: test costs stay hidden until the "
            "test is scored on **Run RL**. Select a row to see why."
        )
        event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key=f"rl_orders_{deploy}")
        st.download_button("Download orders (CSV)", table.assign(rl_run_id=deploy, forecast_run_id=fc_id,
                                                                  gate=", ".join(gate) if gate else "")
                           .to_csv(index=False).encode(), file_name=f"rl_orders_{deploy[:8]}.csv", mime="text/csv")

        picked = event.selection.rows
        row = table.iloc[min(picked[0], len(table) - 1) if picked else 0]
        key = (int(row["store"]), int(row["product"]), pd.Timestamp(row["window_start"]))
        env, obs = _first_round(deploy, key)
        e = _explain(env, agent, obs, row["demand class"], gate)
        cls = f" · {row['demand class']}" if pd.notna(row["demand class"]) else ""
        st.markdown(f"#### Store {key[0]} · product {key[1]}{cls}")
        c = st.columns(5)
        c[0].metric("Delivery-week target", f"{e['target']:.0f}")
        c[1].metric("Projected stock at arrival", f"{e['projected_stock']:.0f}")
        c[2].metric("Base-stock order", e["anchor_order"])
        c[3].metric("RL adjustment", f"{e['adjustment']:+.0f}")
        c[4].metric("RL order", f"{e['rl_order']:.0f}")
        st.markdown(_reason(e, spec))

        c1, c2 = st.columns(2)
        panel = data.panel
        sales = panel[(panel["store"] == key[0]) & (panel["product"] == key[1]) & (panel["date"] < key[2])]
        fc = data.lookup.get(key, {})
        week = pd.Timedelta(weeks=1)
        points = pd.DataFrame([
            {"date": key[2], "value": fc.get(spec.anchor.h0), "label": f"week k forecast ({spec.anchor.h0})"},
            {"date": key[2] + week, "value": fc.get(spec.anchor.h1), "label": f"week k+1 forecast ({spec.anchor.h1})"},
            {"date": key[2] + 2 * week, "value": e["target"], "label": "delivery-week target"},
        ]).dropna(subset=["value"])
        c1.markdown("**Sales before the decision, and the forecasts used**")
        c1.altair_chart(rl_charts.sku_chart(sales.sort_values("date").tail(26)[["date", "sales"]], points),
                        width="stretch")
        what = "order" if spec.method == "raw" else "correction"
        effects = drivers(env, agent, obs).head(6)
        c2.markdown(f"**What moves the RL {what}**")
        if e.get("gated_off"):
            c2.caption("Outside the validation gate: the RL policy isn't used for this demand class.")
        elif effects.empty or effects["effect_units"].abs().max() < 0.05:
            c2.caption(f"No single input moves the RL {what} at this decision.")
        else:
            effects = effects.assign(label=[f"{n} ({v:.0f})" for n, v in zip(effects["input"], effects["value"])])
            c2.altair_chart(rl_charts.drivers_chart(effects, f"Change in the RL {what} (units)"), width="stretch")
            c2.caption(
                f"How the RL {what} would change if one input (current value in brackets) were higher by 10% of this "
                f"SKU's peak weekly sales (+{effects['change'].iloc[0]:.1f} units), the others and the base-stock order "
                f"fixed. A local what-if: inputs interact, so the bars don't add up."
            )
        if pd.notna(row.get("validation € RL")):
            st.caption(
                f"On validation this SKU cost {row['validation € RL']:.2f} € with RL vs {row['validation € base-stock']:.2f} € "
                f"with base-stock ({row['validation € RL'] - row['validation € base-stock']:+.2f} €). A single window: a "
                "hint, not proof."
            )
