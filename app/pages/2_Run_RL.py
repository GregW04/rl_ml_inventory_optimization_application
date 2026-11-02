"""Pick a forecasting run, configure and train RL on it, select on validation, score test once."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

_here = pathlib.Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "src").is_dir() and (_p / "app").is_dir():
        sys.path.insert(0, str(_p))
        break

import pandas as pd
import streamlit as st

from app import lib
from src.rl.selection import ANCHOR, champion
from src.rl.spec import (
    AGENT_LABELS,
    AGENTS,
    ANCHOR_FEATURES,
    CLASS_FEATURES,
    CONTEXT_FEATURES,
    INVENTORY_FEATURES,
    TRAINING_FORECAST_LABELS,
    TRAINING_FORECASTS,
    build_spec,
    default_state,
    is_torch_agent,
    validate_spec,
    windows_per_iteration,
)
from src.simulation.anchor import AnchorSpec, resolve_anchor
from src.simulation.vn2 import VN2Rules

st.set_page_config(page_title="Run RL", page_icon="🚀", layout="wide")
st.title("Run RL")
st.caption(
    "RL trains on the periods and out-of-sample forecasts of one forecasting run. "
    "Configure the state and method, train on the RL train block, compare on validation, "
    "and score the validation champion once on the test block."
)

base_cfg = lib.get_config()
lib.init_tracking()
RULES = VN2Rules.from_config(base_cfg)


def _progress_note() -> None:
    """Tell the user when a forecasting run is in progress, and when it lands."""
    if not lib.forecasts_in_progress():
        return

    @st.fragment(run_every=10)
    def _note() -> None:
        if lib.forecasts_in_progress():
            st.info(
                "A forecasting run is in progress. Press **Refresh** when it finishes to select it."
            )
        else:
            st.success("The forecasting run finished. Press **Refresh** to select it.")

    _note()


# ---------------------------------------------------------------------------
# 1. Forecasting run + frozen split
# ---------------------------------------------------------------------------
c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
if c2.button("Refresh", width="stretch", help="Reload forecasting runs and select the newest"):
    lib.cached_runs.clear()
    lib.run_setup.clear()
    lib.run_has_forecasts.clear()
    lib.measures_of_run.clear()
    lib.rl_leaderboard.clear()
    lib.test_scores.clear()
    st.session_state.pop("rl_fc_run", None)

runs = lib.forecast_runs()
if runs.empty:
    c1.selectbox("Forecasting run", ["No finished forecasting runs yet"], disabled=True)
    _progress_note()
    st.info("Launch one from **Run Forecasting**; it appears here when it finishes.")
    st.stop()

labels = {lib.forecast_run_label(r): r["run_id"] for _, r in runs.iterrows()}
sel = c1.selectbox(
    "Forecasting run", list(labels), key="rl_fc_run", help="Newest first; defaults to the newest"
)
run_id = labels[sel]
_progress_note()

record = lib.run_setup(run_id)
if record is None:
    st.error("This run has no saved period setup (`forecast_setup.json`), so RL cannot use it.")
    st.stop()

resolved = record.get("resolved", {})
st.subheader("How this run split the data")
label_gap = resolved.get("label_gap_weeks")
st.caption(
    f"Frozen from the run's saved setup. Embargo **{label_gap if label_gap is not None else resolved.get('gap_weeks', '?')}** "
    f"weeks before each scored block (the forecast horizon) · {lib.refit_summary(resolved)}."
)
lib.render_split_view(pd.DataFrame(record.get("timeline") or []), resolved)
if lib.run_has_forecasts(run_id):
    level_id, block_err = lib.run_block_bias(run_id, not lib.test_scores(run_id).empty)
    if block_err:
        st.caption(
            f"Forecast error of `{level_id}` by block: "
            + ", ".join(f"{b} {v:+.1f}%" for b, v in block_err.items())
            + " (positive = forecasts too low). Agents trained on forecasts that ran too low can learn to "
            "order more everywhere; *Training forecasts → unbiased* below removes that. "
            "Details: **Forecasts → Forecast drift by fit**."
        )
with st.expander("Features used by this run"):
    st.json(record.get("features") or {})

has_forecasts = lib.run_has_forecasts(run_id)
if not has_forecasts:
    st.error(
        "This run's RL-block forecasts are not available (not on disk and not logged to "
        "MLflow). Re-run forecasting with the same settings to use this split."
    )
    st.stop()

measures = lib.as_measures(lib.measures_of_run(run_id))
measure_ids = [m.id for m in measures]
default_anchor, anchor_warnings = resolve_anchor(measures, RULES)
st.markdown(
    "**Forecast measures:** "
    + ", ".join(f"`{i}`" for i in measure_ids)
    + (f" · primary `{record['primary_measure']}`" if record.get("primary_measure") else "")
    + (" · quantiles calibrated" if any(record.get("calibration") or {}) else "")
)
if not record.get("measures"):
    st.caption("This run predates measure manifests; its measures were read from its forecast columns.")
if default_anchor is None:
    st.error("; ".join(anchor_warnings))
    st.stop()

# ---------------------------------------------------------------------------
# 2. Configure and train
# ---------------------------------------------------------------------------
st.subheader("Configure and train")
o1, o2, o3 = st.columns(3)
method = o1.radio(
    "Method",
    ["residual", "raw"],
    format_func=lambda m: "Residual on the anchor" if m == "residual" else "Raw (learn from scratch)",
    key="rl_method",
    help="Residual: order = anchor + learned correction (a = 0 is the anchor). "
    "Raw: the agent sets the order itself as a fraction of the SKU's cap.",
)
agent = o2.selectbox(
    "Agent", list(AGENTS), format_func=lambda a: AGENT_LABELS[a], key="rl_agent",
    help="PPO and GRPO train on a batched replay of the training windows and start at the "
    "anchor (residual). DQN is the Stable-Baselines3 baseline.",
)
agent_cfg = base_cfg["rl"]["agents"][agent]
if is_torch_agent(agent):
    iterations = o3.number_input(
        "Iterations", min_value=1, max_value=20_000, value=int(agent_cfg.get("iterations", 300)),
        step=10, key=f"rl_iters_{agent}",
        help="Each iteration replays a batch of training windows and updates the policy.",
    )
    early_stop = {"early_stop": o3.checkbox(
        "Stop early on the training sample", value=bool(agent_cfg.get("early_stop", False)), key=f"rl_early_{agent}",
        help="Iterations become the budget. Training stops once the cost on the fixed training windows behind the "
        "learning curve hasn't improved by the minimum for the patience, and keeps the weights at the stop. "
        "Validation is never used.",
    )}
    if early_stop["early_stop"]:
        early_stop["early_stop_patience"] = int(o3.number_input(
            "Patience (iterations)", min_value=1, max_value=20_000,
            value=int(agent_cfg.get("early_stop_patience", 400)), step=50, key=f"rl_es_patience_{agent}",
            help=f"Checked every {int(agent_cfg.get('early_stop_every', 100))} iterations.",
        ))
        early_stop["early_stop_min_delta"] = float(o3.number_input(
            "Minimum improvement (pts)", min_value=0.0, max_value=10.0,
            value=float(agent_cfg.get("early_stop_min_delta", 0.2)), step=0.05, key=f"rl_es_delta_{agent}",
            help="An evaluation counts as an improvement only if the training-sample cost vs anchor beats the best "
            "so far by at least this many percentage points.",
        ))
    timesteps = None
else:
    default_steps = int(agent_cfg.get("total_timesteps", 100_000))
    timesteps = o3.number_input(
        "Timesteps", min_value=1_000, max_value=20_000_000, value=min(default_steps, 50_000),
        step=1_000, key=f"rl_steps_{agent}",
    )
    iterations = None
    early_stop = None

p1, p2 = st.columns(2)
if method == "residual":
    floor_frac = p1.number_input(
        "Residual floor (× demand scale)", 0.05, 2.0,
        float(base_cfg["rl"]["methods"]["residual"]["floor_frac"]), 0.05, key="rl_floor",
        help="Correction step = max(anchor, floor × peak weekly sales).",
    )
    cap_mult = None
else:
    cap_mult = p1.number_input(
        "Order cap (× peak weekly training sales)", 0.25, 5.0,
        float(base_cfg["rl"]["methods"]["raw"]["order_cap_mult"]), 0.25, key="rl_cap",
    )
    floor_frac = None
default_tf = base_cfg["rl"].get("training_forecasts", "as_is")
training_forecasts = p1.selectbox(
    "Training forecasts", list(TRAINING_FORECASTS),
    index=list(TRAINING_FORECASTS).index(default_tf) if default_tf in TRAINING_FORECASTS else 0,
    format_func=lambda v: TRAINING_FORECAST_LABELS[v], key="rl_training_forecasts",
    help="Unbiased: each training fit's average forecast error is removed (per demand class too), so the "
    "agent cannot learn a year-wide offset such as a growth year's under-forecast. Validation and test "
    "always use the real forecasts.",
)
project_seed = int(base_cfg.get("project", {}).get("seed", 42))
seeds_txt = p2.text_input(
    "Seeds", ", ".join(str(s) for s in base_cfg["rl"].get("seeds", [project_seed])), key="rl_seeds",
    help=f"One training run per seed. Seed {project_seed} must be included: its weights are the ones shipped.",
)

with st.expander(f"Anchor inputs · {default_anchor.describe()}"):
    st.caption(
        "The anchor projects stock through weeks k and k+1 with lost sales, then orders up "
        "to the delivery-week target. Residual RL corrects it; for raw RL it is the baseline."
    )
    for w in anchor_warnings:
        st.warning(w)
    level = {h: [m.id for m in measures if m.horizon == h] for h in (0, 1)}
    a1, a2 = st.columns(2)
    h0 = a1.selectbox("Week k demand", level[0], index=level[0].index(default_anchor.h0), key=f"rl_a0_{run_id}")
    h1 = a2.selectbox("Week k+1 demand", level[1], index=level[1].index(default_anchor.h1), key=f"rl_a1_{run_id}")
    targets: list[AnchorSpec] = [
        AnchorSpec(h0, h1, "quantile", (m.id,), m.tau) for m in measures if m.horizon == 2 and m.kind == "quantile"
    ]
    if "mean_h2" in measure_ids:
        targets.append(AnchorSpec(h0, h1, "poisson", ("mean_h2",), RULES.critical_ratio))
    keyed = {(t.target, t.target_ids): t for t in targets}
    default_key = (default_anchor.target, default_anchor.target_ids)
    target_key = st.selectbox(
        "Delivery-week target",
        list(keyed),
        index=list(keyed).index(default_key) if default_key in keyed else 0,
        format_func=lambda key: keyed[key].describe().split("order up to ")[-1],
        key=f"rl_target_{run_id}",
    )
    anchor = keyed[target_key]

st.markdown("**State**")
s1, s2, s3, s4, s5 = st.columns(5)
s1.multiselect("Inventory (always on)", list(INVENTORY_FEATURES), default=list(INVENTORY_FEATURES), disabled=True)
defaults = default_state(method, measure_ids)
fc_state = s2.multiselect(
    "Forecast measures", measure_ids, default=[i for i in measure_ids if i in defaults], key=f"rl_fc_{method}_{run_id}"
)
an_state = s3.multiselect(
    "Anchor-derived", list(ANCHOR_FEATURES), default=[f for f in ANCHOR_FEATURES if f in defaults],
    key=f"rl_an_{method}",
)
cx_state = s4.multiselect(
    "Context", list(CONTEXT_FEATURES), default=[f for f in CONTEXT_FEATURES if f in defaults],
    key=f"rl_cx_{method}",
)
cls_state = s5.multiselect(
    "Demand class", list(CLASS_FEATURES), default=[], key=f"rl_cls_{method}",
    help="One-hot demand pattern of the SKU that week (Smooth, Erratic, Intermittent, Lumpy), "
    "from its trailing 52 weeks of sales.",
)
state = [*INVENTORY_FEATURES, *[i for i in measure_ids if i in fc_state], *an_state, *cx_state, *cls_state]

spec = build_spec(
    base_cfg, run_id, method, agent, state, anchor,
    total_timesteps=int(timesteps) if timesteps else None,
    iterations=int(iterations) if iterations else None,
    floor_frac=floor_frac, order_cap_mult=cap_mult,
    training_forecasts=training_forecasts, early_stop=early_stop,
)
if is_torch_agent(agent):
    per_it = windows_per_iteration(agent, spec.agent_params)
    group = f" ({spec.agent_params.get('batch_windows')} windows × group of {spec.agent_params.get('group_size')})" \
        if agent == "grpo" else ""
    o3.caption(f"{per_it:,} rollouts per iteration{group} · {spec.total_timesteps:,} environment steps")
errors = validate_spec(spec, measure_ids)
try:
    seeds = [int(x) for x in seeds_txt.replace(",", " ").split()]
except ValueError:
    seeds = []
    errors.append("Seeds must be whole numbers.")
if seeds and project_seed not in seeds:
    errors.append(f"Seeds must include the project seed {project_seed}.")
if not seeds:
    errors.append("Give at least one seed.")
for err in errors:
    st.error(err)
st.caption(f"Configuration `{spec.config_key()}` · {len(state)} state features: " + ", ".join(state))

rl_job = lib.job_state("rl")
running = bool(rl_job and rl_job["running"])
if st.button(
    f"Train ({len(seeds)} seed{'s' if len(seeds) != 1 else ''})",
    type="primary", disabled=running or bool(errors),
):
    fd, spec_path = tempfile.mkstemp(suffix=".json", prefix="rl_spec_")
    os.close(fd)
    spec.save(spec_path)
    cmd = ["-m", "src.rl.train", "--config", lib.build_override_config({}), "--spec", spec_path,
           "--seeds", *[str(s) for s in seeds]]
    lib.start_job("rl", f"Training {spec.label()} on run {run_id[:8]} (seeds {', '.join(map(str, seeds))})", [cmd])
    st.rerun()

# ---------------------------------------------------------------------------
# 3. Validation leaderboard
# ---------------------------------------------------------------------------
st.subheader("Validation leaderboard")
board = lib.rl_leaderboard(run_id)
champ = champion(board)
if board.empty:
    st.info("No trained configurations on this forecasting run yet. The anchor is the champion until one beats it.")
else:
    view = board.assign(
        champion=lambda d: ["🏆" if champ is not None and r == champ["candidate"] else "" for r in d["candidate"]],
        val=lambda d: [f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f}" for m, s, n in zip(d["val_mean"], d["val_std"], d["n_seeds"])],
        delta=lambda d: [
            "—" if k == ANCHOR else (f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]" if pd.notna(lo) else f"{x:+.3f}")
            for k, x, lo, hi in zip(d["kind"], d["delta_mean"], d["delta_ci_lo"], d["delta_ci_hi"])
        ],
        transfer=lambda d: [
            "—" if k == ANCHOR or pd.isna(i) else f"{i:+.1f}% → {v:+.1f}%"
            for k, i, v in zip(d["kind"], d["insample_pct"], d["val_pct"])
        ],
        beats=lambda d: ["—" if pd.isna(b) else f"{b:.0%}" for b in d["share_beats"]],
        units=lambda d: ["—" if k == ANCHOR or pd.isna(u) else f"{u:+.1f}%" for k, u in zip(d["kind"], d["units_vs_anchor"])],
        tf=lambda d: [TRAINING_FORECAST_LABELS.get(v, v) if v else "" for v in d["training_forecasts"]],
    )
    st.dataframe(
        view[["champion", "candidate", "seeds", "val", "delta", "transfer", "beats", "units", "class_delta", "tf",
              "state", "anchor", "total_timesteps"]].rename(
            columns={
                "val": "validation € / SKU-window (mean ± std over seeds)",
                "delta": "Δ vs anchor (95% CI, project seed)",
                "transfer": "training sample → validation (Δ vs anchor, mean over seeds)",
                "beats": "seeds beating anchor",
                "units": "units vs anchor (validation)",
                "class_delta": "Δ vs anchor by demand class (project seed)",
                "tf": "training forecasts",
                "total_timesteps": "environment steps",
            }
        ),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Ranked by mean validation cost over seeds; the champion ships its project-seed weights. "
        "Δ < 0 means cheaper than the anchor. The interval is a paired bootstrap over validation episodes. "
        "Training sample → validation: the final weights on 2,000 fixed training windows, then on validation; "
        "a large jump means the training gain did not transfer (details on **Inventory & RL**). "
        "Units vs anchor: how much more (+) or less the project-seed policy ordered than the anchor on the same "
        "validation windows; a large value is a bet on the demand level ('—' for runs saved before units "
        "were recorded). "
        "'RL on …' rows keep the policy only on the demand classes where its configuration beat the anchor on "
        "validation (summed over seeds) and order the anchor on the rest. That choice is fit on validation, so "
        "their validation score is optimistic; the test score is the honest check."
    )

# ---------------------------------------------------------------------------
# 4. Test (production)
# ---------------------------------------------------------------------------
st.subheader("Test (production)")
scores = lib.test_scores(run_id)
champ_txt = "the anchor" if champ is None or champ["kind"] == ANCHOR else f"`{champ['candidate']}`"
if scores.empty:
    st.caption(
        f"Not scored yet. The champion is {champ_txt}. The test block is scored once; "
        "after that the result is locked."
    )
    n_legacy = lib.legacy_test_count(run_id)
    if n_legacy:
        st.warning(
            f"This run's test block was already evaluated {n_legacy}× by the old evaluation, "
            "before the lock existed. A new score will be tagged post-hoc."
        )
    if st.button("Score champion on test", disabled=running):
        cmd = ["-m", "src.rl.evaluate", "--config", lib.build_override_config({}), "--forecast-run", run_id]
        lib.start_job("rl", f"Test scoring on run {run_id[:8]}", [cmd])
        st.rerun()
else:
    latest = scores.iloc[0]
    t1, t2, t3, t4 = st.columns(4)
    who = latest.get("tags.champion", "?")
    t1.metric("Champion", "anchor" if who == ANCHOR else str(who)[:8])
    t2.metric("Test € / SKU-window", f"{latest.get('metrics.test_cost_eur_per_sku', float('nan')):.3f}")
    t3.metric("Anchor on test", f"{latest.get('metrics.test_anchor_cost_eur_per_sku', float('nan')):.3f}")
    vn2 = latest.get("metrics.test_vn2_score_eur")
    t4.metric("VN2 score (€)", f"{vn2:,.1f}" if pd.notna(vn2) else "—")
    lo, hi = latest.get("metrics.test_delta_ci_lo"), latest.get("metrics.test_delta_ci_hi")
    st.caption(
        f"Δ vs anchor {latest.get('metrics.test_minus_anchor_eur', float('nan')):+.3f} € "
        + (f"(95% CI [{lo:+.3f}, {hi:+.3f}])" if pd.notna(lo) else "")
        + f" · scored {latest['start_time']:%Y-%m-%d %H:%M}"
        + (f" · RL on {g}, anchor elsewhere" if isinstance(g := latest.get("tags.champion_gate"), str) and g else "")
        + (" · **post-hoc**" if latest.get("tags.post_hoc") == "true" else "")
    )
    st.warning(
        "Locked: this forecasting run's test block has been scored. Another score is "
        "post-hoc and is tagged as such in MLflow."
    )
    override = st.checkbox("I understand: score again as post-hoc", key=f"rl_override_{run_id}")
    if st.button("Re-score (post-hoc)", disabled=running or not override):
        cmd = ["-m", "src.rl.evaluate", "--config", lib.build_override_config({}), "--forecast-run", run_id,
               "--override-lock"]
        lib.start_job("rl", f"Post-hoc test scoring on run {run_id[:8]}", [cmd])
        st.rerun()
    if len(scores) > 1:
        with st.expander(f"All test scorings ({len(scores)})"):
            st.dataframe(
                scores[[c for c in ["start_time", "tags.champion", "tags.post_hoc",
                                    "metrics.test_cost_eur_per_sku", "metrics.test_anchor_cost_eur_per_sku"]
                        if c in scores]],
                hide_index=True, width="stretch",
            )

job = lib.job_panel("rl")
if job and job["exit_code"] == 0:
    st.page_link("pages/5_Inventory_RL.py", label="Training curves and order recommendations", icon="➡️")
