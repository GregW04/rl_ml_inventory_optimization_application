"""Pick a forecasting run, see how it split the data, then train / evaluate RL on it."""

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

st.set_page_config(page_title="Run RL", page_icon="🚀", layout="wide")
st.title("Run RL")
st.caption(
    "RL trains and is scored on the periods and out-of-sample forecasts of one forecasting "
    "run. Pick the run, check how it split the data, then launch training or evaluation."
)

base_cfg = lib.get_config()
lib.init_tracking()


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


c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
if c2.button("Refresh", width="stretch", help="Reload forecasting runs and select the newest"):
    lib.cached_runs.clear()
    lib.run_setup.clear()
    lib.run_has_forecasts.clear()
    st.session_state.pop("rl_fc_run", None)

runs = lib.forecast_runs()
if runs.empty:
    c1.selectbox("Forecasting run", ["No finished forecasting runs yet"], disabled=True)
    _progress_note()
    st.info("Launch one from **Run Forecasting**; it appears here when it finishes.")
    st.stop()


def _label(r: pd.Series) -> str:
    started = r["start_time"].to_pydatetime().astimezone()
    wql = r.get("metrics.cv_wql_mean")
    score = f" · CV WQL {wql:.3f}" if pd.notna(wql) else ""
    return f"{started:%Y-%m-%d %H:%M} · {r['run_id'][:8]}{score}"


labels = {_label(r): r["run_id"] for _, r in runs.iterrows()}
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
refit = "yes" if resolved.get("refit_per_block", True) else "no"
if label_gap is None:
    st.caption(
        f"Frozen from the run's saved setup. Gap **{resolved.get('gap_weeks', '?')}** weeks · "
        f"feature warm-up **{resolved.get('warmup_weeks', '?')}** weeks · forecaster refit per "
        f"RL block: **{refit}**."
    )
else:
    extra = int(resolved.get("gap_weeks", 0)) - int(label_gap)
    if extra > 0:
        embargo_txt = (
            f"Embargo **{resolved.get('gap_weeks', '?')}** weeks before each scored block "
            f"({extra} from features, {label_gap} label gap)"
        )
    else:
        embargo_txt = (
            f"Embargo **{label_gap}** weeks before each scored block (the forecast horizon)"
        )
    st.caption(
        f"Frozen from the run's saved setup. {embargo_txt} · "
        f"forecaster refit per RL block: **{refit}**."
    )
lib.render_split_view(pd.DataFrame(record.get("timeline") or []), resolved)
with st.expander("Features used by this run"):
    st.json(record.get("features") or {})

has_forecasts = lib.run_has_forecasts(run_id)
if not has_forecasts:
    st.error(
        "This run's RL-block forecasts are not available (not on disk and not logged to "
        "MLflow). Re-run forecasting with the same settings to use this split."
    )

st.subheader("RL run")
o1, o2 = st.columns(2)
agent = o1.selectbox("Agent", ["ppo", "dqn"], key="rl_agent")
default_steps = int(base_cfg["rl"]["agents"][agent].get("total_timesteps", 100_000))
timesteps = o2.number_input(
    "Timesteps",
    min_value=1_000,
    max_value=20_000_000,
    value=min(default_steps, 50_000),
    step=1_000,
    key=f"rl_steps_{agent}",
)

meta = lib.agent_meta(agent)
trained_on = meta.get("forecast_run_id")
if not lib.rl_model_path(agent).exists():
    st.caption(f"No saved {agent.upper()} agent yet; Evaluate scores the baselines only.")
elif trained_on == run_id:
    steps = meta.get("total_timesteps")
    st.caption(
        f"The saved {agent.upper()} agent was trained on this run"
        + (f" ({steps:,} timesteps)." if steps else ".")
    )
elif trained_on:
    st.warning(
        f"The saved {agent.upper()} agent was trained on forecasting run `{trained_on[:8]}`. "
        "Train it on this run before evaluating, or the test score mixes two setups."
    )
else:
    st.warning(
        f"The saved {agent.upper()} agent has no record of the run it was trained on. "
        "Train it on this run before evaluating."
    )

rl_job = lib.job_state("rl")
blocked = not has_forecasts or bool(rl_job and rl_job["running"])
b1, b2, b3, _ = st.columns([1, 1, 1, 2])
do_train = b1.button("Train", type="primary", disabled=blocked, width="stretch")
do_eval = b2.button(
    "Evaluate",
    disabled=blocked,
    width="stretch",
    help="Score the saved agent and the baselines on this run's RL test block",
)
do_both = b3.button("Train + evaluate", disabled=blocked, width="stretch")

if do_train or do_eval or do_both:
    cfg_path = lib.build_override_config(
        {"rl": {"agents": {agent: {"total_timesteps": int(timesteps)}}}}
    )
    train_cmd = [
        "-m", "src.rl.train", "--config", cfg_path, "--agent", agent, "--forecast-run", run_id
    ]
    eval_cmd = [
        "-m", "src.rl.evaluate", "--config", cfg_path, "--agents", agent, "--forecast-run", run_id
    ]
    on_run = f"on run {run_id[:8]}"
    if do_train:
        lib.start_job("rl", f"Training {agent.upper()} {on_run}", [train_cmd])
    elif do_eval:
        lib.start_job("rl", f"Evaluation {on_run}", [eval_cmd])
    else:
        lib.start_job("rl", f"Training + evaluation of {agent.upper()} {on_run}", [train_cmd, eval_cmd])
    st.rerun()

job = lib.job_panel("rl")
if job and job["exit_code"] == 0:
    st.page_link(
        "pages/5_Inventory_RL.py", label="Training curves and order recommendations", icon="➡️"
    )
