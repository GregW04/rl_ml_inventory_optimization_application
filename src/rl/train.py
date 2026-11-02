"""
Train inventory agents on one forecasting run, one MLflow run per seed.

Agents: PPO and GRPO on the batched replay of :mod:`src.rl.batch_env` (:mod:`src.rl.ppo`,
:mod:`src.rl.grpo`). DQN (Stable-Baselines3) was retired on 2026-11-02; its runs keep their
metrics and episodes in MLflow.

Protocol: fit on the ``rl_train`` windows for the full budget, keep the final weights,
then score them once on ``rl_val`` next to the anchor. Every agent is scored by the same
``VN2Env`` rollout. The test block is never loaded here; it is scored once, for the chosen
champion, by ``src.rl.evaluate``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.forecasting.measures import Measure, run_measures
from src.forecasting.periods import fetch_run_file, load_run_setup
from src.forecasting.preprocessing import load_initial_state, load_panel_from_config
from src.rl.agents import model_file
from src.rl.rollout import anchor_baseline, per_class_delta, per_class_units, rollout, units_vs_anchor
from src.rl.spec import PolicySpec, validate_spec
from src.rl.vn2_data import _PANEL_COLS, build_forecast_lookup, build_vn2_episodes, compute_demand_scales
from src.simulation.anchor import AnchorSpec, resolve_anchor
from src.simulation.vn2 import VN2Rules
from src.simulation.vn2_env import VN2Env
from src.utils import tracking
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Forecasting-run data
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """Everything RL needs from one forecasting run."""

    run_id: str | None
    setup: dict
    measures: list[Measure]
    lookup: dict
    panel: pd.DataFrame
    scales: dict[tuple[int, int], float]
    default_anchor: AnchorSpec
    anchor_warnings: list[str]
    rules: VN2Rules

    @property
    def measure_ids(self) -> list[str]:
        return [m.id for m in self.measures]


def official_window_start(config: dict) -> pd.Timestamp | None:
    """First week of the official VN2 window (week after the initial-state date)."""
    init_date = config.get("data", {}).get("initial_state_date")
    return pd.Timestamp(init_date) + pd.Timedelta(weeks=1) if init_date else None


def load_run_data(config: dict, run_id: str | None) -> RunData:
    """Forecasts, measures, panel and demand scales of a forecasting run (None → latest)."""
    setup = load_run_setup(config, run_id)
    path = fetch_run_file(config, run_id, "train_forecasts.parquet")
    if setup is None or path is None:
        which = f"forecasting run {run_id}" if run_id else "the latest forecasting run on disk"
        raise RuntimeError(
            f"No period setup or forecasts found for {which}. "
            "Run the forecasting pipeline from the app or CLI first."
        )
    rules = VN2Rules.from_config(config)
    measures = run_measures(config, run_id)
    anchor, warnings = resolve_anchor(measures, rules)
    if anchor is None:
        raise RuntimeError("; ".join(warnings))
    resolved = setup["resolved"]
    panel = load_panel_from_config(
        config, start=pd.Timestamp(resolved["data_start"]), end=pd.Timestamp(resolved["data_end"])
    )
    if "in_stock" not in panel.columns:
        panel["in_stock"] = True
    forecasts = pd.read_parquet(path)
    return RunData(
        run_id=setup.get("run_id") or run_id,
        setup=setup,
        measures=measures,
        lookup=build_forecast_lookup(forecasts, [m.id for m in measures]),
        panel=panel,
        # Scales come from history up to the end of RL training only (no peeking).
        scales=compute_demand_scales(panel[_PANEL_COLS], cap_end=pd.Timestamp(resolved["rl_train_end"])),
        default_anchor=anchor,
        anchor_warnings=warnings,
        rules=rules,
    )


def block_episodes(config: dict, data: RunData, block: str) -> dict:
    """VN2 episodes of ``rl_train``, ``rl_val`` or ``rl_test`` for a forecasting run."""
    resolved = data.setup["resolved"]
    start, end = pd.Timestamp(resolved[f"{block}_start"]), pd.Timestamp(resolved[f"{block}_end"])
    official_start = official_window_start(config)
    initial_state = None
    if official_start is not None and block == "rl_test" and start <= official_start <= end:
        initial_state = load_initial_state(config["data"]["initial_state_file"])
    return build_vn2_episodes(
        data.panel,
        start,
        end,
        data.lookup,
        config,
        data.default_anchor,
        initial_state=initial_state,
        official_window_start=official_start,
    )


def runs_dir(config: dict) -> Path:
    """Local folder holding one sub-folder of artifacts per RL run."""
    rel = config["rl"].get("artifacts", {}).get("runs_dir", "artifacts/rl/runs")
    return Path(config["_project_root"]) / rel


# ---------------------------------------------------------------------------
# In-sample learning curve (display only: weights are never picked from it)
# ---------------------------------------------------------------------------

INSAMPLE_WINDOWS = 2_000
CURVE_METRIC = "insample_vs_anchor_pct"
UNITS_METRIC = "insample_units_vs_anchor_pct"


def insample_keys(train_eps: dict, n: int = INSAMPLE_WINDOWS, seed: int = 123) -> list:
    """A fixed random sample of training windows for the learning curve."""
    keys = list(train_eps)
    if not keys:
        return []
    rng = np.random.default_rng(seed)
    return [keys[i] for i in rng.choice(len(keys), min(n, len(keys)), replace=False)]


def log_curve_chart(points: list[tuple[int, float]], title: str) -> None:
    """Log the in-sample curve (cost against the anchor, %) as a figure."""
    pts = [(w, v) for w, v in points if w > 0]
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot([w for w, _ in pts], [v for _, v in pts], marker="o", color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("Training windows replayed")
    ax.set_ylabel("Cost vs anchor on 2,000 training windows (%)")
    ax.set_title(f"{title} — in-sample curve (below 0 beats the anchor)")
    tracking.log_figure(fig, artifact_path="rl_charts")


# ---------------------------------------------------------------------------
# Train + validate
# ---------------------------------------------------------------------------

def validate(agent, spec: PolicySpec, data: RunData, episodes: dict) -> tuple[dict, pd.DataFrame]:
    """Score the final agent and its anchor once on ``rl_val``; return metrics + per episode."""
    rl_kpis, rl_eps = rollout(agent, spec, episodes, data.lookup, data.scales, data.rules)
    base_spec, base_policy = anchor_baseline(spec)
    an_kpis, an_eps = rollout(base_policy, base_spec, episodes, data.lookup, data.scales, data.rules)
    per_ep = rl_eps.merge(
        an_eps[["store", "product", "window_start", "cost_eur", "units"]].rename(
            columns={"cost_eur": "anchor_cost_eur", "units": "anchor_units"}
        ),
        on=["store", "product", "window_start"],
    )
    delta = rl_kpis["cost_eur_per_sku"] - an_kpis["cost_eur_per_sku"]
    metrics = {
        "val_cost_eur_per_sku": rl_kpis["cost_eur_per_sku"],
        "val_holding_eur_per_sku": rl_kpis["holding_eur_per_sku"],
        "val_shortage_eur_per_sku": rl_kpis["shortage_eur_per_sku"],
        "val_anchor_cost_eur_per_sku": an_kpis["cost_eur_per_sku"],
        "val_minus_anchor_eur": delta,
        "val_n_episodes": float(rl_kpis["n_episodes"]),
        "beats_anchor": float(delta < 0),
    }
    for cls, value in per_class_delta(per_ep).items():
        metrics[f"val_minus_anchor_pct_{cls.lower()}"] = value
    units = units_vs_anchor(per_ep)
    if units is not None:
        metrics["val_units_vs_anchor_pct"] = units
    for cls, value in per_class_units(per_ep).items():
        metrics[f"val_units_vs_anchor_pct_{cls.lower()}"] = value
    return metrics, per_ep


def _train_torch(spec: PolicySpec, data: RunData, train_eps: dict, sub_keys: list, lookup: dict):
    """PPO or GRPO on the batched replay; returns (TorchPolicy, curve points)."""
    import torch

    from src.rl.batch_env import build_episode_arrays
    from src.rl.grpo import train_grpo
    from src.rl.ppo import train_ppo

    arrays = build_episode_arrays(train_eps, lookup, spec, data.scales, data.rules)
    pos = {k: i for i, k in enumerate(arrays.keys)}
    eval_idx = torch.tensor([pos[k] for k in sub_keys], dtype=torch.long)
    iterations = int(spec.agent_params.get("iterations", 1))
    log_every, print_every = max(1, iterations // 100), max(1, iterations // 20)
    points: list[tuple[int, float]] = []
    name = spec.agent.upper()

    def on_iteration(it: int, stats: dict) -> None:
        if it % log_every == 0 or it == iterations:
            mlflow.log_metrics({k: v for k, v in stats.items() if np.isfinite(v)}, step=it)
        if it % print_every == 0 or it == iterations:
            print(f"[{spec.agent}] iteration {it}/{iterations} · sampled rollouts vs anchor "
                  f"{stats['train_vs_anchor_pct']:+.2f}% · units vs anchor {stats['train_units_vs_anchor_pct']:+.1f}% "
                  f"· policy std {stats['policy_std']:.3f}", flush=True)

    def on_eval(windows: int, pct: float, units: float) -> None:
        mlflow.log_metrics({CURVE_METRIC: pct, UNITS_METRIC: units}, step=int(windows))
        points.append((int(windows), pct))
        print(f"[{spec.agent}] {windows:>10,} windows · training-sample cost vs anchor {pct:+.2f}%", flush=True)

    trainer = train_ppo if spec.agent == "ppo" else train_grpo
    print(f"[{spec.agent}] {name} on {len(arrays):,} training windows, {iterations} iterations", flush=True)
    policy = trainer(spec, arrays, data.rules, seed=spec.seed, eval_idx=eval_idx,
                     on_iteration=on_iteration, on_eval=on_eval)
    ran = int(policy.meta.get("iterations_run", iterations))
    mlflow.log_metric("iterations_run", ran)
    if ran < iterations:
        p = spec.agent_params
        print(f"[{spec.agent}] stopped early at iteration {ran} of {iterations}: the training sample had not improved "
              f"by {p.get('early_stop_min_delta')} pts for {p.get('early_stop_patience')} iterations", flush=True)
    return policy, points


def train_agent(
    spec: PolicySpec,
    config: dict,
    data: RunData | None = None,
    train_eps: dict | None = None,
    val_eps: dict | None = None,
) -> dict:
    """Train one seed of ``spec`` on ``rl_train``, validate once on ``rl_val``, save artifacts."""
    data = data or load_run_data(config, spec.forecast_run_id)
    errors = validate_spec(spec, data.measure_ids)
    if errors:
        raise ValueError("; ".join(errors))
    train_eps = train_eps if train_eps is not None else block_episodes(config, data, "rl_train")
    val_eps = val_eps if val_eps is not None else block_episodes(config, data, "rl_val")
    project_seed = int(config.get("project", {}).get("seed", 42))
    sub_keys = insample_keys(train_eps)
    # Training may use forecasts with each fit's average error removed; validation never does.
    train_lookup, bias_factors = data.lookup, None
    if spec.training_forecasts != "as_is":
        from src.rl.debias import training_lookup

        train_lookup, bias_factors = training_lookup(
            data, config, by_class=spec.training_forecasts == "unbiased_by_class"
        )

    tracking.setup_mlflow(config, "rl")
    resolved = data.setup["resolved"]

    with mlflow.start_run(run_name=f"{spec.method}_{spec.agent}_s{spec.seed}") as run:
        mlflow.set_tags({
            "stage": "train",
            "forecast_run_id": spec.forecast_run_id,
            "config_key": spec.config_key(),
            "seed": spec.seed,
            "method": spec.method,
            "agent": spec.agent,
            "role": "deploy" if spec.seed == project_seed else "seed_check",
        })
        params = {
            "method": spec.method,
            "agent": spec.agent,
            "seed": spec.seed,
            "state": ",".join(spec.state),
            "n_state": len(spec.state),
            "anchor": spec.anchor.describe(),
            "total_timesteps": spec.total_timesteps,
            "forecast_run_id": spec.forecast_run_id,
            "rl_train_start": resolved["rl_train_start"],
            "rl_train_end": resolved["rl_train_end"],
            "rl_val_start": resolved["rl_val_start"],
            "rl_val_end": resolved["rl_val_end"],
            "n_train_episodes": len(train_eps),
            "n_insample_windows": len(sub_keys),
            "training_forecasts": spec.training_forecasts,
            **{f"agent_{k}": v for k, v in spec.agent_params.items()},
        }
        if spec.method == "residual":
            params["floor_frac"] = spec.floor_frac
        else:
            params["order_cap_mult"] = spec.order_cap_mult
        mlflow.log_params(params)

        run_dir = runs_dir(config) / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if bias_factors is not None:
            bias_factors.to_csv(run_dir / "training_bias_factors.csv", index=False)
            print("[train] training forecasts corrected per cell: "
                  + ", ".join(f"{r.fit}/{r.demand_class} ×{r.factor:.3f}" for r in bias_factors.itertuples()),
                  flush=True)
        agent, points = _train_torch(spec, data, train_eps, sub_keys, train_lookup)
        agent.save(run_dir / model_file(spec.agent))
        summary = {"final_insample_vs_anchor_pct": points[-1][1]} if points else {}
        if summary:
            mlflow.log_metrics(summary)
        log_curve_chart(points, spec.label())

        # One validation pass on the final weights: it informs selection, nothing else.
        val, per_ep = validate(agent, spec, data, val_eps)
        mlflow.log_metrics(val)

        spec.save(run_dir / "spec.json")
        per_ep.to_parquet(run_dir / "val_episodes.parquet", index=False)
        # The final weights on the curve's fixed training windows and forecasts, per episode (display only).
        _, insample_eps = validate(agent, spec, replace(data, lookup=train_lookup), {k: train_eps[k] for k in sub_keys})
        insample_eps.to_parquet(run_dir / "insample_episodes.parquet", index=False)
        mlflow.log_artifacts(str(run_dir), artifact_path="policy")

        return {
            "rl_run_id": run.info.run_id,
            "run_dir": str(run_dir),
            "config_key": spec.config_key(),
            "seed": spec.seed,
            **summary,
            **val,
        }


def train_seeds(spec: PolicySpec, seeds: list[int], config: dict) -> list[dict]:
    """Train ``spec`` once per seed on shared episodes; the project seed must be included."""
    project_seed = int(config.get("project", {}).get("seed", 42))
    if project_seed not in seeds:
        raise ValueError(f"Seeds must include the project seed {project_seed} (its weights are shipped).")
    data = load_run_data(config, spec.forecast_run_id)
    errors = validate_spec(spec, data.measure_ids)
    if errors:
        raise ValueError("; ".join(errors))
    train_eps = block_episodes(config, data, "rl_train")
    val_eps = block_episodes(config, data, "rl_val")
    results = []
    for seed in seeds:
        print(f"\n========== {spec.label()} · seed {seed} ==========", flush=True)
        results.append(train_agent(spec.with_seed(seed), config, data, train_eps, val_eps))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inventory RL agents (one run per seed)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--spec", required=True, help="Policy spec JSON (see src/rl/spec.py)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Default: rl.seeds")
    args = parser.parse_args()
    config = load_config(args.config)
    spec = PolicySpec.load(args.spec)
    seeds = args.seeds or [int(s) for s in config["rl"].get("seeds", [spec.seed])]
    results = train_seeds(spec, seeds, config)
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
