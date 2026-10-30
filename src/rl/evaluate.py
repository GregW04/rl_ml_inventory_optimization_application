"""
Score the validation champion once on the test block ("production"), in euros.

The champion was fixed on validation (``src.rl.selection``); nothing here feeds back
into training or selection. Each forecasting run gets one test score. Scoring again
needs ``--override-lock`` and the new run is tagged ``post_hoc=true``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.rl.rollout import ZeroResidual, anchor_baseline, paired_bootstrap_ci, per_class_delta, rollout
from src.rl.selection import ANCHOR, champion, leaderboard, load_policy
from src.rl.spec import PolicySpec, default_state
from src.rl.train import RunData, block_episodes, load_run_data, official_window_start
from src.utils import tracking
from src.utils.config import load_config


class LockedTestError(RuntimeError):
    """The forecasting run's test block was already scored."""


def scored_test_runs(config: dict, forecast_run_id: str) -> pd.DataFrame:
    """Finished test scorings (``stage=test``) of a forecasting run, newest first."""
    tracking.init_tracking(config, "rl")
    return tracking.search_runs(
        config["rl"]["mlflow"]["experiment_name"],
        filter_string=(
            f"tags.stage = 'test' and tags.forecast_run_id = '{forecast_run_id}' "
            "and attributes.status = 'FINISHED'"
        ),
    )


def legacy_test_runs(config: dict, forecast_run_id: str) -> pd.DataFrame:
    """Test-block evaluations logged before the lock existed (``policy_evaluation`` runs)."""
    tracking.init_tracking(config, "rl")
    return tracking.search_runs(
        config["rl"]["mlflow"]["experiment_name"],
        filter_string=(
            f"tags.mlflow.runName = 'policy_evaluation' and params.forecast_run_id = '{forecast_run_id}'"
        ),
    )


def _anchor_spec(data: RunData) -> PolicySpec:
    return PolicySpec(
        forecast_run_id=str(data.run_id),
        method="residual",
        agent="ppo",
        state=default_state("residual", data.measure_ids),
        anchor=data.default_anchor,
    )


def _pick(config: dict, data: RunData, which: str) -> tuple[str, PolicySpec, object, pd.Series | None]:
    """(champion id, spec, policy, leaderboard row) for ``auto``, ``anchor`` or an RL run id."""
    row = None
    if which == "auto":
        row = champion(leaderboard(config, str(data.run_id)))
        which = ANCHOR if row is None or row["kind"] == ANCHOR else row["deploy_run_id"]
    if which == ANCHOR:
        return ANCHOR, _anchor_spec(data), ZeroResidual(), row
    spec, agent = load_policy(config, which)
    if spec.forecast_run_id != str(data.run_id):
        raise ValueError(f"RL run {which} was trained on forecasting run {spec.forecast_run_id}.")
    return which, spec, agent, row


def score_test(
    config: dict,
    forecast_run_id: str,
    champion_id: str = "auto",
    override_lock: bool = False,
) -> dict:
    """Score the champion and its anchor on ``rl_test`` and log one ``policy_test`` run."""
    previous = scored_test_runs(config, forecast_run_id)
    if not previous.empty and not override_lock:
        raise LockedTestError(
            f"The test block of forecasting run {forecast_run_id[:8]} was already scored "
            f"(run {previous.iloc[0]['run_id'][:8]}). Pass --override-lock to score again; "
            "the new score is tagged post_hoc."
        )
    # The test block was already seen if it was scored before the lock existed, too.
    legacy = legacy_test_runs(config, forecast_run_id)
    post_hoc = not previous.empty or not legacy.empty
    data = load_run_data(config, forecast_run_id)
    chosen, spec, policy, row = _pick(config, data, champion_id)
    episodes = block_episodes(config, data, "rl_test")
    if not episodes:
        raise RuntimeError("No RL test episodes; check the forecasting run's periods.")

    kpis, per_ep = rollout(policy, spec, episodes, data.lookup, data.scales, data.rules)
    base_spec, base_policy = anchor_baseline(spec)
    an_kpis, an_eps = rollout(base_policy, base_spec, episodes, data.lookup, data.scales, data.rules)
    per_ep = per_ep.merge(
        an_eps[["store", "product", "window_start", "cost_eur"]].rename(columns={"cost_eur": "anchor_cost_eur"}),
        on=["store", "product", "window_start"],
    )
    lo, hi = paired_bootstrap_ci((per_ep["cost_eur"] - per_ep["anchor_cost_eur"]).to_numpy())
    official = per_ep[per_ep["window_start"] == official_window_start(config)]

    metrics = {
        "test_cost_eur_per_sku": kpis["cost_eur_per_sku"],
        "test_holding_eur_per_sku": kpis["holding_eur_per_sku"],
        "test_shortage_eur_per_sku": kpis["shortage_eur_per_sku"],
        "test_anchor_cost_eur_per_sku": an_kpis["cost_eur_per_sku"],
        "test_minus_anchor_eur": kpis["cost_eur_per_sku"] - an_kpis["cost_eur_per_sku"],
        "test_delta_ci_lo": lo,
        "test_delta_ci_hi": hi,
        "test_n_episodes": float(kpis["n_episodes"]),
    }
    for cls, value in per_class_delta(per_ep).items():
        metrics[f"test_minus_anchor_pct_{cls.lower()}"] = value
    if not official.empty:
        metrics["test_vn2_score_eur"] = float(official["cost_eur"].sum())
        metrics["test_anchor_vn2_score_eur"] = float(official["anchor_cost_eur"].sum())

    resolved = data.setup["resolved"]
    tracking.setup_mlflow(config, "rl")
    with mlflow.start_run(run_name="policy_test") as run:
        mlflow.set_tags({
            "stage": "test",
            "forecast_run_id": forecast_run_id,
            "champion": chosen,
            "champion_config_key": ANCHOR if chosen == ANCHOR else spec.config_key(),
            "post_hoc": str(post_hoc).lower(),
            "legacy_test_evaluations": str(len(legacy)),
        })
        mlflow.log_params({
            "champion": chosen,
            "method": "anchor" if chosen == ANCHOR else spec.method,
            "agent": "" if chosen == ANCHOR else spec.agent,
            "anchor": spec.anchor.describe(),
            "selection": "lowest mean validation cost over seeds" if champion_id == "auto" else "manual",
            "rl_test_start": resolved["rl_test_start"],
            "rl_test_end": resolved["rl_test_end"],
        })
        if row is not None:
            mlflow.log_metrics({"val_mean": float(row["val_mean"]), "val_anchor": float(row["anchor_val"])})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_episodes.parquet"
            per_ep.to_parquet(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="test")
        _log_comparison_chart(chosen, metrics)
        return {"test_run_id": run.info.run_id, "champion": chosen, "post_hoc": post_hoc, **metrics}


def _log_comparison_chart(chosen: str, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    label = "anchor (champion)" if chosen == ANCHOR else f"champion {chosen[:8]}"
    ax.bar([label, "anchor"], [metrics["test_cost_eur_per_sku"], metrics["test_anchor_cost_eur_per_sku"]],
           color=["#F58518", "#4C78A8"])
    ax.set_ylabel("Test cost (€ per SKU-window)")
    ax.set_title("Test block (production) — lower is better")
    tracking.log_figure(fig, artifact_path="test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the validation champion once on the test block")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--forecast-run", required=True)
    parser.add_argument("--champion", default="auto", help="auto (default), anchor, or an RL run id")
    parser.add_argument("--override-lock", action="store_true", help="Score again (tagged post_hoc)")
    args = parser.parse_args()
    try:
        out = score_test(load_config(args.config), args.forecast_run, args.champion, args.override_lock)
    except LockedTestError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
