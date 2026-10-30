"""
Validation leaderboard and champion for one forecasting run.

RL runs that share a ``config_key`` differ only by seed. Each configuration is ranked by
its mean validation cost over seeds; the anchor is a candidate too. The champion ships
the weights of its project-seed run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.rl.agents import load_agent
from src.rl.rollout import format_class_delta, paired_bootstrap_ci, per_class_delta
from src.rl.spec import PolicySpec
from src.utils import tracking

ANCHOR = "anchor"


def train_runs(config: dict, forecast_run_id: str) -> pd.DataFrame:
    """Finished training runs (``stage=train``) on a forecasting run."""
    tracking.init_tracking(config, "rl")
    runs = tracking.search_runs(
        config["rl"]["mlflow"]["experiment_name"],
        max_results=1000,
        filter_string=(
            f"tags.stage = 'train' and tags.forecast_run_id = '{forecast_run_id}' "
            "and attributes.status = 'FINISHED'"
        ),
    )
    if runs.empty or "metrics.val_cost_eur_per_sku" not in runs:
        return pd.DataFrame()
    return runs.dropna(subset=["metrics.val_cost_eur_per_sku"])


def policy_dir(config: dict, rl_run_id: str) -> Path:
    """Local folder with a run's ``model.zip``, ``spec.json`` and validation episodes."""
    from src.rl.train import runs_dir

    local = runs_dir(config) / rl_run_id
    if (local / "spec.json").exists():
        return local
    tracking.init_tracking(config, "rl")
    return Path(tracking.download_artifact(rl_run_id, "policy", dst_path=str(local / "_download")))


def load_policy(config: dict, rl_run_id: str):
    """(spec, agent) saved by a training run (DQN ``model.zip`` or PPO / GRPO ``model.pt``)."""
    d = policy_dir(config, rl_run_id)
    spec = PolicySpec.load(d / "spec.json")
    return spec, load_agent(spec.agent, d)


def _val_summary(config: dict, rl_run_id: str | None) -> tuple[float, float, str]:
    """(CI low, CI high, per-class change) from a run's saved validation episodes."""
    if not rl_run_id:
        return float("nan"), float("nan"), "—"
    try:
        eps = pd.read_parquet(policy_dir(config, rl_run_id) / "val_episodes.parquet")
    except Exception:  # noqa: BLE001 - a missing artifact only hides the interval
        return float("nan"), float("nan"), "—"
    lo, hi = paired_bootstrap_ci((eps["cost_eur"] - eps["anchor_cost_eur"]).to_numpy())
    return lo, hi, format_class_delta(per_class_delta(eps))


def leaderboard(config: dict, forecast_run_id: str) -> pd.DataFrame:
    """
    One row per candidate, best (lowest mean validation cost) first.

    Columns: candidate, kind (``rl`` / ``anchor``), config_key, method, agent, state,
    anchor, n_seeds, seeds, val_mean, val_std, val_min, val_max, anchor_val,
    delta_mean, delta_ci_lo / delta_ci_hi (paired bootstrap on the project-seed run's
    episodes), class_delta (change per demand class, project-seed run), share_beats,
    deploy_run_id and total_timesteps.
    """
    runs = train_runs(config, forecast_run_id)
    if runs.empty or "tags.config_key" not in runs:
        return pd.DataFrame()
    runs = runs.dropna(subset=["tags.config_key"])
    rows = []
    anchors: dict[str, list[float]] = {}
    for key, g in runs.groupby("tags.config_key"):
        cost = g["metrics.val_cost_eur_per_sku"].astype(float)
        anchor_val = float(g["metrics.val_anchor_cost_eur_per_sku"].astype(float).mean())
        deploy = g[g.get("tags.role", pd.Series(index=g.index, dtype=str)) == "deploy"]
        deploy_id = deploy.iloc[0]["run_id"] if not deploy.empty else None
        lo, hi, class_delta = _val_summary(config, deploy_id)
        first = g.iloc[0]
        anchor_desc = first.get("params.anchor", "")
        anchors.setdefault(anchor_desc, []).append(anchor_val)
        rows.append({
            "candidate": f"{first.get('tags.method')} · {str(first.get('tags.agent', '')).upper()} · {key}",
            "kind": "rl",
            "config_key": key,
            "method": first.get("tags.method"),
            "agent": first.get("tags.agent"),
            "state": first.get("params.state"),
            "anchor": anchor_desc,
            "n_seeds": len(g),
            "seeds": ",".join(sorted(str(s) for s in g["tags.seed"])),
            "val_mean": float(cost.mean()),
            "val_std": float(cost.std(ddof=1)) if len(g) > 1 else 0.0,
            "val_min": float(cost.min()),
            "val_max": float(cost.max()),
            "anchor_val": anchor_val,
            "delta_mean": float(cost.mean()) - anchor_val,
            "delta_ci_lo": lo,
            "delta_ci_hi": hi,
            "class_delta": class_delta,
            "share_beats": float((cost < g["metrics.val_anchor_cost_eur_per_sku"].astype(float)).mean()),
            "deploy_run_id": deploy_id,
            "total_timesteps": first.get("params.total_timesteps"),
        })
    for i, (desc, vals) in enumerate(anchors.items()):
        val = float(np.mean(vals))
        rows.append({
            "candidate": "anchor" if len(anchors) == 1 else f"anchor {i + 1}",
            "kind": ANCHOR, "config_key": ANCHOR if len(anchors) == 1 else f"{ANCHOR}{i + 1}",
            "method": "anchor", "agent": "", "state": "", "anchor": desc, "n_seeds": 0, "seeds": "",
            "val_mean": val, "val_std": 0.0, "val_min": val, "val_max": val, "anchor_val": val,
            "delta_mean": 0.0, "delta_ci_lo": 0.0, "delta_ci_hi": 0.0, "class_delta": "—", "share_beats": np.nan,
            "deploy_run_id": None, "total_timesteps": None,
        })
    board = pd.DataFrame(rows)
    # Lowest mean first; on a tie the simpler anchor wins.
    board["_tie"] = (board["kind"] != ANCHOR).astype(int)
    return board.sort_values(["val_mean", "_tie"]).drop(columns="_tie").reset_index(drop=True)


def champion(board: pd.DataFrame) -> pd.Series | None:
    """Best candidate by mean validation cost (an RL row needs its project-seed run)."""
    if board.empty:
        return None
    eligible = board[(board["kind"] == ANCHOR) | board["deploy_run_id"].notna()]
    return eligible.iloc[0] if not eligible.empty else None
