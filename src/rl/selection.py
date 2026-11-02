"""
Validation leaderboard and champion for one forecasting run.

RL runs that share a ``config_key`` differ only by seed. Each configuration is ranked by
its mean validation cost over seeds; the anchor is a candidate too. The champion ships
the weights of its project-seed run.

A configuration that beats the anchor on some demand classes and loses on others also gets
a gated candidate: the policy on the classes where it won on validation, the anchor on the
rest. The gate is fit on validation, so that candidate's validation score is optimistic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.features.demand_profile import DEMAND_CLASSES, UNCLASSIFIED
from src.rl.agents import load_agent
from src.rl.rollout import format_class_delta, paired_bootstrap_ci, per_class_delta, units_vs_anchor
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
    """(spec, agent) saved by a training run (PPO / GRPO ``model.pt``; retired DQN runs raise)."""
    d = policy_dir(config, rl_run_id)
    spec = PolicySpec.load(d / "spec.json")
    return spec, load_agent(spec.agent, d)


def _episodes(config: dict, rl_run_id: str, name: str) -> pd.DataFrame | None:
    """A run's saved episode table (``val_episodes``, ``insample_episodes``), or None if missing."""
    try:
        return pd.read_parquet(policy_dir(config, rl_run_id) / f"{name}.parquet")
    except Exception:  # noqa: BLE001 - a missing artifact only hides what it feeds
        return None


def _val_summary(eps: pd.DataFrame | None) -> tuple[float, float, str, float]:
    """(CI low, CI high, per-class change, units vs anchor %) from a run's validation episodes."""
    if eps is None:
        return float("nan"), float("nan"), "—", float("nan")
    lo, hi = paired_bootstrap_ci((eps["cost_eur"] - eps["anchor_cost_eur"]).to_numpy())
    units = units_vs_anchor(eps)
    return lo, hi, format_class_delta(per_class_delta(eps)), float("nan") if units is None else units


def class_gate(val_eps: list[pd.DataFrame]) -> tuple[str, ...] | None:
    """
    Demand classes on which a configuration beat the anchor on validation: its cost summed
    over seeds below the anchor's. None when the episodes carry no classes.
    """
    need = {"cost_eur", "anchor_cost_eur", "demand_class"}
    if not val_eps or any(not need <= set(e.columns) for e in val_eps):
        return None
    eps = pd.concat(val_eps, ignore_index=True)
    if eps["demand_class"].isna().all():
        return None
    delta = (eps["cost_eur"] - eps["anchor_cost_eur"]).groupby(eps["demand_class"]).sum()
    return tuple(c for c in (*DEMAND_CLASSES, UNCLASSIFIED) if c in delta.index and delta[c] < 0)


def gate_episodes(eps: pd.DataFrame, gate: tuple[str, ...]) -> pd.DataFrame:
    """
    ``eps`` as ordered under ``gate``: episodes of other classes take the anchor's cost and
    units (exact, since SKU-windows don't interact). Holding and shortage are dropped, as the
    anchor's split isn't saved.
    """
    out = eps.drop(columns=["holding_eur", "shortage_eur"], errors="ignore").copy()
    off = ~out["demand_class"].isin(gate)
    out.loc[off, "cost_eur"] = out.loc[off, "anchor_cost_eur"]
    if {"units", "anchor_units"} <= set(out.columns):
        out.loc[off, "units"] = out.loc[off, "anchor_units"]
    return out


def _gated_row(row: dict, gate: tuple[str, ...], val_eps: dict[str, pd.DataFrame],
               ins_eps: list[pd.DataFrame]) -> dict:
    """``row``'s configuration with the policy kept only on the ``gate`` classes."""
    gated = {rid: gate_episodes(e, gate) for rid, e in val_eps.items()}
    cost = np.array([e["cost_eur"].sum() / len(e) for e in gated.values()])
    anchor = np.array([e["anchor_cost_eur"].sum() / len(e) for e in gated.values()])
    pct = 100.0 * (cost / anchor - 1.0)
    ins = [100.0 * (e["cost_eur"].sum() / e["anchor_cost_eur"].sum() - 1.0)
           for e in (gate_episodes(x, gate) for x in ins_eps)]
    lo, hi, class_delta, units = _val_summary(gated.get(row["deploy_run_id"]))
    return {
        **row,
        "candidate": f"{row['candidate']} · RL on {', '.join(gate)}",
        "config_key": f"{row['config_key']}+gate",
        "gate": gate,
        "val_mean": float(cost.mean()),
        "val_std": float(cost.std(ddof=1)) if len(cost) > 1 else 0.0,
        "val_min": float(cost.min()),
        "val_max": float(cost.max()),
        "delta_mean": float(cost.mean()) - row["anchor_val"],
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "class_delta": class_delta,
        "units_vs_anchor": units,
        "insample_pct": float(np.mean(ins)) if ins else float("nan"),
        "val_pct": float(pct.mean()),
        "val_pct_sd": float(pct.std(ddof=1)) if len(pct) > 1 else 0.0,
        "share_beats": float((cost < anchor).mean()),
    }


def leaderboard(config: dict, forecast_run_id: str) -> pd.DataFrame:
    """
    One row per candidate, best (lowest mean validation cost) first.

    Columns: candidate, kind (``rl`` / ``anchor``), config_key, method, agent, state,
    anchor, n_seeds, seeds, val_mean, val_std, val_min, val_max, anchor_val,
    delta_mean, delta_ci_lo / delta_ci_hi (paired bootstrap on the project-seed run's
    episodes), class_delta (change per demand class, project-seed run), units_vs_anchor
    (units ordered against the anchor in %, project-seed run; NaN for runs saved before
    units were recorded), insample_pct (final cost against the anchor on the fixed training
    windows, %, mean over seeds), val_pct / val_pct_sd (validation cost against the anchor,
    %, mean and std over seeds), training_forecasts, share_beats, deploy_run_id,
    total_timesteps and gate (the demand classes a gated candidate keeps the policy on;
    None for the others). A gated candidate's config_key is its configuration's plus "+gate".
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
        val_pct = 100.0 * (cost / g["metrics.val_anchor_cost_eur_per_sku"].astype(float) - 1.0)
        insample = g.get("metrics.final_insample_vs_anchor_pct", pd.Series(np.nan, index=g.index)).astype(float)
        deploy = g[g.get("tags.role", pd.Series(index=g.index, dtype=str)) == "deploy"]
        deploy_id = deploy.iloc[0]["run_id"] if not deploy.empty else None
        val_eps = {rid: e for rid in g["run_id"] if (e := _episodes(config, rid, "val_episodes")) is not None}
        lo, hi, class_delta, units = _val_summary(val_eps.get(deploy_id))
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
            "units_vs_anchor": units,
            "insample_pct": float(insample.mean()),
            "val_pct": float(val_pct.mean()),
            "val_pct_sd": float(val_pct.std(ddof=1)) if len(g) > 1 else 0.0,
            # Runs saved before the option existed have no param (NaN): they trained as forecast.
            "training_forecasts": tf if isinstance(tf := first.get("params.training_forecasts"), str) and tf else "as_is",
            "share_beats": float((cost < g["metrics.val_anchor_cost_eur_per_sku"].astype(float)).mean()),
            "deploy_run_id": deploy_id,
            "total_timesteps": first.get("params.total_timesteps"),
            "gate": None,
        })
        # Gated candidate: only when every seed has classed episodes and the policy won some classes, not all.
        gate = class_gate(list(val_eps.values())) if len(val_eps) == len(g) else None
        present = set(pd.concat(val_eps.values())["demand_class"].dropna()) if gate is not None else set()
        if gate and set(gate) != present:
            ins_eps = [e for rid in g["run_id"] if (e := _episodes(config, rid, "insample_episodes")) is not None]
            rows.append(_gated_row(rows[-1], gate, val_eps, ins_eps))
    for i, (desc, vals) in enumerate(anchors.items()):
        val = float(np.mean(vals))
        rows.append({
            "candidate": "anchor" if len(anchors) == 1 else f"anchor {i + 1}",
            "kind": ANCHOR, "config_key": ANCHOR if len(anchors) == 1 else f"{ANCHOR}{i + 1}",
            "method": "anchor", "agent": "", "state": "", "anchor": desc, "n_seeds": 0, "seeds": "",
            "val_mean": val, "val_std": 0.0, "val_min": val, "val_max": val, "anchor_val": val,
            "delta_mean": 0.0, "delta_ci_lo": 0.0, "delta_ci_hi": 0.0, "class_delta": "—",
            "units_vs_anchor": 0.0, "insample_pct": 0.0, "val_pct": 0.0, "val_pct_sd": 0.0,
            "training_forecasts": "", "share_beats": np.nan,
            "deploy_run_id": None, "total_timesteps": None, "gate": None,
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
