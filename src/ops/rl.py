"""
The RL policy layer of the ops loop.

Every environment starts with the anchor as its policy (version 1), so a fallback is always one
alias move away. Once the as-issued forecast ledger covers enough weeks, a PPO challenger is
trained on it: the forecasts each week's champion forecaster really issued, so the agent learns
from the errors it will face. Training windows lie in the ``train_weeks`` that end ``val_weeks``
before the as-of week; the last ``val_weeks`` are the single validation check, where the gate of
demand classes is picked (as in research) and the challenger must beat the anchor by a margin,
with units close to the anchor's, in most seeds. The gate is optimistic on the weeks that picked
it, so a promoted policy is then watched live against the anchor's shadow (the fallback rule).

The ops recipe is steady state: it leaves out the VN2-window features (order round, error since
the window started), which don't exist week to week. Forecaster versions play the research
pipeline's "fits" when training forecasts are debiased. Torch is imported only inside training
(``src.rl.train``; the data path, ``src.rl.data``, is torch-free), and the retrain step runs in
its own process.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.forecasting.measures import infer_measures
from src.ops.settings import PolicyRules
from src.rl.data import RunData
from src.rl.rollout import units_vs_anchor
from src.rl.selection import class_gate, gate_episodes
from src.rl.spec import ANCHOR_FEATURES, CLASS_FEATURES, INVENTORY_FEATURES, PolicySpec, build_spec
from src.rl.vn2_data import _PANEL_COLS, build_forecast_lookup, compute_demand_scales
from src.simulation.anchor import resolve_anchor
from src.simulation.vn2 import BURN_IN_WEEKS, VN2Rules

WEEK = pd.Timedelta(weeks=1)
KEYS = ["store", "product"]
CONTEXT = ["zero_share", "pos_cv"]


def windows(as_of: pd.Timestamp, rules: PolicyRules) -> dict[str, pd.Timestamp]:
    """Validation = the last ``val_weeks`` up to ``as_of``; training = the ``train_weeks`` before them."""
    val_start = pd.Timestamp(as_of) - (rules.val_weeks - 1) * WEEK
    train_end = val_start - WEEK
    return {"rl_train_start": train_end - (rules.train_weeks - 1) * WEEK, "rl_train_end": train_end,
            "rl_val_start": val_start, "rl_val_end": pd.Timestamp(as_of)}


def eligible(as_of: pd.Timestamp, first_forecast_week: pd.Timestamp, rules: PolicyRules) -> bool:
    """True once the ledger covers the training windows and their burn-in weeks."""
    return windows(as_of, rules)["rl_train_start"] - BURN_IN_WEEKS * WEEK >= pd.Timestamp(first_forecast_week)


def version_fits(forecasts: pd.DataFrame, fit_as_of: dict[str, str], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Contiguous tenures of the forecaster versions within ``[start, end]``, as research block fits."""
    weeks = (forecasts.loc[(forecasts["date"] >= start) & (forecasts["date"] <= end), ["date", "version"]]
             .drop_duplicates().sort_values("date"))
    fits: list[dict] = []
    for _, run in weeks.groupby((weeks["version"] != weeks["version"].shift()).cumsum()):
        v = str(run["version"].iloc[0])
        fits.append({"block": "rl_train", "index": len(fits) + 1, "start": str(run["date"].min().date()),
                     "end": str(run["date"].max().date()), "fit_end": fit_as_of.get(v)})
    return fits


def ledger_run_data(panel: pd.DataFrame, forecasts: pd.DataFrame, as_of: pd.Timestamp, rules: PolicyRules,
                    vn2: VN2Rules, fit_as_of: dict[str, str]) -> RunData:
    """What RL trains and validates on, from the ledger: the issued forecasts, the drop's panel, scales."""
    w = windows(as_of, rules)
    measures = infer_measures(list(forecasts.columns), vn2)
    anchor, warnings = resolve_anchor(measures, vn2)
    if anchor is None:
        raise RuntimeError("; ".join(warnings))
    panel = panel[panel["date"] <= as_of]
    resolved = {**{k: str(v.date()) for k, v in w.items()},
                "block_fits": version_fits(forecasts, fit_as_of, w["rl_train_start"], w["rl_train_end"]),
                "data_start": str(panel["date"].min().date()), "data_end": str(pd.Timestamp(as_of).date())}
    return RunData(
        run_id=None, setup={"run_id": None, "resolved": resolved}, measures=measures,
        lookup=build_forecast_lookup(forecasts, [m.id for m in measures]), panel=panel,
        # Scales from history up to the end of training only, as in research.
        scales=compute_demand_scales(panel[_PANEL_COLS], cap_end=w["rl_train_end"]),
        default_anchor=anchor, anchor_warnings=warnings, rules=vn2,
    )


def ops_spec(config: dict, rules: PolicyRules, data: RunData, env: str) -> PolicySpec:
    """The steady-state recipe: the research state without the VN2-window features."""
    state = [*INVENTORY_FEATURES, *data.measure_ids, *ANCHOR_FEATURES, *CONTEXT, *CLASS_FEATURES]
    cfg = copy.deepcopy(config)
    cfg["rl"]["agents"][rules.agent].update(rules.agent_params)
    # One forecast source per environment: the config key stays the same across retrains (the recipe).
    return build_spec(cfg, f"ledger:{env}", rules.method, rules.agent, state, data.default_anchor,
                      training_forecasts=rules.training_forecasts)


def challenger_report(val_eps: dict[int, pd.DataFrame], rules: PolicyRules) -> dict:
    """The promotion check on the validation weeks: gate, cost and units against the anchor per seed."""
    gate = class_gate(list(val_eps.values()))
    gated = {s: gate_episodes(e, gate) if gate else e for s, e in val_eps.items()}
    pct = {s: float(100.0 * (g["cost_eur"].sum() / g["anchor_cost_eur"].sum() - 1.0)) for s, g in gated.items()}
    units = {s: units_vs_anchor(g) for s, g in gated.items()}
    mean_pct = float(np.mean(list(pct.values())))
    share = float(np.mean([v < 0 for v in pct.values()]))
    checks = {
        f"won some demand classes (gate {gate})": bool(gate),
        f"mean validation cost {mean_pct:+.2f}% <= -{rules.promote_margin_pct}%": mean_pct <= -rules.promote_margin_pct,
        f"units within ±{rules.units_band_pct}% of the anchor in every seed":
            all(u is not None and abs(u) <= rules.units_band_pct for u in units.values()),
        # Rounded, so that 2 of 3 seeds (0.667) meets a 0.67 rule.
        f"{share:.0%} of seeds beat the anchor (needs {rules.min_seed_share:.0%})": round(share, 2) >= rules.min_seed_share,
    }
    return {"gate": gate, "val_pct": pct, "mean_val_pct": mean_pct, "units_pct": units, "share_beating": share,
            "passed": all(checks.values()), "failed": [k for k, ok in checks.items() if not ok]}


def decision_frame(state: pd.DataFrame, forecasts: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """The policy model's input: stock, the decision week's forecasts and context, one row per SKU."""
    ids = [c for c in forecasts.columns if c not in (*KEYS, "date", "version")]
    out = state[[*KEYS, "on_hand", "transit_1", "transit_2"]].merge(forecasts[[*KEYS, *ids]], on=KEYS, validate="one_to_one")
    out = out.merge(context, on=KEYS, how="left")
    return out.fillna({"zero_share": 0.0, "pos_cv": 0.0, "dp_sbc_class": -1}).astype({"dp_sbc_class": "int64"})
