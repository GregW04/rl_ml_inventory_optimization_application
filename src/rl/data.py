"""
What RL trains, validates and tests on: the forecasts, measures, panel and demand scales of one
forecasting run (:func:`load_run_data`), cut into VN2 episodes per block (:func:`block_episodes`).
The ops loop builds the same :class:`RunData` from its forecast ledger (``src/ops/rl.py``).
Torch-free, so any process can load it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.forecasting.measures import Measure, run_measures
from src.forecasting.periods import fetch_run_file, load_run_setup
from src.forecasting.preprocessing import load_initial_state, load_panel_from_config
from src.rl.vn2_data import _PANEL_COLS, build_forecast_lookup, build_vn2_episodes, compute_demand_scales
from src.simulation.anchor import AnchorSpec, resolve_anchor
from src.simulation.vn2 import VN2Rules


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
