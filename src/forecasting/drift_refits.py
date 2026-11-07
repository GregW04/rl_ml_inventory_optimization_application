"""
Drift-triggered refits of a forecasting run, on top of the fixed schedule.

The RL blocks are forecast block by block. A fit is made on the rows known when its first week
starts (``fit_end`` = that week − gap − 1 weeks, like every fit), predicts to the end of the block,
and is cut at the first of

* the end of its scheduled chunk (``cadence_weeks`` weeks: the age backstop; 0 = no limit), or
* the first week a demand class alerts (``drift_alerts.first_alert``: the production bias alarm,
  restarted at this fit), which makes the next fit start the week after.

Nothing after the cut is kept, so every forecast comes from a fit that knew only what was known then.
With no alert the fits are exactly the fixed schedule.

Escalation: if an alert comes back for a class that triggered the previous alert-driven refit, the
walk first asks ``retune`` (tune again on the rows before a holdout of the last 8 weeks known that week;
adopt only a clear win on that holdout),
and the refit uses the adopted settings. At most ``max_retunes`` per run.

The model work is injected (``fit_predict``, ``alert_at``, ``retune``), so the logic is testable
without training anything; src/forecasting/pipeline.py wires the real forecasters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.forecasting.periods import BlockFit

WEEK = pd.Timedelta(weeks=1)


@dataclass
class Retune:
    """
    One retune challenge: when, why, the holdout scores of the current and the new settings, the outcome.
    A challenge that could not use the run's CV settings is recorded as ``skipped`` (it does not count toward the cap).
    """

    week: pd.Timestamp
    block: str
    classes: list[str]
    current_score: float
    new_score: float
    gain_pct: float
    adopted: bool
    # The rows it used, so a run shows which CV settings the tuning had.
    tuned_to: str = ""  # the last row the tuning saw
    holdout_start: str = ""
    holdout_end: str = ""
    folds: int = 0  # CV folds the tuning scored (0: none could be built)
    fold_weeks: int = 0
    skipped: bool = False  # the run's CV settings did not fit before the holdout: nothing was tuned
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "week": str(self.week.date())}


@dataclass
class WalkState:
    """What the walk carries from block to block: the settings in use, the last alert's classes, retunes so far."""

    params: Any = None
    last_alert_classes: set[str] = field(default_factory=set)
    retunes: list[Retune] = field(default_factory=list)


FitPredict = Callable[[pd.Timestamp, pd.Timestamp, pd.Timestamp, Any], pd.DataFrame]
AlertAt = Callable[[pd.DataFrame], tuple[pd.Timestamp | None, list[str]]]
RetuneFn = Callable[[pd.Timestamp, pd.Timestamp, list[str], Any], tuple[Retune, Any]]


def walk_block(
    block: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    gap_weeks: int,
    cadence_weeks: int,
    fit_predict: FitPredict,
    alert_at: AlertAt,
    state: WalkState,
    retune: RetuneFn | None = None,
    max_retunes: int = 1,
    max_fits: int = 12,
) -> tuple[list[BlockFit], list[pd.DataFrame]]:
    """
    The fits of one block and the forecasts each one keeps.

    ``fit_predict(fit_end, start, end, params)`` fits on rows up to ``fit_end`` and returns forecasts for the
    weeks ``start``…``end``; ``alert_at(forecasts)`` returns the first alert week and its classes;
    ``retune(week, fit_end, classes, params)`` returns the challenge outcome and the settings to use next.
    """
    fits: list[BlockFit] = []
    parts: list[pd.DataFrame] = []
    cur, trigger, classes, retuned = start, "start", "", False
    while cur <= end:
        fit_end = cur - (gap_weeks + 1) * WEEK
        scored = fit_predict(fit_end, cur, end, state.params).assign(fit_end=fit_end)
        scheduled_end = min(end, cur + (cadence_weeks - 1) * WEEK) if cadence_weeks > 0 else end
        chunk = scored[scored["date"] <= scheduled_end]
        alert_week, alert_classes = alert_at(chunk) if len(fits) + 1 < max_fits else (None, [])
        chunk_end = alert_week if alert_week is not None else scheduled_end
        fits.append(BlockFit(block, len(fits) + 1, cur, chunk_end, fit_end, trigger, classes, retuned))
        parts.append(scored[scored["date"] <= chunk_end])
        if chunk_end >= end:
            break
        cur = chunk_end + WEEK
        retuned = False
        if alert_week is None:
            trigger, classes = "schedule", ""
            continue
        trigger, classes = "drift", ",".join(alert_classes)
        returns = bool(state.last_alert_classes & set(alert_classes))
        state.last_alert_classes = set(alert_classes)
        if returns and retune is not None and sum(not r.skipped for r in state.retunes) < max_retunes:
            outcome, adopted_params = retune(chunk_end, cur - (gap_weeks + 1) * WEEK, alert_classes, state.params)
            outcome.block = block
            state.retunes.append(outcome)
            if outcome.adopted:
                state.params, retuned = adopted_params, True
    return fits, parts
