"""Resolve forecaster / RL period blocks and timeline for UI and pipelines."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlflow.exceptions import MlflowException

from src.forecasting.validation import TimeSeriesFold, generate_expanding_folds
from src.utils import tracking
from src.utils.config import get_project_root
from src.utils.tracking import config_hash


@dataclass
class Periods:
    """Resolved week boundaries (inclusive Monday dates)."""

    weeks: np.ndarray
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    forecaster_cv_start: pd.Timestamp
    forecaster_cv_end: pd.Timestamp
    gap_weeks: int
    label_gap_weeks: int
    warmup_weeks: int
    rl_train_start: pd.Timestamp
    rl_train_end: pd.Timestamp
    rl_val_start: pd.Timestamp
    rl_val_end: pd.Timestamp
    rl_test_start: pd.Timestamp
    rl_test_end: pd.Timestamp
    fit_end_rl_train: pd.Timestamp
    fit_end_rl_val: pd.Timestamp
    fit_end_rl_test: pd.Timestamp
    refit_per_block: bool
    folds: list[TimeSeriesFold] = field(default_factory=list)
    n_folds_requested: int = 5
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _periods_cfg(config: dict) -> dict:
    return config.get("periods") or {}


def _sales_path(config: dict) -> Path:
    data = config["data"]
    sales = Path(data["sales_file"])
    if sales.is_absolute():
        return sales
    root = Path(config.get("_project_root", get_project_root()))
    return root / sales


def panel_weeks(config: dict) -> np.ndarray:
    """Sorted unique week-start dates from the sales CSV header (no full load)."""
    import csv

    path = _sales_path(config)
    with open(path, encoding="utf-8") as f:
        cols = next(csv.reader(f))
    dates = [c for c in cols if c not in ("Store", "Product")]
    return np.array(sorted(pd.to_datetime(dates)))


def _horizon_gap(config: dict) -> int:
    """How far a target reaches past the feature week."""
    model = config.get("forecasting", {}).get("model", {})
    horizons = model.get("horizons", [0, 1, 2])
    max_h = max(horizons) if horizons else 0
    ltd = model.get("lead_time_demand", {})
    offsets = ltd.get("offsets", [0, 1, 2]) if ltd.get("enabled", False) else [0]
    max_off = max(offsets) if offsets else 0
    return int(max(max_h, max_off))


def lookback_breakdown(config: dict) -> dict[str, int]:
    """Lookback in weeks contributed by each enabled feature family."""
    feat = config.get("forecasting", {}).get("features", {})
    parts: dict[str, int] = {}
    lag_cfg = feat.get("lags", {})
    lags: list[int] = []
    rolls: list[int] = []
    if lag_cfg.get("enabled", True):
        lags = [int(x) for x in (lag_cfg.get("lags") or [])]
        rolls = [int(x) for x in (lag_cfg.get("rolling_windows") or [])]
        if lags:
            parts["lags"] = max(lags)
        if rolls:
            parts["rolling windows"] = max(rolls)
        if lag_cfg.get("expanding_aggs"):
            parts["expanding min periods"] = int(lag_cfg.get("expanding_min_periods", 4))
        if lag_cfg.get("momentum", True):
            peak = max(lags + rolls) if (lags or rolls) else 52
            parts["momentum"] = max(4, 13, peak)
    hier = feat.get("hierarchical", {})
    if hier.get("enabled", True):
        group_lags = [int(x) for x in (hier.get("group_lags") or [])]
        group_rolls = [int(x) for x in (hier.get("group_rolling_windows") or [])]
        if group_lags:
            parts["group lags"] = max(group_lags)
        if group_rolls:
            parts["group rolling"] = max(group_rolls)
    return parts


def feature_lookback(config: dict) -> int:
    """Longest backward window among the feature families that are turned on."""
    parts = lookback_breakdown(config)
    return int(max(parts.values())) if parts else 0


def label_gap_weeks(config: dict) -> int:
    """Weeks a target reaches past the feature week (the forecast horizon)."""
    return _horizon_gap(config)


def required_gap(config: dict) -> int:
    """Weeks left out of the training fit before each scored block.

    This is the forecast horizon. A training row's label reaches that far forward
    (``target_h*`` and lead-time demand). Backward lags, including a 52-week
    seasonal lag, use only earlier sales, so they do not widen the embargo.
    """
    return label_gap_weeks(config)


def _idx_for_date(weeks: np.ndarray, ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    hit = np.where(weeks == ts.to_numpy())[0]
    if len(hit):
        return int(hit[0])
    # nearest Monday on or before
    pos = int(np.searchsorted(weeks, ts, side="right")) - 1
    return max(0, min(pos, len(weeks) - 1))


def resolve_periods(config: dict, weeks: np.ndarray | None = None) -> Periods:
    """Build RL blocks from the end of the panel and CV folds on the forecaster region."""
    weeks = np.sort(weeks if weeks is not None else panel_weeks(config))
    if len(weeks) == 0:
        return Periods(
            weeks=weeks,
            data_start=pd.NaT,
            data_end=pd.NaT,
            forecaster_cv_start=pd.NaT,
            forecaster_cv_end=pd.NaT,
            gap_weeks=0,
            label_gap_weeks=0,
            warmup_weeks=0,
            rl_train_start=pd.NaT,
            rl_train_end=pd.NaT,
            rl_val_start=pd.NaT,
            rl_val_end=pd.NaT,
            rl_test_start=pd.NaT,
            rl_test_end=pd.NaT,
            fit_end_rl_train=pd.NaT,
            fit_end_rl_val=pd.NaT,
            fit_end_rl_test=pd.NaT,
            refit_per_block=True,
            errors=["No weeks found in sales panel."],
        )

    pcfg = _periods_cfg(config)
    gap = required_gap(config)
    label_gap = label_gap_weeks(config)
    lead = int(config.get("rl", {}).get("environment", {}).get("lead_time_weeks", 2))
    min_ep = lead + 2

    data_start = pd.Timestamp(pcfg.get("data_start") or weeks[0])
    data_end = pd.Timestamp(pcfg.get("end") or weeks[-1])
    i0 = _idx_for_date(weeks, data_start)
    i1 = _idx_for_date(weeks, data_end)
    if i0 > i1:
        i0, i1 = i1, i0
    weeks = weeks[i0 : i1 + 1]
    n = len(weeks)

    test_w = int(pcfg.get("rl_test_weeks", 9))
    val_w = int(pcfg.get("rl_val_weeks", 8))
    train_w = int(pcfg.get("rl_train_weeks", 26))
    refit = bool(pcfg.get("refit_per_block", True))

    errors: list[str] = []
    warnings: list[str] = []

    need = test_w + val_w + train_w + gap
    if n < need + 1:
        errors.append(
            f"Panel has {n} weeks but need at least {need + 1} for RL blocks + gap."
        )

    test_end_i = n - 1
    test_start_i = test_end_i - test_w + 1
    val_end_i = test_start_i - 1
    val_start_i = val_end_i - val_w + 1
    train_end_i = val_start_i - 1
    train_start_i = train_end_i - train_w + 1
    forecaster_end_i = train_start_i - gap - 1

    if test_w < min_ep:
        errors.append(f"RL test block ({test_w}w) must be >= {min_ep} (lead_time+2).")
    if val_w < min_ep:
        errors.append(f"RL validation block ({val_w}w) must be >= {min_ep}.")
    if train_w < min_ep:
        errors.append(f"RL train block ({train_w}w) must be >= {min_ep}.")
    if forecaster_end_i < 0:
        errors.append("Forecaster region is empty; shorten RL blocks or widen the date range.")

    def ts(i: int) -> pd.Timestamp:
        return pd.Timestamp(weeks[i])

    fit_end_train_i = train_start_i - gap - 1
    fit_end_val_i = val_start_i - gap - 1
    fit_end_test_i = test_start_i - gap - 1

    if not refit:
        fit_end_val_i = fit_end_test_i = fit_end_train_i

    val_cfg = config.get("forecasting", {}).get("validation", {})
    n_folds_req = int(val_cfg.get("n_folds", 5))
    fold_size = int(val_cfg.get("fold_size_weeks", 13))
    min_train = int(val_cfg.get("min_train_size_weeks", 52))

    fc_weeks = weeks[: forecaster_end_i + 1] if forecaster_end_i >= 0 else weeks[:0]
    folds = (
        generate_expanding_folds(
            fc_weeks,
            n_folds=n_folds_req,
            fold_size_weeks=fold_size,
            embargo_weeks=gap,
            min_train_size_weeks=min_train,
        )
        if len(fc_weeks) > 0
        else []
    )
    embargo_why = f"the {label_gap}-week forecast horizon"
    if not folds and forecaster_end_i >= min_train - 1:
        errors.append(
            f"No CV folds fit. Each fold holds out an embargo of {gap} weeks ({embargo_why}) "
            f"between training and validation, and what remains is under {min_train} training weeks. "
            "Shorten the RL blocks, lower the minimum training length, or reduce the number of folds."
        )
    elif len(folds) < n_folds_req:
        warnings.append(
            f"Only {len(folds)} of {n_folds_req} CV folds fit. The embargo is {gap} weeks "
            f"({embargo_why}) between training and each validation block."
        )

    init_date = config.get("data", {}).get("initial_state_date")
    if init_date:
        init_ts = pd.Timestamp(init_date)
        if ts(test_start_i) != init_ts:
            warnings.append(
                f"RL test starts {ts(test_start_i).date()} but initial_state_date is "
                f"{init_ts.date()}; test block will use warm-start inventory."
            )

    return Periods(
        weeks=weeks,
        data_start=ts(0),
        data_end=ts(n - 1),
        forecaster_cv_start=ts(0),
        forecaster_cv_end=ts(max(0, forecaster_end_i)),
        gap_weeks=gap,
        label_gap_weeks=label_gap,
        warmup_weeks=0,
        rl_train_start=ts(train_start_i),
        rl_train_end=ts(train_end_i),
        rl_val_start=ts(val_start_i),
        rl_val_end=ts(val_end_i),
        rl_test_start=ts(test_start_i),
        rl_test_end=ts(test_end_i),
        fit_end_rl_train=ts(max(0, fit_end_train_i)),
        fit_end_rl_val=ts(max(0, fit_end_val_i)),
        fit_end_rl_test=ts(max(0, fit_end_test_i)),
        refit_per_block=refit,
        folds=folds,
        n_folds_requested=n_folds_req,
        errors=errors,
        warnings=warnings,
    )


def timeline(periods: Periods) -> pd.DataFrame:
    """Long-format segments for charts and tables."""
    if periods.errors and pd.isna(periods.data_start):
        return pd.DataFrame(columns=["lane", "part", "start", "end", "weeks"])

    rows: list[dict[str, Any]] = []

    def add(lane: str, part: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
        if pd.isna(start) or pd.isna(end):
            return
        w = int(np.sum((periods.weeks >= start) & (periods.weeks <= end)))
        rows.append(
            {
                "lane": lane,
                "part": part,
                "start": start,
                "end": end,
                "weeks": w,
            }
        )

    def add_embargo(lane: str, start_i: int, end_i: int) -> None:
        """Label gap against the scored block. Any extra weeks are a feature embargo."""
        if end_i < start_i or start_i < 0:
            return
        end_i = min(end_i, len(periods.weeks) - 1)
        n_hole = end_i - start_i + 1
        gap_n = min(max(periods.label_gap_weeks, 0), n_hole)
        emb_n = n_hole - gap_n
        if emb_n:
            add(
                lane,
                "Embargo",
                pd.Timestamp(periods.weeks[start_i]),
                pd.Timestamp(periods.weeks[start_i + emb_n - 1]),
            )
        if gap_n:
            add(
                lane,
                "Label gap",
                pd.Timestamp(periods.weeks[end_i - gap_n + 1]),
                pd.Timestamp(periods.weeks[end_i]),
            )

    add("Forecaster", "CV fit region", periods.forecaster_cv_start, periods.forecaster_cv_end)
    if periods.gap_weeks and periods.rl_train_start is not pd.NaT:
        gap_start_i = int(np.searchsorted(periods.weeks, np.datetime64(periods.forecaster_cv_end))) + 1
        add_embargo("Forecaster", gap_start_i, gap_start_i + periods.gap_weeks - 1)
    add("RL", "Train", periods.rl_train_start, periods.rl_train_end)
    add("RL", "Validation (checkpoint)", periods.rl_val_start, periods.rl_val_end)
    add("RL", "Test (final score)", periods.rl_test_start, periods.rl_test_end)

    for fold in periods.folds:
        add(
            f"CV fold {fold.fold_id}",
            "Fit",
            pd.Timestamp(fold.train_dates[0]),
            fold.train_end,
        )
        hold_start = int(np.searchsorted(periods.weeks, np.datetime64(fold.embargo_start)))
        hold_end = int(np.searchsorted(periods.weeks, np.datetime64(fold.embargo_end)))
        add_embargo(f"CV fold {fold.fold_id}", hold_start, hold_end)
        add(
            f"CV fold {fold.fold_id}",
            "Validation",
            fold.val_start,
            fold.val_end,
        )

    return pd.DataFrame(rows)


def setup_path(config: dict) -> Path:
    root = Path(config.get("_project_root", get_project_root()))
    rel = config["forecasting"]["artifacts"]["forecasts_dir"]
    return root / rel / "forecast_setup.json"


def periods_to_dict(periods: Periods) -> dict[str, Any]:
    return {
        "data_start": str(periods.data_start.date()),
        "data_end": str(periods.data_end.date()),
        "forecaster_cv_start": str(periods.forecaster_cv_start.date()),
        "forecaster_cv_end": str(periods.forecaster_cv_end.date()),
        "gap_weeks": periods.gap_weeks,
        "label_gap_weeks": periods.label_gap_weeks,
        "warmup_weeks": periods.warmup_weeks,
        "rl_train_start": str(periods.rl_train_start.date()),
        "rl_train_end": str(periods.rl_train_end.date()),
        "rl_val_start": str(periods.rl_val_start.date()),
        "rl_val_end": str(periods.rl_val_end.date()),
        "rl_test_start": str(periods.rl_test_start.date()),
        "rl_test_end": str(periods.rl_test_end.date()),
        "fit_end_rl_train": str(periods.fit_end_rl_train.date()),
        "fit_end_rl_val": str(periods.fit_end_rl_val.date()),
        "fit_end_rl_test": str(periods.fit_end_rl_test.date()),
        "refit_per_block": periods.refit_per_block,
    }


def build_setup_record(
    config: dict,
    periods: Periods,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Serializable setup written beside forecasts for RL and UI staleness checks."""
    hash_payload = {
        "periods": config.get("periods"),
        "forecasting": {
            k: v
            for k, v in config.get("forecasting", {}).items()
            if k not in ("mlflow", "artifacts")
        },
        "data": {
            k: config.get("data", {}).get(k)
            for k in ("sales_file", "initial_state_date")
        },
    }
    return {
        "config_hash": config_hash(hash_payload),
        "run_id": run_id,
        "resolved": periods_to_dict(periods),
        "timeline": timeline(periods).assign(
            start=lambda d: d["start"].astype(str),
            end=lambda d: d["end"].astype(str),
        ).to_dict(orient="records"),
        "periods_input": dict(config.get("periods") or {}),
        "features": config.get("forecasting", {}).get("features"),
    }


def save_setup(config: dict, record: dict) -> Path:
    path = setup_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_setup(config: dict) -> dict | None:
    path = setup_path(config)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_run_file(config: dict, run_id: str | None, name: str) -> Path | None:
    """Local copy of ``forecasts/<name>`` for a forecasting run, or None if it has none.

    ``run_id=None`` means the latest run on disk. Other runs are downloaded from
    MLflow once into ``<forecasts_dir>/runs/<run_id>/``.
    """
    latest = setup_path(config).with_name(name)
    if run_id is None or (load_setup(config) or {}).get("run_id") == run_id:
        return latest if latest.exists() else None
    cache_dir = setup_path(config).parent / "runs" / run_id
    cached = cache_dir / name
    if not cached.exists():
        tracking.init_tracking(config, "forecasting")
        try:
            tracking.download_artifact(run_id, f"forecasts/{name}", dst_path=str(cache_dir))
        except MlflowException:
            with contextlib.suppress(OSError):
                cache_dir.rmdir()
            return None
    return cached if cached.exists() else None


def load_run_setup(config: dict, run_id: str | None) -> dict | None:
    """Setup record of a forecasting run (None → latest on disk)."""
    path = fetch_run_file(config, run_id, "forecast_setup.json")
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def current_setup_hash(config: dict) -> str:
    hash_payload = {
        "periods": config.get("periods"),
        "forecasting": {
            k: v
            for k, v in config.get("forecasting", {}).items()
            if k not in ("mlflow", "artifacts")
        },
        "data": {
            k: config.get("data", {}).get(k)
            for k in ("sales_file", "initial_state_date")
        },
    }
    return config_hash(hash_payload)
