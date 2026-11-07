"""Resolve forecaster / RL period blocks and timeline for UI and pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.validation import TimeSeriesFold, generate_expanding_folds
from src.utils import tracking
from src.utils.config import get_project_root
from src.utils.tracking import config_hash


@dataclass(frozen=True)
class BlockFit:
    """One forecaster fit and the weeks it writes forecasts for.

    The fit uses rows up to ``fit_end``; their labels reach ``gap`` weeks further, so the
    newest label lands the week before ``start``. ``index`` counts fits within the block.
    ``trigger`` says why it was made: ``start`` (the block's first fit), ``schedule`` (the fit
    reached ``refit_every_weeks``) or ``drift`` (an alert, with the alerting ``classes``);
    ``retuned`` marks a fit made with hyperparameters adopted by a retune just before it.
    """

    block: str
    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    fit_end: pd.Timestamp
    trigger: str = "schedule"
    classes: str = ""
    retuned: bool = False

    @property
    def weeks(self) -> int:
        return int((self.end - self.start).days // 7) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "index": self.index,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "fit_end": str(self.fit_end.date()),
            "weeks": self.weeks,
            "trigger": self.trigger,
            "classes": self.classes,
            "retuned": self.retuned,
        }


# Blocks whose forecaster is refit every ``refit_every_weeks``. The test block keeps one fit
# made before its first week, so the VN2 window only uses labels known by week 0.
REFIT_BLOCKS: tuple[str, ...] = ("rl_train", "rl_val")
BLOCK_NAMES: dict[str, str] = {"rl_train": "training", "rl_val": "validation", "rl_test": "test"}
MAX_BLOCK_FITS_WARNING = 12


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
    refit_every_weeks: int
    folds: list[TimeSeriesFold] = field(default_factory=list)
    block_fits: list[BlockFit] = field(default_factory=list)
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
    """How far a target reaches past the feature week (the longest forecast horizon)."""
    from src.forecasting.measures import HORIZONS

    return int(max(HORIZONS))


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
    profile = feat.get("demand_profile", {})
    if profile.get("enabled", False):
        parts["demand profile"] = int(profile.get("window_weeks", 52))
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

    This is the forecast horizon. A training row's label (``target_h*``) reaches that
    far forward. Backward lags, including a 52-week seasonal lag, use only earlier
    sales, so they do not widen the embargo.
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
            refit_every_weeks=0,
            errors=["No weeks found in sales panel."],
        )

    pcfg = _periods_cfg(config)
    gap = required_gap(config)
    label_gap = label_gap_weeks(config)
    from src.simulation.vn2 import VN2Rules

    min_ep = VN2Rules.from_config(config).lead_time_weeks + 2

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
    refit_n = int(pcfg.get("refit_every_weeks", 0) or 0)

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

    min_test_val = 8
    min_train = 12
    if test_w < min_test_val:
        errors.append(f"RL test block ({test_w}w) must be >= {min_test_val} (VN2 window).")
    if val_w < min_test_val:
        errors.append(f"RL validation block ({val_w}w) must be >= {min_test_val}.")
    if train_w < min_train:
        errors.append(f"RL train block ({train_w}w) must be >= {min_train} (burn-in + window).")
    if test_w < min_ep:
        errors.append(f"RL test block ({test_w}w) must be >= {min_ep} (lead_time+2).")
    if val_w < min_ep:
        errors.append(f"RL validation block ({val_w}w) must be >= {min_ep}.")
    if train_w < min_ep:
        errors.append(f"RL train block ({train_w}w) must be >= {min_ep}.")
    if forecaster_end_i < 0:
        errors.append("Forecaster region is empty; shorten RL blocks or widen the date range.")
    if refit_n < 0:
        errors.append("Refit cadence must be 0 (one fit per block) or a positive number of weeks.")
        refit_n = 0

    def ts(i: int) -> pd.Timestamp:
        return pd.Timestamp(weeks[i])

    fit_end_train_i = train_start_i - gap - 1
    fit_end_val_i = val_start_i - gap - 1
    fit_end_test_i = test_start_i - gap - 1

    def _chunks(block: str, start_i: int, end_i: int) -> list[BlockFit]:
        """Consecutive fits of ``refit_n`` weeks from the block start (one if 0 or test)."""
        if end_i < start_i or start_i < 0:
            return []
        length = end_i - start_i + 1
        step = refit_n if (refit_n > 0 and block in REFIT_BLOCKS) else length
        out: list[BlockFit] = []
        for s_i in range(start_i, end_i + 1, step):
            e_i = min(s_i + step - 1, end_i)
            out.append(BlockFit(block, len(out) + 1, ts(s_i), ts(e_i), ts(max(0, s_i - gap - 1)),
                                "start" if not out else "schedule"))
        return out

    block_fits = (
        _chunks("rl_train", train_start_i, train_end_i)
        + _chunks("rl_val", val_start_i, val_end_i)
        + _chunks("rl_test", test_start_i, test_end_i)
    )
    if len(block_fits) > MAX_BLOCK_FITS_WARNING:
        warnings.append(
            f"{len(block_fits)} forecaster fits for the RL blocks. Each fit takes minutes, so "
            "a short refit cadence makes the run much longer."
        )

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
        # The official window starts the week after the initial-state snapshot.
        official_start = pd.Timestamp(init_date) + pd.Timedelta(weeks=1)
        if ts(test_start_i) != official_start:
            warnings.append(
                f"RL test starts {ts(test_start_i).date()} but the official VN2 window starts "
                f"{official_start.date()}; the test block will use warm-start inventory."
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
        refit_every_weeks=refit_n,
        folds=folds,
        block_fits=block_fits,
        n_folds_requested=n_folds_req,
        errors=errors,
        warnings=warnings,
    )


def fit_lane(block: str, index: int, n_in_block: int) -> str:
    """Timeline lane of a block fit, e.g. ``Fit · training 2/4``."""
    name = BLOCK_NAMES.get(block, block)
    return f"Fit · {name}" if n_in_block <= 1 else f"Fit · {name} {index}/{n_in_block}"


def block_fit_counts(resolved: dict) -> dict[str, int]:
    """Fits per block from a saved ``resolved`` record (records without fits: one per block)."""
    fits = resolved.get("block_fits") or []
    counts = {b: sum(1 for f in fits if f.get("block") == b) for b in BLOCK_NAMES}
    return {b: (n or 1) for b, n in counts.items()}


def refit_events(resolved: dict) -> pd.DataFrame:
    """
    When the forecaster is (re)fitted for the RL blocks: one row per block fit with the first week
    it forecasts, its block, its label and the last week of data it saw. Records without fits: none.
    """
    fits = resolved.get("block_fits") or []
    rows = [{
        "date": pd.Timestamp(f["start"]),
        "block": BLOCK_NAMES.get(f["block"], f["block"]),
        "label": fit_lane(f["block"], int(f.get("index", 1)), sum(g.get("block") == f["block"] for g in fits))
        .removeprefix("Fit · "),
        "data_through": pd.Timestamp(f["fit_end"]) if f.get("fit_end") else pd.NaT,
        # Records saved before triggers existed: the first fit of a block started it, the rest were scheduled.
        "trigger": f.get("trigger") or ("start" if int(f.get("index", 1)) == 1 else "schedule"),
        "classes": f.get("classes") or "",
        "retuned": bool(f.get("retuned", False)),
    } for f in fits]
    return pd.DataFrame(rows, columns=["date", "block", "label", "data_through", "trigger", "classes", "retuned"])


def decision_ages(periods: Periods, window_weeks: int = 8, n_rounds: int = 6) -> pd.DataFrame:
    """
    Age of the forecaster behind each order decision, in weeks since its ``fit_end``.

    One row per block and decision Monday. ``windows`` counts the 8-week windows of the
    block that place an order that Monday, so a weighted mean over rows equals the mean
    over every (window, decision) pair the agent sees.
    """
    rows: list[dict[str, Any]] = []
    week = pd.Timedelta(weeks=1)
    for block in BLOCK_NAMES:
        fits = [f for f in periods.block_fits if f.block == block]
        if not fits:
            continue
        start, end = fits[0].start, fits[-1].end
        counts: dict[pd.Timestamp, int] = {}
        s = start
        while s + (window_weeks - 1) * week <= end:
            for j in range(n_rounds):
                d = s + j * week
                counts[d] = counts.get(d, 0) + 1
            s += week
        for d in sorted(counts):
            fit = next((f for f in fits if f.start <= d <= f.end), fits[-1])
            rows.append({
                "block": block,
                "date": d,
                "age_weeks": int((d - fit.fit_end).days // 7),
                "fit": fit.index,
                "windows": counts[d],
            })
    return pd.DataFrame(rows, columns=["block", "date", "age_weeks", "fit", "windows"])


def mean_decision_age(ages: pd.DataFrame) -> dict[str, float]:
    """Mean forecaster age per block over every (window, decision) pair."""
    out: dict[str, float] = {}
    for block, g in ages.groupby("block", sort=False):
        out[str(block)] = float((g["age_weeks"] * g["windows"]).sum() / max(g["windows"].sum(), 1))
    return out


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
    add("RL", "Validation (selection)", periods.rl_val_start, periods.rl_val_end)
    add("RL", "Test (production)", periods.rl_test_start, periods.rl_test_end)

    for fit in periods.block_fits:
        lane = fit_lane(fit.block, fit.index, sum(f.block == fit.block for f in periods.block_fits))
        add(lane, "Fit", periods.data_start, fit.fit_end)
        hole_start = int(np.searchsorted(periods.weeks, np.datetime64(fit.fit_end))) + 1
        add_embargo(lane, hole_start, hole_start + periods.gap_weeks - 1)
        add(lane, "Forecasts written", fit.start, fit.end)

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
        "refit_every_weeks": periods.refit_every_weeks,
        "block_fits": [f.to_dict() for f in periods.block_fits],
    }


def build_setup_record(
    config: dict,
    periods: Periods,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serializable setup written beside forecasts for RL and UI staleness checks.

    ``extra`` carries the run's measure manifest (see ``measures.manifest_record``).
    """
    return {
        **(extra or {}),
        "config_hash": current_setup_hash(config),
        "run_id": run_id,
        "resolved": periods_to_dict(periods),
        "timeline": timeline(periods).assign(
            start=lambda d: d["start"].astype(str),
            end=lambda d: d["end"].astype(str),
        ).to_dict(orient="records"),
        "periods_input": dict(config.get("periods") or {}),
        "features": config.get("forecasting", {}).get("features"),
    }


def save_setup(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def latest_forecast_run_id(config: dict) -> str | None:
    """Newest FINISHED forecasting run that wrote RL forecasts (it logs the RL block dates)."""
    tracking.init_tracking(config)
    runs = tracking.search_runs(
        tracking.experiment_name(config, "forecasting"),
        max_results=50,
        filter_string="attributes.status = 'FINISHED'",
    )
    if runs.empty or "params.rl_train_start" not in runs.columns:
        return None
    runs = runs[runs["params.rl_train_start"].notna()]
    return None if runs.empty else str(runs.iloc[0]["run_id"])


def fetch_run_file(config: dict, run_id: str | None, name: str) -> Path | None:
    """
    Local copy of ``forecasts/<name>`` of a forecasting run (None if the run has none), from
    the run's artifact cache. ``run_id=None`` means the newest finished forecasting run.
    """
    run_id = run_id or latest_forecast_run_id(config)
    if run_id is None:
        return None
    return tracking.cached_artifact(config, run_id, f"forecasts/{name}")


def load_run_setup(config: dict, run_id: str | None) -> dict | None:
    """Setup record of a forecasting run (None → the newest finished one)."""
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
            # Not part of the setup; excluded so hashes match runs made before they moved.
            if k not in ("mlflow", "artifacts", "skip_duplicate_runs")
        },
        "data": {
            k: config.get("data", {}).get(k)
            for k in ("sales_file", "initial_state_date")
        },
    }
    monitoring = config.get("monitoring") or {}
    if monitoring.get("drift_refits"):
        # Alert-driven refits change the forecasts, so they belong to the setup; alerts alone don't.
        hash_payload["drift_refits"] = {
            k: monitoring.get(k) for k in ("retune_on_return", "retune_min_gain_pct", "max_retunes")
        }
    return config_hash(hash_payload)
