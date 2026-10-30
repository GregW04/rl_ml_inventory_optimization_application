"""
Forecast measures: what the forecaster predicts per horizon, and how runs describe it.

A measure is a mean or a quantile of weekly demand at one of the horizons of a VN2 order
(h0 = this week, h1 = next week, h2 = the delivery week). Its id doubles as the forecast
column name and as an RL state feature: ``mean_h0``, ``q50_h1`` (median), ``q83_h2``.

Each forecasting run saves its measures in ``forecast_setup.json``. Runs written before
measures existed are read through :func:`infer_measures`, which recognises the same ids
in their forecast columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.simulation.vn2 import VN2Rules

HORIZONS: tuple[int, ...] = (0, 1, 2)
KINDS: tuple[str, ...] = ("mean", "quantile")
CRITICAL_RATIO = "critical_ratio"

_ID_RE = re.compile(r"^(?:(mean)|q(\d{1,2}))_h(\d)$")


def quantile_label(tau: float) -> str:
    """Column label of a quantile level: 0.5 → ``50``, 0.8333 → ``83``."""
    return str(int(round(float(tau) * 100)))


@dataclass(frozen=True)
class Measure:
    kind: str
    horizon: int
    tau: float | None = None

    @property
    def id(self) -> str:
        if self.kind == "quantile":
            return f"q{quantile_label(self.tau)}_h{self.horizon}"
        return f"{self.kind}_h{self.horizon}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "horizon": self.horizon, "tau": self.tau}

    @classmethod
    def from_dict(cls, d: dict) -> Measure:
        tau = d.get("tau")
        return cls(str(d["kind"]), int(d["horizon"]), None if tau is None else float(tau))


def _sort_key(m: Measure) -> tuple:
    return (m.horizon, KINDS.index(m.kind) if m.kind in KINDS else 99, m.tau or 0.0)


def resolve_tau(value: Any, rules: VN2Rules) -> float:
    """Quantile level from config: a number in (0, 1) or ``critical_ratio``."""
    if isinstance(value, str) and value.strip().lower() == CRITICAL_RATIO:
        return rules.critical_ratio
    return float(value)


def parse_measures(config: dict) -> list[Measure]:
    """Measures listed under ``forecasting.measures`` (sorted by horizon, then kind)."""
    rules = VN2Rules.from_config(config)
    out = []
    for entry in config.get("forecasting", {}).get("measures") or []:
        kind = str(entry.get("kind", "")).lower()
        tau = entry.get("tau")
        out.append(
            Measure(
                kind,
                int(entry.get("horizon", -1)),
                resolve_tau(tau, rules) if kind == "quantile" and tau is not None else None,
            )
        )
    return sorted(out, key=_sort_key)


def validate_measures(measures: list[Measure], primary: str | None = None) -> list[str]:
    """Problems that block a forecasting run (empty list = valid)."""
    errors: list[str] = []
    if not measures:
        return ["Select at least one measure per horizon."]
    for m in measures:
        if m.kind not in KINDS:
            errors.append(f"Unknown measure kind '{m.kind}'.")
        if m.horizon not in HORIZONS:
            errors.append(f"Horizon h{m.horizon} is outside h0–h{HORIZONS[-1]}.")
        if m.kind == "quantile" and (m.tau is None or not 0.0 < m.tau < 1.0):
            errors.append(f"Quantile level {m.tau} at h{m.horizon} must be between 0 and 1.")
    ids = [m.id for m in measures if m.kind in KINDS]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errors.append(
            f"Two measures map to `{dup}`; quantile levels at one horizon must differ by a whole percent."
        )
    for h in HORIZONS:
        if not any(m.horizon == h for m in measures):
            errors.append(
                f"h{h} needs a mean or a quantile: the order decision projects stock through "
                "every week up to delivery."
            )
    if primary is not None and not any(m.id == primary for m in measures):
        errors.append(f"Primary measure `{primary}` is not among the selected measures.")
    return errors


def default_primary(measures: list[Measure], rules: VN2Rules) -> Measure | None:
    """The delivery-week quantile closest to the critical ratio, else the first h2 level."""
    last = HORIZONS[-1]
    quantiles = [m for m in measures if m.kind == "quantile" and m.horizon == last]
    if quantiles:
        return min(quantiles, key=lambda m: abs(m.tau - rules.critical_ratio))
    return max(measures, key=lambda m: m.horizon) if measures else None


def primary_measure(config: dict, measures: list[Measure] | None = None) -> Measure:
    """``forecasting.primary_measure``, or the default when it is unset."""
    measures = parse_measures(config) if measures is None else measures
    wanted = config.get("forecasting", {}).get("primary_measure")
    if wanted:
        for m in measures:
            if m.id == wanted:
                return m
        raise ValueError(f"primary_measure `{wanted}` is not among the configured measures.")
    m = default_primary(measures, VN2Rules.from_config(config))
    if m is None:
        raise ValueError("No mean or quantile measure configured.")
    return m


def score_name(m: Measure) -> str:
    """Strictly consistent score of a measure: pinball for quantiles, RMSE for means."""
    return "pinball" if m.kind == "quantile" else "rmse"


def primary_metric_name(m: Measure) -> str:
    """Metric key minimized by model selection and tuning, e.g. ``pinball_q83_h2``."""
    return f"{score_name(m)}_{m.id}"


# ---------------------------------------------------------------------------
# Run manifests
# ---------------------------------------------------------------------------

def manifest_record(
    measures: list[Measure],
    primary: Measure,
    calibration: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure section of ``forecast_setup.json``."""
    return {
        "schema": 2,
        "measures": [m.to_dict() for m in measures],
        "primary_measure": primary.id,
        "calibration": dict(calibration or {}),
    }


def infer_measures(columns: list[str], rules: VN2Rules | None = None) -> list[Measure]:
    """Measures recognised from forecast column names (runs saved before manifests)."""
    rules = rules or VN2Rules()
    cr_label = quantile_label(rules.critical_ratio)
    out = []
    for col in columns:
        hit = _ID_RE.match(col)
        if not hit or int(hit.group(3)) not in HORIZONS:
            continue
        kind, label, h = hit.group(1), hit.group(2), int(hit.group(3))
        if kind:
            out.append(Measure(kind, h))
        else:
            tau = rules.critical_ratio if label == cr_label else int(label) / 100.0
            out.append(Measure("quantile", h, tau))
    return sorted(set(out), key=_sort_key)


def run_measures(config: dict, run_id: str | None) -> list[Measure]:
    """Measures of a forecasting run: its manifest, else inferred from its forecast columns."""
    from src.forecasting.periods import fetch_run_file, load_run_setup

    record = load_run_setup(config, run_id) or {}
    if record.get("measures"):
        # Older manifests can list kinds that are no longer forecast; skip them.
        kept = (Measure.from_dict(d) for d in record["measures"] if d.get("kind") in KINDS)
        return sorted(kept, key=_sort_key)
    path = fetch_run_file(config, run_id, "train_forecasts.parquet")
    if path is None:
        return []
    import pyarrow.parquet as pq

    return infer_measures(pq.read_schema(path).names, VN2Rules.from_config(config))


def poisson_quantile(mean: np.ndarray | float, tau: float) -> np.ndarray:
    """τ-quantile of a Poisson distribution with the given mean."""
    from scipy.stats import poisson

    mu = np.clip(np.asarray(mean, dtype=float), 0.0, None)
    return np.where(mu > 0, poisson.ppf(tau, np.maximum(mu, 1e-12)), 0.0)
