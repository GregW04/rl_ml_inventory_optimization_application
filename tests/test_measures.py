"""Forecast measure contract: parsing, ids, validation, legacy inference."""

from __future__ import annotations

import json

import pytest

from src.forecasting.measures import (
    Measure,
    default_primary,
    infer_measures,
    parse_measures,
    primary_metric_name,
    run_measures,
    validate_measures,
)
from src.simulation.vn2 import VN2Rules

RULES = VN2Rules()


def _cfg(measures, primary=None):
    return {"forecasting": {"measures": measures, "primary_measure": primary}, "vn2": {}}


def test_ids_and_critical_ratio_token():
    ms = parse_measures(
        _cfg(
            [
                {"kind": "quantile", "horizon": 2, "tau": "critical_ratio"},
                {"kind": "mean", "horizon": 0},
                {"kind": "quantile", "horizon": 1, "tau": 0.5},
            ]
        )
    )
    assert [m.id for m in ms] == ["mean_h0", "q50_h1", "q83_h2"]
    q83 = ms[-1]
    assert q83.tau == pytest.approx(1.0 / 1.2)
    assert Measure.from_dict(q83.to_dict()) == q83


def test_every_horizon_needs_a_level():
    ms = [Measure("mean", 0), Measure("quantile", 2, 0.83)]
    errors = validate_measures(ms)
    assert any("h1 needs a mean or a quantile" in e for e in errors)
    ok = ms + [Measure("quantile", 1, 0.5)]
    assert validate_measures(ok) == []


def test_unknown_kind_is_rejected():
    ms = [Measure("mean", 0), Measure("mean", 1), Measure("mean", 2), Measure("std", 2)]
    assert any("Unknown measure kind 'std'" in e for e in validate_measures(ms))


def test_duplicate_labels_and_bad_primary():
    ms = [Measure("mean", 0), Measure("mean", 1), Measure("quantile", 2, 0.8333), Measure("quantile", 2, 0.83)]
    assert any("q83_h2" in e for e in validate_measures(ms))
    ms = [Measure("mean", 0), Measure("mean", 1), Measure("mean", 2)]
    assert any("not among" in e for e in validate_measures(ms, "q83_h2"))


def test_run_measures_skips_retired_kinds(tmp_path):
    # Older manifests can still list a kind that is no longer forecast.
    setup = tmp_path / "fc" / "forecast_setup.json"
    setup.parent.mkdir()
    setup.write_text(json.dumps({"measures": [
        Measure("mean", 0).to_dict(),
        {"kind": "std", "horizon": 0, "tau": None},
        Measure("quantile", 2, 0.8333).to_dict(),
    ]}))
    cfg = {"_project_root": str(tmp_path), "forecasting": {"artifacts": {"forecasts_dir": "fc"}}, "vn2": {}}
    assert [m.id for m in run_measures(cfg, None)] == ["mean_h0", "q83_h2"]


def test_default_primary_and_metric():
    ms = [Measure("mean", 0), Measure("mean", 1), Measure("quantile", 2, 0.5), Measure("quantile", 2, 0.8333)]
    p = default_primary(ms, RULES)
    assert p.id == "q83_h2" and primary_metric_name(p) == "pinball_q83_h2"
    p = default_primary([Measure("mean", 0), Measure("mean", 1), Measure("mean", 2)], RULES)
    assert p.id == "mean_h2" and primary_metric_name(p) == "rmse_mean_h2"


def test_infer_from_legacy_columns():
    cols = ["store", "product", "date", "sales", "pred_mean_h0", "mean_h0", "mean_h1",
            "pred_q50_h2", "q50_h2", "q83_h2", "mu_h0", "sigma_h0", "block"]
    ms = infer_measures(cols, RULES)
    assert [m.id for m in ms] == ["mean_h0", "mean_h1", "q50_h2", "q83_h2"]
    assert ms[-1].tau == pytest.approx(RULES.critical_ratio)
