"""Ops loop building blocks: ledger = simulator, data contract, decision table, monitors, no torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.forecasting.drift import trigg_signal
from src.forecasting.preprocessing import load_master
from src.ops import contract, drops, ledger, monitor, registry, steps
from src.ops.decide import decide
from src.ops.settings import OpsConfig, Rules
from src.simulation.anchor import AnchorSpec, LostSalesAnchor
from src.simulation.vn2 import InventoryState, VN2Rules, step_one_week
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RULES = VN2Rules()
WEEK = pd.Timedelta(weeks=1)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_realise_matches_the_simulator_week():
    rng = np.random.default_rng(0)
    n = 300
    state = pd.DataFrame({"store": np.arange(n), "product": 1, "on_hand": rng.integers(0, 9, n).astype(float),
                          "transit_1": rng.integers(0, 6, n).astype(float), "transit_2": rng.integers(0, 6, n).astype(float),
                          "order": rng.integers(0, 6, n).astype(float)})
    demand = pd.DataFrame({"store": np.arange(n), "product": 1, "demand": rng.poisson(4, n).astype(float), "censored": False})
    new, costs = ledger.realise(state, demand, RULES)
    for i in range(n):
        r = state.iloc[i]
        s, w = step_one_week(InventoryState(r.on_hand, [r.transit_1, r.transit_2]), demand["demand"].iloc[i], r.order, RULES)
        assert (new["on_hand"].iloc[i], new["transit_1"].iloc[i], new["transit_2"].iloc[i]) == (s.on_hand, *s.pipeline)
        assert costs["cost_eur"].iloc[i] == pytest.approx(w.total_eur)
    with pytest.raises(ValueError):
        ledger.realise(state.assign(order=np.nan), demand, RULES)


def test_anchor_orders_use_this_and_next_weeks_arrivals():
    spec = AnchorSpec("mean_h0", "mean_h1", "quantile", ("q83_h2",), 0.8333)
    anchor = LostSalesAnchor(spec, RULES)
    state = pd.DataFrame({"store": [1, 2], "product": [1, 1], "on_hand": [3.0, 0.0], "transit_1": [2.0, 5.0],
                          "transit_2": [1.0, 0.0], "order": np.nan})
    fc = pd.DataFrame({"store": [1, 2], "product": [1, 1], "mean_h0": [4.0, 1.0], "mean_h1": [2.0, 2.0], "q83_h2": [6.0, 3.0]})
    got = ledger.anchor_orders(state, fc, anchor).tolist()
    want = [anchor.decide(3, 2, 1, {"mean_h0": 4, "mean_h1": 2, "q83_h2": 6}).order,
            anchor.decide(0, 5, 0, {"mean_h0": 1, "mean_h1": 2, "q83_h2": 3}).order]
    assert got == want
    with pytest.raises(ValueError, match="missing"):
        ledger.anchor_orders(state, fc.iloc[:1], anchor)  # a missing forecast is never an order of 0


def test_vn2_initial_state_maps_end_inventory_and_transit():
    cfg = load_config()
    s = ledger.initial_state_vn2(cfg["data"]["initial_state_file"])
    raw = pd.read_csv(cfg["data"]["initial_state_file"])
    assert len(s) == 599 and s["order"].isna().all()
    assert s["on_hand"].sum() == raw["End Inventory"].sum() and s["transit_2"].sum() == raw["In Transit W+2"].sum()


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drop():
    cfg = load_config()
    sales, instock = pd.read_csv(drops.vn2_sales_file(cfg, 0)), pd.read_csv(cfg["data"]["instock_file"])
    as_of = pd.Timestamp("2023-10-09")
    return (drops.cut(sales, as_of), drops.cut(instock, as_of), load_master(cfg["data"]["master_file"]), as_of,
            drops.cut(sales, as_of - WEEK))


def test_a_clean_drop_passes(drop):
    sales, instock, master, as_of, prev = drop
    assert contract.check_drop(sales, instock, master, as_of, prev).ok


@pytest.mark.parametrize("break_it, message", [
    (lambda s, p: (s.assign(**{"2023-01-02": s["2023-01-02"] + 1}), p), "restated"),
    (lambda s, p: (s.drop(columns="2023-05-01"), p), "consecutive"),
    (lambda s, p: (s.assign(**{"2023-10-09": -s["2023-10-09"] - 1}), p), "negative"),
    (lambda s, p: (s.assign(**{"2023-10-09": np.nan}), p), "missing"),
    (lambda s, p: (s.iloc[1:], p), "SKU set"),
    (lambda s, p: (p, p), "last week"),
])
def test_broken_drops_fail(drop, break_it, message):
    sales, instock, master, as_of, prev = drop
    bad_sales, bad_prev = break_it(sales.copy(), prev.copy())
    report = contract.check_drop(bad_sales, instock, master, as_of, bad_prev)
    assert not report.ok and any(message in e for e in report.errors), report.errors


def test_a_catch_up_drop_only_warns(drop):
    sales, instock, master, as_of, _ = drop
    report = contract.check_drop(sales, instock, master, as_of, drops.cut(sales, as_of - 2 * WEEK))
    assert report.ok and any("catch-up" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------

def _signals(age=5, bias=None, coverage=None, share=None, probation=None):
    return {"champion": {"version": "1", "age_weeks": age}, "bias": bias or {}, "coverage": coverage or {},
            "cost_share": share or {"Smooth": 0.5, "Lumpy": 0.05}, "probation": probation}


ALARM = {"Smooth": {"ts": 0.8, "streak": 2, "severe": False}}
SMALL_ALARM = {"Lumpy": {"ts": 0.95, "streak": 3, "severe": True}}


@pytest.mark.parametrize("strategy, signals, action", [
    ("drift_backstop", {}, "bootstrap"),
    ("drift_backstop", _signals(probation={"evidence_weeks": 2, "delta_pct": 3.0, "ci_lo": 0.4, "ci_hi": 5.0}), "rollback"),
    ("drift_backstop", _signals(probation={"evidence_weeks": 2, "delta_pct": 3.0, "ci_lo": -0.4, "ci_hi": 5.0}), "keep"),
    ("drift_backstop", _signals(probation={"evidence_weeks": 1, "delta_pct": 3.0, "ci_lo": 0.4, "ci_hi": 5.0}), "keep"),
    ("never", _signals(age=40, bias=ALARM), "keep"),
    ("fixed", _signals(age=13), "refit"),
    ("fixed", _signals(age=12, bias=ALARM), "keep"),
    ("drift", _signals(age=6, bias=ALARM), "refit"),
    ("drift", _signals(age=40), "keep"),
    ("drift", _signals(age=6, bias=SMALL_ALARM), "keep"),  # Lumpy carries 5% of the cost
    ("drift", _signals(age=2, bias=ALARM), "keep"),  # cooldown
    ("drift", _signals(age=2, bias={"Smooth": {"ts": 0.95, "streak": 2, "severe": True}}), "refit"),
    ("drift", _signals(age=6, coverage={"Smooth": {"value": 0.71, "streak": 2}}), "refit"),
    ("drift", _signals(age=6, coverage={"Smooth": {"value": 0.71, "streak": 1}}), "keep"),
    ("drift_backstop", _signals(age=13), "refit"),
    ("drift_backstop", _signals(age=6, bias=ALARM), "refit"),
])
def test_decision_table(strategy, signals, action):
    assert decide(signals, Rules(strategy=strategy)).action == action


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

def test_tracking_signal_false_alarm_rate_on_unbiased_errors():
    rng = np.random.default_rng(0)
    shares = [np.mean(np.abs(trigg_signal(rng.normal(0, 0.1, 400), 0.2)[3:]) > 0.74) for _ in range(100)]
    assert 0.03 < np.mean(shares) < 0.07  # about 5% of weeks: why persistence and cost weights gate refits


def _frames(n_weeks: int, bias: float, cls: str = "Smooth"):
    start = pd.Timestamp("2023-01-02")
    fc = pd.DataFrame([{"store": s, "product": 1, "date": start + k * WEEK, "mean_h0": 10.0, "q83_h2": 12.0}
                       for k in range(n_weeks) for s in range(20)])
    act = fc[["store", "product", "date"]].assign(sales=10.0 * (1 + bias), demand_class=cls)
    return fc, act


def test_bias_signal_waits_for_warmup_and_flags_a_persistent_bias():
    rules = Rules()
    fc, act = _frames(3, 0.2)
    assert monitor.bias_signals(fc, act, rules)["Smooth"]["ts"] is None  # 3 errors < warm-up of 4
    fc, act = _frames(10, 0.2)
    s = monitor.bias_signals(fc, act, rules)["Smooth"]
    assert s["ts"] > rules.bias.limit and s["streak"] >= 2
    fc, act = _frames(10, 0.0)
    assert monitor.bias_signals(fc, act, rules)["Smooth"]["streak"] == 0


def test_coverage_scores_the_h2_quantile_two_weeks_after_its_row():
    fc, act = _frames(8, 0.0)  # sales 10 <= q83 12 everywhere
    act.loc[act["date"] >= pd.Timestamp("2023-01-30"), "sales"] = 20.0  # weeks 4-7 above the quantile
    cov = monitor.coverage_signals(fc, act, Rules())["Smooth"]
    # Rows dated k score week k+2: weeks 2..7 are scored; the window ending week 7 holds weeks 4-7.
    assert cov["value"] == 0.0 and cov["streak"] >= 1


def test_probation_counts_only_weeks_the_new_orders_can_affect():
    promoted = pd.Timestamp("2023-03-06")
    rows = []
    for k in range(1, 7):
        week = promoted + k * WEEK
        for s in range(30):
            live = 5.0 + (1.0 if k >= 3 else 0.0)  # the new champion costs more from P+3 on
            rows.append({"store": s, "product": 1, "week": week, "live": live, "shadow": 5.0})
    df = pd.DataFrame(rows)
    live = df.rename(columns={"live": "cost_eur"}).assign(censored=False)[["store", "product", "week", "cost_eur", "censored"]]
    shadow = df.rename(columns={"shadow": "cost_eur"}).assign(censored=False)[["store", "product", "week", "cost_eur", "censored"]]
    p = monitor.probation_signal(live, shadow, promoted, Rules())
    assert p["evidence_weeks"] == 4 and p["delta_pct"] == pytest.approx(20.0) and p["ci_lo"] > 0


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------

def test_ops_steps_never_import_torch():
    # The forecaster's LightGBM segfaults next to torch: every ops step must stay torch-free.
    code = "import sys; import src.ops.steps, src.ops.cli; assert 'torch' not in sys.modules, 'torch imported'"
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]


# ---------------------------------------------------------------------------
# Policy layer
# ---------------------------------------------------------------------------

def _policy_signals(kind="anchor", since=None, eligible=True, vs_anchor=None, churn=None):
    return {"policy": {"kind": kind, "weeks_since_rl_attempt": since}, "rl_eligible": eligible,
            "policy_vs_anchor": vs_anchor, "gate_churn": churn}


@pytest.mark.parametrize("signals, action", [
    ({}, "bootstrap"),
    (_policy_signals(eligible=False), "keep"),
    (_policy_signals(), "retrain"),  # eligible, never trained
    (_policy_signals(since=5), "keep"),
    (_policy_signals(since=13), "retrain"),
    (_policy_signals(kind="ppo", since=3, churn=0.15), "retrain"),
    (_policy_signals(kind="ppo", since=3, churn=0.05), "keep"),
    (_policy_signals(kind="ppo", since=3, vs_anchor={"evidence_weeks": 3, "delta_pct": 2.0, "ci_lo": 0.3, "ci_hi": 3.0}), "fallback"),
    (_policy_signals(kind="ppo", since=3, vs_anchor={"evidence_weeks": 3, "delta_pct": 2.0, "ci_lo": -0.3, "ci_hi": 3.0}), "keep"),
    (_policy_signals(kind="ppo", since=3, vs_anchor={"evidence_weeks": 1, "delta_pct": 9.0, "ci_lo": 5.0, "ci_hi": 12.0}), "keep"),
])
def test_policy_decision_table(signals, action):
    from src.ops.decide import decide_policy

    assert decide_policy(signals, Rules()).action == action


def test_rl_windows_and_eligibility():
    from src.ops import rl
    from src.ops.settings import PolicyRules

    pr = PolicyRules(train_weeks=26, val_weeks=8)
    w = rl.windows(pd.Timestamp("2024-04-01"), pr)
    assert (w["rl_val_start"], w["rl_val_end"]) == (pd.Timestamp("2024-02-12"), pd.Timestamp("2024-04-01"))
    assert (w["rl_train_start"], w["rl_train_end"]) == (pd.Timestamp("2023-08-14"), pd.Timestamp("2024-02-05"))
    first = pd.Timestamp("2022-10-10")  # T0 = 2022-10-03: the first decision week
    # 38 ledger weeks, counting both ends: 4 burn-in + 26 training + 8 validation.
    assert not rl.eligible(first + 36 * WEEK, first, pr) and rl.eligible(first + 37 * WEEK, first, pr)


def test_forecaster_tenures_become_training_fits():
    from src.ops import rl

    dates = pd.date_range("2023-01-02", periods=8, freq="7D")
    fc = pd.DataFrame({"date": dates, "version": ["1", "1", "1", "2", "2", "1", "1", "1"]})
    fits = rl.version_fits(fc, {"1": "2022-12-26", "2": "2023-01-16"}, dates[0], dates[-1])
    assert [(f["start"], f["end"], f["fit_end"]) for f in fits] == [
        ("2023-01-02", "2023-01-16", "2022-12-26"), ("2023-01-23", "2023-01-30", "2023-01-16"),
        ("2023-02-06", "2023-02-20", "2022-12-26")]  # a rollback starts a new tenure of version 1


def test_challenger_report_gates_and_checks_every_seed():
    from src.ops import rl
    from src.ops.settings import PolicyRules

    def eps(smooth, lumpy):  # cost per class against an anchor at 10
        return pd.DataFrame({"demand_class": ["Smooth", "Lumpy"], "cost_eur": [smooth, lumpy],
                             "anchor_cost_eur": [10.0, 10.0], "units": [10.0, 10.0], "anchor_units": [10.0, 10.0]})

    good = rl.challenger_report({42: eps(11.0, 8.0), 43: eps(12.0, 8.5)}, PolicyRules())
    assert good["gate"] == ("Lumpy",) and good["passed"]  # RL only on Lumpy: −8.75% on average
    flat = rl.challenger_report({42: eps(10.0, 10.0), 43: eps(10.0, 10.0)}, PolicyRules())
    assert not flat["passed"] and flat["gate"] == ()


class _Echo(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def test_a_promoted_policy_restarts_the_anchor_shadow_from_the_live_state(tmp_path):
    config = load_config()
    config["_project_root"] = str(tmp_path)
    ops = OpsConfig.model_validate({"envs": {"t": {"kind": "sim", "start": "2023-01-02", "end": "2023-06-26"}}})
    c = steps.Cycle(config, ops, "t", pd.Timestamp("2023-04-03"))
    live = pd.DataFrame({"store": [1, 2], "product": [7, 7], "on_hand": [3.0, 0.0], "transit_1": [1.0, 2.0],
                         "transit_2": [0.0, 4.0], "order": [0.0, 0.0]})
    c.write("state_live.parquet", live)
    c.write("state_anchor.parquet", live.assign(on_hand=[9.0, 9.0]))  # left by an earlier RL policy's shadow
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(name="m", python_model=_Echo())
    version = str(registry.register(info.model_uri, c.policy_name, {"kind": "ppo"}).version)

    steps.promote_policy(c, version)
    champion = registry.resolve(c.policy_name, registry.CHAMPION)
    assert str(champion.version) == version and champion.tags["promoted_as_of"] == "2023-04-03"
    pd.testing.assert_frame_equal(c.read("state_anchor.parquet"), live)


def test_an_earlier_week_cannot_run_after_a_later_one(tmp_path):
    config = load_config()
    config["_project_root"] = str(tmp_path)
    ops = OpsConfig.model_validate({"envs": {"t": {"kind": "sim", "start": "2023-01-02", "end": "2023-06-26"}}})
    later = steps.Cycle(config, ops, "t", pd.Timestamp("2023-02-13"))
    later.path("done").parent.mkdir(parents=True)
    later.path("done").write_text("2023-02-13")
    with pytest.raises(RuntimeError, match="has already run 2023-02-13"):
        steps.step_ingest(steps.Cycle(config, ops, "t", pd.Timestamp("2023-02-06")))


def test_a_week_has_one_timestamp_unit_however_it_arrives():
    # The CLI (Airflow tasks) parses a string, the in-process loop walks a date range: same ledgers.
    ops = OpsConfig.model_validate({"envs": {"t": {"kind": "sim", "start": "2023-01-02", "end": "2023-06-26"}}})
    from_cli = steps.Cycle(load_config(), ops, "t", pd.Timestamp("2023-02-06"))
    from_loop = steps.Cycle(load_config(), ops, "t", steps.weeks("2023-01-02", "2023-06-26")[5])
    frames = [pd.DataFrame({"x": [1.0]}).assign(week=c.as_of) for c in (from_cli, from_loop)]
    pd.testing.assert_frame_equal(*frames)
    assert frames[0]["week"].dtype == "datetime64[ns]"


# ---------------------------------------------------------------------------
# Calibration (the decision rule replayed on weekly series)
# ---------------------------------------------------------------------------

def _weekly(n_weeks: int, shift_from: int | None = None, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2022-10-10", periods=n_weeks, freq="7D")
    errors = pd.DataFrame({c: rng.normal(0, 0.04, n_weeks) for c in ("Smooth", "Lumpy")}, index=weeks)
    if shift_from is not None:
        errors.iloc[shift_from:] += shift
    counts = pd.DataFrame(300.0, index=weeks, columns=["Smooth", "Lumpy"])
    hits = pd.DataFrame(rng.binomial(300, 0.833, (n_weeks, 2)).astype(float), index=weeks, columns=["Smooth", "Lumpy"])
    # As in a ledger: q83_h2 is scored from two weeks after the first mean_h0 error.
    return errors, hits.iloc[2:], counts.iloc[2:]


def test_the_drift_rule_refits_after_a_level_shift_and_restarts_its_signal():
    from src.ops import calibrate

    rules = Rules(strategy="drift")
    share = {"Smooth": 0.5, "Lumpy": 0.5}
    # Pure noise: a strict limit never fires (the first values do now and then: what calibration sets).
    assert calibrate.simulate_refits(*_weekly(60), share, Rules(strategy="drift", bias={"limit": 0.95})) == []
    refits = calibrate.simulate_refits(*_weekly(60, shift_from=30, shift=0.3), share, rules)
    # Warm-up (4 errors) and persistence (2 weeks) bound the delay after the shift…
    assert 30 <= [r for r in refits if r >= 30][0] <= 36
    # …and a refit restarts the signal: the next one needs a new warm-up and streak.
    assert all(b - a >= rules.bias.warmup_weeks + rules.bias.persistence_weeks - 1 for a, b in zip(refits, refits[1:]))
    # Classes below the cost-share gate never trigger.
    assert calibrate.simulate_refits(*_weekly(60, shift_from=30, shift=0.3), {"Smooth": 0.1, "Lumpy": 0.1}, rules) == []


def test_false_refits_fall_with_the_limit():
    from src.ops import calibrate

    errors, hits, counts = _weekly(80, seed=3)
    share = {"Smooth": 0.5, "Lumpy": 0.5}
    rate = {limit: calibrate.false_refits_per_26_weeks(errors, hits, counts, share, Rules(strategy="drift", bias={"limit": limit}),
                                                       n_paths=30, length=52)
            for limit in (0.3, 0.74, 0.99)}
    assert rate[0.3] > rate[0.74] >= rate[0.99] and rate[0.99] < 0.1


def test_strategy_comparison_pairs_sku_weeks_and_leaves_out_censored_ones():
    from src.ops import calibrate

    rng = np.random.default_rng(0)
    weeks = pd.date_range("2023-01-02", periods=40, freq="7D")
    base = pd.DataFrame([(s, 1, w) for s in range(50) for w in weeks], columns=["store", "product", "week"])
    a = base.assign(cost_eur=rng.gamma(2, 1, len(base)), censored=False)
    b = a.assign(cost_eur=a["cost_eur"] * 1.02)
    b.loc[b.index[:30], "censored"] = True  # out of stock in b: left out of both
    b.loc[b.index[:30], "cost_eur"] = 1e6
    out = calibrate.compare_strategies({"a": a, "b": b}, reference="a")
    assert out.loc["a", "vs_reference_pct"] == 0.0
    assert out.loc["b", "vs_reference_pct"] == pytest.approx(2.0) and out.loc["b", "ci_lo"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Retune (an operator's challenge)
# ---------------------------------------------------------------------------

def test_a_forecast_changes_costs_only_once_its_order_arrives():
    anchor = LostSalesAnchor(AnchorSpec("mean_h0", "mean_h1", "q83_h2", ("q83_h2",), 0.8333), RULES)
    weeks_ = pd.date_range("2023-01-02", periods=6, freq="7D")
    state = pd.DataFrame({"store": [1], "product": [1], "on_hand": [5.0], "transit_1": [2.0], "transit_2": [2.0], "order": [np.nan]})
    demand = {w: pd.DataFrame({"store": [1], "product": [1], "demand": [3.0], "censored": [False]}) for w in weeks_[1:]}
    low = {w: pd.DataFrame({"store": [1], "product": [1], "mean_h0": [3.0], "mean_h1": [3.0], "q83_h2": [6.0]}) for w in weeks_[:-1]}
    high = {w: f.assign(q83_h2=12.0) for w, f in low.items()}
    a, b = (ledger.anchor_replay(state, f, demand, anchor, RULES).set_index("week")["cost_eur"] for f in (low, high))
    # Cycle A's order arrives at A+3: weeks A+1 and A+2 can't differ, A+3 does (more stock, more holding).
    assert (a.iloc[:2] == b.iloc[:2]).all() and b.iloc[2] > a.iloc[2]


def test_a_won_challenge_is_a_retune_unless_the_champion_must_roll_back():
    rules = Rules()
    champion = {"version": "2", "fit_as_of": "2023-01-02", "promoted_as_of": "2023-01-02", "age_weeks": 3,
                "weeks_since_promotion": 3}
    pending = {"recipe": "r2", "delta_pct": -1.2, "ci_lo": -2.0, "ci_hi": -0.4, "as_of": "2023-01-16", "run_id": "abc"}
    d = decide({"champion": champion, "pending_recipe": pending}, rules)
    assert d.action == "retune" and "r2" in d.reasons[0]
    probation = {"evidence_weeks": 4, "delta_pct": 3.0, "ci_lo": 1.0, "ci_hi": 5.0}
    assert decide({"champion": champion, "pending_recipe": pending, "probation": probation}, rules).action == "rollback"


def test_coverage_profile_finds_no_extra_spread_in_independent_hits():
    from src.ops import calibrate

    _, hits, counts = _weekly(120, seed=5)
    prof = calibrate.coverage_profile(hits, counts, window_weeks=4)
    assert prof["coverage"].between(0.81, 0.86).all()
    # Independent SKU-weeks: the window coverage spreads like a binomial (design effect ≈ 1).
    assert prof["design_effect"].between(0.4, 2.0).all()


def test_threshold_grid_reports_a_rate_and_a_first_refit_per_setting():
    from src.ops import calibrate

    errors, hits, counts = _weekly(70, shift_from=40, shift=0.3, seed=1)
    grid = calibrate.threshold_grid(errors, hits, counts, {"Smooth": 0.5, "Lumpy": 0.5}, Rules(), limits=(0.6, 0.9),
                                    persistences=(2,), half_widths=(0.05,), n_paths=10, length=52)
    bias, cov = grid["bias"], grid["coverage"]
    assert len(bias) == 2 and len(cov) == 1
    # A stricter limit fires less on null paths, and the level shift is caught either way.
    assert bias.loc[0, "false_refits_per_26w"] >= bias.loc[1, "false_refits_per_26w"]
    assert all(pd.Timestamp(d) >= errors.index[30] for d in bias["first_refit"].dropna())


def test_live_runs_only_frozen_rules(tmp_path, monkeypatch):
    from src.utils import tracking

    config = load_config()
    config["_project_root"] = str(tmp_path)
    live = {"kind": "live", "start": "2024-04-08", "end": "2024-06-03"}
    monkeypatch.setattr(tracking, "git_state", lambda: ("abc", False))
    with pytest.raises(RuntimeError, match="aren't the frozen ones"):
        steps.step_ingest(steps.Cycle(config, OpsConfig.model_validate({"envs": {"live": live}}), "live", pd.Timestamp("2024-04-08")))
    frozen = OpsConfig.model_validate({"frozen": True, "frozen_rules": Rules().version(), "envs": {"live": live}})
    monkeypatch.setattr(tracking, "git_state", lambda: ("abc", True))
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        steps.step_ingest(steps.Cycle(config, frozen, "live", pd.Timestamp("2024-04-08")))
    monkeypatch.setattr(tracking, "git_state", lambda: ("abc", False))
    with pytest.raises(FileNotFoundError):  # past the guard: there is no drop yet
        steps.step_ingest(steps.Cycle(config, frozen, "live", pd.Timestamp("2024-04-08")))


def test_coverage_as_a_diagnostic_never_refits():
    champion = {"version": "1", "fit_as_of": "2023-01-02", "promoted_as_of": "2023-01-02", "age_weeks": 8,
                "weeks_since_promotion": 8}
    signals = {"champion": champion, "cost_share": {"Smooth": 0.6},
               "coverage": {"Smooth": {"weeks": 8, "value": 0.70, "streak": 3}}}
    assert decide(signals, Rules(strategy="drift")).action == "refit"
    assert decide(signals, Rules(strategy="drift", coverage={"trigger": False})).action == "keep"


def test_carry_over_starts_an_environment_from_another_ones_champions(tmp_path):
    config = load_config()
    config["_project_root"] = str(tmp_path)
    env = {"kind": "sim", "start": "2023-01-02", "end": "2023-06-26"}
    ops = OpsConfig.model_validate({"envs": {"src": env, "dst": {**env, "start": "2023-07-03", "end": "2023-09-25"}}})

    def version(name: str, tags: dict) -> str:
        with mlflow.start_run():
            info = mlflow.pyfunc.log_model(name="m", python_model=_Echo())
        return str(registry.register(info.model_uri, name, tags).version)

    fc, pol = registry.model_name("src", "forecaster"), registry.model_name("src", "policy")
    registry.promote(fc, version(fc, {"as_of": "2023-04-03", "recipe_key": "abc"}))
    anchor = version(pol, {"kind": "anchor", "as_of": "2023-01-02"})
    registry.promote(pol, anchor)
    registry.promote(pol, version(pol, {"kind": "ppo", "as_of": "2023-06-19", "gate": '["Lumpy"]'}))

    out = steps.carry_over(config, ops, "src", "dst")
    dst_fc = registry.resolve(registry.model_name("dst", "forecaster"), registry.CHAMPION)
    src_fc = registry.resolve(fc, registry.CHAMPION)
    # The same logged model, its fit date kept (the age clock carries on), promoted at the new start.
    assert dst_fc.source == src_fc.source and dst_fc.tags["as_of"] == "2023-04-03"
    assert dst_fc.tags["trigger"] == "carry-over" and dst_fc.tags["promoted_as_of"] == "2023-07-03"
    dst_pol = registry.model_name("dst", "policy")
    champion, previous = (registry.resolve(dst_pol, a) for a in (registry.CHAMPION, registry.PREVIOUS))
    assert champion.tags["kind"] == "ppo" and previous.tags["kind"] == "anchor"  # the fallback target is there
    assert len(out["policy"]) == 2
    with pytest.raises(RuntimeError, match="already has versions"):
        steps.carry_over(config, ops, "src", "dst")
