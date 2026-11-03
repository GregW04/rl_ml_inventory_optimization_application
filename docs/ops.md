# The weekly ops loop

`src/ops` runs the forecaster and the ordering policy the way production would: one cycle per weekly data
drop. Each cycle validates the data, scores last week, reads the monitors, decides whether to retrain, keep or
roll back, and places next week's orders. MLflow (tracking + Model Registry) is the system of record. Rules
and thresholds live in [`configs/ops.yaml`](../configs/ops.yaml).

```bash
make mlflow-server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
python -m src.ops.cli run --env dev                        # every step of every week, in order
python -m src.ops.cli step forecast --env dev --as-of 2023-10-09
python -m src.ops.cli drops release --env live --week 3    # live: publish VN2 file Week3
```

## Environments

An environment has its own registered models (`<env>-vn2-forecaster`), its own MLflow experiment
(`vn2-ops-<env>`) and its own local state (`ops/<env>/`).

- `sim` environments replay history from weekly snapshots cut from the Week0 files, so they never contain the
  VN2 test weeks.
- `live` releases the real `WeekK` files one at a time and starts from the official initial state.

Each environment's rules are the shared `rules` with its overrides applied. `dev` uses a cheap recipe and
`max_age_weeks: 4`, so refits and probation happen within a few weeks.

## A cycle (`as_of = D`, the drop's last week)

| Step | What it does |
|---|---|
| `ingest` | Checks the drop against the data contract (`src/ops/contract.py`): consecutive Mondays, the SKU set equal to the master, non-negative integer sales, the previous drop plus exactly one week, no restated history. It opens the cycle run and logs the drop as an input dataset. A failed contract stops the cycle, and the next week waits. |
| `score` | Realises week D on every trajectory with the order chosen in cycle D−1 (`vn2.step_one_week` semantics). In the first week it sets the start state instead. |
| `monitor` | Computes per-class bias and coverage of the current champion, the cost share per class, probation evidence and input diagnostics (`src/ops/monitor.py`). |
| `decide` | A pure rule table (`src/ops/decide.py`) returns bootstrap / rollback / refit / keep. |
| `refit` | Bootstrap or refit with the champion's recipe: logs a LoggedModel, registers it, runs sanity checks, then promotes. Or it rolls back. |
| `forecast` | Forecasts decision week D+1 from `@champion`, and from `@previous` while the new champion is on probation. |
| `order` | Computes the anchor's order for D+1 on every trajectory; `orders.csv` holds the live ones. |
| `report` | Logs metrics, tags and the cycle's files to the cycle run and to the timeline run (by week index), then closes the run. |

Timing follows the simulator. The order chosen in cycle D is placed at the start of week D+1 and arrives at
D+3. A forecaster promoted in cycle P therefore changes realised costs only from week P+3, so probation counts
its € evidence from then on. Forecast-level monitors see the new model from week P+1.

**Trajectories.** Demand is exogenous and unmet demand is lost, so every policy's stock can be simulated exactly
on the same weekly sales. `live` is the system. While a new champion is on probation, `previous` runs the old
forecaster from the state at promotion: an exact counterfactual, with no off-policy estimate needed.
Out-of-stock weeks are flagged `censored` and left out of error monitors and probation.

## Decisions (first values; calibrated on the shadow replay, then frozen)

| Priority | Situation | Action |
|---|---|---|
| 1 | No champion yet | **bootstrap** |
| 2 | On probation: € worse than `@previous` by more than `rollback_margin_pct`, CI above 0, at least `min_evidence_weeks` | **rollback** |
| 3 | Strategy `never` | **keep** |
| 4 | Age ≥ `max_age_weeks` (`fixed`, `drift_backstop`), or a bias / coverage alarm for `persistence_weeks` in a class with ≥ `min_cost_share` of the cost, after the cooldown (a severe bias alarm skips it) (`drift`, `drift_backstop`) | **refit** |
| 5 | Otherwise; input diagnostics (PSI, zero share, level vs last year, class churn) never retrain on their own | **keep** |

A refit is promoted unless sanity checks fail: missing or negative forecasts, q50 above q83, or a forecast
total more than `max_total_change_pct` away from the champion's. Version tags record the audit trail:
`trigger`, `as_of`, `recipe_key`, `data_digest`, `validation_status`, `promoted_as_of`, and `rejected_reason`
for rejected versions.

## Where things are

- **Local state:**
  - `ops/<env>/drops/`: the weekly snapshots;
  - `ops/<env>/cycles/<D>/`:
    - `contract.json`, `signals.json`, `decision.json`;
    - `state_<trajectory>.parquet`, `costs_<trajectory>.parquet`, `forecasts_<alias>.parquet`;
    - `orders.csv`;
    - `done`.

  Every cycle folder is also logged to its MLflow cycle run (`cycle/`).
- **MLflow:**
  - an `<env>-timeline` run whose metrics are indexed by week (cost per trajectory, bias, coverage, age,
    diagnostics);
  - a `cycle-<D>` run per week (child of the timeline);
  - `bootstrap-<D>` / `refit-<D>` runs (children of their cycle) with the training data as an input and the
    model.
- **Models:** downloaded versions are cached under `artifacts/cache/models/<model_id>/`.

## First dev run (2023-10-02 → 2023-12-11, 11 weeks, 2 min)

- **Decisions:** bootstrap, then age refits on 10-30 and 11-27, both promoted.
- **Shadow costs:** the previous forecaster's shadow costs equal live for two weeks after each promotion, then
  diverge. On 11-20, v2 cost 2.038 vs v1's 2.232 €/SKU.
- **Probation:** v2's evidence after two weeks was −0.07% with a CI of [−2.7%, +2.1%], so it was kept.
- **Calibration notes for Phase 6:**
  - a bias alarm needs at least warm-up + persistence = 6 weeks after a promotion;
  - two evidence weeks give a CI about ±2.4 points wide, much wider than the 1% rollback margin.
