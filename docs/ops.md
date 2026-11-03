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
make airflow                                               # the same steps as Airflow DAGs (see Orchestration)
```

## Environments

An environment has its own registered models (`<env>-vn2-forecaster`, `<env>-vn2-policy`), its own MLflow
experiment (`vn2-ops-<env>`) and its own local state (`ops/<env>/`).

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
| `monitor` | Computes per-class bias and coverage of the current champion, the cost share per class, probation evidence and input diagnostics (`src/ops/monitor.py`). For the policy: whether the ledger is long enough to train RL, an RL policy's cost against its anchor shadow, and gate churn. |
| `decide` | Two pure rule tables (`src/ops/decide.py`): bootstrap / rollback / refit / keep for the forecaster, and bootstrap / fallback / retrain / keep for the policy. |
| `refit` | Bootstrap or refit with the champion's recipe: logs a LoggedModel, registers it, runs sanity checks, then promotes. Or it rolls back. |
| `forecast` | Forecasts decision week D+1 from `@champion`, and from `@previous` while the new champion is on probation. |
| `retrain` | The policy layer. It registers the anchor as policy version 1 (bootstrap), moves `@champion` back to the anchor (fallback), or trains a PPO challenger on the forecast ledger, checks it, registers it and promotes it if it passes. Training imports torch, so this step runs in its own process. |
| `order` | Orders for D+1 from `<env>-vn2-policy@champion` on `live` (and on `previous`), and from the anchor on the `anchor` shadow. `orders.csv` holds the live orders with their parts: `anchor_order`, `adjustment` and `rl` (whether the learned policy decided). |
| `report` | Logs metrics, tags and the cycle's files to the cycle run and to the timeline run (by week index), then closes the run. |

Timing follows the simulator. The order chosen in cycle D is placed at the start of week D+1 and arrives at
D+3. A forecaster or policy promoted in cycle P therefore changes realised costs only from week P+3, so
probation and the fallback check count their € evidence from then on. Forecast-level monitors see a new
forecaster from week P+1.

**Trajectories.** Demand is exogenous and unmet demand is lost, so every policy's stock can be simulated exactly
on the same weekly sales. `live` is the system. While a new forecaster is on probation, `previous` runs the old
forecaster (under the same policy) from the state at promotion. While an RL policy serves, `anchor` runs the
anchor on the champion forecaster's forecasts from the state at the policy's promotion. Both are exact
counterfactuals, with no off-policy estimate needed. Out-of-stock weeks are flagged `censored` and left out of
error monitors and probation.

## Decisions (first values; calibrated on the shadow replay, then frozen)

**Forecaster**

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

**Policy** (`rules.policy`)

| Priority | Situation | Action |
|---|---|---|
| 1 | No policy registered | **bootstrap**: the anchor becomes version 1, so a fallback is always one alias move away |
| 2 | An RL policy serves and costs more than its anchor shadow by more than `fallback_margin_pct` over the latest `evidence_weeks`, CI above 0, at least `min_evidence_weeks` | **fallback** to the anchor; the RL version is tagged `rejected` |
| 3 | `enabled: false`, or the ledger doesn't cover 4 burn-in + `train_weeks` + `val_weeks` weeks yet | **keep** |
| 4 | No RL challenger yet, or the last one `retrain_every_weeks` ago (rejected ones count), or more than `gate_churn_limit` of the SKUs crossed the serving policy's gate since it was trained | **retrain** |
| 5 | Otherwise | **keep** |

The fallback check runs every week while an RL policy serves, not only during a probation window. Both
trajectories read the same champion forecaster, so a forecaster refit doesn't break the pairing.

## The RL challenger (`src/ops/rl.py`)

- **Data: the forecast ledger.** The challenger trains on the forecasts each week's champion really issued,
  so it learns from the errors it will face in service. Training windows start in the `train_weeks` that end
  `val_weeks` before the as-of week. The last `val_weeks` are the single validation check.
  - The data goes through the same `RunData` → `block_episodes` path as research (`src/rl/data.py`).
  - Forecaster versions play the research pipeline's block fits when training forecasts are debiased
    (`training_forecasts: unbiased_by_class`).
  - Demand scales come from history up to the end of training.
- **Recipe: steady state.** The state is the research state without `round_frac` and `forecast_error`. Both are
  VN2-window constructs: the order round doesn't exist week to week, and the error since the window started
  is 0 in every training window's first week, but real at serving.
  - `recipe_key` is the spec hash with one forecast source per environment (`ledger:<env>`), so every retrain
    of the recipe shares it.
  - The hyperparameters are the research champion's: σ 0.30 at the start, early stopping on the training
    sample, a 2,000-iteration budget.
- **Seeds.** Each seed trains in its own child run of the cycle (`stage=rl-train`). The project seed's final
  weights are served. The other seeds are the robustness check; weights are never picked by validation score.
- **Promotion check** on the validation weeks, all of:
  - the demand-class gate is picked there, as in research, and isn't empty;
  - the mean gated cost is ≤ the anchor's − `promote_margin_pct`;
  - units are within ±`units_band_pct` of the anchor's in every seed;
  - at least `min_seed_share` of the seeds beat the anchor.

  The gate is optimistic on the weeks that picked it. That's why a promoted policy is then watched live
  against the anchor's shadow (the fallback rule).
- **Registration.** Every challenger is registered, passed or not. Its tags are `kind=ppo`, `trigger`,
  `as_of`, `recipe_key`, `gate`, `val_vs_anchor_pct`, `seeds_beating_anchor`, `validated_with_forecaster`,
  `training_runs` and `validation_status`, plus `rejected_reason` when it failed.
- **Serving without torch.** The policy pyfunc (`src/serving/policy_model.py`) holds the actor as numpy
  weights and computes the observation and the order with the training environment's functions
  (`src/rl/observation.py`). So the order step shares a process with any forecaster.
  - `tests/test_serving_policy.py` checks that the served orders equal the training environment's:
    the anchor on all 599 SKUs, and PPO on 60 states including the gate.

## Where things are

- **Local state:**
  - `ops/<env>/drops/`: the weekly snapshots;
  - `ops/<env>/cycles/<D>/`:
    - `contract.json`, `signals.json`, `decision.json` (with the policy's decision under `policy`);
    - `state_<trajectory>.parquet`, `costs_<trajectory>.parquet`, `forecasts_<alias>.parquet`;
    - `orders.csv`;
    - `done`.

  Every cycle folder is also logged to its MLflow cycle run (`cycle/`).
- **MLflow:**
  - an `<env>-timeline` run whose metrics are indexed by week (cost per trajectory, bias, coverage, age,
    diagnostics, the share of SKUs RL decides);
  - a `cycle-<D>` run per week (child of the timeline);
  - children of their cycle:
    - `bootstrap-<D>` / `refit-<D>` runs, with the training data as an input and the forecaster;
    - `policy-bootstrap-<D>` (the anchor policy);
    - for a retrain, one `rl-train` run per seed and a `policy-<D>` run with the validation metrics and the
      policy model.
- **Models:** downloaded versions are cached under `artifacts/cache/models/<model_id>/`.

## Orchestration (Airflow)

`python -m src.ops.cli run` is the whole loop without an orchestrator. Airflow runs the same steps as one DAG
per environment, with retries left to the operator, a UI per week and task, and one heavy job at a time
across environments.

```bash
make airflow-install   # once: Airflow 3.3.2 in .venv-airflow, with the official constraints
make airflow-init      # once: the metadata DB (airflow_home/) and the 1-slot heavy pool
make airflow           # standalone on http://127.0.0.1:8080 (starts the MLflow server too)
make airflow-stop      # stop it when idle (about 1 GB)
```

Log in as `admin`; the password is in `airflow_home/simple_auth_manager_passwords.json.generated`. Everything
listens on 127.0.0.1 only.

- **DAGs** (`airflow/dags/vn2_ops.py`): `vn2_<env>` for every environment in `configs/ops.yaml`. A new
  environment appears on the next parse, paused.
- **Shape:**
  `drop` (sim) or `wait_for_drop` (live) → `ingest` → `score` → `monitor` → `decide` → `route_forecaster`
  → `refit` → `forecast` → `route_policy` → `retrain` → `order` → `report`.
  - Each step is `.venv/bin/python -m src.ops.cli step <name> …` in its own process. Airflow's venv never
    imports the project.
  - The two routes read `decision.json`. A refit or retrain that was decided against shows as *skipped*.
- **Schedule:** one run per Monday from the environment's `start` to its `end`; the logical date is the
  as-of week.
  - Unpausing a sim DAG replays it (catchup), one week at a time.
  - A failed week stops the next one (`depends_on_past`). Fix the cause and clear the failed task to resume.
- **Live:** each run waits for its drop (a sensor in reschedule mode, which holds no worker slot while
  waiting) until `python -m src.ops.cli drops release --env live --week K` publishes it. Drops are written
  aside and renamed, so a waiting run never reads half of one.
- **Resources:** `refit` and `retrain` take the 1-slot `heavy` pool, shared by all DAGs. At most 2 tasks run
  at a time.
- **Re-running:** clearing a task re-runs that step from the same inputs. `ingest` refuses a week when a
  later week of the environment is done, because the registry already holds that later state; replaying from
  an earlier week needs a fresh environment.
  - In Airflow 3.3.2, `airflow tasks clear -t` matches task ids by **substring**, despite the flag's name
    `--task-regex`: use `-t score -d` (with downstream), not `-t '^score$'`.
- **Tests:** `.venv-airflow/bin/python -m pytest airflow/tests` covers the import, the graph, pools, dates and
  routing. It runs in CI as the `dags` job.

**Checked on `airflow-dev`** (2023-10-02 → 10-23, 2026-11-03):

- **The run:** the 4 weeks ran in 4.6 min, about 45 s per week plus a CatBoost fit in weeks 1 and 3.
  `cli run` takes about 10 s per week, so the extra time is process start-up and scheduler hops.
  - Decisions: bootstrap, keep, refit (age 2), keep. Forecaster v2 is champion with v1 previous; the
    policy is the anchor v1.
- **Reproducibility:** each week's decisions, forecasts, orders and stock states are byte-identical to
  `cli run` on a scratch store.
  - The check caught one skew, now fixed: a step run from the CLI wrote the `week` column in a different
    timestamp unit (equal values, different bytes).
  - Clearing `score` and its downstream tasks twice gave the same 10 files both times.
- **Memory** (physical footprint):
  - Airflow standalone: 1.04 GB (scheduler, API server, DAG processor, triggerer and 2 executor workers).
  - The MLflow server: 0.40 GB, plus 1.16 GB when its job runner is on. The runner only serves GenAI
    features, so `make mlflow-server` turns it off (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`).

## First dev run (2023-10-02 → 2023-12-11, 11 weeks, 2 min)

- **Decisions:** bootstrap, then age refits on 10-30 and 11-27, both promoted.
- **Shadow costs:** the previous forecaster's shadow costs equal live for two weeks after each promotion, then
  diverge. On 11-20, v2 cost 2.038 vs v1's 2.232 €/SKU.
- **Probation:** v2's evidence after two weeks was −0.07% with a CI of [−2.7%, +2.1%], so it was kept.
- **Calibration notes for Phase 6:**
  - a bias alarm needs at least warm-up + persistence = 6 weeks after a promotion;
  - two evidence weeks give a CI about ±2.4 points wide, much wider than the 1% rollback margin.

## First real-size RL retrain (2026-11-03, isolated store)

The environment ran 2022-10-03 → 2023-07-03 with a cheap CatBoost recipe refit every 13 weeks and the policy
rules of `configs/ops.yaml`.

- **When:** the ledger first covered 4 burn-in + 26 training + 8 validation weeks on 2023-06-26, 37 weeks
  after the first forecast week, and the first challenger trained then.
- **Cost:** 3 seeds of 146–178 s each, and 531 s for the whole cycle (budget: 20 min). The other 38 cycles took
  about 9 s each.
- **Outcome:** the gate was Smooth only. The mean validation cost was −0.83% vs the anchor (seeds +0.23,
  −2.13 and −0.60%), with units within ±2%.
  - It was rejected only because 2 of 3 seeds (0.667) fell short of `min_seed_share: 0.67`. That was a
    rounding bug, now fixed.
  - Under the fixed rule it would have been promoted, and the fallback check would have watched it from
    there.
