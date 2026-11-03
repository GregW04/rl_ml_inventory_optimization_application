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

**Recipes.** An environment names a shared forecaster recipe (`recipe:` → `recipes.<name>` in
`configs/ops.yaml`). A recipe is a set of overrides of `config.yaml`'s forecasting section (features, measures,
backends, objective) plus `model_params`, the hyperparameters per ensemble member. The loop only *refits* a
recipe; it never changes it. A recipe is (re)tuned by an operator on the data known at a week:

```bash
python -m src.ops.cli tune --recipe champion --as-of 2022-10-03 --trials 20 --folds 2 --fold-weeks 8
```

- Optuna tunes each ensemble member on the last CV folds before the fit cut, `as_of − 2 weeks`, the same cut
  every refit uses.
- The run goes to the experiment `vn2-ops-recipes` with `recipe.json`, its `recipe_key` and the
  `model_params` to paste into the recipe.
- `champion` is research run 74602b75's structure: CatBoost + LightGBM + XGBoost, Tweedie mean loss,
  demand-profile features, 6 measures and the same 116 features. It is tuned at T0 = 2022-10-03 on data up to
  2022-09-19 only.
  - Only 76 weeks exist then, so 2 folds of 8 weeks fit after 52 training weeks.
  - 20 trials per member, stopping after 10 trials without gain.

**Retune: an operator's decision, checked on a holdout** (`src/ops/retune.py`). Add the new recipe to
`recipes`, tune it, then challenge the environment's champion with it:

```bash
python -m src.ops.cli tune --recipe champion-2023q4 --as-of 2023-12-04
python -m src.ops.cli challenge --env sim-final --recipe champion-2023q4 [--holdout-weeks 8] [--margin-pct 0.5]
```

- **The holdout.** The challenger is fit as cycle A's refit would have been (A = `holdout_weeks` ago). It then
  issues every cycle's forecasts since, from that week's data. Both it and the champion's as-issued forecasts
  drive the anchor from cycle A's live stock, so only the forecasts differ.
- **The test.** Their € is paired by SKU on weeks A+3 … latest (6 weeks for an 8-week holdout), with a
  bootstrap CI. The run is logged as `stage=challenge`.
- **The outcome.** A win by `margin_pct` with the CI below 0 leaves a request. The next cycle decides
  **retune**, and the refit step fits the new recipe on all the data, sanity-checks it and promotes it, with the
  usual probation. The version is tagged `trigger=retune`, `recipe`, `challenge_run_id` and
  `holdout_delta_pct`.
- **The constraint.** A new recipe has to forecast every measure the champion does, because the anchor and the
  policy read them.

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
| 3 | An operator's challenge won with a new recipe (see Retune) | **retune** |
| 4 | Strategy `never` | **keep** |
| 5 | Age ≥ `max_age_weeks` (`fixed`, `drift_backstop`), or a bias / coverage alarm for `persistence_weeks` in a class with ≥ `min_cost_share` of the cost, after the cooldown (a severe bias alarm skips it) (`drift`, `drift_backstop`) | **refit** |
| 6 | Otherwise; input diagnostics (PSI, zero share, level vs last year, class churn) never retrain on their own | **keep** |

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

Reset and re-run on 2026-11-03 with the policy layer, it reproduced every decision, forecast, order and
cost of the first run exactly.

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

## Shadow replay and calibration (Phase 6, 2026-11-03)

All runs replay 2022-10-03 → 2024-04-01 (79 cycles) with the `champion` recipe, which was tuned at T0 on data up
to 2022-09-19 only.

**The never-refit T0 model** (`sim-never`) shows the regime change the planning evidence predicted. Mean
relative error of `mean_h0`, (sales − forecast) / forecast, by quarter:

| class (cost share) | 22Q4 | 23Q1 | 23Q2 | 23Q3 | 23Q4 | 24Q1 | 24Q2 |
|---|---|---|---|---|---|---|---|
| Smooth (37%) | +3.0 | −2.6 | +3.5 | −1.4 | +6.6 | −1.6 | +3.4 |
| Intermittent (26%) | +5.7 | +9.4 | +6.2 | +2.5 | −9.6 | −15.5 | −5.4 |
| Erratic (21%) | −3.0 | +25.1 | +5.4 | +10.9 | +26.4 | +7.5 | +62.2 |
| Lumpy (16%) | +12.3 | +4.5 | +18.7 | +5.0 | +2.6 | −10.5 | −27.9 |

- **Intermittent and Lumpy** are under-forecast through the 2023 growth and over-forecast in the 2024 decline.
  Their `q83_h2` coverage then rises above nominal (0.875 and 0.943).
- **Erratic** is a small class, about 42 SKU-weeks a week, and is noisy and mostly under-forecast.

**Calibration** (`python -m src.ops.cli calibrate --env sim-never`; MLflow experiment `vn2-ops-calibration`):

- **The null.** The replay's own weekly class series, with the slow drift removed (a centred 13-week mean),
  resampled in 4-week blocks jointly across classes. The decision rule then runs on 200 null paths of
  78 weeks with the loop's own code.
- **Bias.** False refits per 26 weeks:

  | limit | persistence 1 | 2 | 3 |
  |---|---|---|---|
  | 0.70 | 2.07 | 1.11 | 0.73 |
  | 0.74 | 1.56 | 0.89 | 0.50 |
  | 0.80 | 0.92 | 0.48 | 0.26 |
  | 0.85 | 0.51 | 0.26 | 0.15 |

  The budget is ≤ 1 per 26 weeks, and the most sensitive setting within it, 0.74 / 2, is kept. On the real
  series it first fires on 2022-11-07, five weeks after T0, on a Smooth over-forecast (tracking signal
  −0.93). 0.80 / 2 would first fire on 2023-04-03.
- **Coverage becomes a diagnostic** (`coverage.trigger: false`). No band meets the budget: ±0.03 … ±0.08
  around 0.833 or around the recipe's own 0.820 gives 1.34–3.39 false refits per 26 weeks.
  - The 4-week coverage swings 1.5–2.1× more than a binomial: design effects 2.1–4.4, from weekly shocks the
    SKUs share.
  - The classes sit at different levels: Erratic 0.778, Smooth 0.797, Lumpy 0.818, Intermittent 0.838.
- **The class biases are partly structural.** A fresh refit keeps them: Lumpy +13.6% (v1) and +10.6% (v2), and
  Erratic +26% (v2) in `sim-fixed13`. Trigg's signal flags any persistent bias, so the drift rule refits again
  and again on these classes. The strategy comparison below shows what that is worth.
- **Probation** in `sim-fixed13`: refits gained 0.4–1.2% against the previous model in every evidence week.
  The 2023-04-03 refit was rolled back on 2023-05-01: +3.05% after one evidence week and +3.02% after two
  (CI [+0.01%, +5.85%]). It was consistently worse, but the CI only just cleared 0.

**Strategy comparison** (`python -m src.ops.cli compare …`): the four backtests have the anchor as the policy and
identical demand, so their costs pair on the same in-stock SKU-weeks. The CI is a bootstrap over 4-week blocks.

| strategy | € / SKU-week | vs fixed13 | 95% CI | refits | rollbacks |
|---|---|---|---|---|---|
| never | 1.2977 | +1.54% | +0.41 … +2.95 | 0 | 0 |
| **fixed13** | **1.2780** | — | — | **6** | 1 |
| drift | 1.2795 | +0.12% | −0.40 … +0.61 | 9 | 0 |
| drift_backstop | 1.2784 | +0.03% | −0.45 … +0.48 | 9 | 0 |

- **The rule, fixed before the results:** the lowest €, unless its edge over a cheaper strategy has a CI that
  includes 0. **fixed13** wins on both counts: it beats the only cheaper strategy (never) clearly, and the drift
  strategies tie it with 50% more refits.
- **Freshness again.** The gain over never grows with the model's age, from 0% in 2022Q4 to −4.9% in 2023Q4.
  It matches the research result: refits every 13 weeks gave −1.49% on `rl_train`.
- **What the drift alarms did.** They caught the real shifts, for example 2023-10-16 (Intermittent −0.99,
  Lumpy −0.87, the start of the decline). But refitting on them didn't beat a steady 13-week rhythm.
- **What stays on.** The monitors keep running as logged signals (`bias_ts_*` and `coverage_q83_*` on the
  timeline run); only the trigger is age.
- **What was deleted.** After the comparison, the model files and registered models of `sim-never`, `sim-drift`
  and `sim-drift-backstop` were removed (24 models, 7.05 GB). Their runs, metrics and ledgers remain, and the
  models refit bit for bit from the recipe and the drops. `sim-fixed13` and `sim-final` keep theirs.

**The final replay** (`sim-final`: the chosen rules with RL on, run as the Airflow DAG `vn2_sim-final`, 79 weeks):

- **Same forecaster as the backtest.** It is byte-identical to `sim-fixed13` in all 79 decisions, forecasts and
  realised costs. The ensemble refits are deterministic across processes and orchestrators, so RL is the only
  difference.
- **RL challengers.** One every 13 weeks once the ledger was long enough:

  | as of | validation-picked gate | vs anchor | seeds beating it | outcome |
  |---|---|---|---|---|
  | 2023-06-26 | Smooth | −0.40% | 2 of 3 | rejected: margin (needs −0.5%) |
  | 2023-09-25 | Smooth, Intermittent, Lumpy | −2.27% | 3 of 3 | rejected: units (Intermittent −7…−10%, Lumpy −6…−13%) |
  | 2023-12-25 | Intermittent, Lumpy | −0.74% | 3 of 3 | rejected: units (−5…−9%) |
  | 2024-03-25 | Intermittent, Lumpy | −0.81% | 3 of 3 | **promoted** (policy v5) |

  - **The units rule.** In the decline, the challengers learned to undercut a forecaster that over-forecasts
    Intermittent and Lumpy. The units rule (within ±2% of the anchor in every seed) refused that trend bet
    until a challenger won without it.
  - **The promoted policy.** v5 decides the orders of 447–450 of the 599 SKUs, its gate. Its first orders
    arrive after the replay ends, so the live run is its first realised test, against the anchor's shadow.
- **Freeze.** `frozen: true` and `frozen_rules: 03f2617e8ceb` (the version of the rules `sim-final` ran). The
  `live` environment runs only these rules, from a clean tree.

**Disk.** The backtests filled the disk. The tuned ensemble's `forecaster.joblib` was 394 MB, about 40 versions
were made, and each model was stored twice: in `mlartifacts/` and in the download cache.

- Forecasters are now saved with zlib (`SAVE_COMPRESSION = 3`): 28 MB, and loading is faster (1.2 s instead of
  5.4 s).
- While the disk was full, Airflow's scheduler died: standalone couldn't write its log. `make airflow-stop` and
  `make airflow` restarted it; the queued runs resumed, after clearing the tasks that had failed.

## The ops test: the live VN2 run (Phase 7, 2026-11-03)

**Setup.** Run once, under the frozen rules (tag `ops-rules-v1`, rules `03f2617e8ceb`), from a clean tree, as the
`vn2_live` DAG.

- **Data.** The nine organizer files were released one at a time, each after the previous week had finished.
  Checked beforehand: each file only appends one week to an unchanged history.
- **Start.** `cli carry-over --from sim-final --to live`:
  - the forecaster fit on 2024-02-05;
  - RL policy v5, gated on Intermittent + Lumpy;
  - the anchor as the fallback, and as the exact shadow from the official initial state.

**What happened.** 9 cycles, 24 min.

- RL decided the round-1 orders of 455 of the 599 SKUs.
- The age refit on 2024-05-06 was promoted. Its probation evidence was −0.43% against the old forecaster
  (CI −2.03 … +1.07), so it was kept.
- The RL fallback check at the end: −1.28% against the anchor's shadow (CI −1.99 … −0.59), so RL kept serving.

**Score** (`cli score --env live`, run `19e79a67`, scored once):

| official window, weeks 3–8, 599 SKUs | € | € / SKU-window |
|---|---|---|
| live: RL v5 and the refit forecaster | **3654.2** | **6.101** |
| the anchor's shadow on the same forecasts | 3698.0 | 6.174 |
| live vs anchor (paired, CI over SKUs) | **−1.18%** (−1.77 … −0.62) | |
| research's locked test (run `b9950526`): RL vs anchor | 3582.8 vs 3618.2 (−0.98%) | 5.981 vs 6.040 |

- **By class:** Intermittent −2.85% and Lumpy −2.27% (the gate), Smooth +0.10% and Erratic −0.07%. Units were
  −1.07% against the anchor's.
- **RL's gain over the anchor holds on the real weeks,** at a similar size to research: −1.18% against −0.98%.
  Here it comes from a system that trained, validated, promoted and monitored the policy itself on data before
  the window, and served it without torch.
- **Both ops numbers are about 2% above research's** (the anchor: 3698.0 against 3618.2). The ops forecaster was
  tuned only on the 76 weeks up to 2022-09-19, while research's 74602b75 was tuned with CV data up to
  2023-07-31. That gap is the price of an honest go-live.
- **Disclosure.** Research's test lock had already scored these weeks once (run `b9950526`). The ops rules were
  set on the shadow replay (up to 2024-04-01), then frozen and tagged before the live run. Nothing was changed
  after it.

**Seasonally adjusted level diagnostic** (`diag_level_change_13w_pct_<class>`, logged only, never a trigger).
For each class, the weekly sales of its current SKUs, minus the seasonal component of an MSTL decomposition
(yearly period, robust), as the mean of the last 4 weeks against the 4 weeks ending 13 weeks earlier.

- **It needs two full years** of history, so it starts in 2023-08.
- **Why not the decomposition's trend:** it is smoothed over more than a year. On synthetic data it showed a
  +30% shift from 10 weeks ago as +0.9%.
- **Read it qualitatively.** With only 2–3 yearly cycles, the seasonal estimate is weak around holidays. On
  2024-02-26 it gives Intermittent +44%, which is implausible.
- **Moving holidays distort year-on-year numbers too.** Easter fell on 9 April in 2023 and on 31 March in 2024,
  so part of the "−23.5% against last year by the end of March 2024" in the planning evidence is the holiday
  moving, not demand.
