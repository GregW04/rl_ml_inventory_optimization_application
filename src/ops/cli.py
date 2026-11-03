"""
Command line of the ops loop (what the Airflow tasks call):

    python -m src.ops.cli drops make --env dev                  # sim drops for the env's weeks
    python -m src.ops.cli drops release --env live --week 3     # publish VN2 file Week3
    python -m src.ops.cli step forecast --env dev --as-of 2023-10-09
    python -m src.ops.cli run --env dev [--start …] [--end …] [--force]   # all steps, week by week
    python -m src.ops.cli tune --recipe champion --as-of 2022-10-03        # (re)tune a recipe
    python -m src.ops.cli challenge --env live --recipe champion-2024      # a new recipe vs the champion
    python -m src.ops.cli calibrate --env sim-never                        # drift-threshold tables
    python -m src.ops.cli compare --envs sim-never sim-fixed13 --reference sim-fixed13
    python -m src.ops.cli carry-over --from sim-final --to live            # go-live
    python -m src.ops.cli score --env live                                 # the ops test, once

The tracking store comes from ``MLFLOW_TRACKING_URI`` (``make mlflow-server``) or the config.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd
import yaml

from src.ops import calibrate, drops, retune, score
from src.ops.settings import dump, env_settings, load_ops_config, weeks
from src.ops.steps import STEPS, Cycle, carry_over, run_env, run_step
from src.ops.tune import tune_recipe
from src.utils.config import load_config

# MLflow notes that don't apply here, kept out of every task log: the drops' integer columns never
# hold NaN (the data contract), a local CSV source matches two identical source classes, and the
# pyfuncs declare their schema in the signature rather than in type hints.
QUIET = (".*Inferred schema contains integer column", ".*dataset source can be interpreted in multiple ways",
         ".*Add type hints to the `predict` method")


def main(argv: list[str] | None = None) -> None:
    for message in QUIET:
        warnings.filterwarnings("ignore", message=message)
    parser = argparse.ArgumentParser(description="Weekly ops loop (drops, steps, runs)")
    parser.add_argument("--config", help="default: the repo's configs/config.yaml (from any directory)")
    parser.add_argument("--ops-config", help="default: the repo's configs/ops.yaml")
    parser.add_argument("--project-root", help="where ops/ state and the artifact cache live (default: the repo)")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("drops", help="publish weekly data drops")
    d.add_argument("action", choices=("make", "release"))
    d.add_argument("--env", required=True)
    d.add_argument("--start")
    d.add_argument("--end")
    d.add_argument("--week", type=int, help="VN2 file number for release")

    s = sub.add_parser("step", help="run one step of one week")
    s.add_argument("name", choices=STEPS)
    s.add_argument("--env", required=True)
    s.add_argument("--as-of", required=True)

    r = sub.add_parser("run", help="run every step of every week in order")
    r.add_argument("--env", required=True)
    r.add_argument("--start")
    r.add_argument("--end")
    r.add_argument("--force", action="store_true",
                   help="re-run the latest finished week (earlier weeks need a fresh environment)")

    t = sub.add_parser("tune", help="tune a recipe of configs/ops.yaml on the data known at a week")
    t.add_argument("--recipe", required=True)
    t.add_argument("--as-of", required=True)
    t.add_argument("--trials", type=int, default=20, help="Optuna trials per backend (stops after 10 without gain)")
    t.add_argument("--folds", type=int, default=2, help="the last CV folds before the fit cut")
    t.add_argument("--fold-weeks", type=int, default=8)
    t.add_argument("--min-train-weeks", type=int, default=52)

    ch = sub.add_parser("challenge", help="score a new recipe against an environment's champion on its latest weeks")
    ch.add_argument("--env", required=True)
    ch.add_argument("--recipe", required=True)
    ch.add_argument("--holdout-weeks", type=int, default=8)
    ch.add_argument("--margin-pct", type=float, default=0.5, help="the € gain it needs, with the CI below 0")

    cal = sub.add_parser("calibrate", help="threshold calibration tables from an environment's replay")
    cal.add_argument("--env", required=True)
    cal.add_argument("--paths", type=int, default=200, help="null paths per setting")

    cmp_ = sub.add_parser("compare", help="refit strategies side by side: paired €, refits, compute")
    cmp_.add_argument("--envs", nargs="+", required=True)
    cmp_.add_argument("--reference", required=True)

    co = sub.add_parser("carry-over", help="start an environment from another's champions (live from sim-final)")
    co.add_argument("--from", dest="source", required=True)
    co.add_argument("--to", dest="target", required=True)

    sc = sub.add_parser("score", help="the ops test: official VN2 score of a live environment, once")
    sc.add_argument("--env", default="live")
    sc.add_argument("--override-lock", action="store_true", help="score again (tagged post_hoc)")

    args = parser.parse_args(argv)
    config, ops = load_config(args.config), load_ops_config(args.ops_config)
    if args.project_root:
        config["_project_root"] = args.project_root
    if args.command == "drops":
        if args.action == "release":
            print(drops.release(config, args.env, args.week))
        else:
            s_ = env_settings(ops, args.env)
            for path in drops.make(config, args.env, weeks(args.start or s_.start, args.end or s_.end)):
                print(path)
    elif args.command == "step":
        run_step(Cycle(config, ops, args.env, pd.Timestamp(args.as_of)), args.name)
    elif args.command == "score":
        print(dump(score.score_live(config, ops, args.env, args.override_lock)))
    elif args.command == "carry-over":
        for layer, lines in carry_over(config, ops, args.source, args.target).items():
            print(layer, *lines, sep="\n  ")
    elif args.command == "compare":
        print(calibrate.strategy_report(config, ops, args.envs, args.reference).round(4).to_string())
    elif args.command == "calibrate":
        for name, table in calibrate.calibration_run(config, ops, args.env, args.paths).items():
            print(f"\n{name}\n{table.to_string(index=False)}")
    elif args.command == "challenge":
        print(dump(retune.challenge(config, ops, args.env, args.recipe, args.holdout_weeks, args.margin_pct)))
    elif args.command == "tune":
        params = tune_recipe(config, ops, args.recipe, args.as_of, args.trials, args.folds, args.fold_weeks,
                             args.min_train_weeks)
        print(yaml.safe_dump({"model_params": params}, sort_keys=False))
    else:
        for week, status in zip(weeks(args.start or env_settings(ops, args.env).start, args.end or env_settings(ops, args.env).end),
                                run_env(config, ops, args.env, args.start, args.end, args.force)):
            print(f"{week.date()}  {status}")


if __name__ == "__main__":
    main()
