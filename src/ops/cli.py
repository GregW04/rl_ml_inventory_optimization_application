"""
Command line of the ops loop (what the Airflow tasks call):

    python -m src.ops.cli drops make --env dev                  # sim drops for the env's weeks
    python -m src.ops.cli drops release --env live --week 3     # publish VN2 file Week3
    python -m src.ops.cli step forecast --env dev --as-of 2023-10-09
    python -m src.ops.cli run --env dev [--start …] [--end …] [--force]   # all steps, week by week

The tracking store comes from ``MLFLOW_TRACKING_URI`` (``make mlflow-server``) or the config.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.ops import drops
from src.ops.settings import env_settings, load_ops_config, weeks
from src.ops.steps import STEPS, Cycle, run_env, run_step
from src.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Weekly ops loop (drops, steps, runs)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ops-config", default="configs/ops.yaml")
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
    r.add_argument("--force", action="store_true", help="re-run weeks that already finished")

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
    else:
        for week, status in zip(weeks(args.start or env_settings(ops, args.env).start, args.end or env_settings(ops, args.env).end),
                                run_env(config, ops, args.env, args.start, args.end, args.force)):
            print(f"{week.date()}  {status}")


if __name__ == "__main__":
    main()
