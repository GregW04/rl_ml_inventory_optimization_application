"""
The data provider of the ops loop: one cumulative sales snapshot per week, in the VN2 format.

* ``make`` (sim environments) cuts the snapshot of every as-of week from the Week0 files, so a
  week publishes only what was known then. The in-stock file runs to 2024-06-03; cutting it too
  keeps future stock-outs out of the past. Sim drops never contain the VN2 test weeks.
* ``release`` (live) copies the real VN2 ``WeekK`` sales file, with the in-stock flags cut to its
  last week: one file per week, as the organizer published them.

A drop is ``sales_<as_of>.csv`` and ``instock_<as_of>.csv`` in ``ops/<env>/drops/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from src.ops.settings import env_dir

ID_COLS = ["Store", "Product"]


def drop_paths(config: dict, env: str, as_of: pd.Timestamp) -> tuple[Path, Path]:
    folder, tag = env_dir(config, env) / "drops", str(pd.Timestamp(as_of).date())
    return folder / f"sales_{tag}.csv", folder / f"instock_{tag}.csv"


def vn2_sales_file(config: dict, week: int) -> Path:
    matches = sorted(Path(config["data"]["data_dir"]).glob(f"Week{week}-*-Sales.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one data/Week{week}-*-Sales.csv, found {len(matches)}.")
    return matches[0]


def cut(wide: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The ID columns and the week columns up to ``as_of``."""
    keep = [c for c in wide.columns if c in ID_COLS or pd.Timestamp(c) <= pd.Timestamp(as_of)]
    return wide[keep]


def _write(config: dict, env: str, as_of: pd.Timestamp, sales: pd.DataFrame, instock: pd.DataFrame) -> Path:
    sales_path, instock_path = drop_paths(config, env, as_of)
    sales_path.parent.mkdir(parents=True, exist_ok=True)
    # Each file appears whole (written aside, then renamed), the in-stock flags first: whoever
    # waits for a drop (the live DAG) never reads half of one.
    for frame, path in ((cut(instock, as_of), instock_path), (sales, sales_path)):
        tmp = path.with_name(f".{path.name}.tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    return sales_path


def make(config: dict, env: str, weeks: list[pd.Timestamp]) -> list[Path]:
    """Write the drops of a sim environment for ``weeks`` from the Week0 sales and in-stock files."""
    sales = pd.read_csv(vn2_sales_file(config, 0))
    instock = pd.read_csv(config["data"]["instock_file"])
    last = pd.Timestamp(sales.columns[-1])
    late = [w for w in weeks if w > last]
    if late:
        raise ValueError(f"{late[0].date()} is after the Week0 file ({last.date()}): VN2 weeks are released, not made.")
    return [_write(config, env, w, cut(sales, w), instock) for w in weeks]


def release(config: dict, env: str, week: int) -> Path:
    """Publish VN2 file ``WeekK`` as the drop of its last week (live environment)."""
    source = vn2_sales_file(config, week)
    sales = pd.read_csv(source)
    path = _write(config, env, pd.Timestamp(sales.columns[-1]), sales, pd.read_csv(config["data"]["instock_file"]))
    shutil.copystat(source, path)
    return path
