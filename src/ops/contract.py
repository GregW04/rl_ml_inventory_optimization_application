"""
Data contract of a weekly drop, checked before anything learns from it or orders with it.

Errors stop the cycle: the step fails and the loop doesn't advance until the drop is fixed
(an orchestrator shows the failed task; the next week waits). Warnings are logged with the
cycle. A drop must be the previous drop plus exactly one new week: history is never restated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.ops.drops import ID_COLS


class ContractError(RuntimeError):
    """The drop breaks the data contract; nothing downstream may use it."""


@dataclass
class ContractReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self)


def _weeks(wide: pd.DataFrame, name: str, report: ContractReport) -> pd.DatetimeIndex | None:
    if list(wide.columns[:2]) != ID_COLS:
        report.errors.append(f"{name}: the first two columns must be {ID_COLS}.")
        return None
    try:
        weeks = pd.DatetimeIndex(pd.to_datetime(wide.columns[2:]))
    except (ValueError, TypeError):
        report.errors.append(f"{name}: week columns must be dates.")
        return None
    if (weeks.weekday != 0).any():
        report.errors.append(f"{name}: week columns must be Mondays.")
    if len(weeks) > 1 and (np.diff(weeks.values) != np.timedelta64(7, "D")).any():
        report.errors.append(f"{name}: weeks must be consecutive and in order.")
    if wide.duplicated(subset=ID_COLS).any():
        report.errors.append(f"{name}: duplicated Store/Product rows.")
    return weeks


def check_drop(
    sales: pd.DataFrame,
    instock: pd.DataFrame,
    master: pd.DataFrame,
    as_of: pd.Timestamp,
    previous: pd.DataFrame | None = None,
) -> ContractReport:
    report = ContractReport()
    as_of = pd.Timestamp(as_of)
    weeks = _weeks(sales, "sales", report)
    in_weeks = _weeks(instock, "in-stock", report)
    if weeks is None or in_weeks is None:
        return report
    if weeks[-1] != as_of:
        report.errors.append(f"sales: the last week is {weeks[-1].date()}, the drop is for {as_of.date()}.")
    if in_weeks[-1] < as_of:
        report.errors.append(f"in-stock: the flags end at {in_weeks[-1].date()}, before {as_of.date()}.")

    skus = set(map(tuple, sales[ID_COLS].to_numpy()))
    known = set(map(tuple, master[["store", "product"]].to_numpy()))
    if skus != known:
        report.errors.append(f"sales: SKU set differs from the master ({len(skus - known)} new, {len(known - skus)} missing).")
    if set(map(tuple, instock[ID_COLS].to_numpy())) != skus:
        report.errors.append("in-stock: SKU set differs from the sales.")
    values = sales.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    if values.isna().to_numpy().any():
        report.errors.append(f"sales: {int(values.isna().to_numpy().sum())} missing or non-numeric values.")
    elif (values.to_numpy() < 0).any():
        report.errors.append("sales: negative values.")
    elif not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        report.errors.append("sales: non-integer values.")
    flags = instock.iloc[:, 2:]
    if not flags.isin([True, False]).to_numpy().all():
        report.errors.append("in-stock: flags must be True or False.")

    if previous is not None:
        prev_cols = list(previous.columns[2:])
        missing = [c for c in prev_cols if c not in sales.columns]
        if missing:
            report.errors.append(f"sales: {len(missing)} weeks of the previous drop are missing.")
        else:
            merged = previous.merge(sales[ID_COLS + prev_cols], on=ID_COLS, suffixes=("_old", "_new"))
            old = merged[[f"{c}_old" for c in prev_cols]].to_numpy()
            new = merged[[f"{c}_new" for c in prev_cols]].to_numpy()
            changed = int((old != new).sum())
            if changed:
                report.errors.append(f"sales: history restated in {changed} cells.")
        n_new = len(weeks) - len(prev_cols)
        if n_new < 1:
            report.errors.append("sales: no new week since the previous drop.")
        elif n_new > 1:
            report.warnings.append(f"sales: {n_new} new weeks since the previous drop (catch-up).")

    if report.ok and len(weeks) > 8:
        latest, trailing = values.iloc[:, -1], values.iloc[:, -9:-1]
        ratio = float(latest.sum()) / max(float(trailing.sum(axis=0).mean()), 1e-9)
        if not 0.3 <= ratio <= 3.0:
            report.warnings.append(f"sales: this week's total is {ratio:.2f}x the trailing 8-week mean.")
        spikes = int(((latest > 10 * values.iloc[:, :-1].max(axis=1)) & (latest > 10)).sum())
        if spikes:
            report.warnings.append(f"sales: {spikes} SKUs sold over 10x their previous maximum.")
    return report
