"""Load VN2 wide-format CSVs and build a long weekly panel with OOS masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config


def _melt_wide_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide sales panel to long format."""
    raw_id = ["Store", "Product"]
    value_vars = [c for c in df.columns if c not in raw_id]
    long_df = df.melt(
        id_vars=raw_id,
        value_vars=value_vars,
        var_name="date",
        value_name="sales",
    )
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.rename(
        columns={"Store": "store", "Product": "product"}
    )
    long_df["sales"] = pd.to_numeric(long_df["sales"], errors="coerce")
    return long_df


def _melt_instock(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide in-stock panel to long format."""
    raw_id = ["Store", "Product"]
    value_vars = [c for c in df.columns if c not in raw_id]
    long_df = df.melt(
        id_vars=raw_id,
        value_vars=value_vars,
        var_name="date",
        value_name="in_stock",
    )
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.rename(
        columns={"Store": "store", "Product": "product"}
    )
    long_df["in_stock"] = long_df["in_stock"].map(
        {"True": True, "False": False, True: True, False: False}
    )
    return long_df


def load_master(master_path: str | Path) -> pd.DataFrame:
    """Load hierarchy master and normalize column names."""
    master = pd.read_csv(master_path)
    master = master.rename(
        columns={
            "Store": "store",
            "Product": "product",
            "ProductGroup": "product_group",
            "Division": "division",
            "Department": "department",
            "DepartmentGroup": "department_group",
            "StoreFormat": "store_format",
            "Format": "format",
        }
    )
    return master


def load_initial_state(path: str | Path) -> pd.DataFrame:
    """Load week-0 inventory snapshot."""
    state = pd.read_csv(path)
    state = state.rename(
        columns={
            "Store": "store",
            "Product": "product",
            "Start Inventory": "start_inventory",
            "Sales": "sales_w0",
            "Missed Sales": "missed_sales_w0",
            "End Inventory": "end_inventory",
            "In Transit W+1": "in_transit_1w",
            "In Transit W+2": "in_transit_2w",
            "Holding Cost": "holding_cost",
            "Shortage Cost": "shortage_cost",
        }
    )
    return state


def load_sales_panel(
    sales_path: str | Path,
    instock_path: str | Path,
    master_path: str | Path,
    mask_oos: bool = True,
) -> pd.DataFrame:
    """
    Build long weekly panel: sales, in_stock, hierarchy attributes.

    When `mask_oos=True`, sets `sales=NaN` where `in_stock=False` (censored demand).
    """
    sales_wide = pd.read_csv(sales_path)
    instock_wide = pd.read_csv(instock_path)
    master = load_master(master_path)

    sales = _melt_wide_sales(sales_wide)
    instock = _melt_instock(instock_wide)

    panel = sales.merge(instock, on=["store", "product", "date"], how="left")
    panel = panel.merge(master, on=["store", "product"], how="left")

    if mask_oos:
        panel.loc[~panel["in_stock"].fillna(True), "sales"] = np.nan

    panel = panel.sort_values(["store", "product", "date"]).reset_index(drop=True)
    return panel


def build_target_columns(
    panel: pd.DataFrame,
    horizons: list[int],
    target_col: str = "sales",
) -> pd.DataFrame:
    """
    Add direct multi-horizon targets: target_h0 = sales at t, target_h1 = sales at t+1, etc.
    """
    df = panel.copy()
    for h in horizons:
        df[f"target_h{h}"] = df.groupby(["store", "product"], observed=True)[
            target_col
        ].shift(-h)
    return df


def build_lead_time_target_column(
    panel: pd.DataFrame,
    offsets: list[int],
    target_col: str = "sales",
    name: str = "target_S",
) -> pd.DataFrame:
    """
    Add protection-period demand target: sum of sales at t+offset for each offset.

    Any NaN week in the window makes the sum NaN (same OOS masking as weekly targets).
    """
    df = panel.copy()
    g = df.groupby(["store", "product"], observed=True)[target_col]
    parts = [g.shift(-o) for o in offsets]
    df[name] = sum(parts)
    return df


def load_panel_from_config(
    config: dict | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convenience loader using project config; optional inclusive date filter."""
    cfg = config or load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["forecasting"]["model"]
    horizons = model_cfg["horizons"]

    panel = load_sales_panel(
        sales_path=data_cfg["sales_file"],
        instock_path=data_cfg["instock_file"],
        master_path=data_cfg["master_file"],
        mask_oos=True,
    )
    panel["date"] = pd.to_datetime(panel["date"])
    if start is not None:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel = build_target_columns(panel, horizons=horizons)

    ltd_cfg = model_cfg.get("lead_time_demand", {})
    if ltd_cfg.get("enabled", False):
        offsets = ltd_cfg.get("offsets", [0, 1, 2])
        panel = build_lead_time_target_column(panel, offsets=offsets)

    return panel
