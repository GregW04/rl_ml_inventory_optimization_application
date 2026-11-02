"""
MLflow pyfunc of the VN2 forecaster, logged *from code*: MLflow rebuilds the model from this file
and the project's ``src`` package (logged with it), not from a pickled object.

* input: weekly sales history up to the as-of week, one row per SKU and week
  (``store``, ``product``, ``date``, ``sales``, ``in_stock``);
* param ``as_of``: ISO date of the last week with sales (empty: the latest date in the input);
* output: one row per SKU for the decision week ``as_of + 1 week``, one column per measure.

Hierarchy attributes come from the master table stored with the model. Load it in a process
without torch: LightGBM and torch segfault together on macOS.
"""

from __future__ import annotations

import json

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel


class VN2Forecaster(PythonModel):
    def load_context(self, context):
        from src.forecasting.forecaster import load_forecaster

        self.forecaster = load_forecaster(context.artifacts["forecaster"])
        self.master = pd.read_csv(context.artifacts["master"])
        with open(context.artifacts["meta"], encoding="utf-8") as f:
            self.meta = json.load(f)

    def predict(self, context, model_input, params=None):
        from src.forecasting.recipe import forecast, panel_from_history

        history = model_input.copy()
        history["date"] = pd.to_datetime(history["date"])
        as_of = pd.Timestamp((params or {}).get("as_of") or history["date"].max())
        return forecast(self.forecaster, panel_from_history(history, self.master), as_of, self.meta["config"])


mlflow.models.set_model(VN2Forecaster())
