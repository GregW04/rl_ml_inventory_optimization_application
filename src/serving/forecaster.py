"""Log a fitted forecaster as an MLflow model: a pyfunc logged from code, with its recipe and data."""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from src.forecasting.recipe import PANEL_COLUMNS, Forecaster, Recipe, forecast, panel_from_history

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = Path(__file__).with_name("forecaster_model.py")
# What loading and serving the forecaster imports (torch-free on purpose).
SERVING_PACKAGES = (
    "mlflow", "pandas", "numpy", "scipy", "pyarrow", "PyYAML", "joblib",
    "scikit-learn", "lightgbm", "xgboost", "catboost",
)
EXAMPLE_SKUS, EXAMPLE_WEEKS = 2, 60


def pip_requirements() -> list[str]:
    """The serving packages pinned to the versions of this environment (= uv.lock)."""
    return [f"{p}=={version(p)}" for p in SERVING_PACKAGES]


def serving_config(recipe: Recipe, config: dict) -> dict:
    """What the model needs at serving time: the recipe's feature settings and the VN2 costs."""
    return recipe.apply({"forecasting": {}, "vn2": config.get("vn2", {})})


def log_forecaster(
    forecaster: Forecaster,
    recipe: Recipe,
    config: dict,
    history: pd.DataFrame,
    master: pd.DataFrame,
    as_of: pd.Timestamp | str,
    data_digest: str,
    name: str = "forecaster",
) -> ModelInfo:
    """
    Log a fitted forecaster in the active run as a pyfunc LoggedModel. ``history`` (raw weekly
    rows up to ``as_of``) supplies a small input example from which the signature is inferred.
    """
    as_of = pd.Timestamp(as_of)
    cfg = serving_config(recipe, config)
    skus = history[["store", "product"]].drop_duplicates().head(EXAMPLE_SKUS)
    example = history.merge(skus, on=["store", "product"])[PANEL_COLUMNS]
    example = example[(example["date"] <= as_of) & (example["date"] > as_of - pd.Timedelta(weeks=EXAMPLE_WEEKS))]
    output = forecast(forecaster, panel_from_history(example, master), as_of, cfg)
    # Default "": the latest week in the input is the as-of week.
    signature = infer_signature(example, output, params={"as_of": ""})
    meta = {"recipe": recipe.to_dict(), "recipe_key": recipe.key(), "as_of": str(as_of.date()),
            "data_digest": data_digest, "config": cfg}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        forecaster.save(tmp / "forecaster.joblib")
        master.to_csv(tmp / "master.csv", index=False)
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return mlflow.pyfunc.log_model(
            name=name,
            python_model=str(MODEL_CODE),
            artifacts={"forecaster": str(tmp / "forecaster.joblib"), "master": str(tmp / "master.csv"),
                       "meta": str(tmp / "meta.json")},
            code_paths=[str(ROOT / "src")],
            signature=signature,
            input_example=(example, {"as_of": str(as_of.date())}),
            pip_requirements=pip_requirements(),
            model_type="vn2-forecaster",
            params={"recipe_key": recipe.key(), "model_name": recipe.model_name,
                    "as_of": str(as_of.date()), "data_digest": data_digest},
        )
