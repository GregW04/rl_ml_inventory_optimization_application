"""
Tune a forecaster recipe on the data known at an as-of week (``python -m src.ops.cli tune``): the
bootstrap at T0, or an operator's retune later (the loop itself only refits a frozen recipe).

The recipe's structure comes from ``recipes.<name>`` in ``configs/ops.yaml``; Optuna tunes each
ensemble member (``src/forecasting/tuning.py``) on the last CV folds before the fit cut,
``as_of − 2 weeks``, the same cut every refit uses. The run is logged in ``vn2-ops-recipes`` with
the resulting recipe (``recipe.json``) and the ``model_params`` to paste into the recipe.
"""

from __future__ import annotations

import time

import mlflow
import pandas as pd
import yaml

from src.forecasting.forecaster import ensemble_settings
from src.forecasting.measures import primary_measure, primary_metric_name
from src.forecasting.model import build_feature_matrix
from src.forecasting.preprocessing import load_panel_from_config
from src.forecasting.recipe import Recipe, fit_end, panel_digest
from src.forecasting.tuning import tune_model
from src.ops.settings import OpsConfig, recipe_config
from src.utils import tracking

EXPERIMENT = "vn2-ops-recipes"


def tune_recipe(config: dict, ops: OpsConfig, name: str, as_of, n_trials: int, n_folds: int, fold_weeks: int,
                min_train_weeks: int = 52) -> dict:
    """Tune recipe ``name`` at ``as_of``; return its ``model_params`` (per member for an ensemble)."""
    as_of = pd.Timestamp(as_of)
    cfg = recipe_config(config, ops, name)
    cfg["forecasting"]["validation"].update(n_folds=n_folds, fold_size_weeks=fold_weeks,
                                            min_train_size_weeks=min_train_weeks)
    cfg["forecasting"]["tuning"].update(enabled=True, n_trials=n_trials, n_cv_folds=n_folds)
    panel = load_panel_from_config(cfg, end=as_of)
    features = build_feature_matrix(panel, cfg)
    data = features[features["date"] <= fit_end(as_of)]
    ensemble, members = ensemble_settings(cfg)
    members = members if ensemble else [cfg["forecasting"]["model"]["name"]]
    metric = primary_metric_name(primary_measure(cfg))

    tracking.init_tracking(cfg)
    day = str(as_of.date())
    with mlflow.start_run(run_name=f"tune-{name}-{day}", experiment_id=tracking.get_or_create_experiment(EXPERIMENT),
                          tags={"stage": "tune", "recipe": name, "as_of": day}):
        tracking.log_provenance(cfg)
        mlflow.log_params({"recipe": name, "as_of": day, "fit_end": str(fit_end(as_of).date()),
                           "members": ",".join(members), "n_trials": n_trials, "n_folds": n_folds,
                           "fold_weeks": fold_weeks, "min_train_weeks": min_train_weeks, "metric": metric,
                           "data_digest": panel_digest(panel)})
        params = {}
        for member in members:
            start = time.time()
            result = tune_model(member, data, cfg)
            params[member] = result["best_params"]
            mlflow.log_metrics({f"tuned_{metric}_{member}": result["best_value"],
                                f"trials_{member}": result["n_trials_completed"],
                                f"minutes_{member}": (time.time() - start) / 60})
        model_params = params if ensemble else params[members[0]]
        recipe = Recipe.from_config(cfg, model_params)
        mlflow.set_tag("recipe_key", recipe.key())
        mlflow.log_dict(recipe.to_dict(), "recipe.json")
        mlflow.log_text(yaml.safe_dump({"model_params": model_params}, sort_keys=False), "model_params.yaml")
    return model_params
