"""Model registry: candidate regressors and their Optuna search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ModelSpec:
    """A boosting backend: quantile, Poisson and Tweedie (count-mean) estimators, plus its search space."""

    name: str
    make_quantile_estimator: Callable[[float, dict], Any]  # (quantile, params) -> estimator
    make_poisson_estimator: Callable[[dict], Any]  # params -> count-mean estimator
    make_tweedie_estimator: Callable[[float, dict], Any]  # (variance power, params) -> count-mean estimator
    default_params: dict
    # search_space: param_name -> ("int", lo, hi)
    #                            ("float", lo, hi)
    #                            ("float_log", lo, hi)  -- log-uniform
    #                            ("categorical", [choices])
    search_space: dict


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _lgb_factory(quantile: float, params: dict) -> Any:
    import lightgbm as lgb  # type: ignore

    p = {**params, "objective": "quantile", "alpha": quantile, "verbose": -1}
    return lgb.LGBMRegressor(**p)


def _lgb_poisson_factory(params: dict) -> Any:
    import lightgbm as lgb  # type: ignore

    p = {**params, "objective": "poisson", "verbose": -1}
    return lgb.LGBMRegressor(**p)


def _lgb_tweedie_factory(power: float, params: dict) -> Any:
    import lightgbm as lgb  # type: ignore

    p = {**params, "objective": "tweedie", "tweedie_variance_power": power, "verbose": -1}
    return lgb.LGBMRegressor(**p)


def _xgb_factory(quantile: float, params: dict) -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError("xgboost>=2.0 is required. pip install xgboost") from exc

    p = {
        **params,
        "objective": "reg:quantileerror",
        "quantile_alpha": quantile,
        "verbosity": 0,
        "tree_method": params.get("tree_method", "hist"),
        "enable_categorical": True,
    }
    return xgb.XGBRegressor(**p)


def _xgb_poisson_factory(params: dict) -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError("xgboost>=2.0 is required. pip install xgboost") from exc

    p = {
        **params,
        "objective": "count:poisson",
        "verbosity": 0,
        "tree_method": params.get("tree_method", "hist"),
        "enable_categorical": True,
    }
    return xgb.XGBRegressor(**p)


def _xgb_tweedie_factory(power: float, params: dict) -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as exc:
        raise ImportError("xgboost>=2.0 is required. pip install xgboost") from exc

    p = {
        **params,
        "objective": "reg:tweedie",
        "tweedie_variance_power": power,
        "verbosity": 0,
        "tree_method": params.get("tree_method", "hist"),
        "enable_categorical": True,
    }
    return xgb.XGBRegressor(**p)


def _catboost_factory(quantile: float, params: dict) -> Any:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError("catboost is required. pip install catboost") from exc

    p = {**params, "loss_function": f"Quantile:alpha={quantile}", "verbose": 0}
    return CatBoostRegressor(**p)


def _catboost_poisson_factory(params: dict) -> Any:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError("catboost is required. pip install catboost") from exc

    p = {**params, "loss_function": "Poisson", "verbose": 0}
    return CatBoostRegressor(**p)


def _catboost_tweedie_factory(power: float, params: dict) -> Any:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError("catboost is required. pip install catboost") from exc

    # Like Poisson, CatBoost predicts Tweedie models on the demand scale (exponent) by default.
    p = {**params, "loss_function": f"Tweedie:variance_power={power}", "verbose": 0}
    return CatBoostRegressor(**p)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, ModelSpec] = {
    "lightgbm": ModelSpec(
        name="lightgbm",
        make_quantile_estimator=_lgb_factory,
        make_poisson_estimator=_lgb_poisson_factory,
        make_tweedie_estimator=_lgb_tweedie_factory,
        default_params={
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
        search_space={
            "num_leaves": ("int", 16, 128),
            "learning_rate": ("float_log", 0.01, 0.2),
            "n_estimators": ("int", 200, 1000),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_samples": ("int", 5, 100),
            "reg_alpha": ("float_log", 1e-4, 10.0),
            "reg_lambda": ("float_log", 1e-4, 10.0),
        },
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        make_quantile_estimator=_xgb_factory,
        make_poisson_estimator=_xgb_poisson_factory,
        make_tweedie_estimator=_xgb_tweedie_factory,
        default_params={
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
        search_space={
            "max_depth": ("int", 3, 10),
            "learning_rate": ("float_log", 0.01, 0.2),
            "n_estimators": ("int", 200, 1000),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_weight": ("int", 1, 20),
            "reg_alpha": ("float_log", 1e-4, 10.0),
            "reg_lambda": ("float_log", 1e-4, 10.0),
        },
    ),
    "catboost": ModelSpec(
        name="catboost",
        make_quantile_estimator=_catboost_factory,
        make_poisson_estimator=_catboost_poisson_factory,
        make_tweedie_estimator=_catboost_tweedie_factory,
        default_params={
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "rsm": 0.8,
        },
        search_space={
            "iterations": ("int", 200, 1000),
            "learning_rate": ("float_log", 0.01, 0.2),
            "depth": ("int", 3, 10),
            "l2_leaf_reg": ("float_log", 0.1, 20.0),
            "subsample": ("float", 0.5, 1.0),
            "rsm": ("float", 0.5, 1.0),
        },
    ),
}


def get_spec(model_name: str) -> ModelSpec:
    if model_name not in REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(REGISTRY)}"
        )
    return REGISTRY[model_name]


def available_candidates(names: list[str]) -> list[str]:
    """Return only those candidate names whose backend can actually be imported."""
    result = []
    for name in names:
        if name not in REGISTRY:
            continue
        try:
            REGISTRY[name].make_quantile_estimator(0.5, REGISTRY[name].default_params)
            result.append(name)
        except ImportError:
            pass
        except Exception:
            result.append(name)
    return result


def sample_params(trial: Any, model_name: str) -> dict:
    """Sample hyperparameters for `model_name` from an Optuna trial."""
    spec = get_spec(model_name)
    params: dict = {}
    for pname, entry in spec.search_space.items():
        kind = entry[0]
        if kind == "int":
            params[pname] = trial.suggest_int(pname, entry[1], entry[2])
        elif kind == "float":
            params[pname] = trial.suggest_float(pname, entry[1], entry[2])
        elif kind == "float_log":
            params[pname] = trial.suggest_float(pname, entry[1], entry[2], log=True)
        elif kind == "categorical":
            params[pname] = trial.suggest_categorical(pname, entry[1])
    return params
