"""Log an ordering policy as an MLflow model: a pyfunc from code (the anchor, or a numpy PPO actor)."""

from __future__ import annotations

import json
import tempfile
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from src.simulation.anchor import AnchorSpec
from src.simulation.vn2 import VN2Rules

ROOT = Path(__file__).resolve().parents[2]
MODEL_CODE = Path(__file__).with_name("policy_model.py")
SERVING_PACKAGES = ("mlflow", "pandas", "numpy", "PyYAML")
# Features of a VN2 8-week window (order round, error since the window started) that don't exist
# week to week; a served policy can't be given them.
WINDOW_FEATURES = ("round_frac", "forecast_error")
INPUT_COLUMNS = ["store", "product", "on_hand", "transit_1", "transit_2", "zero_share", "pos_cv", "dp_sbc_class"]


def actor_arrays(policy) -> dict[str, np.ndarray]:
    """A trained ``TorchPolicy``'s actor as numpy arrays (``w0``/``b0``…, ``wh``/``bh``)."""
    sd = {k: v.detach().cpu().numpy() for k, v in policy.actor.state_dict().items()}
    linear = sorted({int(k.split(".")[1]) for k in sd if k.startswith("body.") and k.endswith(".weight")})
    out = {}
    for i, j in enumerate(linear):
        out[f"w{i}"], out[f"b{i}"] = sd[f"body.{j}.weight"], sd[f"body.{j}.bias"]
    out["wh"], out["bh"] = sd["head.weight"], sd["head.bias"]
    return out


def log_policy(
    kind: str,
    anchor: AnchorSpec,
    rules: VN2Rules,
    example: pd.DataFrame,
    name: str = "policy",
    spec=None,
    actor: dict[str, np.ndarray] | None = None,
    scales: dict | None = None,
    gate: tuple[str, ...] | None = None,
    params: dict | None = None,
) -> ModelInfo:
    """
    Log a policy in the active run as a pyfunc LoggedModel. ``example`` is a decision frame (the
    model's input) for a few SKUs; the signature is inferred from it.
    """
    if kind == "ppo" and any(f in spec.state for f in WINDOW_FEATURES):
        raise ValueError(f"A served policy can't read the VN2-window features {WINDOW_FEATURES}.")
    meta = {"kind": kind, "rules": asdict(rules), "anchor": {**asdict(anchor), "target_ids": list(anchor.target_ids)},
            "gate": None if gate is None else list(gate)}
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column")
        tmp = Path(tmp)
        artifacts = {"policy": str(tmp / "policy.json")}
        if kind == "ppo":
            meta.update(spec=spec.to_dict(), n_layers=sum(1 for k in actor if k.startswith("w") and k != "wh"))
            np.savez(tmp / "actor.npz", **actor)
            pd.DataFrame([{"store": s, "product": p, "scale": v} for (s, p), v in scales.items()]).to_csv(
                tmp / "scales.csv", index=False)
            artifacts.update(actor=str(tmp / "actor.npz"), scales=str(tmp / "scales.csv"))
        (tmp / "policy.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        # Output types only (importing the model code would run its set_model outside a load).
        output = pd.DataFrame({"store": example["store"], "product": example["product"], "order": 0.0,
                               "anchor_order": 0.0, "adjustment": 0.0, "rl": False})
        signature = infer_signature(example, output)
        return mlflow.pyfunc.log_model(
            name=name,
            python_model=str(MODEL_CODE),
            artifacts=artifacts,
            code_paths=[str(ROOT / "src")],
            signature=signature,
            input_example=example,
            pip_requirements=[f"{p}=={version(p)}" for p in SERVING_PACKAGES],
            model_type="vn2-policy",
            params={"kind": kind, **(params or {})},
        )
