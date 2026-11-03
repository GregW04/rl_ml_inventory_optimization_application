"""The served policy (numpy, torch-free) orders exactly what the anchor and the training environment order."""

from __future__ import annotations

import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import torch

from src.ops import ledger
from src.rl.policy import Actor, TorchPolicy
from src.rl.spec import ANCHOR_FEATURES, CLASS_FEATURES, INVENTORY_FEATURES, PolicySpec
from src.rl.vn2_env import VN2Env
from src.serving.policy import actor_arrays, log_policy
from src.simulation.anchor import LostSalesAnchor
from src.utils.config import load_config
from tests.test_env import ANCHOR, DATES, MEASURES, RULES, _episode

STATE = [*INVENTORY_FEATURES, *MEASURES, *ANCHOR_FEATURES, "zero_share", "pos_cv", *CLASS_FEATURES]


def _served(kind: str, example: pd.DataFrame, **kw):
    with mlflow.start_run():
        info = log_policy(kind, ANCHOR, RULES, example.head(3), **kw)
    return mlflow.pyfunc.load_model(info.model_uri)


def test_served_anchor_equals_the_anchor_on_every_sku():
    state = ledger.initial_state_vn2(load_config()["data"]["initial_state_file"])
    rng = np.random.default_rng(0)
    fc = state[["store", "product"]].assign(mean_h0=rng.gamma(2, 2, len(state)), mean_h1=rng.gamma(2, 2, len(state)),
                                            q83_h2=rng.gamma(2, 3, len(state)) + 1)
    frame = state[["store", "product", "on_hand", "transit_1", "transit_2"]].merge(fc)
    frame = frame.assign(zero_share=0.2, pos_cv=0.5, dp_sbc_class=2)
    got = _served("anchor", frame).predict(frame)
    want = ledger.anchor_orders(state, fc, LostSalesAnchor(ANCHOR, RULES))
    assert len(got) == 599 and got["order"].tolist() == want.tolist() and not got["rl"].any()


def test_served_ppo_orders_what_the_training_environment_orders(tmp_path):
    torch.manual_seed(0)
    spec = PolicySpec("ledger:test", "residual", "ppo", STATE, ANCHOR)
    actor = Actor(len(STATE), (16, 16))
    torch.nn.init.normal_(actor.head.weight, std=0.5)  # a policy that really corrects the anchor
    policy = TorchPolicy(actor, {"obs_dim": len(STATE), "hidden": [16, 16]})
    rng = np.random.default_rng(1)
    rows, want, scales = [], [], {}
    for i in range(60):
        ctx = {"zero_share": float(rng.uniform(0, 0.6)), "pos_cv": float(rng.uniform(0, 2)), "dp_sbc_class": int(rng.integers(-1, 4))}
        start = {"end_inventory": [float(rng.integers(0, 12))] + [np.nan] * 7,
                 "in_transit_1w": [float(rng.integers(0, 6))] + [np.nan] * 7,
                 "in_transit_2w": [float(rng.integers(0, 6))] + [np.nan] * 7}
        fc = {"mean_h0": float(rng.gamma(2, 3)), "mean_h1": float(rng.gamma(2, 3)), "q83_h2": float(rng.gamma(2, 4) + 1)}
        scales[(i, 2)] = float(rng.uniform(0.5, 30))
        env = VN2Env(spec, _episode().assign(store=i, **start, **ctx), {(i, 2, d): fc for d in DATES},
                     demand_scale=scales[(i, 2)], rules=RULES)
        obs, _ = env.reset()
        want.append(env.resolve_order(policy.predict(obs)[0]))
        rows.append({"store": i, "product": 2, "on_hand": env.inventory, "transit_1": env.in_transit[0],
                     "transit_2": env.in_transit[1], **fc, **ctx})
    frame = pd.DataFrame(rows)
    served = _served("ppo", frame, spec=spec, actor=actor_arrays(policy), scales=scales)
    out = served.predict(frame)
    assert out["order"].tolist() == want and out["rl"].all()
    assert (out["order"] != out["anchor_order"]).mean() > 0.5  # the actor does correct the anchor

    gated = _served("ppo", frame, spec=spec, actor=actor_arrays(policy), scales=scales, gate=("Intermittent",))
    g = gated.predict(frame)
    inside = frame["dp_sbc_class"] == 2  # Intermittent
    assert (g.loc[~inside, "order"] == g.loc[~inside, "anchor_order"]).all() and not g.loc[~inside, "rl"].any()
    assert g.loc[inside, "order"].tolist() == out.loc[inside, "order"].tolist() and g.loc[inside, "rl"].all()

    # Serving needs neither torch nor gymnasium (the model's requirements leave them out), and the
    # model runs on its own copy of src/ (the subprocess can't see the repo's).
    with mlflow.start_run():
        info = log_policy("ppo", ANCHOR, RULES, frame.head(3), spec=spec, actor=actor_arrays(policy), scales=scales)
    local = mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri, dst_path=str(tmp_path / "model"))
    frame.to_parquet(tmp_path / "frame.parquet")
    code = (f"import sys, mlflow, pandas as pd; m = mlflow.pyfunc.load_model({local!r}); "
            f"out = m.predict(pd.read_parquet({str(tmp_path / 'frame.parquet')!r})); "
            f"assert out['order'].tolist() == {want!r}; "
            "bad = {'torch', 'gymnasium'} & set(sys.modules); assert not bad, bad")
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
