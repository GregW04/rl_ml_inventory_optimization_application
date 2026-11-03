"""
MLflow pyfunc of the VN2 ordering policy, logged *from code*. Torch-free on purpose: a PPO actor
is served as numpy weights, so the order step can share a process with any forecaster (LightGBM
and torch segfault together on macOS). It computes the observation and the order with the same
functions the training environment uses (``src.rl.observation``).

* input: one row per SKU for the decision week: ``store``, ``product``, the stock (``on_hand``,
  ``transit_1`` arriving this week, ``transit_2`` next week), the forecasts the anchor and the
  policy read, and ``zero_share``, ``pos_cv``, ``dp_sbc_class`` (context of the week);
* output: ``order`` and its parts: ``anchor_order``, ``adjustment`` (order − anchor) and ``rl``
  (whether the learned policy decided: false for the anchor and outside the policy's gate).

Kinds: ``anchor`` (the lost-sales anchor) and ``ppo`` (an actor with an optional gate of demand
classes it acts on; the anchor orders for the others).
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel


class VN2Policy(PythonModel):
    def load_context(self, context):
        from src.rl.spec import PolicySpec
        from src.simulation.anchor import AnchorSpec
        from src.simulation.vn2 import VN2Rules

        with open(context.artifacts["policy"], encoding="utf-8") as f:
            self.meta = json.load(f)
        self.kind = self.meta["kind"]
        self.rules = VN2Rules(**self.meta["rules"])
        a = self.meta["anchor"]
        self.anchor_spec = AnchorSpec(a["h0"], a["h1"], a["target"], tuple(a["target_ids"]), float(a["tau"]))
        if self.kind == "ppo":
            self.spec = PolicySpec.from_dict(self.meta["spec"])
            w = np.load(context.artifacts["actor"])
            self.layers = [(w[f"w{i}"], w[f"b{i}"]) for i in range(int(self.meta["n_layers"]))]
            self.head = (w["wh"], w["bh"])
            scales = pd.read_csv(context.artifacts["scales"])
            self.scales = {(int(r.store), int(r.product)): float(r.scale) for r in scales.itertuples(index=False)}
            self.gate = None if self.meta.get("gate") is None else set(self.meta["gate"])

    def actions(self, obs: np.ndarray) -> np.ndarray:
        """The actor's mean action per row: Linear → tanh layers, a linear head, clipped to [-1, 1]."""
        h = obs.astype(np.float32)
        for w, b in self.layers:
            h = np.tanh(h @ w.T + b)
        return np.clip((h @ self.head[0].T + self.head[1]).reshape(-1), -1.0, 1.0)

    def predict(self, context, model_input, params=None):
        from src.forecasting.features.demand_profile import demand_class_name
        from src.rl.observation import effective_scale, observation, order_for_action, state_values
        from src.simulation.anchor import LostSalesAnchor

        df = model_input.reset_index(drop=True)
        needed = sorted(self.anchor_spec.inputs())
        if df[needed].isna().to_numpy().any():
            raise ValueError(f"Forecasts {needed} are missing for some SKUs.")
        anchor = LostSalesAnchor(self.anchor_spec, self.rules)
        fc_cols = [c for c in df.columns if c not in ("store", "product", "on_hand", "transit_1", "transit_2",
                                                      "zero_share", "pos_cv", "dp_sbc_class")]
        rows = df.to_dict("records")
        decisions = [anchor.decide(r["on_hand"], r["transit_1"], r["transit_2"], {c: r[c] for c in fc_cols}) for r in rows]
        anchor_orders = np.array([d.order for d in decisions], dtype=float)
        orders, rl = anchor_orders.copy(), np.zeros(len(df), dtype=bool)
        if self.kind == "ppo" and len(df):
            scales = [effective_scale(self.scales.get((int(r["store"]), int(r["product"])), 1.0)) for r in rows]
            obs = np.stack([
                observation(self.spec, state_values(r["on_hand"], r["transit_1"], r["transit_2"], d, r["zero_share"],
                                                    r["pos_cv"], r["dp_sbc_class"]), {c: r[c] for c in fc_cols}, s)
                for r, d, s in zip(rows, decisions, scales)
            ])
            for i, (r, a, s) in enumerate(zip(rows, self.actions(obs), scales)):
                if self.gate is None or demand_class_name(r["dp_sbc_class"]) in self.gate:
                    orders[i], rl[i] = order_for_action(self.spec, anchor_orders[i], s, a), True
        return pd.DataFrame({"store": df["store"], "product": df["product"], "order": orders,
                             "anchor_order": anchor_orders, "adjustment": orders - anchor_orders, "rl": rl})


mlflow.models.set_model(VN2Policy())
