"""Lost-sales order-up-to anchor for VN2, built from whichever forecast measures a run has."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.forecasting.measures import Measure, poisson_quantile
from src.simulation.vn2 import VN2Rules

# Two quantile levels this close count as the same service level (0.83 ≈ 0.8333).
_TAU_TOL = 0.005


@dataclass(frozen=True)
class AnchorSpec:
    """
    Which forecast measures feed the anchor.

    ``h0`` / ``h1`` are the demand levels used to project stock through weeks k and k+1
    with lost sales. The delivery-week target is one of:

    - ``quantile``: order up to ``target_ids[0]`` (a quantile measure of h2);
    - ``poisson``: the Poisson τ-quantile of ``target_ids[0]`` (a mean of h2).
    """

    h0: str
    h1: str
    target: str
    target_ids: tuple[str, ...]
    tau: float

    def inputs(self) -> set[str]:
        return {self.h0, self.h1, *self.target_ids}

    def describe(self) -> str:
        if self.target == "quantile":
            goal = f"{self.target_ids[0]}"
        else:
            goal = f"Poisson({self.target_ids[0]}) at {self.tau:.3f}"
        return f"project with {self.h0}, {self.h1}; order up to {goal}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_ids"] = list(self.target_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AnchorSpec:
        return cls(d["h0"], d["h1"], d["target"], tuple(d["target_ids"]), float(d["tau"]))


def _level_id(measures: list[Measure], h: int) -> str | None:
    """Demand level of a horizon: its mean, else its median, else the quantile nearest 0.5."""
    at_h = [m for m in measures if m.horizon == h]
    for m in at_h:
        if m.kind == "mean":
            return m.id
    quantiles = [m for m in at_h if m.kind == "quantile"]
    return min(quantiles, key=lambda m: abs(m.tau - 0.5)).id if quantiles else None


def resolve_anchor(measures: list[Measure], rules: VN2Rules) -> tuple[AnchorSpec | None, list[str]]:
    """
    Default anchor for a set of measures, plus warnings.

    Delivery-week target, in order of preference: a quantile at the critical ratio, a
    Poisson quantile of the mean, or else the quantile closest to the critical ratio
    (with a warning, since it is not cost-optimal).
    """
    tau = rules.critical_ratio
    h0, h1 = _level_id(measures, 0), _level_id(measures, 1)
    if h0 is None or h1 is None:
        return None, ["The anchor needs a mean or a quantile at h0 and h1."]
    at2 = {m.id: m for m in measures if m.horizon == 2}
    quantiles = [m for m in at2.values() if m.kind == "quantile"]
    exact = [m for m in quantiles if abs(m.tau - tau) <= _TAU_TOL]
    if exact:
        return AnchorSpec(h0, h1, "quantile", (exact[0].id,), exact[0].tau), []
    if "mean_h2" in at2:
        return AnchorSpec(h0, h1, "poisson", ("mean_h2",), tau), []
    if quantiles:
        q = min(quantiles, key=lambda m: abs(m.tau - tau))
        return AnchorSpec(h0, h1, "quantile", (q.id,), q.tau), [
            f"No delivery-week measure at the critical ratio {tau:.3f}; the anchor orders up "
            f"to {q.id} (τ = {q.tau:.3f}), which is not cost-optimal."
        ]
    return None, ["The anchor needs a mean or a quantile at h2."]


@dataclass
class AnchorDecision:
    order: int
    projected_stock_at_arrival: float
    projected_lost_sales_before_arrival: float


def _fc_get(fc: dict, key: str) -> float:
    v = fc.get(key, 0.0)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


class LostSalesAnchor:
    """
    Project stock through weeks k and k+1 with lost sales, then order up to the
    delivery-week target of week k+2.
    """

    def __init__(self, spec: AnchorSpec, rules: VN2Rules | None = None):
        self.spec = spec
        self.rules = rules or VN2Rules()

    def target_level(self, fc: dict) -> float:
        s = self.spec
        if s.target == "quantile":
            return max(0.0, _fc_get(fc, s.target_ids[0]))
        mean = max(0.0, _fc_get(fc, s.target_ids[0]))
        return float(poisson_quantile(mean, s.tau))

    def decide(self, on_hand: float, arrive_k: float, arrive_k1: float, fc: dict) -> AnchorDecision:
        """``arrive_k`` lands this week (k), ``arrive_k1`` next week (k+1)."""
        d0 = max(0.0, _fc_get(fc, self.spec.h0))
        d1 = max(0.0, _fc_get(fc, self.spec.h1))

        start_k = on_hand + arrive_k
        end_k = max(0.0, start_k - d0)
        lost_k = max(0.0, d0 - start_k)

        start_k1 = end_k + arrive_k1
        end_k1 = max(0.0, start_k1 - d1)
        lost_k1 = max(0.0, d1 - start_k1)

        order = max(0, int(round(self.target_level(fc) - end_k1)))
        return AnchorDecision(
            order=order,
            projected_stock_at_arrival=end_k1,
            projected_lost_sales_before_arrival=lost_k + lost_k1,
        )

    def order_quantity(self, on_hand: float, arrive_k: float, arrive_k1: float, fc: dict) -> int:
        return self.decide(on_hand, arrive_k, arrive_k1, fc).order
