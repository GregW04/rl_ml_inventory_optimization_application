"""
The decision engine: a pure function from this week's signals to one action, in priority order.
It reads and writes nothing, so every branch is unit-tested (tests/test_ops_decide.py).

1. no champion yet → **bootstrap**
2. the champion is on probation and its realised € is worse than the shadow on the previous
   forecaster by more than the margin, with the whole CI above 0 → **rollback**
3. strategy ``never`` → **keep**
4. a due refit: age ≥ ``max_age_weeks`` (strategies ``fixed``, ``drift_backstop``), or a drift
   alarm (``drift``, ``drift_backstop``) after the cooldown; a severe bias alarm skips the
   cooldown. Drift alarms only count in classes carrying ≥ ``min_cost_share`` of the cost
   → **refit** (same recipe)
5. otherwise → **keep**. Input-side diagnostics alone never retrain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.ops.settings import Rules

ACTIONS = ("bootstrap", "rollback", "refit", "keep")


@dataclass
class Decision:
    action: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def drift_alarms(signals: dict, rules: Rules) -> tuple[list[str], bool]:
    """Reasons from bias and coverage alarms in classes that carry enough of the cost, and severity."""
    share = signals.get("cost_share", {})
    big = {c for c, s in share.items() if s >= rules.bias.min_cost_share}
    reasons, severe = [], False
    for cls, b in signals.get("bias", {}).items():
        if cls in big and b.get("streak", 0) >= rules.bias.persistence_weeks:
            reasons.append(f"bias {cls}: tracking signal {b['ts']:+.2f} past ±{rules.bias.limit} for {b['streak']} weeks")
            severe = severe or bool(b.get("severe"))
    for cls, c in signals.get("coverage", {}).items():
        if cls in big and c.get("streak", 0) >= rules.coverage.persistence_weeks:
            lo, hi = rules.coverage.band
            reasons.append(f"coverage {cls}: {c['value']:.3f} outside [{lo}, {hi}] for {c['streak']} weeks")
    return reasons, severe


def decide(signals: dict, rules: Rules) -> Decision:
    champion = signals.get("champion")
    if not champion:
        return Decision("bootstrap", ["no champion yet"])

    p = signals.get("probation")
    if p and p.get("evidence_weeks", 0) >= rules.probation.min_evidence_weeks:
        if p["delta_pct"] > rules.probation.rollback_margin_pct and p["ci_lo"] > 0:
            return Decision("rollback", [
                f"probation: realised € {p['delta_pct']:+.2f}% vs the previous forecaster over "
                f"{p['evidence_weeks']} weeks (CI {p['ci_lo']:+.2f}…{p['ci_hi']:+.2f}%)"])

    if rules.strategy == "never":
        return Decision("keep", ["strategy never"])

    reasons = []
    age = champion["age_weeks"]
    if rules.strategy in ("fixed", "drift_backstop") and age >= rules.max_age_weeks:
        reasons.append(f"age {age} weeks ≥ {rules.max_age_weeks}")
    if rules.strategy in ("drift", "drift_backstop"):
        alarms, severe = drift_alarms(signals, rules)
        if alarms and age < rules.cooldown_weeks and not severe and not reasons:
            return Decision("keep", [f"drift alarms wait for the cooldown ({age} < {rules.cooldown_weeks} weeks)", *alarms])
        reasons.extend(alarms)
    return Decision("refit", reasons) if reasons else Decision("keep", ["no trigger"])
