"""
The decision engine: a pure function from this week's signals to one action, in priority order.
It reads and writes nothing, so every branch is unit-tested (tests/test_ops_units.py).

1. no champion yet → **bootstrap**
2. the champion is on probation and its realised € is worse than the shadow on the previous
   forecaster by more than the margin, with the whole CI above 0 → **rollback**
3. an operator's challenge won with a new recipe (``src/ops/retune.py``) → **retune**
4. strategy ``never`` → **keep**
5. a due refit: age ≥ ``max_age_weeks`` (strategies ``fixed``, ``drift_backstop``), or a drift
   alarm (``drift``, ``drift_backstop``) after the cooldown; a severe bias alarm skips the
   cooldown. Drift alarms only count in classes carrying ≥ ``min_cost_share`` of the cost
   → **refit** (same recipe)
6. otherwise → **keep**. Input-side diagnostics alone never retrain.

Under every strategy the cycle also records the drift alarm (``drift_alert``) next to the decision.
Only the strategies above act on it; under ``fixed`` and ``never`` it is an alert and nothing more.

The policy layer decides separately (``decide_policy``):

1. no policy yet → **bootstrap** (the anchor, version 1)
2. an RL policy whose realised € is worse than the anchor's shadow by more than the margin over
   the latest evidence weeks, with the CI above 0 → **fallback** to the anchor
3. RL disabled, or the forecast ledger too short to train on → **keep**
4. never trained, the last challenger ``retrain_every_weeks`` ago, or (RL serving) more than
   ``gate_churn_limit`` of the SKUs crossed its gate since it was trained → **retrain**
5. otherwise → **keep**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.ops.settings import Rules

ACTIONS = ("bootstrap", "rollback", "retune", "refit", "keep")
POLICY_ACTIONS = ("bootstrap", "fallback", "retrain", "keep")


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
    for cls, c in signals.get("coverage", {}).items() if rules.coverage.trigger else ():
        if cls in big and c.get("streak", 0) >= rules.coverage.persistence_weeks:
            lo, hi = rules.coverage.band
            reasons.append(f"coverage {cls}: {c['value']:.3f} outside [{lo}, {hi}] for {c['streak']} weeks")
    return reasons, severe


def drift_alert(signals: dict, rules: Rules) -> dict:
    """The drift alarm as every cycle records it, whatever the strategy: an alert for the operator, never an action."""
    reasons, severe = drift_alarms(signals, rules)
    return {"active": bool(reasons), "severe": severe, "reasons": reasons}


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

    r = signals.get("pending_recipe")
    if r:
        return Decision("retune", [
            f"operator: recipe {r['recipe']} beat the champion by {r['delta_pct']:+.2f}% on the holdout to "
            f"{r['as_of']} (CI {r['ci_lo']:+.2f}…{r['ci_hi']:+.2f}%, challenge run {r['run_id']})"])

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


def decide_policy(signals: dict, rules: Rules) -> Decision:
    pr = rules.policy
    policy = signals.get("policy")
    if not policy:
        return Decision("bootstrap", ["no policy yet: the anchor"])
    if policy["kind"] == "ppo":
        e = signals.get("policy_vs_anchor")
        if e and e.get("evidence_weeks", 0) >= pr.min_evidence_weeks and e["delta_pct"] > pr.fallback_margin_pct and e["ci_lo"] > 0:
            return Decision("fallback", [
                f"RL costs {e['delta_pct']:+.2f}% vs the anchor over {e['evidence_weeks']} weeks "
                f"(CI {e['ci_lo']:+.2f}…{e['ci_hi']:+.2f}%)"])
    if not pr.enabled:
        return Decision("keep", ["RL disabled"])
    if not signals.get("rl_eligible"):
        return Decision("keep", ["the forecast ledger is too short to train RL"])
    since = policy.get("weeks_since_rl_attempt")
    if since is None:
        return Decision("retrain", ["first RL challenger"])
    if since >= pr.retrain_every_weeks:
        return Decision("retrain", [f"last RL challenger {since} weeks ago (every {pr.retrain_every_weeks})"])
    churn = signals.get("gate_churn")
    if policy["kind"] == "ppo" and churn is not None and churn > pr.gate_churn_limit:
        return Decision("retrain", [f"{churn:.1%} of SKUs crossed the gate (limit {pr.gate_churn_limit:.0%})"])
    return Decision("keep", ["no trigger"])

