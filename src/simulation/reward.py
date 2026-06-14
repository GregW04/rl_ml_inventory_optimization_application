"""Inventory step reward: holding and shortage costs."""

from __future__ import annotations


def compute_reward(
    leftover: float,
    shortage: float,
    holding_cost: float = 0.2,
    shortage_cost: float = 1.0,
) -> float:
    """
    R_t = -(holding_cost * leftover + shortage_cost * shortage)

    Matches plan objective: -0.2 * Leftover - 1.0 * Shortage.
    """
    return -(holding_cost * leftover + shortage_cost * shortage)
