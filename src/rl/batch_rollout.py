"""Rollouts on :class:`BatchVN2Env` shared by PPO and GRPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.rl.batch_env import BatchVN2Env
from src.rl.policy import Actor


@dataclass
class Batch:
    """One batch of episodes: ``R`` decisions per rollout, ``M`` rollouts."""

    obs: torch.Tensor      # [R, M, obs_dim] float32
    actions: torch.Tensor  # [R, M] sampled (unclipped) actions
    logp: torch.Tensor     # [R, M] log-probability under the sampling policy
    rewards: torch.Tensor  # [R, M] negative euro cost credited to each decision
    cost: torch.Tensor     # [M] scored cost of the rollout (weeks 3-8)
    shadow: torch.Tensor   # [M] the anchor's scored cost on the same window


def collect(
    env: BatchVN2Env,
    actor: Actor,
    idx: torch.Tensor,
    gen: torch.Generator | None,
    *,
    stochastic: bool = True,
    paired: bool = False,
) -> Batch:
    """
    Roll ``actor`` through the windows ``idx`` (repeats allowed).

    Each decision is credited with the cost of the week it is placed in; weeks after the
    last order round are credited to the last decision, so every decision's reward-to-go
    is the remaining scored cost. ``paired`` subtracts the shadow anchor's weekly cost.
    """
    rounds = env.rounds
    obs = env.reset(idx)
    m = len(idx)
    obs_l, act_l, logp_l = [], [], []
    rewards = torch.zeros(rounds, m)
    cost = torch.zeros(m, dtype=torch.float64)
    shadow = torch.zeros(m, dtype=torch.float64)
    for t in range(env.weeks):
        if t < rounds:
            with torch.no_grad():
                dist = actor.dist(obs)
                if stochastic:
                    a = dist.mean + dist.stddev * torch.randn(dist.mean.shape, generator=gen)
                else:
                    a = dist.mean
                lp = dist.log_prob(a)
            obs_l.append(obs)
            act_l.append(a)
            logp_l.append(lp)
            nxt, c, sh = env.step(a)
        else:
            nxt, c, sh = env.step(None)
        r = -(c - sh) if paired else -c
        rewards[min(t, rounds - 1)] += r.to(torch.float32)
        cost += c
        shadow += sh
        obs = nxt
    return Batch(torch.stack(obs_l), torch.stack(act_l), torch.stack(logp_l), rewards, cost, shadow)


def evaluate_actor(env: BatchVN2Env, actor: Actor, idx: torch.Tensor) -> tuple[float, float]:
    """(policy cost, anchor cost) summed over ``idx`` with deterministic actions."""
    b = collect(env, actor, idx, None, stochastic=False)
    return float(b.cost.sum()), float(b.shadow.sum())


def raw_init_mean(env: BatchVN2Env, sample: int = 20_000, seed: int = 0) -> float:
    """
    Starting action for the raw method: the median anchor order at the first round, as
    the action that orders it (``order = (a + 1) / 2 · cap · scale``). Untrained raw
    policies therefore start with typical-sized orders instead of half the cap.
    """
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(env.n, generator=gen)[: min(sample, env.n)]
    env.reset(idx)
    cap = env.spec.order_cap_mult * env.scale[idx]
    f = torch.clamp(env.dec[0] / cap, 0.0, 1.0)
    return float(torch.clamp(torch.median(2.0 * f - 1.0), -0.95, 0.95))
