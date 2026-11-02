"""
PPO on batched replay of VN2 windows.

Lessons from the HDPO prototype built in: the residual policy starts exactly at the
anchor, every update replays thousands of windows, the reward is paired with an anchor run
on the same demand (removing the window's own noise), costs stay in euros with only
batch-level advantage scaling, actions are continuous with small initial noise, and the
final weights are kept. Rewards are undiscounted by default: the objective is the window
cost, and weeks after the last order are credited to the last decision.

With ``early_stop`` on, ``iterations`` is the budget: training stops once the training sample
(``eval_idx``) stops improving (:class:`src.rl.batch_rollout.EarlyStop`). Validation is never
looked at, and the weights at the stop are the final weights.
"""

from __future__ import annotations

import copy
import itertools
from typing import Callable

import torch
from torch.distributions import kl_divergence
from torch.nn.utils import clip_grad_norm_

from src.rl.batch_env import BatchVN2Env, EpisodeArrays
from src.rl.batch_rollout import EarlyStop, collect, evaluate_actor, raw_init_mean
from src.rl.policy import Actor, Critic, TorchPolicy
from src.rl.spec import PolicySpec
from src.simulation.vn2 import VN2Rules

DEFAULTS = {
    "iterations": 300, "batch_windows": 2048, "epochs": 4, "minibatch_size": 4096,
    "learning_rate": 3e-4, "gamma": 1.0, "gae_lambda": 0.95, "clip_range": 0.2,
    "vf_coef": 0.5, "ent_coef": 0.0, "init_log_std": -2.3, "hidden": [64, 64],
    "max_grad_norm": 0.5, "paired_reward": True,
    "early_stop": False, "early_stop_patience": 400, "early_stop_min_delta": 0.2, "early_stop_every": 100,
}


def _gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    rounds = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(rounds)):
        nxt = values[t + 1] if t + 1 < rounds else torch.zeros_like(values[t])
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def _with_round(obs: torch.Tensor) -> torch.Tensor:
    """Append the order round as a one-hot (critic input): [R, M, d] -> [R, M, d + R]."""
    rounds, m = obs.shape[0], obs.shape[1]
    eye = torch.eye(rounds).unsqueeze(1).expand(rounds, m, rounds)
    return torch.cat([obs, eye], dim=2)


def train_ppo(
    spec: PolicySpec,
    arrays: EpisodeArrays,
    rules: VN2Rules,
    *,
    seed: int,
    eval_idx: torch.Tensor | None = None,
    on_iteration: Callable[[int, dict], None] | None = None,
    on_eval: Callable[[int, float, float], None] | None = None,
) -> TorchPolicy:
    p = {**DEFAULTS, **spec.agent_params}
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    env = BatchVN2Env(spec, arrays, rules)
    init_mean = 0.0 if spec.method == "residual" else raw_init_mean(env, seed=seed)
    hidden = tuple(int(h) for h in p["hidden"])
    actor = Actor(env.obs_dim, hidden, init_mean, float(p["init_log_std"]))
    # Frozen starting policy (the anchor for residual), only to log how far training moved.
    ref = copy.deepcopy(actor).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    critic = Critic(env.obs_dim + env.rounds, hidden)
    params = list(itertools.chain(actor.parameters(), critic.parameters()))
    opt = torch.optim.Adam(params, lr=float(p["learning_rate"]))
    iterations, batch = int(p["iterations"]), int(p["batch_windows"])
    clip, mb_size = float(p["clip_range"]), int(p["minibatch_size"])
    stopper = EarlyStop(int(p["early_stop_patience"]), float(p["early_stop_min_delta"])) if p["early_stop"] else None
    if stopper is not None and eval_idx is None:
        raise ValueError("Early stopping scores the training sample, so it needs eval_idx.")
    eval_every = int(p["early_stop_every"]) if stopper is not None else max(1, iterations // 10)

    def _eval(it: int) -> float | None:
        """Training-sample cost vs anchor in %, reported to ``on_eval``; None when nothing needs it."""
        if eval_idx is None or (on_eval is None and stopper is None):
            return None
        pol, anc, units = evaluate_actor(env, actor, eval_idx)
        pct = 100.0 * (pol / anc - 1.0) if anc > 0 else 0.0
        if on_eval is not None:
            on_eval(it * batch, pct, units)
        return pct

    start = _eval(0)
    if stopper is not None:
        stopper.should_stop(0, start)
    it = 0
    for it in range(1, iterations + 1):
        idx = torch.randint(0, env.n, (batch,), generator=gen)
        b = collect(env, actor, idx, gen, stochastic=True, paired=bool(p["paired_reward"]))
        crit_in = _with_round(b.obs)
        with torch.no_grad():
            values = critic(crit_in)
        adv, ret = _gae(b.rewards, values, float(p["gamma"]), float(p["gae_lambda"]))
        # Share of the return variance the critic explains before this update (SB3's definition).
        explained = 1.0 - float((ret - values).var() / ret.var()) if float(ret.var()) > 0 else float("nan")
        obs, act, logp_old = b.obs.flatten(0, 1), b.actions.flatten(), b.logp.flatten()
        crit_in, adv, ret = crit_in.flatten(0, 1), adv.flatten(), ret.flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        sums, n_mb = {"approx_kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}, 0
        for _ in range(int(p["epochs"])):
            perm = torch.randperm(len(adv), generator=gen)
            for start in range(0, len(perm), mb_size):
                mb = perm[start:start + mb_size]
                dist = actor.dist(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                surr = torch.min(ratio * adv[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb])
                v_loss = ((critic(crit_in[mb]) - ret[mb]) ** 2).mean()
                loss = -surr.mean() + float(p["vf_coef"]) * v_loss - float(p["ent_coef"]) * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(params, float(p["max_grad_norm"]))
                opt.step()
                with torch.no_grad():
                    sums["approx_kl"] += float(((ratio - 1) - (logp - logp_old[mb])).mean())
                    sums["clip_frac"] += float(((ratio - 1).abs() > clip).float().mean())
                    sums["grad_norm"] += float(grad_norm)
                n_mb += 1
        if on_iteration is not None:
            with torch.no_grad():
                kl_to_start = float(kl_divergence(actor.dist(obs), ref.dist(obs)).mean())
            on_iteration(it, {
                "train_vs_anchor_pct": float(100.0 * (b.cost.sum() / b.shadow.sum() - 1.0)) if b.shadow.sum() > 0 else 0.0,
                "train_units_vs_anchor_pct": b.units_vs_anchor_pct(),
                "policy_std": float(actor.log_std.detach().exp()),
                **{k: v / max(n_mb, 1) for k, v in sums.items()},
                "kl_to_start": kl_to_start,
                "explained_variance": explained,
            })
        if it % eval_every == 0 or it == iterations:
            value = _eval(it)
            if stopper is not None and stopper.should_stop(it, value):
                break
    meta = {"obs_dim": env.obs_dim, "hidden": list(hidden), "agent": "ppo", "method": spec.method,
            "state": list(spec.state), "iterations_run": it}
    return TorchPolicy(actor, meta)
