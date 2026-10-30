"""
GRPO (Group Relative Policy Optimization) on batched replay of VN2 windows.

Each sampled SKU-window is replayed ``group_size`` times with the stochastic policy, so
the group sees the same demand and only the orders differ. A rollout's advantage is the
group's mean cost minus its own cost, in euros, applied to all of that rollout's decisions;
there is no critic. Updates use the clipped surrogate plus a KL penalty to the frozen
initial policy, which for the residual method is the anchor.

Advantages are scaled by one batch-wide factor (``advantage_scale: batch``), so windows
keep their euro weight; dividing by each group's own spread (``group``, the textbook
variant) would weight every window equally, which hurt in earlier experiments here.
"""

from __future__ import annotations

import copy
from typing import Callable

import torch
from torch.distributions import kl_divergence
from torch.nn.utils import clip_grad_norm_

from src.rl.batch_env import BatchVN2Env, EpisodeArrays
from src.rl.batch_rollout import collect, evaluate_actor, raw_init_mean
from src.rl.policy import Actor, TorchPolicy
from src.rl.spec import PolicySpec
from src.simulation.vn2 import VN2Rules

DEFAULTS = {
    "iterations": 300, "batch_windows": 256, "group_size": 16, "epochs": 4, "minibatch_size": 4096,
    "learning_rate": 3e-4, "clip_range": 0.2, "kl_coef": 0.02, "init_log_std": -2.3,
    "hidden": [64, 64], "max_grad_norm": 0.5, "advantage_scale": "batch",
}


def train_grpo(
    spec: PolicySpec,
    arrays: EpisodeArrays,
    rules: VN2Rules,
    *,
    seed: int,
    eval_idx: torch.Tensor | None = None,
    on_iteration: Callable[[int, dict], None] | None = None,
    on_eval: Callable[[int, float], None] | None = None,
) -> TorchPolicy:
    p = {**DEFAULTS, **spec.agent_params}
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    env = BatchVN2Env(spec, arrays, rules)
    init_mean = 0.0 if spec.method == "residual" else raw_init_mean(env, seed=seed)
    hidden = tuple(int(h) for h in p["hidden"])
    actor = Actor(env.obs_dim, hidden, init_mean, float(p["init_log_std"]))
    ref = copy.deepcopy(actor).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=float(p["learning_rate"]))
    iterations, batch, group = int(p["iterations"]), int(p["batch_windows"]), int(p["group_size"])
    clip, mb_size, beta = float(p["clip_range"]), int(p["minibatch_size"]), float(p["kl_coef"])
    eval_every = max(1, iterations // 10)

    def _eval(it: int) -> None:
        if eval_idx is not None and on_eval is not None:
            pol, anc = evaluate_actor(env, actor, eval_idx)
            on_eval(it * batch * group, 100.0 * (pol / anc - 1.0) if anc > 0 else 0.0)

    _eval(0)
    for it in range(1, iterations + 1):
        idx = torch.randint(0, env.n, (batch,), generator=gen).repeat_interleave(group)
        b = collect(env, actor, idx, gen, stochastic=True, paired=False)
        costs = b.cost.view(batch, group)
        adv = (costs.mean(dim=1, keepdim=True) - costs)
        if p["advantage_scale"] == "group":
            adv = adv / (costs.std(dim=1, unbiased=False, keepdim=True) + 1e-6)
        else:
            adv = adv / (adv.std() + 1e-8)
        adv = adv.reshape(-1).to(torch.float32).unsqueeze(0).expand(env.rounds, -1)
        obs, act, logp_old, adv = b.obs.flatten(0, 1), b.actions.flatten(), b.logp.flatten(), adv.flatten()
        clip_frac, kl_ref = 0.0, 0.0
        for _ in range(int(p["epochs"])):
            perm = torch.randperm(len(adv), generator=gen)
            for start in range(0, len(perm), mb_size):
                mb = perm[start:start + mb_size]
                dist = actor.dist(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                surr = torch.min(ratio * adv[mb], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb])
                with torch.no_grad():
                    ref_dist = ref.dist(obs[mb])
                kl = kl_divergence(dist, ref_dist).mean()
                loss = -surr.mean() + beta * kl
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(actor.parameters(), float(p["max_grad_norm"]))
                opt.step()
                with torch.no_grad():
                    clip_frac = float(((ratio - 1).abs() > clip).float().mean())
                    kl_ref = float(kl)
        if on_iteration is not None:
            on_iteration(it, {
                "train_vs_anchor_pct": float(100.0 * (b.cost.sum() / b.shadow.sum() - 1.0)) if b.shadow.sum() > 0 else 0.0,
                "policy_std": float(actor.log_std.detach().exp()),
                "kl_to_start": kl_ref,
                "clip_frac": clip_frac,
            })
        if it % eval_every == 0 or it == iterations:
            _eval(it)
    meta = {"obs_dim": env.obs_dim, "hidden": list(hidden), "agent": "grpo", "method": spec.method,
            "state": list(spec.state)}
    return TorchPolicy(actor, meta)
