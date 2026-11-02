"""
Torch policies for PPO and GRPO.

The actor is a Gaussian over one action in [-1, 1] (the same action ``VN2Env`` takes from
continuous agents). For the residual method its output layer starts at zero, so an untrained
policy orders exactly the anchor. ``TorchPolicy`` wraps a trained actor with the
``predict(obs, deterministic)`` interface :func:`src.rl.rollout.rollout` expects, so
validation, test scoring and the order page treat PPO and GRPO like any other agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, int(h)), nn.Tanh()]
        d = int(h)
    return nn.Sequential(*layers), d


class Actor(nn.Module):
    """Gaussian policy: state-dependent mean, one learned log standard deviation."""

    def __init__(self, obs_dim: int, hidden=(64, 64), init_mean: float = 0.0, init_log_std: float = -2.3):
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        self.body, last = _mlp(obs_dim, self.hidden)
        self.head = nn.Linear(last, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, float(init_mean))
        self.log_std = nn.Parameter(torch.full((1,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs)).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))


class Critic(nn.Module):
    """State value for PPO; sees the observation plus the order round (one-hot)."""

    def __init__(self, in_dim: int, hidden=(64, 64)):
        super().__init__()
        self.body, last = _mlp(in_dim, tuple(int(h) for h in hidden))
        self.head = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class TorchPolicy:
    """A trained actor behind the ``predict`` interface of Stable-Baselines3 agents."""

    def __init__(self, actor: Actor, meta: dict):
        self.actor = actor.eval()
        self.meta = dict(meta)

    def predict(self, obs, deterministic: bool = True):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            a = self.actor(x) if deterministic else self.actor.dist(x).sample()
        return np.clip(a.numpy().astype(np.float32), -1.0, 1.0), None

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.actor.state_dict(), "meta": self.meta}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> TorchPolicy:
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
        meta = blob["meta"]
        actor = Actor(int(meta["obs_dim"]), tuple(meta["hidden"]))
        actor.load_state_dict(blob["state_dict"])
        return cls(actor, meta)
