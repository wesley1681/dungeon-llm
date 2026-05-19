"""Behaviour-cloning training loop.

Treats action prediction as three independent classification tasks
(skill / entity / grid). Cross-entropy loss summed across the three heads.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .model import CombatPolicyNet


def bc_loss_step(net: CombatPolicyNet, obs: dict,
                 actions: torch.Tensor,
                 optim: torch.optim.Optimizer) -> torch.Tensor:
    """One gradient step. Returns scalar loss (detached)."""
    net.train()
    skill_logits, entity_logits, grid_logits = net(obs)
    loss = (
        F.cross_entropy(skill_logits, actions[:, 0])
        + F.cross_entropy(entity_logits, actions[:, 1])
        + F.cross_entropy(grid_logits, actions[:, 2])
    )
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.detach()


def train_bc(dataset: dict, *, epochs: int = 10, batch_size: int = 64,
             lr: float = 3e-4, device: str = "cuda",
             save_path: str | None = None) -> dict:
    """Run BC over a stacked dataset (from bc_collect.collect_bc_dataset).

    Returns a history dict with per-epoch mean loss.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    actions = torch.from_numpy(dataset["actions"]).long()
    n = actions.shape[0]
    idx = np.arange(n)

    net = CombatPolicyNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        np.random.shuffle(idx)
        ep_losses = []
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            obs_b = {k: torch.from_numpy(v[sel]).to(device)
                     for k, v in dataset["obs"].items()}
            act_b = actions[sel].to(device)
            loss = bc_loss_step(net, obs_b, act_b, optim)
            ep_losses.append(float(loss))
        losses.append(float(np.mean(ep_losses)))

    if save_path:
        torch.save(net.state_dict(), save_path)
    return {"losses": losses, "model": net}
