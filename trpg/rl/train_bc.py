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
                 optim: torch.optim.Optimizer) -> tuple[torch.Tensor, dict]:
    """One gradient step. Returns (loss, per-head accuracy dict).

    Two-head decomposition:
      - end_head: binary CE over all pairs (target = action[:,0] == 0)
      - skill_head / entity_head / grid_head: multiclass CE over act-only
        pairs (action[:,0] != 0). End pairs don't carry meaningful
        target/coord, so excluding them keeps those heads clean.
    """
    net.train()
    end_logit, skill_logits, entity_logits, grid_logits = net(obs)

    is_end = (actions[:, 0] == 0)
    end_target = is_end.float()
    end_loss = F.binary_cross_entropy_with_logits(end_logit, end_target)

    act_mask = ~is_end
    if act_mask.any():
        skill_loss  = F.cross_entropy(skill_logits[act_mask],  actions[act_mask, 0])
        entity_loss = F.cross_entropy(entity_logits[act_mask], actions[act_mask, 1])
        grid_loss   = F.cross_entropy(grid_logits[act_mask],   actions[act_mask, 2])
    else:
        zero = torch.zeros((), device=end_logit.device)
        skill_loss = entity_loss = grid_loss = zero

    loss = end_loss + skill_loss + entity_loss + grid_loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        end_pred = (torch.sigmoid(end_logit) > 0.5)
        acc = {
            "end":    (end_pred == is_end).float().mean().item(),
            "skill":  ((skill_logits.argmax(-1)  == actions[:, 0])[act_mask]
                       .float().mean().item() if act_mask.any() else float("nan")),
            "entity": ((entity_logits.argmax(-1) == actions[:, 1])[act_mask]
                       .float().mean().item() if act_mask.any() else float("nan")),
            "grid":   ((grid_logits.argmax(-1)   == actions[:, 2])[act_mask]
                       .float().mean().item() if act_mask.any() else float("nan")),
        }
    return loss.detach(), acc


def train_bc(dataset: dict, *, epochs: int = 10, batch_size: int = 64,
             lr: float = 3e-4, device: str = "cuda",
             save_path: str | None = None,
             verbose: bool = True) -> dict:
    """Run BC over a stacked dataset (from bc_collect.collect_bc_dataset).

    Returns a history dict with per-epoch mean loss and per-head accuracy.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    actions = torch.from_numpy(dataset["actions"]).long()
    n = actions.shape[0]
    idx = np.arange(n)

    net = CombatPolicyNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    losses = []
    accs_end, accs_skill, accs_entity, accs_grid = [], [], [], []
    for epoch in range(epochs):
        np.random.shuffle(idx)
        ep_losses = []
        ep_acc_end, ep_acc_skill, ep_acc_entity, ep_acc_grid = [], [], [], []
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            obs_b = {k: torch.from_numpy(v[sel]).to(device)
                     for k, v in dataset["obs"].items()}
            act_b = actions[sel].to(device)
            loss, acc = bc_loss_step(net, obs_b, act_b, optim)
            ep_losses.append(float(loss))
            ep_acc_end.append(acc["end"])
            if acc["skill"] == acc["skill"]:  # nan check
                ep_acc_skill.append(acc["skill"])
                ep_acc_entity.append(acc["entity"])
                ep_acc_grid.append(acc["grid"])
        losses.append(float(np.mean(ep_losses)))
        accs_end.append(float(np.mean(ep_acc_end)))
        accs_skill.append(float(np.mean(ep_acc_skill)) if ep_acc_skill else float("nan"))
        accs_entity.append(float(np.mean(ep_acc_entity)) if ep_acc_entity else float("nan"))
        accs_grid.append(float(np.mean(ep_acc_grid)) if ep_acc_grid else float("nan"))
        if verbose:
            print(f"   epoch {epoch+1:3d}/{epochs}  loss={losses[-1]:.3f}  "
                  f"acc end={accs_end[-1]:.3f} skill={accs_skill[-1]:.3f} "
                  f"entity={accs_entity[-1]:.3f} grid={accs_grid[-1]:.3f}")

    if save_path:
        torch.save(net.state_dict(), save_path)
    return {
        "losses":      losses,
        "acc_end":     accs_end,
        "acc_skill":   accs_skill,
        "acc_entity":  accs_entity,
        "acc_grid":    accs_grid,
        "model":       net,
    }
