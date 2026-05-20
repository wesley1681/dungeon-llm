"""Behaviour-cloning training loop.

Hierarchical 4-head action factorisation:
  - end_head:    trains on ALL pairs (binary: end vs act)
  - skill_head:  trains on ACT pairs (multiclass over real skills)
  - entity_head: trains on pairs where the chosen skill targets a creature
                 (SINGLE_ENEMY/ALLY, MULTI_ENEMY/ALLY)
  - grid_head:   trains on pairs where the chosen skill targets a position
                 (POINT, LINE, CONE)

Per-head conditional loss eliminates the marginal-vs-joint pollution: each
head only sees samples where its output is meaningful for the chosen skill.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .model import CombatPolicyNet
from ..engine.skill import TargetType


_ENTITY_TARGET_TYPES = (
    int(TargetType.SINGLE_ENEMY), int(TargetType.SINGLE_ALLY),
    int(TargetType.MULTI_ENEMY),  int(TargetType.MULTI_ALLY),
)
_GRID_TARGET_TYPES = (
    int(TargetType.POINT), int(TargetType.LINE), int(TargetType.CONE),
)


def bc_loss_step(net: CombatPolicyNet, obs: dict,
                 actions: torch.Tensor,
                 target_types: torch.Tensor,
                 optim: torch.optim.Optimizer,
                 skill_weight: torch.Tensor | None = None,
                 ) -> tuple[torch.Tensor, dict]:
    """One gradient step. Returns (loss, per-head accuracy dict).

    Hierarchical conditional loss:
      end_head:    all pairs                          (binary CE)
      skill_head:  act pairs                          (multiclass CE)
      entity_head: pairs where target_type ∈ {SINGLE_*, MULTI_*}
      grid_head:   pairs where target_type ∈ {POINT, LINE, CONE}

    ``skill_weight`` (optional, shape [N_SKILL_SLOTS]) re-weights the skill
    cross-entropy by class — used to counter dataset imbalance where rare
    abilities (vow_of_enmity, rage, sacred_weapon — fired ~once per fight)
    get drowned by the ~80% weapon-attack samples.
    """
    net.train()
    end_logit, skill_logits, entity_logits, grid_logits = net(obs)
    # entity_logits: [B, N_SKILL, N_ENTITY], grid_logits: [B, N_SKILL, N_GRID*N_GRID]

    is_end = (actions[:, 0] == 0)
    end_target = is_end.float()
    end_loss = F.binary_cross_entropy_with_logits(end_logit, end_target)

    act_mask = ~is_end
    entity_mask = torch.zeros_like(is_end)
    grid_mask   = torch.zeros_like(is_end)
    for tt in _ENTITY_TARGET_TYPES:
        entity_mask = entity_mask | (target_types == tt)
    for tt in _GRID_TARGET_TYPES:
        grid_mask = grid_mask | (target_types == tt)

    # Gather the per-skill heads using the *target* skill index (teacher
    # forcing). At inference the caller will gather using argmax skill.
    B = end_logit.shape[0]
    skill_target = actions[:, 0].clamp(min=0)
    batch_idx = torch.arange(B, device=end_logit.device)
    chosen_entity_logits = entity_logits[batch_idx, skill_target]   # [B, N_ENTITY]
    chosen_grid_logits   = grid_logits[batch_idx, skill_target]     # [B, N_GRID*N_GRID]

    zero = torch.zeros((), device=end_logit.device)
    skill_loss  = (F.cross_entropy(skill_logits[act_mask],  actions[act_mask, 0],
                                    weight=skill_weight)
                   if act_mask.any() else zero)
    entity_loss = (F.cross_entropy(chosen_entity_logits[entity_mask],
                                   actions[entity_mask, 1])
                   if entity_mask.any() else zero)
    grid_loss   = (F.cross_entropy(chosen_grid_logits[grid_mask],
                                   actions[grid_mask, 2])
                   if grid_mask.any() else zero)

    loss = end_loss + skill_loss + entity_loss + grid_loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        end_pred = (torch.sigmoid(end_logit) > 0.5)
        def _acc(logits, targets, mask):
            if not mask.any():
                return float("nan")
            return (logits.argmax(-1) == targets)[mask].float().mean().item()
        acc = {
            "end":    (end_pred == is_end).float().mean().item(),
            "skill":  _acc(skill_logits,        actions[:, 0], act_mask),
            "entity": _acc(chosen_entity_logits, actions[:, 1], entity_mask),
            "grid":   _acc(chosen_grid_logits,   actions[:, 2], grid_mask),
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
    target_types = torch.from_numpy(dataset["target_types"]).long()
    n = actions.shape[0]
    idx = np.arange(n)

    # Inverse-frequency class weight for the skill_head, sqrt-softened.
    # Pure inverse-frequency (1/count) gives the rarest skills ~50x the
    # weight of common ones, which causes degenerate "always hide" /
    # "always vow" policies. Taking sqrt compresses the dynamic range to
    # ~7x — rare skills still boosted, but not enough to spam them.
    # Skills never seen in the dataset get weight 0 (no gradient, no NaN).
    from .obs import N_SKILL_SLOTS
    act_actions = actions[actions[:, 0] > 0, 0]
    counts = torch.bincount(act_actions, minlength=N_SKILL_SLOTS).float()
    n_active_classes = (counts > 0).sum().clamp(min=1)
    n_act = counts.sum().clamp(min=1)
    raw_w = n_act / (n_active_classes * counts.clamp(min=1))
    skill_weight = torch.where(
        counts > 0, raw_w.sqrt(), torch.zeros_like(counts)
    ).to(device)
    if verbose:
        top = sorted(enumerate(counts.tolist()), key=lambda x: -x[1])[:6]
        print(f"   class-weight (inverse-freq): top counts {top}")
        print(f"   resulting weights (top→bottom): "
              + ", ".join(f"idx{i}:{skill_weight[i]:.2f}" for i, _ in top))

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
            tt_b  = target_types[sel].to(device)
            loss, acc = bc_loss_step(net, obs_b, act_b, tt_b, optim,
                                       skill_weight=skill_weight)
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
