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
                 skill_sample_w: torch.Tensor | None = None,
                 ) -> tuple[torch.Tensor, dict]:
    """One gradient step. Returns (loss, per-head accuracy dict).

    Hierarchical conditional loss:
      end_head:    all pairs                          (binary CE)
      skill_head:  act pairs                          (multiclass CE)
      entity_head: pairs where target_type ∈ {SINGLE_*, MULTI_*}
      grid_head:   pairs where target_type ∈ {POINT, LINE, CONE}

    ``skill_sample_w`` (optional, shape [B]) is a PER-SAMPLE weight on the skill
    cross-entropy. It must be per-sample, NOT per-slot: ``available_skills``
    orders slots [END, MOVE, weapons, abilities, DODGE, HIDE], so the same slot
    index is a DIFFERENT skill across archetypes (slot 4 = action_surge for
    champion but trip_attack for battle_master). A per-slot weight therefore
    pools unrelated skills and fails to boost the rare combo skills
    (action_surge / maneuver / smite) by their true identity — which is exactly
    why combo classes under-learn them. The caller computes the weight from each
    sample's chosen skill FEATURE VECTOR (identity-stable across archetypes).
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
    if act_mask.any():
        _sl = F.cross_entropy(skill_logits[act_mask], actions[act_mask, 0],
                              reduction="none")
        if skill_sample_w is not None:
            w = skill_sample_w[act_mask]
            skill_loss = (_sl * w).sum() / w.sum().clamp(min=1e-6)
        else:
            skill_loss = _sl.mean()
    else:
        skill_loss = zero
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
             use_skill_weight: bool = True,
             verbose: bool = True) -> dict:
    """Run BC over a stacked dataset (from bc_collect.collect_bc_dataset).

    ``use_skill_weight``: when False, every act-sample gets weight 1 (no
    inverse-frequency boost). The boost helps rare combo skills be learned, but
    it also over-boosts a rare-but-cheap skill like action_surge, making the
    clone fire it ~1.5x more than the expert (champion over-surge); disabling it
    is the ablation that tests whether that over-surge is what caps champion.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    actions = torch.from_numpy(dataset["actions"]).long()
    target_types = torch.from_numpy(dataset["target_types"]).long()
    n = actions.shape[0]
    idx = np.arange(n)

    # Inverse-frequency weight on the skill_head, sqrt-softened, keyed by skill
    # IDENTITY (the chosen skill's feature vector) rather than slot index.
    # Slot index is archetype-dependent (slot 4 = action_surge for champion but
    # trip_attack for battle_master), so a per-slot weight pools unrelated skills
    # and never correctly boosts the rare combo skills the combo classes need.
    # Keying by the feature vector groups identical skills across archetypes.
    # sqrt compresses pure 1/count (~50x) to ~7x so rare skills are boosted but
    # not spammed into degenerate "always vow / always hide" policies.
    from collections import Counter
    skills_arr = dataset["obs"]["skills"]                 # [n, N_SKILL_SLOTS, F]
    act_pos = np.where(dataset["actions"][:, 0] > 0)[0]
    chosen_feats = skills_arr[act_pos, dataset["actions"][act_pos, 0]]  # [n_act, F]
    keys = [k.tobytes() for k in np.round(chosen_feats, 3)]
    cnt = Counter(keys)
    n_classes = max(1, len(cnt))
    n_act_total = max(1, len(keys))
    id_w = {k: float(np.sqrt(n_act_total / (n_classes * c))) for k, c in cnt.items()}
    sample_w = np.zeros(n, dtype=np.float32)
    for j, i in enumerate(act_pos):
        sample_w[i] = id_w[keys[j]] if use_skill_weight else 1.0
    if verbose:
        top = sorted(cnt.items(), key=lambda x: -x[1])[:6]
        print(f"   identity-weight: {len(cnt)} distinct skills, "
              f"top counts {[c for _, c in top]}, "
              f"weight range [{min(id_w.values()):.2f}, {max(id_w.values()):.2f}]")

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
            sw_b = torch.from_numpy(sample_w[sel]).to(device)
            loss, acc = bc_loss_step(net, obs_b, act_b, tt_b, optim,
                                       skill_sample_w=sw_b)
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
