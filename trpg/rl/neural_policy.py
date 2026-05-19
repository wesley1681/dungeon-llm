"""Wraps a CombatPolicyNet as a CombatPolicy for the engine.

Lets the opponent in CombatEnvV2 be driven by a (frozen) snapshot of the
policy network instead of a hand-written ArchetypePolicy — enables self-play.

The actor's own perspective is rebuilt via build_obs(ws, actor_id, ...) so the
same network can drive either side of the fight.
"""
from __future__ import annotations
import torch

from ..engine.combat_policy import CombatPolicy, CombatDecision
from ..engine.character import Character
from ..engine.world_state import WorldState

MOVE_BUDGET_M = 9.0


class NeuralCombatPolicy(CombatPolicy):
    """CombatPolicy that delegates decisions to a CombatPolicyNet checkpoint."""

    def __init__(self, net, device: str = "cpu"):
        self.net = net
        self.device = device
        # State carried across sub-actions of a single turn so we can detect
        # voluntary end (action=None) and propagate it as ended=True.

    def decide(self, actor_id: str, actor: Character, world_state: WorldState,
               resources: dict, round_num: int) -> CombatDecision:
        from .obs import build_obs
        from .model import apply_resource_mask, apply_entity_mask
        from .action import decode_action

        obs = build_obs(world_state, actor_id, resources)
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                 for k, v in obs.items()}
        with torch.no_grad():
            skill_l, entity_l, grid_l = self.net(obs_t)
        skill_l  = apply_resource_mask(skill_l, resources, world_state, actor_id)
        entity_l = apply_entity_mask(entity_l, obs_t)
        action = [int(skill_l[0].argmax()),
                  int(entity_l[0].argmax()),
                  int(grid_l.argmax())]
        action_dict = decode_action(action, world_state, actor_id)
        if action_dict is None:
            return CombatDecision(ended=True)
        return CombatDecision(action=action_dict, ended=False)
