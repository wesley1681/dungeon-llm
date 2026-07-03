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


# Self-identity channel offsets (see obs._entity_row / END_FEATURES_DIM and the
# matching constants in distill_routed). A BLIND unified student is trained with
# these columns zeroed; a self-play opponent driven by such a net MUST see the
# same blinded obs or it reads identity one-hots its weights never adapted to.
_ENT_ARCH_START = 7   # entities row 0 (self) archetype one-hot start
_END_ARCH_START = 8   # end_features archetype one-hot start


class NeuralCombatPolicy(CombatPolicy):
    """CombatPolicy that delegates decisions to a CombatPolicyNet checkpoint."""

    def __init__(self, net, device: str = "cpu", blind: bool = False):
        self.net = net
        self.device = device
        # When True, zero the actor's own archetype one-hot before forward so a
        # BLIND capability-conditioned net sees its training-time obs contract.
        self.blind = blind
        # State carried across sub-actions of a single turn so we can detect
        # voluntary end (action=None) and propagate it as ended=True.

    def decide(self, actor_id: str, actor: Character, world_state: WorldState,
               resources: dict, round_num: int) -> CombatDecision:
        from .obs import build_obs, N_ARCHETYPES
        from .model import apply_resource_mask, apply_entity_mask
        from .action import decode_action

        obs = build_obs(world_state, actor_id, resources)
        if self.blind:
            obs["entities"][0,
                _ENT_ARCH_START:_ENT_ARCH_START + N_ARCHETYPES] = 0.0
            obs["end_features"][
                _END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device)
                 for k, v in obs.items()}
        from .model import pick_action
        with torch.no_grad():
            end_l, skill_l, entity_l, grid_l = self.net(obs_t)
        skill_l  = apply_resource_mask(skill_l, resources, world_state, actor_id)
        entity_l = apply_entity_mask(entity_l, obs_t, world_state, actor_id)
        action = list(pick_action(end_l[0], skill_l[0], entity_l[0], grid_l[0],
                                  ws=world_state, agent_id=actor_id))
        action_dict = decode_action(action, world_state, actor_id)
        if action_dict is None:
            return CombatDecision(ended=True)
        return CombatDecision(action=action_dict, ended=False)
