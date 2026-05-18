"""Combat decision policies — what every combatant calls to pick its sub-action.

A CombatPolicy returns a structured action dict (the same shape execute_action
consumes) directly. No LLM is invoked in this layer. The LLM's only combat role
is post-hoc narration once the action has resolved.

Implementations included here:
  - EndTurnPolicy   trivial stub; ends the turn immediately
  - HeuristicCombatPolicy  rule-based baseline used as the default until a
                           trained RL policy plugs in. Also doubles as a
                           scripted opponent for RL training environments.

The RL controller will land as a separate subclass that wraps a trained model.
"""
from __future__ import annotations
from dataclasses import dataclass

from .character import Character


@dataclass
class CombatDecision:
    action: dict | None = None   # execute_action-shape dict, or None to skip
    ended: bool = False          # turn ends after this sub-action
    fled: bool = False           # actor leaves combat entirely


class CombatPolicy:
    """Abstract policy. Given the actor's observation, return a CombatDecision."""

    def decide(self, actor_id: str, actor: Character, world_state,
               resources: dict, round_num: int) -> CombatDecision:
        raise NotImplementedError


class EndTurnPolicy(CombatPolicy):
    """Always ends the turn. Used as a placeholder for human-controlled PCs
    until a structured-action UI is wired up; once that exists, swap in a
    HumanInputPolicy that reads structured choices from the event queue."""

    def decide(self, actor_id, actor, world_state, resources, round_num):
        return CombatDecision(ended=True)


class HeuristicCombatPolicy(CombatPolicy):
    """Greedy melee baseline.

    Picks the nearest valid target and:
      - attacks if in weapon reach (and action available)
      - otherwise moves toward them (if movement available)
      - otherwise ends the turn
    Does not consider spells, dodging, or cover yet. Good enough as the default
    NPC opponent and as a scripted baseline to compare RL training runs against.
    """

    def decide(self, actor_id, actor, world_state, resources, round_num):
        ws = world_state
        is_party = ws.is_party_ally(actor_id)
        room = ws.dungeon_map.current_room if ws.dungeon_map else None
        room_ids = set(room.npc_ids) if room else set(ws.characters.keys())

        candidates: list[tuple[str, Character]] = []
        for cid, c in ws.characters.items():
            if cid == actor_id or not c.is_alive():
                continue
            other_party = ws.is_party_ally(cid)
            if is_party:
                if c.is_npc and c.attitude == 0 and cid in room_ids:
                    candidates.append((cid, c))
            else:
                if other_party:
                    candidates.append((cid, c))

        if not candidates:
            return CombatDecision(ended=True)

        target_id, target = min(
            candidates, key=lambda kv: actor.position.distance_to(kv[1].position)
        )

        weapon = actor.get_weapon() if actor.weapons else None
        if weapon is None:
            return CombatDecision(ended=True)

        d = actor.position.distance_to(target.position)
        reach = weapon.range_normal or 1.5

        if d <= reach + 1e-6 and resources.get("action", 0) > 0:
            return CombatDecision(action={
                "type":     "ATTACK",
                "attacker": actor_id,
                "target":   target_id,
                "weapon":   weapon.name,
                "consumes": ["action"],
            })

        if resources.get("movement", 0.0) > 1e-6:
            return CombatDecision(action={
                "type":        "MOVE",
                "character":   actor_id,
                "target":      target_id,
                "description": "靠近",
                "consumes":    ["movement"],
            })

        return CombatDecision(ended=True)
