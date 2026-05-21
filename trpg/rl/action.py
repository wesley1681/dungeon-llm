"""Phase 2 RL action encode/decode.

The agent emits a triplet ``[skill_idx, entity_idx, grid_cell]``. We translate
that into an engine action dict using the skill's TargetType:

  SELF / SINGLE_*  → ignore grid_cell; pass entity slot's char_id as target
  POINT            → ignore entity_idx; decode grid_cell → (x, y) world coords
  MULTI_*          → simplified to first entity (entity_idx)

Any out-of-range or invalid action returns None (treated as end-of-turn /
no-op by the env).
"""
from __future__ import annotations
from typing import Sequence

from ..engine.world_state import WorldState
from ..engine.skill import available_skills, TargetType
from ..engine.vec2 import Vec2
from .obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, N_GRID, GRID_CELL_SIZE_M, partition_entities


ACTION_DIMS: tuple[int, int, int] = (N_SKILL_SLOTS, N_ENTITY_SLOTS, N_GRID * N_GRID)


def _entity_id_at_slot(ws: WorldState, agent_id: str, slot: int) -> str | None:
    """Reverse-lookup the canonical char_id for an entity-slot index.

    Slot ordering must match :func:`trpg.rl.obs.entities_obs`.
    """
    if slot == 0:
        return agent_id
    allies, enemies = partition_entities(ws, agent_id)

    if 1 <= slot <= 2:
        idx = slot - 1
        return allies[idx] if idx < len(allies) else None
    if 3 <= slot <= 5:
        idx = slot - 3
        return enemies[idx] if idx < len(enemies) else None
    return None


def _grid_cell_to_xy(cell: int) -> tuple[float, float]:
    """Decode a 0..N_GRID*N_GRID-1 cell index to (x, y) world coords at cell centre."""
    ix = cell // N_GRID
    iy = cell % N_GRID
    return (ix + 0.5) * GRID_CELL_SIZE_M, (iy + 0.5) * GRID_CELL_SIZE_M


def decode_action(action: Sequence[int], ws: WorldState, agent_id: str) -> dict | None:
    """Translate ``[skill_idx, entity_idx, grid_cell]`` into an engine action dict.

    Returns None if the chosen skill_idx is out of range (treated as END turn).
    """
    skill_idx, entity_idx, grid_cell = int(action[0]), int(action[1]), int(action[2])
    agent = ws.characters[agent_id]
    skills = available_skills(agent, ws)
    if skill_idx >= len(skills):
        return None
    sk = skills[skill_idx]
    if sk.skill_id == "end":
        return None

    target_id = _entity_id_at_slot(ws, agent_id, entity_idx)
    coord_x, coord_y = _grid_cell_to_xy(grid_cell)
    bf = ws.combat.battlefield if ws.combat else None

    tt = sk.features.target_type
    if tt == TargetType.SELF:
        return sk.build_action(agent_id, agent_id, agent.position)
    if tt in (TargetType.SINGLE_ALLY, TargetType.MULTI_ALLY):
        is_party = ws.is_party_ally(agent_id)
        if target_id is None or ws.is_party_ally(target_id) != is_party:
            target_id = agent_id
        target_pos = ws.characters[target_id].position
        return sk.build_action(agent_id, target_id, target_pos)
    if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
        if target_id is None:
            return None
        target_pos = ws.characters[target_id].position
        if bf is not None and not bf.has_line_of_sight(agent.position, target_pos):
            return None   # LOS-blocked targets become a no-op turn end
        return sk.build_action(agent_id, target_id, target_pos)
    if tt == TargetType.POINT:
        coord = _clamp_to_range(Vec2(coord_x, coord_y), agent.position,
                                sk.features.range_m)
        if bf is not None:
            if bf.is_blocked(coord):
                return None
            # Movement doesn't require line of sight to the destination —
            # you walk on the ground, so a pillar between you and your
            # target doesn't stop you from walking around it. LOS gating
            # only applies to ranged abilities (spells, AOEs).
            if sk.skill_id != "move" and not bf.has_line_of_sight(
                    agent.position, coord):
                return None
        return sk.build_action(agent_id, None, coord)
    # LINE/CONE fall back to POINT semantics
    coord = _clamp_to_range(Vec2(coord_x, coord_y), agent.position,
                            sk.features.range_m)
    if bf is not None:
        if bf.is_blocked(coord) or not bf.has_line_of_sight(agent.position, coord):
            return None
    return sk.build_action(agent_id, target_id, coord)


def _clamp_to_range(target: Vec2, origin: Vec2, range_m: float) -> Vec2:
    """Pull `target` along the (origin → target) ray to within `range_m`.

    Random grid samples are usually farther than a spell's range; clamping
    here keeps decode_action from ever producing an out-of-range POINT action.
    """
    if range_m <= 0.0:
        return target
    dx = target.x - origin.x
    dy = target.y - origin.y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= range_m - 1e-3:
        return target
    # Pull strictly inside the range circle — engine compares with strict
    # inequality, so landing exactly on the boundary can still trip on FP drift.
    scale = (range_m - 1e-3) / dist
    return Vec2(origin.x + dx * scale, origin.y + dy * scale)


def _entity_slot_of(ws: WorldState, agent_id: str, target_id: str) -> int:
    """Reverse of _entity_id_at_slot."""
    if target_id == agent_id:
        return 0
    allies, enemies = partition_entities(ws, agent_id)
    if target_id in allies[:2]:
        return 1 + allies.index(target_id)
    if target_id in enemies[:3]:
        return 3 + enemies.index(target_id)
    return 0


def _xy_to_grid_cell(x: float, y: float) -> int:
    """Inverse of _grid_cell_to_xy. Clamps to grid range."""
    ix = max(0, min(N_GRID - 1, int(x / GRID_CELL_SIZE_M)))
    iy = max(0, min(N_GRID - 1, int(y / GRID_CELL_SIZE_M)))
    return ix * N_GRID + iy


def encode_action(action_dict: dict | None, ws: WorldState, agent_id: str) -> tuple[int, int, int]:
    """Reverse-map an engine action dict into [skill_idx, entity_idx, grid_cell].

    Every action dict carries ``skill_id`` (auto-embedded by Skill.build_action
    / Ability.build_action, enforced by execute_action) so this is a
    direct lookup — no pattern-matching, no reverse inference.

    Returns (0, 0, 0) for None / end-turn. Returns (-1, -1, -1) when the
    skill is no longer in the agent's current available list (e.g. ability
    was used up after the expert committed to it).
    """
    if action_dict is None:
        return (0, 0, 0)

    skill_id = action_dict.get("skill_id")
    if not skill_id:
        # Action dict was built outside the Skill abstraction — engine
        # validator should have caught this. Bail loudly rather than guess.
        raise ValueError(
            f"encode_action: dict missing skill_id field — must be built via "
            f"Skill.build_action() / Ability.build_action(). Got: {action_dict!r}"
        )

    agent = ws.characters[agent_id]
    skills = available_skills(agent, ws)
    skill_idx = next((i for i, s in enumerate(skills) if s.skill_id == skill_id), -1)
    if skill_idx < 0:
        return (-1, -1, -1)

    # Entity slot
    target_id = (action_dict.get("target") or action_dict.get("character")
                 or action_dict.get("caster") or agent_id)
    if isinstance(target_id, str) and target_id in ws.characters:
        entity_idx = _entity_slot_of(ws, agent_id, target_id)
    else:
        entity_idx = 0

    # Grid cell — the spatial label the grid_head learns to predict.
    # Priority:
    #   1. explicit target_position in the action (POINT spells, MOVE-to-coord)
    #   2. target entity's position (MOVE-toward-entity — engine resolves the
    #      destination from target_id, but encode_action needs an explicit
    #      cell so BC's grid_head sees the right supervision)
    #   3. fallback 0 (only for actions where grid genuinely is unused)
    if "target_position" in action_dict:
        x, y = action_dict["target_position"][0], action_dict["target_position"][1]
        grid_cell = _xy_to_grid_cell(float(x), float(y))
    elif (isinstance(target_id, str)
          and target_id != agent_id
          and target_id in ws.characters):
        tpos = ws.characters[target_id].position
        grid_cell = _xy_to_grid_cell(float(tpos.x), float(tpos.y))
    else:
        grid_cell = 0
    return (skill_idx, entity_idx, grid_cell)
