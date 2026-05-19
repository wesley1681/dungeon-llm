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

    tt = sk.features.target_type
    if tt == TargetType.SELF:
        return sk.builder(agent_id, agent_id, agent.position)
    if tt in (TargetType.SINGLE_ENEMY, TargetType.SINGLE_ALLY, TargetType.MULTI_ENEMY, TargetType.MULTI_ALLY):
        if target_id is None:
            return None
        target_pos = ws.characters[target_id].position
        return sk.builder(agent_id, target_id, target_pos)
    if tt == TargetType.POINT:
        return sk.builder(agent_id, None, Vec2(coord_x, coord_y))
    # LINE/CONE fall back to POINT semantics
    return sk.builder(agent_id, target_id, Vec2(coord_x, coord_y))
