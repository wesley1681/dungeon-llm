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


def _match_skill_idx(action_dict: dict, agent, agent_id: str, ws: WorldState) -> int:
    """Find the skill_idx whose builder would have produced this action dict.

    Strategy: match on action ``type`` plus the most-discriminative key
    (``weapon_index`` / ``modifier`` / ``spell_name``). Falls back to skill_id
    inference for MOVE-style actions.

    For weapon/spell matches we prefer exact ``skill_id == "weapon:<name>"`` /
    ``"spell:<name>"`` equality; a substring fallback only kicks in when no
    exact match exists, to avoid collisions like "短劍" ⊂ "雙手短劍".
    """
    skills = available_skills(agent, ws)
    a_type = action_dict.get("type")

    # MOVE → 'move' skill
    if a_type == "MOVE":
        for i, s in enumerate(skills):
            if s.skill_id == "move":
                return i
        return 0

    if a_type == "ATTACK":
        weapon_name = action_dict.get("weapon", "")
        target_skill_id = f"weapon:{weapon_name}"
        # Exact match first.
        for i, s in enumerate(skills):
            if s.skill_id == target_skill_id:
                return i
        # Substring fallback for safety.
        for i, s in enumerate(skills):
            if s.skill_id.startswith("weapon:") and weapon_name and weapon_name in s.skill_id:
                return i
        # Fall through: some ClassAbility builders also emit ATTACK
        # (e.g. reckless_attack). Probe each builder with the real agent_id.
        for i, s in enumerate(skills):
            try:
                probe = s.builder(agent_id, action_dict.get("target"), None)
                if probe and probe.get("type") == "ATTACK":
                    return i
            except Exception:
                continue
        return 0

    # SPELL by spell_name
    if a_type == "SPELL":
        spell_name = action_dict.get("spell_name", "")
        target_skill_id = f"spell:{spell_name}"
        for i, s in enumerate(skills):
            if s.skill_id == target_skill_id:
                return i
        for i, s in enumerate(skills):
            if s.skill_id.startswith("spell:") and spell_name and spell_name in s.skill_id:
                return i

    # APPLY_MOD by modifier name
    if a_type == "APPLY_MOD":
        mod = action_dict.get("modifier", "")
        for i, s in enumerate(skills):
            if s.skill_id == mod or s.skill_id.endswith(mod):
                return i

    # DODGE / HIDE / DISENGAGE — match by type name
    for i, s in enumerate(skills):
        if s.skill_id == a_type.lower():
            return i
    return 0


def encode_action(action_dict: dict | None, ws: WorldState, agent_id: str) -> tuple[int, int, int]:
    """Reverse-map an engine action dict into [skill_idx, entity_idx, grid_cell].

    Returns (0, 0, 0) for None / end-turn. Used by BC data collection.
    """
    if action_dict is None:
        return (0, 0, 0)

    agent = ws.characters[agent_id]
    skill_idx = _match_skill_idx(action_dict, agent, agent_id, ws)

    # Entity slot
    target_id = (action_dict.get("target") or action_dict.get("character")
                 or action_dict.get("caster") or agent_id)
    if isinstance(target_id, str) and target_id in ws.characters:
        entity_idx = _entity_slot_of(ws, agent_id, target_id)
    else:
        entity_idx = 0

    # Grid cell (for POINT-target spells and MOVE-with-coord)
    if "target_position" in action_dict:
        x, y = action_dict["target_position"][0], action_dict["target_position"][1]
        grid_cell = _xy_to_grid_cell(float(x), float(y))
    else:
        grid_cell = 0
    return (skill_idx, entity_idx, grid_cell)
