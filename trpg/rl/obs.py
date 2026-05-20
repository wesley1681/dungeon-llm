"""Phase 2 RL observation extractors.

Each function reads from WorldState and produces a single numpy array. The
shapes and dtypes match the spec at docs/superpowers/specs/2026-05-19-rl-env-v2-design.md.
"""
from __future__ import annotations
import numpy as np

from ..engine.character import Character
from ..engine.world_state import WorldState
from ..engine.skill import available_skills, SKILL_FEATURE_DIM, STATUS_SLOTS
from ..engine.combat import MOVE_BUDGET_M
from ..engine.status import MODIFIER_CLASSES
from ..engine.vec2 import Battlefield, TerrainType, Vec2
from ..scenarios.archetypes import ARCHETYPE_FACTORIES


# ── Schema constants ─────────────────────────────────────────────────────────

N_SKILL_SLOTS = 20            # max skills per turn (pad with zeros)
N_ENTITY_SLOTS = 6            # self + 2 allies + 3 enemies
# Stable archetype ordering for the per-entity one-hot. Multi-hot in shape so a
# future multiclass character (paladin/sorcerer) can light up multiple bits;
# single-class characters just have one bit set. Sorted alphabetically so the
# mapping is deterministic across Python versions / dict orderings.
ARCHETYPE_OBS_LIST: tuple[str, ...] = tuple(sorted(ARCHETYPE_FACTORIES.keys()))
N_ARCHETYPES = len(ARCHETYPE_OBS_LIST)   # 12

# Per-entity status multi-hot. Union of canonical D&D conditions and every
# implemented buff/debuff in the engine registry so adding a new StatusEffect
# subclass (with a MODIFIER_CLASSES entry) auto-extends the obs.
# D&D-fair: these correspond to visible buff/condition icons at the table.
RL_STATUS_NAMES: tuple[str, ...] = tuple(
    sorted(set(STATUS_SLOTS) | set(MODIFIER_CLASSES.keys()))
)
N_RL_STATUS = len(RL_STATUS_NAMES)

ENTITY_DIM = 7 + N_ARCHETYPES + N_RL_STATUS + 1
# layout: 7 base + archetype multi-hot + status multi-hot + is_concentrating
N_GRID = 30                   # battlefield grid resolution (1m cells)
BATTLEFIELD_SIZE_M = 30.0     # matches Battlefield default
GRID_CELL_SIZE_M = BATTLEFIELD_SIZE_M / N_GRID    # 1.0m
N_GRID_CELLS = N_GRID * N_GRID                    # 900

# Entity-grid overlay channels: a parallel 30x30 representation of where
# self/allies/enemies are. Gives the model the same positional info as the
# continuous entity rows BUT pre-discretised to the same cell grid that
# grid_head outputs — so grid_head doesn't have to learn the continuous→
# discrete map with a Linear (which is what bottlenecked the previous arch).
N_ENTITY_GRID_CHANNELS = 3       # [self, ally, enemy]

OBS_KEYS = ("skills", "skill_mask", "entities", "resources", "terrain",
            "entity_grid")


def terrain_obs(battlefield: Battlefield) -> np.ndarray:
    """Sample the battlefield at each grid cell centre.

    Returns: float32[N_GRID, N_GRID] with values
       0.0 = open
       0.5 = difficult terrain
       1.0 = obstacle / wall
      -1.0 = dangerous terrain
    """
    grid = np.zeros((N_GRID, N_GRID), dtype=np.float32)
    for ix in range(N_GRID):
        for iy in range(N_GRID):
            cx = (ix + 0.5) * GRID_CELL_SIZE_M
            cy = (iy + 0.5) * GRID_CELL_SIZE_M
            p = Vec2(cx, cy)
            if battlefield.is_blocked(p):
                grid[ix, iy] = 1.0
                continue
            t = battlefield.terrain_at(p)
            if t == TerrainType.DANGEROUS:
                grid[ix, iy] = -1.0
            elif t == TerrainType.DIFFICULT:
                grid[ix, iy] = 0.5
    return grid


def partition_entities(ws: WorldState, agent_id: str) -> tuple[list[str], list[str]]:
    """Partition live characters (excluding self) into (allies, enemies_sorted_by_distance).

    Both lists contain char_ids. Enemy ordering is the slot ordering used by
    entities_obs (slot 3 = enemies[0] = closest).
    """
    self_char = ws.characters[agent_id]
    is_party = ws.is_party_ally(agent_id)
    allies: list[str] = []
    enemies: list[str] = []
    for cid, c in ws.characters.items():
        if cid == agent_id or not c.is_alive():
            continue
        other_is_party = ws.is_party_ally(cid)
        if is_party == other_is_party:
            allies.append(cid)
        else:
            enemies.append(cid)
    enemies.sort(key=lambda cid: self_char.position.distance_to(ws.characters[cid].position))
    return allies, enemies


def _entity_row(char: Character, self_char: Character,
                bf_size_x: float, bf_size_y: float,
                is_self: bool, is_enemy: bool) -> np.ndarray:
    """Build one entity row for the entities observation.

    bf_size_x / bf_size_y are the battlefield dimensions used to normalise
    coordinates to [0, 1]. Distance is normalised by the diagonal so that
    the max possible distance maps to 1.0.

    Layout (ENTITY_DIM = 7 + N_ARCHETYPES + N_RL_STATUS + 1):
        [0..6]  base features (no debuff bit — replaced by status multi-hot)
        [7..7+N_ARCHETYPES-1]              archetype multi-hot
        [7+N_ARCHETYPES..+N_RL_STATUS-1]   status multi-hot (per-status bits;
                                            replaces the old single has_debuff
                                            so timing-dependent buffs like
                                            vow_target / sacred_weapon_buff /
                                            raging are visible to the policy)
        [-1]                                is_concentrating
    """
    bf_diag = (bf_size_x ** 2 + bf_size_y ** 2) ** 0.5 or 1.0
    base = np.array([
        char.hp / max(1, char.max_hp),
        char.position.x / bf_size_x if bf_size_x else 0.0,
        char.position.y / bf_size_y if bf_size_y else 0.0,
        self_char.position.distance_to(char.position) / bf_diag,
        1.0 if is_enemy else 0.0,
        1.0 if char.is_alive() else 0.0,
        1.0 if is_self else 0.0,
    ], dtype=np.float32)

    # Archetype multi-hot. Multi-hot rather than one-hot so a hypothetical
    # multiclass character (e.g. paladin/sorcerer) can set both bits.
    arch_oh = np.zeros(N_ARCHETYPES, dtype=np.float32)
    arch_id = getattr(char, "archetype_id", "") or ""
    # Accept "vengeance,sorcerer" comma-separated form for future multiclass —
    # split and light each component that we recognise.
    for part in arch_id.split(","):
        part = part.strip()
        if part in ARCHETYPE_OBS_LIST:
            arch_oh[ARCHETYPE_OBS_LIST.index(part)] = 1.0

    status_mh = np.zeros(N_RL_STATUS, dtype=np.float32)
    for i, name in enumerate(RL_STATUS_NAMES):
        if char.has_status(name):
            status_mh[i] = 1.0

    is_concentrating = np.array([1.0 if char.concentrating_on else 0.0],
                                 dtype=np.float32)
    return np.concatenate([base, arch_oh, status_mh, is_concentrating])


def entities_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Build the entities observation matrix.

    Row order: [self, ally_1, ally_2, enemy_1, enemy_2, enemy_3].
    Enemies sorted by distance to self ascending. Missing rows are zero.
    """
    self_char = ws.characters[agent_id]
    # Use the actual battlefield dimensions so non-default sizes still
    # produce normalised coords in [0, 1]. Fall back to the engine default
    # when combat hasn't been set up (e.g. agent observed out of combat).
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    bf_size_x = bf.width if bf else BATTLEFIELD_SIZE_M
    bf_size_y = bf.height if bf else BATTLEFIELD_SIZE_M
    out = np.zeros((N_ENTITY_SLOTS, ENTITY_DIM), dtype=np.float32)

    # Row 0: self
    out[0] = _entity_row(self_char, self_char, bf_size_x, bf_size_y,
                         is_self=True, is_enemy=False)

    ally_ids, enemy_ids = partition_entities(ws, agent_id)

    # Rows 1..2: allies (truncate at 2)
    for i, cid in enumerate(ally_ids[:2]):
        out[1 + i] = _entity_row(ws.characters[cid], self_char, bf_size_x, bf_size_y,
                                  is_self=False, is_enemy=False)
    # Rows 3..5: enemies (truncate at 3)
    for i, cid in enumerate(enemy_ids[:3]):
        out[3 + i] = _entity_row(ws.characters[cid], self_char, bf_size_x, bf_size_y,
                                  is_self=False, is_enemy=True)
    return out


def skills_obs(ws: WorldState, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (skills_matrix, skill_mask).

    skills_matrix : float32[N_SKILL_SLOTS, SKILL_FEATURE_DIM]
    skill_mask    : float32[N_SKILL_SLOTS]    1=valid, 0=padding
    """
    agent = ws.characters[agent_id]
    skills = available_skills(agent, ws)
    skill_mat = np.zeros((N_SKILL_SLOTS, SKILL_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(N_SKILL_SLOTS, dtype=np.float32)

    for i, sk in enumerate(skills[:N_SKILL_SLOTS]):
        skill_mat[i] = np.asarray(sk.features.as_vector(), dtype=np.float32)
        mask[i] = 1.0
    return skill_mat, mask


def resources_obs(resources: dict, round_number: int) -> np.ndarray:
    """Build the resources observation vector (length 4)."""
    return np.array([
        1.0 if resources.get("action", 0) > 0 else 0.0,
        1.0 if resources.get("bonus_action", 0) > 0 else 0.0,
        max(0.0, min(1.0, resources.get("movement", 0.0) / MOVE_BUDGET_M)),
        min(1.0, round_number / 10.0),
    ], dtype=np.float32)


def entity_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Build the entity-grid overlay.

    Returns: float32[N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — self position   (1.0 at self's cell, 0 elsewhere)
      channel 1 — allies          (sum of ally markers per cell)
      channel 2 — enemies         (sum of enemy markers per cell)

    Each cell value is the count of matching characters whose centre falls in
    that cell — typically 0 or 1, occasionally >1 if two share a cell.
    """
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    cell_size = bf.width / N_GRID if bf else GRID_CELL_SIZE_M
    out = np.zeros((N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)

    self_char = ws.characters[agent_id]
    ally_ids, enemy_ids = partition_entities(ws, agent_id)

    def _mark(ch: int, pos) -> None:
        ix = max(0, min(N_GRID - 1, int(pos.x / cell_size)))
        iy = max(0, min(N_GRID - 1, int(pos.y / cell_size)))
        out[ch, ix, iy] += 1.0

    _mark(0, self_char.position)
    for cid in ally_ids:
        c = ws.characters[cid]
        if c.is_alive():
            _mark(1, c.position)
    for cid in enemy_ids:
        c = ws.characters[cid]
        if c.is_alive():
            _mark(2, c.position)
    return out


def build_obs(ws: WorldState, agent_id: str, resources: dict) -> dict:
    """Assemble the full Phase 2 Dict observation."""
    skills_mat, skill_mask = skills_obs(ws, agent_id)
    round_num = ws.combat.round_number if ws.combat else 0
    return {
        "skills":      skills_mat,
        "skill_mask":  skill_mask,
        "entities":    entities_obs(ws, agent_id),
        "resources":   resources_obs(resources, round_num),
        "terrain":     terrain_obs(ws.combat.battlefield),
        "entity_grid": entity_grid_obs(ws, agent_id),
    }
