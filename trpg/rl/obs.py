"""Phase 2 RL observation extractors.

Each function reads from WorldState and produces a single numpy array. The
shapes and dtypes match the spec at docs/superpowers/specs/2026-05-19-rl-env-v2-design.md.
"""
from __future__ import annotations
import numpy as np

from ..engine.character import Character
from ..engine.world_state import WorldState
from ..engine.skill import available_skills, SKILL_FEATURE_DIM
from ..engine.combat import MOVE_BUDGET_M
from ..engine.vec2 import Battlefield, TerrainType, Vec2


# ── Schema constants ─────────────────────────────────────────────────────────

N_SKILL_SLOTS = 20            # max skills per turn (pad with zeros)
N_ENTITY_SLOTS = 6            # self + 2 allies + 3 enemies
ENTITY_DIM = 8                # per-entity feature width
N_GRID = 20                   # battlefield grid resolution
BATTLEFIELD_SIZE_M = 30.0     # matches Battlefield default
GRID_CELL_SIZE_M = BATTLEFIELD_SIZE_M / N_GRID    # 1.5m
N_GRID_CELLS = N_GRID * N_GRID                    # 400

OBS_KEYS = ("skills", "skill_mask", "entities", "resources", "terrain")


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


_DEBUFF_NAMES = frozenset({"paralyzed", "restrained", "stunned", "poisoned",
                            "frightened", "charmed", "prone", "blinded"})


def _entity_row(char: Character, self_char: Character, bf_size: float,
                is_self: bool, is_enemy: bool) -> np.ndarray:
    """Build one entity row for the entities observation."""
    has_debuff = any(
        getattr(fx, "name", None) in _DEBUFF_NAMES
        for fx in char.status_effects
    )
    return np.array([
        char.hp / max(1, char.max_hp),
        char.position.x / bf_size,
        char.position.y / bf_size,
        self_char.position.distance_to(char.position) / bf_size,
        1.0 if is_enemy else 0.0,
        1.0 if char.is_alive() else 0.0,
        1.0 if is_self else 0.0,
        1.0 if has_debuff else 0.0,
    ], dtype=np.float32)


def entities_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Build the entities observation matrix.

    Row order: [self, ally_1, ally_2, enemy_1, enemy_2, enemy_3].
    Enemies sorted by distance to self ascending. Missing rows are zero.
    """
    self_char = ws.characters[agent_id]
    bf_size = BATTLEFIELD_SIZE_M
    out = np.zeros((N_ENTITY_SLOTS, ENTITY_DIM), dtype=np.float32)

    # Row 0: self
    out[0] = _entity_row(self_char, self_char, bf_size, is_self=True, is_enemy=False)

    # Partition others
    allies, enemies = [], []
    is_party = ws.is_party_ally(agent_id)
    for cid, c in ws.characters.items():
        if cid == agent_id or not c.is_alive():
            continue
        other_is_party = ws.is_party_ally(cid)
        if is_party == other_is_party:
            allies.append(c)
        else:
            enemies.append(c)

    enemies.sort(key=lambda c: self_char.position.distance_to(c.position))

    # Rows 1..2: allies (truncate at 2)
    for i, ally in enumerate(allies[:2]):
        out[1 + i] = _entity_row(ally, self_char, bf_size,
                                  is_self=False, is_enemy=False)
    # Rows 3..5: enemies (truncate at 3)
    for i, enemy in enumerate(enemies[:3]):
        out[3 + i] = _entity_row(enemy, self_char, bf_size,
                                  is_self=False, is_enemy=True)
    return out
