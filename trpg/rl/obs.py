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

ENTITY_DIM = 8 + N_ARCHETYPES + 1   # 21: base + archetype multi-hot + is_concentrating
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


# Status names in STATUS_SLOTS (canonical 5e conditions) that are NOT debuffs.
# Used to derive `_DEBUFF_NAMES` for STATUS_SLOTS entries that don't have an
# implementing StatusEffect subclass yet (so they have no `kind` field to
# read). Tiny explicit list — extending STATUS_SLOTS with a new non-debuff
# requires adding it here, but failures are explicit (RL obs flags it as a
# debuff until corrected, which is preferable to silent misclassification).
_NON_DEBUFF_SLOT_NAMES: frozenset[str] = frozenset({
    "dodging",     # 5e Dodge action — defensive buff
    "hidden",      # successful Hide — tactical buff
    "invisible",   # buff: attackers vs you have disadvantage
})


def _debuff_names() -> frozenset[str]:
    """All known status names that count as debuffs for RL observation.

    Two sources, unioned:
      1. `MODIFIER_CLASSES` entries whose `kind` is 'debuff' — the registry
         of every status the engine can actually attach via APPLY_MOD.
      2. `STATUS_SLOTS` entries minus `_NON_DEBUFF_SLOT_NAMES` — covers
         canonical condition names that have no implementing subclass yet
         (e.g. blinded, deafened, grappled) but can still be applied as
         bare-string statuses via the LLM tag parser.

    Adding a new debuff means appending it to STATUS_SLOTS (if novel) or
    registering it in MODIFIER_CLASSES — no separate allowlist to update.
    """
    names: set[str] = set()
    for cls in MODIFIER_CLASSES.values():
        try:
            fx = cls()
        except TypeError:
            fx = cls(applied_round=0)
        if getattr(fx, "kind", "debuff") == "debuff":
            names.add(fx.name)
    for slot in STATUS_SLOTS:
        if slot not in _NON_DEBUFF_SLOT_NAMES:
            names.add(slot)
    return frozenset(names)


_DEBUFF_NAMES = _debuff_names()


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

    Layout (ENTITY_DIM = 8 + N_ARCHETYPES + 1):
        [0..7]  base features
        [8..8+N_ARCHETYPES-1]  archetype multi-hot (D&D-fair: visible from look)
        [-1]    is_concentrating (D&D-fair: DM normally tells players)
    """
    has_debuff = any(char.has_status(name) for name in _DEBUFF_NAMES)
    bf_diag = (bf_size_x ** 2 + bf_size_y ** 2) ** 0.5 or 1.0
    base = np.array([
        char.hp / max(1, char.max_hp),
        char.position.x / bf_size_x if bf_size_x else 0.0,
        char.position.y / bf_size_y if bf_size_y else 0.0,
        self_char.position.distance_to(char.position) / bf_diag,
        1.0 if is_enemy else 0.0,
        1.0 if char.is_alive() else 0.0,
        1.0 if is_self else 0.0,
        1.0 if has_debuff else 0.0,
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

    is_concentrating = np.array([1.0 if char.concentrating_on else 0.0],
                                 dtype=np.float32)
    return np.concatenate([base, arch_oh, is_concentrating])


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


def build_obs(ws: WorldState, agent_id: str, resources: dict) -> dict:
    """Assemble the full Phase 2 Dict observation."""
    skills_mat, skill_mask = skills_obs(ws, agent_id)
    round_num = ws.combat.round_number if ws.combat else 0
    return {
        "skills":     skills_mat,
        "skill_mask": skill_mask,
        "entities":   entities_obs(ws, agent_id),
        "resources":  resources_obs(resources, round_num),
        "terrain":    terrain_obs(ws.combat.battlefield),
    }
