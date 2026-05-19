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
from ..engine.vec2 import TerrainType


# ── Schema constants ─────────────────────────────────────────────────────────

N_SKILL_SLOTS = 20            # max skills per turn (pad with zeros)
N_ENTITY_SLOTS = 6            # self + 2 allies + 3 enemies
ENTITY_DIM = 8                # per-entity feature width
N_GRID = 20                   # battlefield grid resolution
BATTLEFIELD_SIZE_M = 30.0     # matches Battlefield default
GRID_CELL_SIZE_M = BATTLEFIELD_SIZE_M / N_GRID    # 1.5m
N_GRID_CELLS = N_GRID * N_GRID                    # 400

OBS_KEYS = ("skills", "skill_mask", "entities", "resources", "terrain")
