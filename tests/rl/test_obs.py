import pytest
from trpg.rl.obs import (
    N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM,
    N_GRID, GRID_CELL_SIZE_M, BATTLEFIELD_SIZE_M, OBS_KEYS,
)


def test_schema_constants():
    assert N_SKILL_SLOTS == 20
    assert N_ENTITY_SLOTS == 6
    assert ENTITY_DIM == 8
    assert N_GRID == 20
    assert GRID_CELL_SIZE_M == 1.5
    assert BATTLEFIELD_SIZE_M == 30.0
    assert set(OBS_KEYS) == {"skills", "skill_mask", "entities", "resources", "terrain"}


import numpy as np
from trpg.engine.vec2 import Battlefield, TerrainType
from trpg.rl.obs import terrain_obs, N_GRID


def test_terrain_obs_shape_and_dtype():
    bf = Battlefield(width=30.0, height=30.0)
    grid = terrain_obs(bf)
    assert grid.shape == (N_GRID, N_GRID)
    assert grid.dtype == np.float32


def test_terrain_obs_open_field_is_zero():
    bf = Battlefield(width=30.0, height=30.0)
    grid = terrain_obs(bf)
    assert np.all(grid == 0.0)


def test_terrain_obs_obstacle_is_one():
    bf = Battlefield(width=30.0, height=30.0)
    bf.add_rect_obstacle(0.0, 0.0, 3.0, 3.0)
    grid = terrain_obs(bf)
    # (0..3m, 0..3m) covers cells (0..2, 0..2) at 1.5m/cell
    assert grid[0, 0] == 1.0
    assert grid[1, 1] == 1.0
    # outside the obstacle stays 0
    assert grid[5, 5] == 0.0


def test_terrain_obs_dangerous_is_negative():
    bf = Battlefield(width=30.0, height=30.0)
    bf.add_rect_terrain(0.0, 0.0, 3.0, 3.0, TerrainType.DANGEROUS)
    grid = terrain_obs(bf)
    assert grid[0, 0] == -1.0


def test_terrain_obs_difficult_is_half():
    bf = Battlefield(width=30.0, height=30.0)
    bf.add_rect_terrain(0.0, 0.0, 3.0, 3.0, TerrainType.DIFFICULT)
    grid = terrain_obs(bf)
    assert grid[0, 0] == 0.5


from trpg.engine.character import Character, Stats
from trpg.engine.world_state import WorldState, CombatState
from trpg.engine.combat import setup_combat_positions
from trpg.engine.items import WEAPON_DEFS
from trpg.rl.obs import entities_obs, N_ENTITY_SLOTS, ENTITY_DIM


def _build_1v1_world():
    a = Character(name="agent", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14, DEX=12, CON=14),
                  hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    b = Character(name="enemy", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12),
                  hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b"], round_number=1)
    setup_combat_positions(ws, ws.combat)
    return ws


def test_entities_obs_shape_and_dtype():
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    assert obs.shape == (N_ENTITY_SLOTS, ENTITY_DIM)
    assert obs.dtype == np.float32


def test_entities_obs_self_row_zero():
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    # row 0 is self
    assert obs[0, 6] == 1.0   # is_self
    assert obs[0, 4] == 0.0   # not enemy
    assert obs[0, 5] == 1.0   # alive
    assert obs[0, 0] == pytest.approx(1.0)   # full hp


def test_entities_obs_enemy_in_enemy_slots():
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    # rows 3..5 are enemies, slot 3 is closest
    assert obs[3, 4] == 1.0   # is_enemy
    assert obs[3, 6] == 0.0   # not self
    assert obs[3, 5] == 1.0   # alive
    assert obs[3, 0] == pytest.approx(1.0)


def test_entities_obs_padding_is_zero():
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    # no allies → rows 1, 2 are padding
    assert np.all(obs[1] == 0.0)
    assert np.all(obs[2] == 0.0)
    # only 1 enemy → rows 4, 5 are padding
    assert np.all(obs[4] == 0.0)
    assert np.all(obs[5] == 0.0)
