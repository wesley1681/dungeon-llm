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
