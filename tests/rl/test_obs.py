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
