import pytest
from trpg.engine.vec2 import Battlefield, Vec2, TerrainType
from trpg.sandbox.terrain import TERRAIN_PRESETS


def _fresh_bf() -> Battlefield:
    return Battlefield(width=30.0, height=30.0, grid_resolution=0.5)


def test_terrain_presets_keys():
    assert set(TERRAIN_PRESETS) == {"empty", "lava_strip", "pillars", "corridor", "arena"}


def test_empty_preset_is_noop():
    bf = _fresh_bf()
    TERRAIN_PRESETS["empty"](bf)
    assert all(cell == TerrainType.NORMAL for row in bf.cells for cell in row)


def test_lava_strip_paints_dangerous_band():
    bf = _fresh_bf()
    TERRAIN_PRESETS["lava_strip"](bf)
    # A 1m-wide horizontal lava strip at the middle (y ≈ 15)
    assert bf.terrain_at(Vec2(15.0, 15.0)) == TerrainType.DANGEROUS
    # And no lava at the corners
    assert bf.terrain_at(Vec2(0.5, 0.5)) == TerrainType.NORMAL
    assert bf.terrain_at(Vec2(29.5, 29.5)) == TerrainType.NORMAL


def test_pillars_paints_four_walls():
    bf = _fresh_bf()
    TERRAIN_PRESETS["pillars"](bf)
    for x, y in [(7.5, 7.5), (22.5, 7.5), (7.5, 22.5), (22.5, 22.5)]:
        assert bf.terrain_at(Vec2(x, y)) == TerrainType.BLOCKED


def test_arena_paints_perimeter_walls():
    bf = _fresh_bf()
    TERRAIN_PRESETS["arena"](bf)
    assert bf.terrain_at(Vec2(0.5, 15.0)) == TerrainType.BLOCKED   # left edge
    assert bf.terrain_at(Vec2(29.5, 15.0)) == TerrainType.BLOCKED  # right edge
    assert bf.terrain_at(Vec2(15.0, 0.5)) == TerrainType.BLOCKED   # bottom edge
    assert bf.terrain_at(Vec2(15.0, 29.5)) == TerrainType.BLOCKED  # top edge
    assert bf.terrain_at(Vec2(15.0, 15.0)) == TerrainType.NORMAL   # interior


def test_corridor_paints_narrow_passage():
    bf = _fresh_bf()
    TERRAIN_PRESETS["corridor"](bf)
    # Walls forming a 3m-wide corridor along the middle
    assert bf.terrain_at(Vec2(15.0, 5.0)) == TerrainType.BLOCKED
    assert bf.terrain_at(Vec2(15.0, 25.0)) == TerrainType.BLOCKED
    assert bf.terrain_at(Vec2(15.0, 15.0)) == TerrainType.NORMAL
