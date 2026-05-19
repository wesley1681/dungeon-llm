import pytest
import numpy as np

from trpg.engine.character import Character, Stats
from trpg.engine.combat import setup_combat_positions
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.skill import SKILL_FEATURE_DIM
from trpg.engine.status import Paralyzed
from trpg.engine.vec2 import Battlefield, TerrainType, Vec2
from trpg.engine.world_state import CombatState, WorldState
from trpg.rl.obs import (
    BATTLEFIELD_SIZE_M,
    ENTITY_DIM,
    GRID_CELL_SIZE_M,
    N_ENTITY_SLOTS,
    N_GRID,
    N_SKILL_SLOTS,
    OBS_KEYS,
    build_obs,
    entities_obs,
    resources_obs,
    skills_obs,
    terrain_obs,
)


def test_schema_constants():
    assert N_SKILL_SLOTS == 20
    assert N_ENTITY_SLOTS == 6
    assert ENTITY_DIM == 8
    assert N_GRID == 20
    assert GRID_CELL_SIZE_M == 1.5
    assert BATTLEFIELD_SIZE_M == 30.0
    assert set(OBS_KEYS) == {"skills", "skill_mask", "entities", "resources", "terrain"}


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


def test_entities_obs_sorts_enemies_by_distance():
    """When multiple enemies exist, slot 3 = closest, slot 4 = farther."""
    a = Character(name="agent", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14), hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    far = Character(name="far", race="哥布林", class_="戰士", level=1,
                    stats=Stats(STR=10), hp=10, max_hp=10, ac=12,
                    weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    close = Character(name="close", race="哥布林", class_="戰士", level=1,
                      stats=Stats(STR=10), hp=10, max_hp=10, ac=12,
                      weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "far": far, "close": close},
                    scene="test", pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "close", "far"], round_number=1)
    setup_combat_positions(ws, ws.combat)
    # Manual placement
    a.position = Vec2(5.0, 15.0)
    close.position = Vec2(10.0, 15.0)
    far.position = Vec2(20.0, 15.0)

    obs = entities_obs(ws, "a")
    # slot 3 should be "close" (distance 5), slot 4 should be "far" (distance 15)
    # We can verify via the dist_norm column (index 3)
    assert obs[3, 3] < obs[4, 3], "closer enemy should have smaller dist_norm"


def test_entities_obs_has_debuff_flag():
    """has_debuff flag is set when a debuff status is on the character."""
    ws = _build_1v1_world()
    ws.characters["b"].status_effects.append(Paralyzed())
    obs = entities_obs(ws, "a")
    # enemy "b" is at slot 3, has_debuff is column 7
    assert obs[3, 7] == 1.0


def test_entities_obs_dead_chars_excluded():
    """Dead characters are not placed in slots."""
    ws = _build_1v1_world()
    ws.characters["b"].hp = 0
    obs = entities_obs(ws, "a")
    # slot 3 (enemy_1) should be all zeros since b is dead
    assert np.all(obs[3] == 0.0)


def test_skills_obs_shapes():
    ws = _build_1v1_world()
    skills, mask = skills_obs(ws, "a")
    assert skills.shape == (N_SKILL_SLOTS, SKILL_FEATURE_DIM)
    assert mask.shape == (N_SKILL_SLOTS,)
    assert skills.dtype == np.float32
    assert mask.dtype == np.float32


def test_skills_obs_mask_marks_valid_slots():
    ws = _build_1v1_world()
    skills, mask = skills_obs(ws, "a")
    # Agent is L3 fighter, should have several skills (end, move, weapon, dodge, hide)
    n_valid = int(mask.sum())
    assert n_valid >= 4
    # mask[0] = end skill (always present)
    assert mask[0] == 1.0
    # padding slots have all-zero features
    if n_valid < N_SKILL_SLOTS:
        assert np.all(skills[n_valid:] == 0.0)
        assert np.all(mask[n_valid:] == 0.0)


def test_resources_obs_full_budget():
    res = {"action": 1, "bonus_action": 1, "movement": 9.0}
    out = resources_obs(res, round_number=1)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert out[0] == 1.0   # action available
    assert out[1] == 1.0   # bonus available
    assert out[2] == pytest.approx(1.0)   # movement ratio (9.0 / MOVE_BUDGET_M)
    assert out[3] == pytest.approx(0.1)   # round 1 / 10


def test_resources_obs_after_action_spent():
    res = {"action": 0, "bonus_action": 1, "movement": 0.0}
    out = resources_obs(res, round_number=5)
    assert out[0] == 0.0
    assert out[1] == 1.0
    assert out[2] == 0.0
    assert out[3] == pytest.approx(0.5)


def test_partition_entities_excludes_self_and_dead():
    from trpg.rl.obs import partition_entities
    ws = _build_1v1_world()
    allies, enemies = partition_entities(ws, "a")
    assert "a" not in allies and "a" not in enemies
    assert allies == []
    assert enemies == ["b"]


def test_partition_entities_sorts_enemies_by_distance():
    from trpg.rl.obs import partition_entities
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.vec2 import Vec2
    a = Character(name="a", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14), hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    close = Character(name="c", race="哥布林", class_="戰士", level=1,
                      stats=Stats(STR=10), hp=10, max_hp=10, ac=12,
                      weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    far = Character(name="f", race="哥布林", class_="戰士", level=1,
                    stats=Stats(STR=10), hp=10, max_hp=10, ac=12,
                    weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "close": close, "far": far},
                    scene="test", pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a","close","far"], round_number=1)
    setup_combat_positions(ws, ws.combat)
    a.position = Vec2(5.0, 15.0)
    close.position = Vec2(10.0, 15.0)
    far.position = Vec2(20.0, 15.0)
    _, enemies = partition_entities(ws, "a")
    assert enemies == ["close", "far"]


def test_build_obs_returns_all_5_keys():
    ws = _build_1v1_world()
    res = {"action": 1, "bonus_action": 1, "movement": 9.0}
    obs = build_obs(ws, "a", res)
    for k in OBS_KEYS:
        assert k in obs
    assert obs["skills"].shape == (N_SKILL_SLOTS, SKILL_FEATURE_DIM)
    assert obs["skill_mask"].shape == (N_SKILL_SLOTS,)
    assert obs["entities"].shape == (N_ENTITY_SLOTS, ENTITY_DIM)
    assert obs["resources"].shape == (4,)
    assert obs["terrain"].shape == (N_GRID, N_GRID)
