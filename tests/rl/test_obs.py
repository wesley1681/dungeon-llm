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
    ENEMY_SLOT_START,
    GRID_CELL_SIZE_M,
    N_ALLY_SLOTS,
    N_ENEMY_SLOTS,
    N_ENTITY_SLOTS,
    N_GRID,
    N_SKILL_SLOTS,
    N_V3_EXTRA,
    OBS_KEYS,
    build_obs,
    entities_obs,
    resources_obs,
    skills_obs,
    terrain_obs,
)


def test_schema_constants():
    assert N_SKILL_SLOTS == 20
    assert N_ENTITY_SLOTS == 1 + N_ALLY_SLOTS + N_ENEMY_SLOTS
    assert ENEMY_SLOT_START == 1 + N_ALLY_SLOTS
    from trpg.rl.obs import (N_RL_STATUS, N_V4_DESC, N_V5_TRAIT, N_V6_CIMMUN,
                             N_V7_ABILITY, ENT_V3_TAIL_START, ENT_DESC_START,
                             ENT_TRAIT_START, ENT_CIMMUN_START, ENT_ABILITY_START)
    # 7 base + 12 archetype multi-hot + N_RL_STATUS status multi-hot
    # + 1 concentrating + v3 tail (level, max_hp, ac, dying, death saves)
    # + v4 capability descriptor + v5 passive-trait descriptor
    # + v6 condition-immunity descriptor + v7 ability-modifier descriptor
    assert ENTITY_DIM == (7 + 12 + N_RL_STATUS + 1 + N_V3_EXTRA + N_V4_DESC
                          + N_V5_TRAIT + N_V6_CIMMUN + N_V7_ABILITY)
    assert N_V7_ABILITY == 6
    assert ENT_CIMMUN_START == ENT_TRAIT_START + N_V5_TRAIT
    assert ENT_ABILITY_START == ENT_CIMMUN_START + N_V6_CIMMUN
    assert ENT_V3_TAIL_START == 7 + 12 + N_RL_STATUS + 1
    assert ENT_DESC_START == ENT_V3_TAIL_START + N_V3_EXTRA
    assert ENT_TRAIT_START == ENT_DESC_START + N_V4_DESC
    assert N_GRID == 30
    assert GRID_CELL_SIZE_M == 1.0
    assert BATTLEFIELD_SIZE_M == 30.0
    assert set(OBS_KEYS) == {"skills", "skill_mask",
                             "entity_skills", "entity_skill_mask",  # obs v8
                             "entities", "resources",
                             "terrain", "entity_grid", "distance_grid",
                             "los_grid", "reach_grid", "threat_grid",
                             "end_features", "decision_context"}


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
    # (0..3m, 0..3m) covers cells (0..2, 0..2) at 1.0m/cell
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
    # enemy rows start at ENEMY_SLOT_START, first is closest
    e0 = ENEMY_SLOT_START
    assert obs[e0, 4] == 1.0   # is_enemy
    assert obs[e0, 6] == 0.0   # not self
    assert obs[e0, 5] == 1.0   # alive
    assert obs[e0, 0] == pytest.approx(1.0)


def test_entities_obs_padding_is_zero():
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    # no allies → all ally rows are padding
    for r in range(1, ENEMY_SLOT_START):
        assert np.all(obs[r] == 0.0)
    # only 1 enemy → remaining enemy rows are padding
    for r in range(ENEMY_SLOT_START + 1, N_ENTITY_SLOTS):
        assert np.all(obs[r] == 0.0)


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
    # first enemy slot should be "close" (distance 5), next "far" (distance 15)
    # We can verify via the dist_norm column (index 3)
    e0 = ENEMY_SLOT_START
    assert obs[e0, 3] < obs[e0 + 1, 3], "closer enemy should have smaller dist_norm"


def test_entities_obs_status_multi_hot():
    """Status multi-hot lights the bit matching the attached status."""
    from trpg.rl.obs import RL_STATUS_NAMES, N_ARCHETYPES
    ws = _build_1v1_world()
    ws.characters["b"].status_effects.append(Paralyzed())
    obs = entities_obs(ws, "a")
    status_start = 7 + N_ARCHETYPES
    para_idx = status_start + RL_STATUS_NAMES.index("paralyzed")
    # enemy "b" is at the first enemy slot; only paralyzed bit set in status block
    e0 = ENEMY_SLOT_START
    assert obs[e0, para_idx] == 1.0
    for i, name in enumerate(RL_STATUS_NAMES):
        if name != "paralyzed":
            assert obs[e0, status_start + i] == 0.0


def test_entities_obs_dead_chars_excluded():
    """Dead characters are not placed in slots."""
    ws = _build_1v1_world()
    ws.characters["b"].hp = 0
    obs = entities_obs(ws, "a")
    # first enemy slot should be all zeros since b (NPC) is dead at 0 HP
    assert np.all(obs[ENEMY_SLOT_START] == 0.0)


def test_entities_obs_v3_threat_features():
    """v3 tail: level / max_hp / ac normalised scalars at the row tail."""
    from trpg.rl.obs import LEVEL_NORM, MAXHP_NORM, AC_NORM, ENT_V3_TAIL_START, N_V3_EXTRA as _NV3
    ws = _build_1v1_world()
    obs = entities_obs(ws, "a")
    # self: L3, 24 max_hp, AC 14 (from _build_1v1_world)
    tail = ENT_V3_TAIL_START
    assert obs[0, tail + 0] == pytest.approx(3 / LEVEL_NORM)
    assert obs[0, tail + 1] == pytest.approx(24 / MAXHP_NORM)
    assert obs[0, tail + 2] == pytest.approx(14 / AC_NORM)
    # enemy: L1, 10 max_hp, AC 12 — a full-HP L1 and L3 are now distinguishable
    e0 = ENEMY_SLOT_START
    assert obs[e0, tail + 0] == pytest.approx(1 / LEVEL_NORM)
    assert obs[e0, tail + 1] == pytest.approx(10 / MAXHP_NORM)
    assert obs[e0, tail + 2] == pytest.approx(12 / AC_NORM)
    # nobody dying → dying / death-save features all zero
    assert np.all(obs[0, tail + 3:tail + _NV3] == 0.0)
    assert np.all(obs[e0, tail + 3:tail + _NV3] == 0.0)


def test_entities_obs_v3_dying_features():
    """A dying PC shows is_dying=1 and its death-save counters."""
    ws = _build_1v1_world()
    a = ws.characters["a"]
    a.hp = 0                      # PC at 0 HP → dying, not dead
    a.death_saves = {"successes": 1, "failures": 2}
    obs = entities_obs(ws, "a")
    from trpg.rl.obs import ENT_V3_TAIL_START
    tail = ENT_V3_TAIL_START
    assert obs[0, 5] == 0.0                       # is_alive = 0
    assert obs[0, tail + 3] == 1.0                # is_dying
    assert obs[0, tail + 4] == pytest.approx(1 / 3)
    assert obs[0, tail + 5] == pytest.approx(2 / 3)


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
