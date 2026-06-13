import numpy as np
import pytest
from trpg.engine.character import Character, Stats
from trpg.engine.world_state import WorldState, CombatState
from trpg.engine.combat import setup_combat_positions
from trpg.engine.items import WEAPON_DEFS
from trpg.rl.action import decode_action, ACTION_DIMS


def _world():
    a = Character(name="a", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14, DEX=12, CON=14),
                  hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    b = Character(name="b", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12),
                  hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a","b"], round_number=1)
    setup_combat_positions(ws, ws.combat)
    return ws


def test_action_dims_shape():
    from trpg.rl.obs import N_ENTITY_SLOTS
    assert ACTION_DIMS == (20, N_ENTITY_SLOTS, 900)


def test_decode_skill_0_returns_none_end_turn():
    ws = _world()
    out = decode_action([0, 0, 0], ws, "a")
    assert out is None   # end turn


def test_decode_skill_out_of_range_returns_none():
    ws = _world()
    # slot 19 is past available_skills() length → invalid
    out = decode_action([19, 0, 0], ws, "a")
    assert out is None


def test_decode_attack_targets_first_enemy_slot():
    # The first enemy slot (ENEMY_SLOT_START) is enemy_1.
    from trpg.engine.skill import available_skills
    from trpg.rl.obs import ENEMY_SLOT_START
    ws = _world()
    skills = available_skills(ws.characters["a"], ws)
    weapon_idx = next(
        i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:")
    )
    out = decode_action([weapon_idx, ENEMY_SLOT_START, 0], ws, "a")
    assert out is not None
    assert out["type"] == "ATTACK"
    assert out["attacker"] == "a"
    assert out["target"] == "b"


from trpg.rl.action import encode_action


def test_encode_end_turn():
    ws = _world()
    enc = encode_action(None, ws, "a")
    assert enc == (0, 0, 0)   # slot 0 = end skill


def test_encode_attack_action_roundtrip():
    ws = _world()
    from trpg.engine.skill import available_skills
    skills = available_skills(ws.characters["a"], ws)
    weapon_idx = next(i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:"))
    # Build the action the policy would emit
    weapon_name = ws.characters["a"].weapons[0].name
    action_dict = {
        "type": "ATTACK", "skill_id": f"weapon:{weapon_name}",
        "attacker": "a", "target": "b",
        "weapon": weapon_name,
        "consumes": ["action"],
    }
    enc = encode_action(action_dict, ws, "a")
    # skill_idx matches weapon attack, entity_idx = first enemy slot
    from trpg.rl.obs import ENEMY_SLOT_START
    assert enc[0] == weapon_idx
    assert enc[1] == ENEMY_SLOT_START


def test_encode_move_action():
    ws = _world()
    from trpg.engine.skill import available_skills
    skills = available_skills(ws.characters["a"], ws)
    move_idx = next(i for i, s in enumerate(skills) if s.skill_id == "move")
    action_dict = {
        "type": "MOVE", "skill_id": "move", "character": "a", "target": "b",
        "consumes": ["movement"],
    }
    enc = encode_action(action_dict, ws, "a")
    assert enc[0] == move_idx


def test_decode_move_skill_via_grid_cell():
    """MOVE skill with POINT target uses grid_cell, not entity_idx."""
    ws = _world()
    from trpg.engine.skill import available_skills
    skills = available_skills(ws.characters["a"], ws)
    # Find move skill (target_type might be SINGLE_ENEMY for "move to creature" or POINT for raw coord)
    move_skills = [(i, s) for i, s in enumerate(skills) if s.skill_id == "move"]
    if not move_skills:
        return  # No move skill available
    move_idx, move_sk = move_skills[0]
    # Decode action — should produce a valid action dict without TypeError
    out = decode_action([move_idx, 0, 100], ws, "a")
    # Either None (if skill_type expects entity but slot 0 is self → no move) or a MOVE action
    if out is not None:
        assert out["type"] == "MOVE"


def test_encode_action_roundtrip_via_grid():
    """Encode → decode round trip for POINT-target actions preserves grid cell."""
    ws = _world()
    from trpg.engine.skill import available_skills
    # Find a POINT-target skill if any; otherwise skip
    skills = available_skills(ws.characters["a"], ws)
    point_skills = [(i, s) for i, s in enumerate(skills)
                    if hasattr(s.features, "target_type")
                    and s.features.target_type.name == "POINT"]
    if not point_skills:
        return  # No POINT skill in this fixture
    # Otherwise verify grid encoding round-trips
    from trpg.rl.action import _xy_to_grid_cell, _grid_cell_to_xy
    for cell in [0, 50, 100, 200, 899]:
        x, y = _grid_cell_to_xy(cell)
        assert _xy_to_grid_cell(x, y) == cell


def test_point_validity_mask_open_field_all_valid():
    """No walls → every aim cell is legal (fast path)."""
    from trpg.rl.action import point_validity_mask
    from trpg.engine.skill import available_skills
    ws = _world()
    skills = available_skills(ws.characters["a"], ws)
    move = next(s for s in skills if s.skill_id == "move")
    inv = point_validity_mask(ws, "a", move)
    assert not inv.any()


def test_point_validity_mask_matches_decode_action():
    """The mask must mirror decode_action: every cell it marks INVALID is a
    cell decode_action rejects (None), and every valid cell decodes to a real
    action. Single-source-of-truth regression for the silent-no-op loop
    (28-round fireball-at-wall standoff)."""
    from trpg.rl.action import point_validity_mask, decode_action
    from trpg.engine.skill import available_skills
    ws = _world()
    # Wall between the combatants (mirrors env_v2's "walls" layout geometry).
    bf = ws.combat.battlefield
    bf.add_rect_obstacle(7.5, 0.0, 8.5, 30.0)
    ws.characters["a"].position = type(ws.characters["a"].position)(5.0, 15.0)
    ws.characters["b"].position = type(ws.characters["b"].position)(11.0, 15.0)
    skills = available_skills(ws.characters["a"], ws)
    move_idx, move = next((i, s) for i, s in enumerate(skills)
                          if s.skill_id == "move")
    inv = point_validity_mask(ws, "a", move)
    assert inv.any(), "wall cells must be masked"
    assert not inv.all(), "open cells must stay legal"
    #

    for cell in range(0, 900, 37):   # sample across the grid
        decoded = decode_action([move_idx, 0, cell], ws, "a")
        if inv[cell]:
            assert decoded is None, f"cell {cell}: masked but decodes"
        else:
            assert decoded is not None, f"cell {cell}: legal but rejected"
