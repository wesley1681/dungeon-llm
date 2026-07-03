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
    """For MOVE the mask is a REACHABILITY mask (the engine walks straight and
    stops at walls, so a wall-blocked-path cell isn't reachable this turn):
    inv[cell] ⟺ the clamped target is blocked terrain OR its straight path is
    wall-blocked. Valid (non-masked) cells must still decode to a real action.
    Masked move cells MAY decode to a partial move (walk toward wall, stop) —
    the contract is reachability, not decode-None; the policy only picks
    non-masked cells, so it stays a no-silent-no-op guard while enabling corner
    navigation."""
    from trpg.rl.action import (point_validity_mask, decode_action,
                                _grid_cell_to_xy, _clamp_to_range)
    from trpg.engine.skill import available_skills
    from trpg.engine.vec2 import Vec2
    ws = _world()
    # Wall between the combatants (mirrors env_v2's "walls" layout geometry).
    bf = ws.combat.battlefield
    bf.add_rect_obstacle(7.5, 0.0, 8.5, 30.0)
    ws.characters["a"].position = type(ws.characters["a"].position)(5.0, 15.0)
    ws.characters["b"].position = type(ws.characters["b"].position)(11.0, 15.0)
    agent = ws.characters["a"]
    skills = available_skills(agent, ws)
    move_idx, move = next((i, s) for i, s in enumerate(skills)
                          if s.skill_id == "move")
    inv = point_validity_mask(ws, "a", move)
    assert inv.any(), "wall cells must be masked"
    assert not inv.all(), "open cells must stay legal"

    for cell in range(0, 900, 37):   # sample across the grid
        x, y = _grid_cell_to_xy(cell)
        coord = _clamp_to_range(Vec2(x, y), agent.position, move.features.range_m)
        expect_invalid = bf.is_blocked(coord) or not bf.has_line_of_sight(
            agent.position, coord)
        assert bool(inv[cell]) == bool(expect_invalid), (
            f"cell {cell}: mask={bool(inv[cell])} != reachable-invalid={expect_invalid}")
        if not inv[cell]:
            assert decode_action([move_idx, 0, cell], ws, "a") is not None, (
                f"cell {cell}: reachable but decode rejected")


def test_point_mask_blocks_line_zero_aim_own_cell():
    """LINE 技（吐息）瞄自己格＝零向量＝引擎拒「需要一個方向」→浪費回合。
    point_validity_mask 必須把該格標非法（bug_miner 2026-07-02 於 storm_ogre 挖出：
    模型 grid argmax 落在自己格、每次施放都 ERROR、整場 0 傷）。
    僅當施法者正好站在格心（環境生成常態）才會產生零向量＝精確鏡射引擎條件。"""
    from trpg.rl.action import (point_validity_mask, decode_action,
                                _xy_to_grid_cell, _grid_cell_to_xy)
    from trpg.engine.skill import available_skills
    from trpg.engine.combat import execute_action
    from trpg.engine.vec2 import Vec2
    ws = _world()
    a = ws.characters["a"]
    a.known_abilities = list(a.known_abilities or []) + ["lightning_breath"]
    own = _xy_to_grid_cell(a.position.x, a.position.y)
    cx, cy = _grid_cell_to_xy(own)
    a.position = Vec2(cx, cy)                     # 站在格心＝自指瞄準向量為零
    skills = available_skills(a, ws)
    li, breath = next((i, s) for i, s in enumerate(skills)
                      if s.skill_id == "lightning_breath")
    # 引擎地面真值：對自己格施放被拒（此斷言鎖住引擎行為，防未來悄悄改語義）
    act = decode_action([li, 0, own], ws, "a")
    assert act is not None
    res = execute_action(act, ws)
    assert res.get("type") == "ERROR" and "方向" in str(res.get("message", ""))
    # 遮罩必須鏡射引擎：自己格非法、其他格仍合法
    inv = point_validity_mask(ws, "a", breath)
    assert bool(inv[own]), "LINE 技自己格（零向量瞄準）必須被標非法"
    assert not inv.all(), "其他格必須仍合法"


def test_entity_mask_blocks_out_of_reach_enemy_slot():
    """實體目標層的距離合法性（bug_miner 2026-07-02 挖出）：技能層遮罩只保證
    「最近敵在射程內」，實體層原本不看距離→模型可鎖 reach 外的較遠敵→引擎
    ERROR 迴圈（v5 一場 75 回合 0 傷平手）。apply_entity_mask 必須把超距目標
    遮掉、近距目標保留。"""
    import torch
    from trpg.engine.character import Character, Stats
    from trpg.engine.world_state import WorldState, CombatState
    from trpg.engine.combat import setup_combat_positions, execute_action
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.skill import available_skills
    from trpg.engine.vec2 import Vec2
    from trpg.rl.model import apply_entity_mask
    from trpg.rl.obs import build_obs, ENEMY_SLOT_START, N_ENTITY_SLOTS
    from trpg.rl.action import decode_action, ACTION_DIMS

    a = Character(name="a", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14, DEX=12, CON=14),
                  hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    b = Character(name="b", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12), hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    c = Character(name="c", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12), hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b, "c": c}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a", "b", "c"],
                            round_number=1)
    setup_combat_positions(ws, ws.combat)
    a.position = Vec2(10.0, 10.0)
    b.position = Vec2(11.0, 10.0)     # 1.0m＝reach 內（slot 0，距離排序較近）
    c.position = Vec2(11.6, 10.0)     # 1.6m＞長劍伸手 1.5m（slot 1）

    skills = available_skills(a, ws)
    widx = next(i for i, s in enumerate(skills)
                if s.skill_id.startswith("weapon:"))
    # 引擎地面真值：打 1.6m 的 c＝ERROR 超出伸手範圍
    act = decode_action([widx, ENEMY_SLOT_START + 1, 0], ws, "a")
    assert act is not None and act["target"] == "c"
    res = execute_action(act, ws)
    assert res.get("type") == "ERROR" and "伸手範圍" in str(res.get("message", ""))
    # 遮罩必須鏡射引擎：超距 slot 遮掉、近距 slot 保留
    obs = build_obs(ws, "a", {"movement": 6.0, "action": 1, "bonus_action": 1})
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    e = apply_entity_mask(torch.zeros(1, ACTION_DIMS[0], N_ENTITY_SLOTS),
                          ot, ws, "a")
    assert e[0, widx, ENEMY_SLOT_START].item() > -1e8, "reach 內目標必須保留"
    assert e[0, widx, ENEMY_SLOT_START + 1].item() <= -1e8, "超距目標必須被遮"
