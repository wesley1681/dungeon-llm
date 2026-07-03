"""NvM 波守門三修（bug_miner seed666 n500 新空間挖出）：

A. 魅惑方向（engine）：`Charmed` docstring 是規格＝「受魅者不能攻擊施魅者」；
   實作反了（擋施魅者打受魅者、放行受魅者反打）——mine|666|0230 眼魔魅惑
   Evoker 後自己咬不了對方、7 次 dead_action。修 combat.py ATTACK 判定方向
   ＋apply_entity_mask 鏡射（受魅的 agent 不會把攻擊鎖在施魅者身上）。

B. 實體層免疫空砸（decision layer）：skill 層 null_dmg_idx 只擋「對所有敵
   全免」的技能；目標層沒有 sibling——mine|666|0065 shadow 反覆摸免疫的敵
   shadow、旁邊站著能打的 orc（0-16 輸掉）。修 apply_entity_mask per-(skill,
   slot)：純傷害技對倍率≈0 的目標遮掉（有非 null 合法目標時＝嚴格支配；
   全遮則還原＝不製造無合法目標）。

C. move 自格重選（decision layer）：mine|666|0136 wyvern+war 疊同格刷 50
   回合「move 到自己格」（位移 0＝引擎可證 null）。pick_action 網格分支：
   move 選中自格且存在其他合法格→遮自格由模型 logits 重選。
"""
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()                      # 怪物武器（生命吸取/鬼爪…）進 WEAPON_DEFS

from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2, Battlefield
from trpg.engine.world_state import WorldState
from trpg.engine.combat import execute_action
from trpg.engine.status import Charmed
from trpg.engine.skill import available_skills
from trpg.engine.items import WEAPON_DEFS
from trpg.rl.model import apply_entity_mask, pick_action
from trpg.rl.obs import build_obs, ENEMY_SLOT_START, N_GRID, GRID_CELL_SIZE_M


def _fighter(name, cid_pos, hp=60, **kw):
    c = Character(name=name, race="", class_="戰士", level=5,
                  stats=Stats(STR=16, DEX=12), hp=hp, max_hp=hp, ac=14,
                  is_npc=kw.pop("is_npc", True),
                  attitude=kw.pop("attitude", 0), **kw)
    c.weapons = [WEAPON_DEFS["長劍"]]
    c.position = Vec2(*cid_pos)
    return c


def _world(chars: dict, party):
    ws = WorldState(characters=chars, scene="", dungeon_map=None)
    ws.party_ids = list(party)
    ws.combat = CombatState(initiative_order=list(chars.keys()))
    ws.combat.battlefield = Battlefield()          # 全 NORMAL 開闊地
    return ws


def _atk(attacker_id, target_id):
    return {"type": "ATTACK", "attacker": attacker_id, "target": target_id,
            "weapon": "", "consumes": ["action"], "skill_id": "weapon:長劍"}


# ── A. 魅惑方向（engine） ────────────────────────────────────────────────────

def test_charmed_victim_cannot_attack_charmer():
    a = _fighter("受魅者", (5.0, 5.0), is_npc=False)
    b = _fighter("施魅者", (6.0, 5.0))
    a.add_status(Charmed(source_id="b"))
    ws = _world({"a": a, "b": b}, ["a"])
    res = execute_action(_atk("a", "b"), ws)
    assert isinstance(res, dict) and res.get("type") == "ERROR", (
        f"受魅者攻擊施魅者必須被拒（Charmed 規格），卻執行了: {res}")


def test_charmer_can_attack_charmed_victim():
    a = _fighter("受魅者", (5.0, 5.0), is_npc=False)
    b = _fighter("施魅者", (6.0, 5.0))
    a.add_status(Charmed(source_id="b"))
    ws = _world({"a": a, "b": b}, ["a"])
    res = execute_action(_atk("b", "a"), ws)
    assert isinstance(res, dict) and res.get("type") != "ERROR", (
        f"施魅者攻擊自己的受魅目標是合法的（0230 眼魔咬不了獵物＝方向反了）: {res}")


def test_charmed_can_still_attack_third_party():
    a = _fighter("受魅者", (5.0, 5.0), is_npc=False)
    b = _fighter("施魅者", (6.0, 5.0))
    c = _fighter("路人", (5.0, 6.0))
    a.add_status(Charmed(source_id="b"))
    ws = _world({"a": a, "b": b, "c": c}, ["a"])
    res = execute_action(_atk("a", "c"), ws)
    assert isinstance(res, dict) and res.get("type") != "ERROR", (
        f"魅惑只限制打施魅者，打第三方必須合法: {res}")


# ── 遮罩測試共用 ─────────────────────────────────────────────────────────────

_RES = {"action": 1, "bonus_action": 1, "movement": 9.0}


def _mask_inputs(ws, aid):
    ob = build_obs(ws, aid, _RES)          # 遮罩只讀 presence/self 位，盲化無關
    ot = {k: torch.from_numpy(np.asarray(v)).unsqueeze(0) for k, v in ob.items()}
    sks = available_skills(ws.characters[aid], ws)
    n_ent = ot["entities"].shape[1]
    e = torch.zeros((1, len(sks), n_ent))
    return ot, sks, e


def _weapon_row(sks):
    return next(i for i, s in enumerate(sks) if s.skill_id.startswith("weapon:"))


# ── A. 魅惑鏡射（mask） ──────────────────────────────────────────────────────

def test_entity_mask_blocks_charmer_slot_for_charmed_agent():
    a = _fighter("受魅者", (5.0, 5.0), is_npc=False)
    b = _fighter("施魅者", (6.0, 5.0))
    c = _fighter("另一敵", (5.0, 6.0))
    a.add_status(Charmed(source_id="b"))
    ws = _world({"a": a, "b": b, "c": c}, ["a"])
    ot, sks, e = _mask_inputs(ws, "a")
    out = apply_entity_mask(e, ot, ws, "a")
    i = _weapon_row(sks)
    # 敵 slot 順序＝partition_entities＝characters 插入序（b 先 c 後）
    assert out[0, i, ENEMY_SLOT_START].item() <= -1e8, (
        "受魅 agent 的攻擊列必須遮掉施魅者 slot（引擎會拒＝dead_action 接縫）")
    assert out[0, i, ENEMY_SLOT_START + 1].item() > -1e8, "第三方敵不得被遮"


def test_entity_mask_charm_full_row_reverts():
    # 只剩施魅者一個敵：全遮則還原（不製造無合法目標）
    a = _fighter("受魅者", (5.0, 5.0), is_npc=False)
    b = _fighter("施魅者", (6.0, 5.0))
    a.add_status(Charmed(source_id="b"))
    ws = _world({"a": a, "b": b}, ["a"])
    ot, sks, e = _mask_inputs(ws, "a")
    out = apply_entity_mask(e, ot, ws, "a")
    i = _weapon_row(sks)
    assert out[0, i, ENEMY_SLOT_START].item() > -1e8, (
        "唯一敵人是施魅者時該列必須還原（維持 decode/引擎後盾語義）")


# ── B. 實體層免疫空砸 ────────────────────────────────────────────────────────

def _necro_fighter(name, pos, **kw):
    c = _fighter(name, pos, **kw)
    c.weapons = [WEAPON_DEFS["暗影之觸"]] if "暗影之觸" in WEAPON_DEFS else c.weapons
    return c


def test_entity_mask_blocks_resist_null_target():
    a = _necro_fighter("死靈手", (5.0, 5.0), is_npc=False)
    imm = _fighter("免疫者", (6.0, 5.0))
    imm.damage_multipliers = {a.get_weapon("").damage_type: 0.0}
    soft = _fighter("可打者", (5.0, 6.0))
    ws = _world({"a": a, "imm": imm, "soft": soft}, ["a"])
    ot, sks, e = _mask_inputs(ws, "a")
    out = apply_entity_mask(e, ot, ws, "a")
    i = _weapon_row(sks)
    assert out[0, i, ENEMY_SLOT_START].item() <= -1e8, (
        "純傷害技對倍率 0 的目標＝null 攻擊，有可打目標在場時必須遮"
        "（0065：shadow 反覆摸免疫敵、旁邊站著 orc）")
    assert out[0, i, ENEMY_SLOT_START + 1].item() > -1e8, "可打目標不得被遮"


def test_resource_mask_blocks_null_skill_when_only_immune_reachable():
    """0150 家族（skill 層）：雙武器身體、當下唯一搆得到的敵人免疫其中一型——
    「黯蝕打免疫者=0」被「斬擊打同一人=正傷」嚴格支配，null 型必須被遮。
    dominance 只在「當下合法可執行」的選項間比（遠處的高 EV 目標不算替代）。"""
    from trpg.rl.model import apply_resource_mask
    a = _fighter("雙刀鬼", (5.0, 5.0), is_npc=False)
    a.weapons = [WEAPON_DEFS["生命吸取"], WEAPON_DEFS["鬼爪"]]
    imm = _fighter("免疫者", (6.0, 5.0))            # 貼臉
    imm.damage_multipliers = {"黯蝕": 0.0}
    far = _fighter("遠敵", (25.0, 25.0))            # reach 外＝不是合法替代
    ws = _world({"a": a, "imm": imm, "far": far}, ["a"])
    sks = available_skills(a, ws)
    s = torch.zeros((1, len(sks)))
    out = apply_resource_mask(s, dict(_RES), ws, "a")
    i_null = next(i for i, k in enumerate(sks) if k.skill_id == "weapon:生命吸取")
    i_ok = next(i for i, k in enumerate(sks) if k.skill_id == "weapon:鬼爪")
    assert out[0, i_null].item() <= -1e8, (
        "當下唯一合法目標免疫該型、同目標有正傷替代＝null 型必須遮（0150）")
    assert out[0, i_ok].item() > -1e8, "正傷武器不得被遮"


def test_resource_mask_keeps_monotype_swing_vs_immune():
    """單型 kit 面對免疫：永不製造「無合法輸出」——揮擊保留（既有守門鐵律）。"""
    from trpg.rl.model import apply_resource_mask
    a = _fighter("單刀鬼", (5.0, 5.0), is_npc=False)
    a.weapons = [WEAPON_DEFS["生命吸取"]]
    imm = _fighter("免疫者", (6.0, 5.0))
    imm.damage_multipliers = {"黯蝕": 0.0}
    ws = _world({"a": a, "imm": imm}, ["a"])
    sks = available_skills(a, ws)
    s = torch.zeros((1, len(sks)))
    out = apply_resource_mask(s, dict(_RES), ws, "a")
    i_null = next(i for i, k in enumerate(sks) if k.skill_id == "weapon:生命吸取")
    assert out[0, i_null].item() > -1e8, "單型 kit 的唯一揮擊不得被剝奪"


def test_entity_mask_resist_null_full_row_reverts():
    a = _necro_fighter("死靈手", (5.0, 5.0), is_npc=False)
    imm = _fighter("免疫者", (6.0, 5.0))
    imm.damage_multipliers = {a.get_weapon("").damage_type: 0.0}
    ws = _world({"a": a, "imm": imm}, ["a"])
    ot, sks, e = _mask_inputs(ws, "a")
    out = apply_entity_mask(e, ot, ws, "a")
    i = _weapon_row(sks)
    assert out[0, i, ENEMY_SLOT_START].item() > -1e8, (
        "全場只剩免疫敵＝該列全遮則還原（單型 kit 不許被剝奪唯一揮擊）")


# ── C. move 自格重選 ─────────────────────────────────────────────────────────

def _own_cell(ch):
    gx = max(0, min(N_GRID - 1, int(ch.position.x / GRID_CELL_SIZE_M)))
    gy = max(0, min(N_GRID - 1, int(ch.position.y / GRID_CELL_SIZE_M)))
    return gx * N_GRID + gy


def _move_pick(ws, a, attempts):
    a._outgoing_attempts = attempts
    sks = available_skills(a, ws)
    move_i = next(i for i, s in enumerate(sks) if s.skill_id == "move")
    el = torch.full((1,), -50.0)                      # 不 END
    s = torch.full((len(sks),), -50.0); s[move_i] = 50.0   # 強選 move
    e = torch.zeros((len(sks), 10))
    g = torch.full((len(sks), N_GRID * N_GRID), -10.0)
    g[move_i, _own_cell(a)] = 50.0                    # 模型最愛自格（0136 形態）
    return move_i, pick_action(el, s, e, g, ws=ws, agent_id="a")


def _cell_center(cell):
    gx, gy = cell // N_GRID, cell % N_GRID
    return (gx + 0.5) * GRID_CELL_SIZE_M, (gy + 0.5) * GRID_CELL_SIZE_M


def test_pick_action_breaks_own_cell_deadlock_toward_enemy():
    # 0136 形態：整場零攻擊嘗試＋敵在 reach+移動預算內＋move 選自格＝死鎖，
    # 必須往「離最近敵更近」的格重選（不是任意次愛格＝v1 亂走回歸的教訓）。
    a = _fighter("龜縮者", (5.0, 5.0), is_npc=False)
    b = _fighter("近敵", (7.6, 5.0))                   # 2.6m＝0136 同距
    ws = _world({"a": a, "b": b}, ["a"])
    move_i, act = _move_pick(ws, a, attempts=0)
    own = _own_cell(a)
    assert act[0] == move_i and act[2] != own, "死鎖態必須離開自格"
    cx, cy = _cell_center(act[2])
    d_new = ((cx - b.position.x) ** 2 + (cy - b.position.y) ** 2) ** 0.5
    assert d_new < a.position.distance_to(b.position) - 1e-3, (
        "重選必須嚴格向敵靠近（防 v1 的亂走/逃跑回歸）")


def test_pick_action_stand_ground_stays_legal_after_output():
    # 站樁＝合法戰術站姿（讓近戰走進 reach）：只要這場出過手，自格照選——
    # v1 全域遮自格實測 seed0 no_engage 0→60、受影響場敗率 58%，此測固化教訓。
    a = _fighter("站樁者", (5.0, 5.0), is_npc=False)
    b = _fighter("近敵", (7.6, 5.0))
    ws = _world({"a": a, "b": b}, ["a"])
    _, act = _move_pick(ws, a, attempts=1)
    assert act[2] == _own_cell(a), "出過手之後站樁必須保持合法"


def test_pick_action_stand_ground_when_enemy_unreachable():
    # 敵在 reach+移動預算外＝這回合怎麼走都打不到：不強迫接近（風箏/守點自由）
    a = _fighter("守點者", (5.0, 5.0), is_npc=False)
    b = _fighter("遠敵", (25.0, 25.0))
    ws = _world({"a": a, "b": b}, ["a"])
    _, act = _move_pick(ws, a, attempts=0)
    assert act[2] == _own_cell(a), "敵不可及時不得強迫移動"
