"""倒地/瀕死/救起機制（5e 死亡豁免）— 引擎、env、obs/mask、專家各層測試。

規則來源（5e PHB）：
  - PC 降到 0 HP → 瀕死（昏迷），每輪擲死亡豁免；3 失敗死亡、3 成功穩定、
    nat 20 回 1 HP、nat 1 算 2 次失敗。
  - 溢出傷害 ≥ 最大 HP → 即死。
  - 瀕死者受傷 = 1 次豁免失敗（暴擊 2 次）；近戰命中昏迷者自動暴擊；
    攻擊昏迷者有優勢；昏迷者 STR/DEX 豁免自動失敗。
  - 任意治療使瀕死者以治療量 HP 回到戰鬥（救起）。
"""
import pytest
from unittest.mock import patch

from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.combat import (
    execute_action, apply_damage, apply_heal, roll_death_save,
    make_saving_throw,
)


def _char(name, *, hp=20, max_hp=20, ac=10, is_npc=False, pos=(5, 5), **kw):
    c = Character(name=name, race="", class_="", level=3,
                  stats=Stats(), hp=hp, max_hp=max_hp, ac=ac,
                  is_npc=is_npc, **kw)
    c.position = Vec2(*pos)
    return c


def _world(chars: dict, party: list[str]):
    ws = WorldState(characters=chars, scene="", dungeon_map=None)
    ws.party_ids = list(party)
    ws.combat = CombatState(initiative_order=list(chars.keys()))
    return ws


# ── apply_damage：進入瀕死 / 即死 / 瀕死中受傷 ────────────────────────────────

def test_pc_drops_to_dying_npc_dies():
    pc = _char("pc", hp=5)
    npc = _char("npc", hp=5, is_npc=True)
    apply_damage(pc, 5)
    apply_damage(npc, 5)
    assert pc.is_dying() and not pc.is_dead()
    assert npc.is_dead() and not npc.is_dying()


def test_massive_overflow_damage_is_instant_death():
    pc = _char("pc", hp=5, max_hp=20)
    apply_damage(pc, 26)   # 溢出 21 ≥ max_hp 20 → 即死
    assert pc.is_dead()
    pc2 = _char("pc2", hp=5, max_hp=20)
    apply_damage(pc2, 24)  # 溢出 19 < 20 → 瀕死
    assert pc2.is_dying()


def test_damage_while_dying_adds_failures_crit_doubles():
    pc = _char("pc", hp=0)
    assert pc.is_dying()
    apply_damage(pc, 4)
    assert pc.death_saves["failures"] == 1
    apply_damage(pc, 4, crit=True)
    assert pc.death_saves["failures"] == 3
    assert pc.is_dead()


def test_damage_at_least_maxhp_while_dying_kills():
    pc = _char("pc", hp=0, max_hp=20)
    apply_damage(pc, 20)
    assert pc.is_dead()


def test_damage_breaks_stabilized():
    pc = _char("pc", hp=0)
    pc.death_saves = {"successes": 3, "failures": 0, "stable": True}
    apply_damage(pc, 4)
    assert "stable" not in pc.death_saves
    assert pc.death_saves["failures"] == 1


# ── roll_death_save ───────────────────────────────────────────────────────────

def test_death_save_nat20_revives_at_1hp():
    pc = _char("pc", hp=0)
    with patch("trpg.engine.combat.roll", return_value=20):
        r = roll_death_save(pc)
    assert r["outcome"] == "revived" and pc.hp == 1 and pc.is_alive()
    assert pc.death_saves == {"successes": 0, "failures": 0}


def test_death_save_nat1_double_failure_and_death():
    pc = _char("pc", hp=0)
    with patch("trpg.engine.combat.roll", return_value=1):
        assert roll_death_save(pc)["outcome"] == "failure"
        assert pc.death_saves["failures"] == 2
        assert roll_death_save(pc)["outcome"] == "dead"
    assert pc.is_dead()


def test_death_save_three_successes_stabilize_then_skip():
    pc = _char("pc", hp=0)
    with patch("trpg.engine.combat.roll", return_value=12):
        roll_death_save(pc); roll_death_save(pc)
        assert roll_death_save(pc)["outcome"] == "stable"
    assert pc.death_saves.get("stable") is True
    # 穩定後不再擲（不會累積失敗死掉）
    r = roll_death_save(pc)
    assert r["outcome"] == "skipped_stable"
    assert pc.is_dying() and not pc.is_dead()


def test_death_save_low_roll_failure():
    pc = _char("pc", hp=0)
    with patch("trpg.engine.combat.roll", return_value=9):
        assert roll_death_save(pc)["outcome"] == "failure"
    assert pc.death_saves["failures"] == 1


# ── 豁免：昏迷自動失敗 STR/DEX ────────────────────────────────────────────────

def test_dying_autofails_str_dex_saves_only():
    pc = _char("pc", hp=0)
    ok_dex, total = make_saving_throw(pc, "DEX", 10)
    assert ok_dex is False and total == -999
    ok_str, _ = make_saving_throw(pc, "STR", 10)
    assert ok_str is False
    with patch("trpg.engine.combat.roll", return_value=20):
        ok_con, _ = make_saving_throw(pc, "CON", 10)
    assert ok_con is True   # CON 不自動失敗


# ── ATTACK：可打瀕死者、優勢、近戰自動暴擊、屍體不可打 ───────────────────────

def _attack(ws, attacker="a", target="b"):
    return execute_action(
        {"type": "ATTACK", "skill_id": "t", "attacker": attacker,
         "target": target, "weapon": "短劍", "consumes": ["action"]}, ws)


def test_melee_attack_on_dying_autocrits_two_failures():
    A = _char("A", is_npc=True, pos=(5, 5), weapons=[WEAPON_DEFS["短劍"]])
    B = _char("B", hp=0, ac=10, pos=(6, 5))   # 瀕死 PC，1m 內
    ws = _world({"a": A, "b": B}, party=["b"])
    with patch("trpg.engine.combat.roll_d20", return_value=10), \
         patch("trpg.engine.combat.roll", return_value=3):
        res = _attack(ws)
    assert res["hit"] is True
    assert res.get("auto_crit") is True
    assert res["advantage_mode"] == "advantage"
    assert B.death_saves["failures"] == 2
    assert B.hp == 0


def test_attack_dead_target_rejected():
    A = _char("A", is_npc=True, weapons=[WEAPON_DEFS["短劍"]])
    B = _char("B", hp=0, pos=(6, 5))
    B.death_saves["failures"] = 3
    ws = _world({"a": A, "b": B}, party=["b"])
    res = _attack(ws)
    assert res["type"] == "ERROR" and "已死亡" in res["message"]


# ── HEAL / LAY_ON_HANDS：救起 ────────────────────────────────────────────────

def test_heal_revives_dying_resets_saves():
    C = _char("C", pos=(5, 5))
    B = _char("B", hp=0, pos=(6, 5))
    B.death_saves = {"successes": 1, "failures": 2}
    ws = _world({"c": C, "b": B}, party=["c", "b"])
    with patch("trpg.engine.combat.roll", return_value=7):
        res = execute_action({"type": "HEAL", "skill_id": "t", "caster": "c",
                              "target": "b", "dice": "1d8", "range_m": 1.5}, ws)
    assert res["type"] == "HEAL" and res["revived"] is True
    assert B.hp == 7 and B.is_alive()
    assert B.death_saves == {"successes": 0, "failures": 0}


def test_heal_dead_target_rejected():
    C = _char("C")
    B = _char("B", hp=0, pos=(6, 5))
    B.death_saves["failures"] = 3
    ws = _world({"c": C, "b": B}, party=["c", "b"])
    res = execute_action({"type": "HEAL", "skill_id": "t", "caster": "c",
                          "target": "b", "dice": "1d8", "range_m": 1.5}, ws)
    assert res["type"] == "ERROR" and "已死亡" in res["message"]


def test_lay_on_hands_revives_dying():
    P = _char("P", pos=(5, 5))
    P.lay_on_hands_pool = 15
    B = _char("B", hp=0, pos=(6, 5))
    ws = _world({"p": P, "b": B}, party=["p", "b"])
    res = execute_action({"type": "LAY_ON_HANDS", "skill_id": "t",
                          "caster": "p", "target": "b", "amount": 5}, ws)
    assert res["type"] == "LAY_ON_HANDS" and res["revived"] is True
    assert B.hp == 5 and B.is_alive()


# ── 偷襲鄰接判定改為攻擊者陣營（兄弟 bug 修正回歸） ──────────────────────────

def test_sneak_attack_not_triggered_by_targets_own_allies():
    # NPC 盜賊攻擊 PC；旁邊站的是「目標自己的隊友」——不該觸發偷襲。
    A = _char("A", is_npc=True, pos=(5, 5), weapons=[WEAPON_DEFS["短劍"]])
    A.sneak_attack_dice = "2d6"
    B = _char("B", hp=30, max_hp=30, ac=5, pos=(6, 5))
    ally_of_b = _char("C", pos=(7, 5))
    ws = _world({"a": A, "b": B, "c": ally_of_b}, party=["b", "c"])
    with patch("trpg.engine.combat.roll_d20", return_value=10), \
         patch("trpg.engine.combat.roll", return_value=3):
        res = _attack(ws)
    assert res["hit"] is True
    assert "sneak_attack_damage" not in res


# ── 專家政策：actor 相對陣營 + 撈倒地 ────────────────────────────────────────

def test_find_heal_target_is_actor_side_relative():
    from trpg.engine.combat_policy import _ClericBase
    pol = _ClericBase()
    # 對手側牧師：自己隊友殘血、敵方（party）也殘血 → 應選自己隊友
    cleric = _char("OC", is_npc=True, pos=(5, 5))
    own_ally = _char("OA", hp=4, max_hp=20, is_npc=True, pos=(6, 5))
    enemy_pc = _char("E", hp=2, max_hp=20, pos=(7, 5))
    ws = _world({"oc": cleric, "oa": own_ally, "e": enemy_pc}, party=["e"])
    assert pol._find_heal_target("oc", ws, threshold=0.5) == "oa"


def test_find_heal_target_prefers_dying_ally():
    from trpg.engine.combat_policy import _ClericBase
    pol = _ClericBase()
    cleric = _char("C", pos=(5, 5))
    wounded = _char("W", hp=4, max_hp=20, pos=(6, 5))
    dying = _char("D", hp=0, max_hp=20, pos=(7, 5))
    ws = _world({"c": cleric, "w": wounded, "d": dying}, party=["c", "w", "d"])
    assert pol._find_heal_target("c", ws, threshold=0.5) == "d"


def test_life_cleric_expert_revives_adjacent_dying_ally():
    from trpg.engine.combat_policy import make_archetype_policy
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    cleric = ARCHETYPE_FACTORIES["life"](level=5)
    cleric.position = Vec2(5, 5)
    ally = _char("Ally", hp=0, max_hp=30, pos=(6, 5))
    enemy = _char("E", is_npc=True, ac=10, pos=(15, 5),
                  weapons=[WEAPON_DEFS["短劍"]])
    enemy.attitude = 0
    ws = _world({"cl": cleric, "al": ally, "e": enemy}, party=["cl", "al"])
    pol = make_archetype_policy("life")
    dec = pol.decide("cl", cleric, ws,
                     {"action": 1, "bonus_action": 1, "movement": 9.0}, 1)
    assert dec.action is not None
    assert dec.action["type"] in ("HEAL", "LAY_ON_HANDS")
    assert dec.action["target"] == "al"


def test_life_cleric_expert_walks_toward_far_dying_ally():
    from trpg.engine.combat_policy import make_archetype_policy
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    cleric = ARCHETYPE_FACTORIES["life"](level=5)
    cleric.position = Vec2(5, 5)
    ally = _char("Ally", hp=0, max_hp=30, pos=(11, 5))   # 6m，超出觸碰
    enemy = _char("E", is_npc=True, ac=10, pos=(20, 5),
                  weapons=[WEAPON_DEFS["短劍"]])
    enemy.attitude = 0
    ws = _world({"cl": cleric, "al": ally, "e": enemy}, party=["cl", "al"])
    pol = make_archetype_policy("life")
    dec = pol.decide("cl", cleric, ws,
                     {"action": 1, "bonus_action": 1, "movement": 9.0}, 1)
    assert dec.action is not None and dec.action["type"] == "MOVE"


def test_devotion_paladin_picks_up_adjacent_dying_ally():
    from trpg.engine.combat_policy import make_archetype_policy
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    pal = ARCHETYPE_FACTORIES["devotion"](level=5)
    pal.position = Vec2(5, 5)
    ally = _char("Ally", hp=0, max_hp=30, pos=(6, 5))
    enemy = _char("E", is_npc=True, ac=10, pos=(6.5, 5),
                  weapons=[WEAPON_DEFS["短劍"]])
    enemy.attitude = 0
    ws = _world({"p": pal, "al": ally, "e": enemy}, party=["p", "al"])
    pol = make_archetype_policy("devotion")
    dec = pol.decide("p", pal, ws,
                     {"action": 1, "bonus_action": 1, "movement": 9.0}, 1)
    assert dec.action is not None
    assert dec.action["type"] == "LAY_ON_HANDS"
    assert dec.action["target"] == "al"


# ── obs / mask / env ─────────────────────────────────────────────────────────

def test_partition_includes_dying_excludes_dead():
    from trpg.rl.obs import partition_entities, entities_obs
    me = _char("Me", pos=(5, 5))
    dying_ally = _char("D", hp=0, pos=(6, 5))
    dead_ally = _char("X", hp=0, pos=(7, 5))
    dead_ally.death_saves["failures"] = 3
    npc = _char("N", is_npc=True, pos=(8, 5))
    ws = _world({"me": me, "d": dying_ally, "x": dead_ally, "n": npc},
                party=["me", "d", "x"])
    allies, enemies = partition_entities(ws, "me")
    assert "d" in allies and "x" not in allies and enemies == ["n"]
    rows = entities_obs(ws, "me")
    assert rows[1][0] == 0.0      # 瀕死隊友 hp_frac = 0
    assert rows[1][5] == 0.0      # is_alive 位 = 0
    assert rows[1].sum() > 0      # 但列非空（可被選取）


def test_entity_mask_allows_dying_ally_slot():
    torch = pytest.importorskip("torch")
    import numpy as np
    from trpg.rl.obs import entities_obs
    from trpg.rl.model import apply_entity_mask
    me = _char("Me", pos=(5, 5))
    dying_ally = _char("D", hp=0, pos=(6, 5))
    npc = _char("N", is_npc=True, pos=(8, 5))
    ws = _world({"me": me, "d": dying_ally, "n": npc}, party=["me", "d"])
    from trpg.rl.obs import N_ENTITY_SLOTS
    batched = {"entities": np.expand_dims(entities_obs(ws, "me"), 0)}
    logits = torch.zeros((1, N_ENTITY_SLOTS))
    out = apply_entity_mask(logits, batched)
    assert out[0, 1] > -1e8   # 瀕死隊友槽（slot 1）可選
    assert out[0, 2] < -1e8   # 空槽仍被遮
    assert out[0, 0] < -1e8   # self 仍被遮


def test_env_ticks_death_saves_for_dying_agent():
    from trpg.rl.env_v2 import CombatEnvV2
    env = CombatEnvV2(seed=7, n_agents=2, n_opps=1)
    env.reset(agent_archs=["champion", "life"], opp_archs=["champion"],
              level=5, layout="open")
    other = next(a for a in env.agent_ids if a != env.current_agent_id)
    c = env.ws.characters[other]
    c.hp = 0
    assert c.is_dying()
    for _ in range(6):
        _, _, term, trunc, _ = env.step([0, 0, 0])   # 連續結束回合
        if term or trunc:
            break
    saves_progressed = (c.death_saves != {"successes": 0, "failures": 0}
                        or c.hp >= 1)
    assert saves_progressed
