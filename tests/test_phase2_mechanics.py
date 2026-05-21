import pytest
from unittest.mock import patch
from trpg.engine.character import Character, Stats, CombatState, rest_character
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.combat import execute_action


def _two_char_world(attacker_sneak: str = "", target_ac: int = 10):
    A = Character(name="A", race="", class_="盜賊", level=3,
                  stats=Stats(DEX=16), hp=20, max_hp=20, ac=12, is_npc=False,
                  weapons=[WEAPON_DEFS["短劍"]])
    A.sneak_attack_dice = attacker_sneak
    B = Character(name="B", race="", class_="", level=1,
                  stats=Stats(), hp=30, max_hp=30, ac=target_ac,
                  is_npc=True, attitude=0)
    A.position = Vec2(5, 5)
    B.position = Vec2(6, 5)
    ws = WorldState(characters={"a": A, "b": B}, scene="", dungeon_map=None)
    ws.party_ids = ["a"]
    ws.combat = CombatState(initiative_order=["a", "b"])
    return ws, A, B


def _add_ally_adjacent(ws, target_pos):
    ally = Character(name="C", race="", class_="", level=1,
                     stats=Stats(), hp=20, max_hp=20, ac=12, is_npc=False)
    ally.position = Vec2(target_pos.x + 1.0, target_pos.y)
    ws.characters["c"] = ally
    ws.party_ids.append("c")


def test_sneak_attack_default_empty():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.sneak_attack_dice == ""


def test_sneak_attack_fires_when_ally_adjacent_to_target():
    ws, A, B = _two_char_world(attacker_sneak="2d6")
    _add_ally_adjacent(ws, B.position)
    B.hp = 100; B.max_hp = 100
    with patch("trpg.engine.combat.roll_d20", return_value=20), \
         patch("trpg.engine.combat.roll", side_effect=[4, 4, 3, 3]):
        # roll calls: d20=20 triggers crit → 2× weapon 1d6 (4,4), then 2× 1d6 sneak (3,3)
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "a", "target": "b",
             "weapon": "短劍", "consumes": ["action"]}, ws
        )
    assert res["hit"] is True
    assert res.get("sneak_attack_damage", 0) > 0


def test_sneak_attack_does_not_fire_without_condition():
    ws, A, B = _two_char_world(attacker_sneak="2d6")
    # No ally adjacent, no disadvantage → no sneak
    with patch("trpg.engine.combat.roll_d20", return_value=20), \
         patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "a", "target": "b",
             "weapon": "短劍", "consumes": ["action"]}, ws
        )
    assert res.get("sneak_attack_damage", 0) == 0


def test_sneak_attack_does_not_fire_without_sneak_dice():
    ws, A, B = _two_char_world(attacker_sneak="")
    _add_ally_adjacent(ws, B.position)
    with patch("trpg.engine.combat.roll_d20", return_value=20), \
         patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "a", "target": "b",
             "weapon": "短劍", "consumes": ["action"]}, ws
        )
    assert res.get("sneak_attack_damage", 0) == 0


def test_channel_divinity_uses_tracked_in_ability_uses():
    c = Character(name="x", race="", class_="牧師", level=2,
                  stats=Stats(), hp=20, max_hp=20, ac=12, is_npc=False)
    c.ability_uses["channel_divinity"] = 1
    assert c.ability_uses["channel_divinity"] == 1


def test_channel_divinity_resets_on_short_rest():
    c = Character(name="x", race="", class_="牧師", level=2,
                  stats=Stats(), hp=20, max_hp=20, ac=12, is_npc=False)
    c.known_abilities = ["channel_divinity"]
    c.ability_uses["channel_divinity"] = 0
    rest_character(c, "short")
    assert c.ability_uses.get("channel_divinity") == 1


def test_lay_on_hands_pool_default_zero():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.lay_on_hands_pool == 0


def _paladin_world():
    from trpg.engine.character import CombatState
    pal = Character(name="P", race="", class_="聖騎士", level=3,
                    stats=Stats(STR=16), hp=25, max_hp=25, ac=18, is_npc=False,
                    weapons=[WEAPON_DEFS["長劍"]])
    pal.lay_on_hands_pool = 15  # 5 × level 3
    ally = Character(name="A", race="", class_="", level=1,
                     stats=Stats(), hp=5, max_hp=20, ac=12, is_npc=False)
    pal.position = Vec2(5, 5)
    ally.position = Vec2(5.5, 5)
    ws = WorldState(characters={"p": pal, "a": ally}, scene="", dungeon_map=None)
    ws.party_ids = ["p", "a"]
    ws.combat = CombatState(initiative_order=["p", "a"])
    return ws, pal, ally


def test_lay_on_hands_heals_target_from_pool():
    ws, pal, ally = _paladin_world()
    res = execute_action(
        {"type": "LAY_ON_HANDS", "skill_id": "test_handwritten", "caster": "p", "target": "a",
         "amount": 10, "consumes": ["action"]}, ws
    )
    assert res["type"] == "LAY_ON_HANDS"
    assert ally.hp == 15
    assert pal.lay_on_hands_pool == 5  # 15 - 10 = 5


def test_lay_on_hands_cannot_exceed_pool():
    ws, pal, ally = _paladin_world()
    res = execute_action(
        {"type": "LAY_ON_HANDS", "skill_id": "test_handwritten", "caster": "p", "target": "a",
         "amount": 20, "consumes": ["action"]}, ws
    )
    assert res["type"] == "ERROR"
    assert "不足" in res["message"]


from trpg.engine.status import Evasion, HuntersMark


def test_evasion_status_exists():
    e = Evasion(applied_round=1)
    assert e.name == "evasion"


def test_evasion_registered_in_modifier_classes():
    from trpg.engine.status import MODIFIER_CLASSES
    assert "evasion" in MODIFIER_CLASSES


def test_sculpt_spells_default_false():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.sculpt_spells is False


def test_sculpt_spells_excludes_allies_from_aoe():
    caster = Character(name="S", race="", class_="法師", level=7,
                       stats=Stats(INT=14), hp=20, max_hp=20, ac=12, is_npc=False,
                       spellcasting_ability="INT",
                       spell_slots={3: 2})
    caster.sculpt_spells = True
    ally = Character(name="R", race="", class_="", level=7,
                     stats=Stats(DEX=18), hp=40, max_hp=40, ac=14, is_npc=False)
    enemy = Character(name="E", race="", class_="", level=1,
                      stats=Stats(DEX=8), hp=30, max_hp=30, ac=10,
                      is_npc=True, attitude=0)
    caster.position = Vec2(0, 0)
    ally.position = Vec2(10, 10)
    enemy.position = Vec2(10, 10)
    ws = WorldState(characters={"s": caster, "r": ally, "e": enemy},
                    scene="", dungeon_map=None)
    ws.party_ids = ["s", "r"]
    ws.combat = CombatState(initiative_order=["s", "r", "e"])
    ally_hp_before = ally.hp
    with patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action({
            "type": "SPELL", "skill_id": "test_handwritten", "caster": "s", "spell_name": "火球術",
            "target_position": [10.0, 10.0], "consumes": ["action"]
        }, ws)
    assert ally.hp == ally_hp_before, "sculpt_spells should protect ally"
    assert enemy.hp < 30, "enemy should still take damage"


# ── Task 5: Aura of Protection ────────────────────────────────────────────────

def test_aura_of_protection_default_zero():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.aura_of_protection_bonus == 0


def _aura_world():
    from trpg.engine.character import CombatState
    paladin = Character(name="P", race="", class_="聖騎士", level=6,
                        stats=Stats(CHA=16), hp=40, max_hp=40, ac=18, is_npc=False)
    paladin.aura_of_protection_bonus = 3  # CHA 16 → mod +3
    ally = Character(name="A", race="", class_="", level=1,
                     stats=Stats(CON=10), hp=20, max_hp=20, ac=12, is_npc=False)
    paladin.position = Vec2(5, 5)
    ally.position = Vec2(5.5, 5)  # 0.5m — within 3m aura
    ws = WorldState(characters={"p": paladin, "a": ally}, scene="", dungeon_map=None)
    ws.party_ids = ["p", "a"]
    ws.combat = CombatState(initiative_order=["p", "a"])
    return ws, paladin, ally


def test_aura_adds_cha_mod_to_ally_saves():
    ws, paladin, ally = _aura_world()
    from trpg.engine.combat import make_saving_throw
    with patch("trpg.engine.combat.roll", return_value=10):
        # CON mod=0, no prof. Without aura: total=10. With aura (+3): total=13.
        success, total = make_saving_throw(ally, "CON", 12, world_state=ws)
    assert total == 13


def test_aura_does_not_apply_when_paladin_out_of_range():
    ws, paladin, ally = _aura_world()
    paladin.position = Vec2(20, 20)  # far away
    from trpg.engine.combat import make_saving_throw
    with patch("trpg.engine.combat.roll", return_value=10):
        success, total = make_saving_throw(ally, "CON", 12, world_state=ws)
    assert total == 10


# ── Task 6: Hunter's Mark ─────────────────────────────────────────────────────

def test_hunters_mark_status_exists():
    hm = HuntersMark(source_id="ranger1", applied_round=1)
    assert hm.name == "hunters_mark"
    assert hm.source_id == "ranger1"


def _hunters_world():
    from trpg.engine.character import CombatState
    ranger = Character(name="R", race="", class_="遊俠", level=3,
                       stats=Stats(STR=14), hp=25, max_hp=25, ac=14, is_npc=False,
                       weapons=[WEAPON_DEFS["長劍"]])
    quarry = Character(name="Q", race="", class_="", level=1,
                       stats=Stats(), hp=100, max_hp=100, ac=10,
                       is_npc=True, attitude=0)
    ranger.position = Vec2(5, 5)
    quarry.position = Vec2(6, 5)
    ws = WorldState(characters={"r": ranger, "q": quarry}, scene="", dungeon_map=None)
    ws.party_ids = ["r"]
    ws.combat = CombatState(initiative_order=["r", "q"])
    # Mark the quarry
    quarry.add_status(HuntersMark(source_id="r", applied_round=1))
    ranger.concentrating_on = "hunters_mark"
    return ws, ranger, quarry


def test_hunters_mark_adds_1d6_on_hit():
    ws, ranger, quarry = _hunters_world()
    with patch("trpg.engine.combat.roll_d20", return_value=15), \
         patch("trpg.engine.combat.roll", side_effect=[5, 4]):
        # roll calls: weapon 1d8 (5), then 1d6 hunter's mark (4)
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "r", "target": "q",
             "weapon": "長劍", "consumes": ["action"]}, ws
        )
    assert res.get("hunters_mark_damage", 0) == 4


def test_hunters_mark_does_not_fire_on_unmarked_target():
    ws, ranger, quarry = _hunters_world()
    other = Character(name="O", race="", class_="", level=1,
                      stats=Stats(), hp=100, max_hp=100, ac=10,
                      is_npc=True, attitude=0)
    other.position = Vec2(6.5, 5)
    ws.characters["o"] = other
    with patch("trpg.engine.combat.roll_d20", return_value=15), \
         patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "r", "target": "o",
             "weapon": "長劍", "consumes": ["action"]}, ws
        )
    assert res.get("hunters_mark_damage", 0) == 0


# ── Task 7: Portent (Divination Wizard) ──────────────────────────────────────

def test_portent_dice_default_empty():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.portent_dice == []
    assert c.pending_portent is None


def _portent_wizard():
    from trpg.engine.character import CombatState
    wiz = Character(name="W", race="", class_="法師", level=2,
                    stats=Stats(INT=16), hp=12, max_hp=12, ac=12, is_npc=False,
                    spellcasting_ability="INT", spell_slots={1: 4, 2: 3})
    goblin = Character(name="G", race="", class_="", level=1,
                       stats=Stats(WIS=8), hp=7, max_hp=7, ac=13,
                       is_npc=True, attitude=0)
    wiz.position = Vec2(0, 0)
    goblin.position = Vec2(5, 0)
    ws = WorldState(characters={"w": wiz, "g": goblin}, scene="", dungeon_map=None)
    ws.party_ids = ["w"]
    ws.combat = CombatState(initiative_order=["w", "g"])
    return ws, wiz, goblin


def test_portent_substitutes_save_roll():
    """Portent die 1 on goblin → goblin auto-fails DC 13 hold_person save."""
    ws, wiz, goblin = _portent_wizard()
    # pending_portent is set on the target (goblin), not the wizard
    goblin.pending_portent = 1  # override: goblin rolls 1 on its next save
    # WIS mod = -1, DC = 8+2+3=13. Save total = 1+(-1) = 0 < 13 → fail → paralyzed
    res = execute_action(
        {"type": "SPELL", "skill_id": "test_handwritten", "caster": "w", "spell_name": "定身術",
         "target": "g", "consumes": ["action"]}, ws
    )
    assert goblin.has_status("paralyzed")
    assert goblin.pending_portent is None   # cleared after use


def test_portent_pending_portent_clears_after_use():
    ws, wiz, goblin = _portent_wizard()
    goblin.pending_portent = 19  # high value → goblin succeeds most saves
    with patch("trpg.engine.combat.roll", return_value=5):
        execute_action(
            {"type": "SPELL", "skill_id": "test_handwritten", "caster": "w", "spell_name": "定身術",
             "target": "g", "consumes": ["action"]}, ws
        )
    assert goblin.pending_portent is None


def test_portent_action_applies_die_to_target():
    ws, wiz, goblin = _portent_wizard()
    wiz.portent_dice = [3, 18]
    res = execute_action(
        {"type": "PORTENT", "skill_id": "test_handwritten", "caster": "w", "target": "g", "die_value": 3}, ws
    )
    assert res["type"] == "PORTENT"
    assert goblin.pending_portent == 3
    assert 3 not in wiz.portent_dice   # die consumed from caster's list


# ── Task 8: Counterspell reaction ────────────────────────────────────────────

def _counterspell_world():
    from trpg.engine.character import CombatState
    enemy_caster = Character(name="EC", race="", class_="法師", level=5,
                             stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
                             is_npc=True, attitude=0,
                             spellcasting_ability="INT",
                             spell_slots={3: 1})
    defender = Character(name="D", race="", class_="法師", level=5,
                         stats=Stats(INT=14), hp=20, max_hp=20, ac=12, is_npc=False,
                         spellcasting_ability="INT",
                         spell_slots={3: 2},
                         reactions=["counterspell"])
    party_member = Character(name="PM", race="", class_="", level=1,
                             stats=Stats(), hp=20, max_hp=20, ac=12, is_npc=False)
    enemy_caster.position = Vec2(0, 0)
    defender.position = Vec2(5, 0)
    party_member.position = Vec2(5.5, 0)
    ws = WorldState(characters={"ec": enemy_caster, "d": defender, "pm": party_member},
                    scene="", dungeon_map=None)
    ws.party_ids = ["d", "pm"]
    ws.combat = CombatState(initiative_order=["ec", "d", "pm"])
    return ws, enemy_caster, defender


def test_counterspell_cancels_enemy_spell():
    ws, enemy_caster, defender = _counterspell_world()
    res = execute_action(
        {"type": "SPELL", "skill_id": "test_handwritten", "caster": "ec", "spell_name": "火球術",
         "target_position": [5.0, 0.0], "consumes": ["action"]}, ws
    )
    assert res["type"] == "COUNTERSPELLED"
    assert defender.reaction_used is True
    assert defender.spell_slots[3] == 1  # consumed one 3rd-level slot


def test_counterspell_does_not_fire_when_reaction_used():
    ws, enemy_caster, defender = _counterspell_world()
    defender.reaction_used = True
    with patch("trpg.engine.combat.roll", return_value=3):
        res = execute_action(
            {"type": "SPELL", "skill_id": "test_handwritten", "caster": "ec", "spell_name": "火球術",
             "target_position": [5.0, 0.0], "consumes": ["action"]}, ws
        )
    assert res["type"] == "SPELL"
