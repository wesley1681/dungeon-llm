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
            {"type": "ATTACK", "attacker": "a", "target": "b",
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
            {"type": "ATTACK", "attacker": "a", "target": "b",
             "weapon": "短劍", "consumes": ["action"]}, ws
        )
    assert res.get("sneak_attack_damage", 0) == 0


def test_sneak_attack_does_not_fire_without_sneak_dice():
    ws, A, B = _two_char_world(attacker_sneak="")
    _add_ally_adjacent(ws, B.position)
    with patch("trpg.engine.combat.roll_d20", return_value=20), \
         patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action(
            {"type": "ATTACK", "attacker": "a", "target": "b",
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
