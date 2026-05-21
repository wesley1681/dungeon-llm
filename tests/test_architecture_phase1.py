import pytest
from trpg.engine.abilities import CLASS_ABILITIES, ClassAbility
from trpg.engine.skill import SkillFeatures, TargetType, SaveStat
from trpg.engine.character import Character, Stats


def test_class_ability_has_min_level():
    ab = CLASS_ABILITIES["second_wind"]
    assert hasattr(ab, "min_level")
    assert ab.min_level == 1


def test_class_ability_has_archetype_id():
    ab = CLASS_ABILITIES["trip_attack"]
    assert hasattr(ab, "archetype_id")
    assert ab.archetype_id == "battle_master"


def test_action_surge_min_level_2():
    assert CLASS_ABILITIES["action_surge"].min_level == 2


def test_base_class_abilities_have_empty_archetype():
    for sid in ("second_wind", "magic_missile", "cure_wounds", "rage"):
        assert CLASS_ABILITIES[sid].archetype_id == ""


def _make_wizard(level: int = 3, int_score: int = 14) -> Character:
    c = Character(
        name="W", race="", class_="法師", level=level,
        stats=Stats(INT=int_score), hp=20, max_hp=20, ac=12,
        spellcasting_ability="INT", spell_slots={1: 4, 2: 3},
        is_npc=False,
    )
    c.ability_uses["hold_person"] = 2
    return c


def test_materialize_save_dc_from_caster_stats():
    # INT 14 → mod +2; proficiency L3 = +2; DC = 8+2+2 = 12
    char = _make_wizard(level=3, int_score=14)
    ab = CLASS_ABILITIES["hold_person"]
    mat = ab.features.materialize(char, "hold_person")
    assert mat.save_dc == 12.0


def test_materialize_save_dc_higher_level():
    # INT 16 → mod +3; proficiency L5 = +3; DC = 8+3+3 = 14
    char = _make_wizard(level=5, int_score=16)
    ab = CLASS_ABILITIES["hold_person"]
    mat = ab.features.materialize(char, "hold_person")
    assert mat.save_dc == 14.0


def test_materialize_remaining_uses_unlimited_ability():
    # hold_person max_uses == 0 (spell, unlimited) → field unchanged
    char = _make_wizard()
    char.ability_uses["hold_person"] = 1
    ab = CLASS_ABILITIES["hold_person"]
    mat = ab.features.materialize(char, "hold_person")
    assert mat.remaining_uses == ab.features.remaining_uses


def test_materialize_cantrip_scales_at_level_5():
    char = _make_wizard(level=5)
    ab = CLASS_ABILITIES["sacred_flame"]
    mat = ab.features.materialize(char, "sacred_flame")
    # expected_damage should double at level 5
    assert mat.expected_damage == pytest.approx(ab.features.expected_damage * 2, rel=0.01)


def test_materialize_cantrip_no_scale_below_5():
    char = _make_wizard(level=3)
    ab = CLASS_ABILITIES["sacred_flame"]
    mat = ab.features.materialize(char, "sacred_flame")
    assert mat.expected_damage == pytest.approx(ab.features.expected_damage, rel=0.01)


def test_materialize_returns_copy_not_original():
    char = _make_wizard(level=5, int_score=18)
    ab = CLASS_ABILITIES["hold_person"]
    original_dc = ab.features.save_dc
    _ = ab.features.materialize(char, "hold_person")
    assert ab.features.save_dc == original_dc  # original unchanged


def _make_fighter(level: int = 3) -> Character:
    from trpg.engine.items import WEAPON_DEFS
    return Character(
        name="F", race="", class_="戰士", level=level,
        stats=Stats(STR=14), hp=30, max_hp=30, ac=16,
        weapons=[WEAPON_DEFS["長劍"]],
        is_npc=False,
    )


def test_second_wind_dice_scales_with_level():
    char = _make_fighter(level=5)
    ab = CLASS_ABILITIES["second_wind"]
    action = ab.builder("f", None, None, char=char)
    assert action["dice"] == "1d10+5"


def test_second_wind_dice_level_3():
    char = _make_fighter(level=3)
    ab = CLASS_ABILITIES["second_wind"]
    action = ab.builder("f", None, None, char=char)
    assert action["dice"] == "1d10+3"


def test_builder_without_char_uses_fallback():
    ab = CLASS_ABILITIES["second_wind"]
    action = ab.builder("f", None, None)
    assert "dice" in action
    assert "1d10" in action["dice"]


from unittest.mock import patch
from trpg.engine.world_state import WorldState
from trpg.engine.combat import execute_action
from trpg.engine.vec2 import Vec2
from trpg.engine.items import WEAPON_DEFS


def _combat_world(attacker_attacks: int = 1):
    from trpg.engine.character import CombatState
    A = Character(name="A", race="", class_="", level=5,
                  stats=Stats(STR=14), hp=30, max_hp=30, ac=10, is_npc=False,
                  weapons=[WEAPON_DEFS["長劍"]])
    A.attacks_per_action = attacker_attacks
    B = Character(name="B", race="", class_="", level=1,
                  stats=Stats(), hp=20, max_hp=20, ac=10, is_npc=True, attitude=0)
    A.position = Vec2(5, 5)
    B.position = Vec2(6, 5)
    ws = WorldState(characters={"a": A, "b": B}, scene="", dungeon_map=None)
    ws.party_ids = ["a"]
    ws.combat = CombatState(initiative_order=["a", "b"])
    return ws, A, B


def test_single_attack_returns_attack_type():
    ws, A, B = _combat_world(attacker_attacks=1)
    with patch("trpg.engine.combat.roll_d20", return_value=20):
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "a", "target": "b",
             "weapon": "長劍", "consumes": ["action"]}, ws
        )
    assert res["type"] == "ATTACK"


def test_extra_attack_returns_multi_attack_type():
    ws, A, B = _combat_world(attacker_attacks=2)
    B.hp = 50  # ensure target survives both hits
    B.max_hp = 50
    with patch("trpg.engine.combat.roll_d20", return_value=20), \
         patch("trpg.engine.combat.roll", return_value=4):
        res = execute_action(
            {"type": "ATTACK", "skill_id": "test_handwritten", "attacker": "a", "target": "b",
             "weapon": "長劍", "consumes": ["action"]}, ws
        )
    assert res["type"] == "MULTI_ATTACK"
    assert len(res["attacks"]) == 2


def test_extra_attack_character_field_default_is_1():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.attacks_per_action == 1


def test_crit_range_default_is_20():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.crit_range == 20


def _cantrip_world(caster_level: int):
    from trpg.engine.character import CombatState
    caster = Character(
        name="C", race="", class_="牧師", level=caster_level,
        stats=Stats(INT=14), hp=20, max_hp=20, ac=12, is_npc=False,
        spellcasting_ability="INT", spell_slots={},
    )
    target = Character(
        name="T", race="", class_="", level=1,
        stats=Stats(DEX=4), hp=50, max_hp=50, ac=10,
        is_npc=True, attitude=0,
    )
    caster.position = Vec2(0, 0)
    target.position = Vec2(1, 0)
    ws = WorldState(characters={"c": caster, "t": target}, scene="", dungeon_map=None)
    ws.party_ids = ["c"]
    ws.combat = CombatState(initiative_order=["c", "t"])
    return ws, caster, target


def test_cantrip_uses_1d8_at_level_3():
    ws, caster, target = _cantrip_world(caster_level=3)
    # roll side_effect: save d20 roll (1 → fail), then 1 damage die
    with patch("trpg.engine.combat.roll", side_effect=[1, 8]):
        res = execute_action(
            {"type": "SPELL", "skill_id": "test_handwritten", "caster": "c", "spell_name": "神聖光輝", "target": "t"}, ws
        )
    assert res["target_results"][0]["damage"] == 8


def test_cantrip_uses_2d8_at_level_5():
    ws, caster, target = _cantrip_world(caster_level=5)
    # save d20 (1 → fail), then 2 damage dice each returning 4 = 8 total
    with patch("trpg.engine.combat.roll", side_effect=[1, 4, 4]):
        res = execute_action(
            {"type": "SPELL", "skill_id": "test_handwritten", "caster": "c", "spell_name": "神聖光輝", "target": "t"}, ws
        )
    assert res["target_results"][0]["damage"] == 8


def test_cantrip_uses_3d8_at_level_11():
    ws, caster, target = _cantrip_world(caster_level=11)
    # save d20 (1 → fail), then 3 dice each returning 3 = 9 total
    with patch("trpg.engine.combat.roll", side_effect=[1, 3, 3, 3]):
        res = execute_action(
            {"type": "SPELL", "skill_id": "test_handwritten", "caster": "c", "spell_name": "神聖光輝", "target": "t"}, ws
        )
    assert res["target_results"][0]["damage"] == 9
