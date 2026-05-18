import pytest
from trpg.engine.character import Character, Stats
from trpg.engine.abilities import CLASS_ABILITIES
from trpg.engine.skill import available_skills
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.vec2 import Vec2


def test_archetype_id_default_empty():
    c = Character(name="x", race="", class_="", level=1,
                  stats=Stats(), hp=10, max_hp=10, ac=10)
    assert c.archetype_id == ""


def test_archetype_id_can_be_set():
    c = Character(name="x", race="", class_="戰士", level=3,
                  stats=Stats(), hp=30, max_hp=30, ac=16)
    c.archetype_id = "battle_master"
    assert c.archetype_id == "battle_master"


def _make_cleric(wis: int = 16, level: int = 3) -> Character:
    return Character(
        name="C", race="", class_="牧師", level=level,
        stats=Stats(WIS=wis), hp=25, max_hp=25, ac=14, is_npc=False,
        spellcasting_ability="WIS", spell_slots={1: 4},
    )


def test_materialize_expected_healing_uses_actual_spell_mod():
    # WIS 16 → mod +3. Template already uses +3, so no change.
    char_wis16 = _make_cleric(wis=16)
    ab = CLASS_ABILITIES["cure_wounds"]
    mat16 = ab.features.materialize(char_wis16, "cure_wounds")
    assert mat16.expected_healing == pytest.approx(7.5, rel=0.01)

    # WIS 20 → mod +5. Delta = +5 - +3 = +2. expected_healing = 7.5 + 2 = 9.5.
    char_wis20 = _make_cleric(wis=20)
    mat20 = ab.features.materialize(char_wis20, "cure_wounds")
    assert mat20.expected_healing == pytest.approx(9.5, rel=0.01)

    # WIS 10 → mod 0. Delta = 0 - +3 = -3. expected_healing = 7.5 - 3 = 4.5.
    char_wis10 = _make_cleric(wis=10)
    mat10 = ab.features.materialize(char_wis10, "cure_wounds")
    assert mat10.expected_healing == pytest.approx(4.5, rel=0.01)


def _make_battle_master(level: int = 3) -> Character:
    c = Character(
        name="BM", race="", class_="戰士", level=level,
        stats=Stats(STR=16, DEX=12, CON=14, INT=10, WIS=10, CHA=10),
        hp=28, max_hp=28, ac=16, is_npc=False,
        weapons=[WEAPON_DEFS["長劍"]],
        known_abilities=["second_wind", "action_surge",
                         "trip_attack", "menacing_attack"],
    )
    c.archetype_id = "battle_master"
    c.position = Vec2(5, 5)
    return c


def test_available_skills_includes_class_abilities():
    char = _make_battle_master()
    skills = available_skills(char)
    skill_ids = [s.skill_id for s in skills]
    assert "menacing_attack" in skill_ids
    assert "trip_attack" in skill_ids
    assert "second_wind" in skill_ids    # archetype_id="" → always included
    assert "action_surge" in skill_ids   # archetype_id="" → always included


def test_available_skills_excludes_wrong_archetype():
    char = _make_battle_master()
    char.archetype_id = "champion"
    skills = available_skills(char)
    skill_ids = [s.skill_id for s in skills]
    assert "menacing_attack" not in skill_ids  # archetype="battle_master" ≠ "champion"
    assert "trip_attack" not in skill_ids
    assert "second_wind" in skill_ids           # archetype="" → always included


def test_available_skills_excludes_below_min_level():
    char = _make_battle_master(level=1)
    skills = available_skills(char)
    skill_ids = [s.skill_id for s in skills]
    assert "menacing_attack" not in skill_ids  # min_level=3 > level=1
    assert "second_wind" in skill_ids           # min_level=1 ≤ level=1


def test_available_skills_excludes_depleted_uses():
    char = _make_battle_master()
    char.ability_uses["second_wind"] = 0    # max_uses=1, 0 remaining
    skills = available_skills(char)
    skill_ids = [s.skill_id for s in skills]
    assert "second_wind" not in skill_ids


def test_available_skills_ability_features_are_materialized():
    char = _make_battle_master()
    skills = available_skills(char)
    sw = next((s for s in skills if s.skill_id == "second_wind"), None)
    assert sw is not None
    # attack_vs_ac = STR mod (+3) + prof (+2) = +5
    assert sw.features.attack_vs_ac == pytest.approx(5.0, rel=0.01)


def test_available_skills_reactions_excluded():
    char = Character(
        name="W", race="", class_="法師", level=1,
        stats=Stats(INT=14), hp=8, max_hp=8, ac=12, is_npc=False,
        weapons=[], spellcasting_ability="INT", spell_slots={1: 4},
        known_abilities=["magic_missile", "shield_spell"],
    )
    char.archetype_id = "evocation"
    char.position = Vec2(0, 0)
    skills = available_skills(char)
    skill_ids = [s.skill_id for s in skills]
    assert "magic_missile" in skill_ids
    assert "shield_spell" not in skill_ids   # is_reaction=True → excluded
