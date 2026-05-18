import pytest
from trpg.engine.character import Character, Stats
from trpg.engine.abilities import CLASS_ABILITIES


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
