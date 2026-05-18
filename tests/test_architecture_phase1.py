import pytest
from trpg.engine.abilities import CLASS_ABILITIES, ClassAbility


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
