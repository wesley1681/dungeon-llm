import pytest
from trpg.engine.abilities import CLASS_ABILITIES
from trpg.engine.spells import SPELLS
from trpg.engine.status import MODIFIER_CLASSES


def _check_registered(skill_ids: list, class_id: str, archetype_id: str,
                       min_level: int = None):
    for sid in skill_ids:
        ab = CLASS_ABILITIES.get(sid)
        assert ab is not None, f"'{sid}' missing from CLASS_ABILITIES"
        assert ab.class_id == class_id, f"{sid}: expected class_id='{class_id}', got '{ab.class_id}'"
        assert ab.archetype_id == archetype_id, (
            f"{sid}: expected archetype_id='{archetype_id}', got '{ab.archetype_id}'"
        )
        if min_level is not None:
            assert ab.min_level == min_level, (
                f"{sid}: expected min_level={min_level}, got {ab.min_level}"
            )


def test_new_status_classes_registered():
    for name in ["blurred", "baned", "sacred_weapon_buff"]:
        assert name in MODIFIER_CLASSES, f"'{name}' missing from MODIFIER_CLASSES"


def test_new_spells_registered():
    for spell_name in ["燃燒之手", "蜘蛛網", "冰風暴", "定怪術", "治療語"]:
        assert spell_name in SPELLS, f"'{spell_name}' missing from SPELLS"
