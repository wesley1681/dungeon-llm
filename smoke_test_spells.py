"""Smoke test for the Phase 1 spell system.

Built up incrementally across tasks in docs/superpowers/plans/2026-05-17-spell-system-phase-1.md.
No LLM calls; all assertions are deterministic with random.seed where dice are rolled.
"""
import sys
import random
sys.path.insert(0, ".")


def test_spell_dataclass_and_catalog() -> None:
    """Task 1: Spell dataclass exists, SPELLS catalog has 火球術 with expected fields."""
    from trpg.engine.spells import Spell, SPELLS

    fireball = SPELLS["火球術"]
    assert isinstance(fireball, Spell)
    assert fireball.name == "火球術"
    assert fireball.level == 3
    assert fireball.range_m == 45.0
    assert fireball.aoe_radius_m == 6.0
    assert fireball.attack_type == "save"
    assert fireball.save_ability == "DEX"
    assert fireball.damage_dice == "8d6"
    assert fireball.damage_type == "fire"
    print("Spell dataclass + SPELLS catalog: OK")


def test_character_spell_fields() -> None:
    """Task 2: Character carries spells / spellcasting_ability / spell_slots."""
    from trpg.engine.character import Character, Stats

    # Default values — non-caster character
    plain = Character(
        name="平民", race="人類", class_="—", level=1,
        stats=Stats(), hp=4, max_hp=4, ac=10,
    )
    assert plain.spells == []
    assert plain.spellcasting_ability == ""
    assert plain.spell_slots == {}

    # Caster — explicit setup
    caster = Character(
        name="法師", race="人類", class_="法師", level=5,
        stats=Stats(INT=16), hp=20, max_hp=20, ac=12,
        spells=["火球術"],
        spell_slots={1: 4, 2: 3, 3: 2},
        spellcasting_ability="INT",
    )
    assert caster.spells == ["火球術"]
    assert caster.spellcasting_ability == "INT"
    assert caster.spell_slots[3] == 2
    print("Character spell fields: OK")


def main() -> int:
    test_spell_dataclass_and_catalog()
    test_character_spell_fields()
    print("\n=== ALL SPELL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
