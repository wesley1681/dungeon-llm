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


def main() -> int:
    test_spell_dataclass_and_catalog()
    print("\n=== ALL SPELL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
