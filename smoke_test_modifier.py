"""Smoke test for Modifier base class.

Verifies all hook methods exist with the documented signatures and default
to pass-through (mode unchanged, amount unchanged, AC unchanged).
"""
import sys
sys.path.insert(0, ".")

from trpg.engine.modifier import Modifier


def main() -> int:
    m = Modifier()
    assert m.on_compute_ac(None, 15) == 15
    assert m.on_outgoing_attack(None, None, None, "advantage") == "advantage"
    assert m.on_incoming_attack(None, None, None, "disadvantage") == "disadvantage"
    assert m.on_outgoing_damage(None, None, 10, "slashing") == 10
    assert m.on_incoming_damage(None, None, 7, "fire") == 7
    # on_critical_hit is None-returning; just verify it doesn't raise
    m.on_critical_hit(None, None, None)
    print("Modifier default hooks pass-through: OK")

    # Subclass override
    class HalfFireResist(Modifier):
        def on_incoming_damage(self, target, attacker, amount, dtype):
            return amount // 2 if dtype == "fire" else amount

    h = HalfFireResist()
    assert h.on_incoming_damage(None, None, 10, "fire") == 5
    assert h.on_incoming_damage(None, None, 10, "slashing") == 10
    print("Modifier subclass override: OK")
    print("\n=== ALL MODIFIER TESTS PASSED ===")

    # ── 7. Modifier wired into resolve_attack ─────────────────────────────────
    import random
    from trpg.engine import combat
    from trpg.engine.character import Character, Stats
    from trpg.engine.items import WEAPON_DEFS

    class AdvantageAura(Modifier):
        """Defender modifier that forces advantage on incoming attacks (for testing)."""
        def on_incoming_attack(self, defender, attacker, weapon, mode):
            from trpg.engine.dice import combine_advantage
            return combine_advantage(mode, "advantage")

    attacker = Character(name="A", race="人類", class_="戰士", level=1,
                         stats=Stats(STR=14), hp=20, max_hp=20, ac=15,
                         weapons=[WEAPON_DEFS["長劍"]])
    target = Character(name="T", race="人類", class_="戰士", level=1,
                       stats=Stats(), hp=20, max_hp=20, ac=15)
    target.equipment.append(AdvantageAura())
    attacker.position = 0.0
    target.position = 1.0

    # Run many attacks; average roll should reflect advantage (~13.83) not normal (~10.5)
    random.seed(0)
    totals = []
    for _ in range(500):
        _, total = combat.resolve_attack(attacker, target, attacker.weapons[0], mode="normal")
        totals.append(total)
    avg = sum(totals) / len(totals)
    # The Modifier should push us to advantage; mean of 2d20-max + STR(+2) + prof(+2) ≈ 17.8
    assert avg > 16.0, f"expected mean ~17.8 with advantage, got {avg:.2f}"
    print(f"AdvantageAura via on_incoming_attack: avg total={avg:.2f}  OK")

    # ── 8. Modifier wired into apply_damage ──────────────────────────────────
    class HalfResist(Modifier):
        def on_incoming_damage(self, target, attacker, amount, dtype):
            return amount // 2

    t = Character(name="R", race="人類", class_="戰士", level=1,
                  stats=Stats(), hp=20, max_hp=20, ac=15)
    t.equipment.append(HalfResist())
    combat.apply_damage(t, 10, dtype="slashing", attacker=None)
    assert t.hp == 15, f"expected hp=15 after halved 10 dmg, got {t.hp}"
    print("HalfResist via on_incoming_damage: OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
