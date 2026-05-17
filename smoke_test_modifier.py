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
    return 0


if __name__ == "__main__":
    sys.exit(main())
