"""Smoke test for StatusEffect lifecycle.

Verifies:
  - StatusEffect is a Modifier subclass
  - tick_status_effects handles phase-based expiry (self_turn_start, etc.)
  - tick_status_effects handles rounds_remaining countdown at round_end
  - tick_status_effects handles save_each at self_turn_end
  - Dodging applies disadvantage via Modifier hook
"""
import sys
import random
sys.path.insert(0, ".")

from trpg.engine.modifier import Modifier
from trpg.engine.status import StatusEffect, Dodging, tick_status_effects
from trpg.engine.character import Character, Stats


def make_char(name="testchar") -> Character:
    return Character(
        name=name, race="人類", class_="戰士", level=3,
        stats=Stats(STR=14, DEX=12, CON=14, INT=10, WIS=10, CHA=10),
        hp=20, max_hp=20, ac=15,
    )


def main() -> int:
    # ── 1. StatusEffect is a Modifier ─────────────────────────────────────────
    fx = StatusEffect(name="testing", expires_on="never")
    assert isinstance(fx, Modifier)
    print("StatusEffect is Modifier: OK")

    # ── 2. Phase-based expiry ─────────────────────────────────────────────────
    char = make_char()
    char.status_effects.append(Dodging(applied_round=1))
    assert char.has_status("dodging")
    # Wrong phase — should NOT remove
    tick_status_effects(char, "round_end", round_num=1)
    assert char.has_status("dodging"), "dodging should survive round_end tick"
    # Correct phase — should remove
    tick_status_effects(char, "self_turn_start", round_num=2)
    assert not char.has_status("dodging"), "dodging should expire on self_turn_start"
    print("Phase-based expiry (Dodging): OK")

    # ── 3. Rounds remaining countdown ─────────────────────────────────────────
    char = make_char()
    char.status_effects.append(StatusEffect(name="blessed", rounds_remaining=3))
    for n in range(1, 4):
        tick_status_effects(char, "round_end", round_num=n)
        if n < 3:
            assert char.has_status("blessed"), f"blessed should still be active at round {n}"
    assert not char.has_status("blessed"), "blessed should expire after 3 rounds"
    print("Rounds-remaining countdown: OK")

    # ── 4. save_each on self_turn_end ─────────────────────────────────────────
    random.seed(0)
    char = make_char()
    # Easy DC — should expire quickly
    char.status_effects.append(StatusEffect(name="stunned", save_each="CON DC5"))
    for _ in range(10):
        tick_status_effects(char, "self_turn_end", round_num=1)
        if not char.has_status("stunned"):
            break
    assert not char.has_status("stunned"), "stunned should expire on low-DC save"
    print("save_each lifecycle: OK")

    # ── 5. Dodging Modifier hook applies disadvantage ─────────────────────────
    from trpg.engine.dice import combine_advantage
    d = Dodging(applied_round=1)
    mode = d.on_incoming_attack(defender=None, attacker=None, weapon=None, mode="normal")
    assert mode == "disadvantage", f"Dodging should give disadvantage, got {mode}"
    print("Dodging Modifier hook: OK")

    # ── 6. combat_end phase clears combat_end effects ────────────────────────
    char = make_char()
    char.status_effects.append(StatusEffect(name="raging", expires_on="combat_end"))
    tick_status_effects(char, "round_end", round_num=5)
    assert char.has_status("raging")
    tick_status_effects(char, "combat_end", round_num=5)
    assert not char.has_status("raging")
    print("combat_end phase: OK")

    # ── 7. End-to-end: Dodging via Character.add_status + iter_modifiers ─────
    char = make_char()
    char.add_status(Dodging(applied_round=1))
    # iter_modifiers should yield the Dodging instance
    mods = list(char.iter_modifiers())
    dodging_mods = [m for m in mods if isinstance(m, Dodging)]
    assert len(dodging_mods) == 1, f"expected 1 Dodging modifier, got {len(dodging_mods)}"
    # Calling on_incoming_attack on the yielded modifier should give disadvantage
    mode = dodging_mods[0].on_incoming_attack(defender=char, attacker=None, weapon=None, mode="normal")
    assert mode == "disadvantage", f"expected disadvantage, got {mode}"
    # add_status is idempotent — re-adding shouldn't duplicate
    char.add_status(Dodging(applied_round=2))
    assert sum(1 for fx in char.status_effects if isinstance(fx, Dodging)) == 1
    print("End-to-end Dodging via iter_modifiers: OK")

    print("\n=== ALL STATUS EFFECT TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
