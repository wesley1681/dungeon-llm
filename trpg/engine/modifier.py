"""Modifier — unified hook interface for equipment, status effects, racial/class
features. Anything that wants to influence combat resolution implements one or
more of the hook methods below.

Hook points (called by engine/combat.py):
  on_compute_ac        — Character AC computation
  on_outgoing_attack   — resolve_attack, attacker side, before roll
  on_incoming_attack   — resolve_attack, target side, before roll
  on_outgoing_damage   — resolve_attack, attacker side, after hit (e.g. sneak attack)
  on_incoming_damage   — apply_damage, target side (e.g. resistance)
  on_critical_hit      — resolve_attack, on natural 20 (notification only; return value ignored)

Convention: the first positional argument of every hook is the Character this
modifier is currently attached to (the "subject"). The opposing combatant
is always passed too. Character/Weapon parameters are intentionally untyped
to avoid forward-reference imports — call sites pass real objects.

All hooks default to no-op so subclasses only override what they care about.
"""


class Modifier:
    def on_compute_ac(self, char, current_ac: int) -> int:
        return current_ac

    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return mode

    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return mode

    def on_outgoing_attack_total(self, attacker, target, weapon, total: int) -> int:
        """Modify the numeric d20 total *after* the roll (e.g. Bless's +1d4)."""
        return total

    def on_outgoing_damage(self, attacker, target, amount: int, dtype: str) -> int:
        return amount

    def on_incoming_damage(self, target, attacker, amount: int, dtype: str) -> int:
        return amount

    def on_saving_throw(self, char, stat: str, modifier: int) -> int:
        """Modify the bonus added to a saving throw (e.g. Bless's +1d4)."""
        return modifier

    def on_auto_fail_save(self, char, stat: str) -> bool:
        """Return True to auto-fail a saving throw for this stat (e.g. paralyzed
        auto-fails STR and DEX saves in 5e)."""
        return False

    def on_speed_multiplier(self, char) -> float:
        """Multiply the character's base movement speed (0.0 = immobilised)."""
        return 1.0

    def on_critical_hit(self, attacker, target, weapon) -> None:
        pass
