"""Modifier — unified hook interface for equipment, status effects, racial/class
features. Anything that wants to influence combat resolution implements one or
more of the hook methods below.

Hook points (called by engine/combat.py):
  on_compute_ac        — Character AC computation
  on_outgoing_attack   — resolve_attack, attacker side, before roll
  on_incoming_attack   — resolve_attack, target side, before roll
  on_outgoing_damage   — resolve_attack, attacker side, after hit (e.g. sneak attack)
  on_incoming_damage   — apply_damage, target side (e.g. resistance)
  on_critical_hit      — resolve_attack, on natural 20

All hooks default to no-op so subclasses only override what they care about.
"""


class Modifier:
    def on_compute_ac(self, char, current_ac: int) -> int:
        return current_ac

    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return mode

    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return mode

    def on_outgoing_damage(self, attacker, target, amount: int, dtype: str) -> int:
        return amount

    def on_incoming_damage(self, target, attacker, amount: int, dtype: str) -> int:
        return amount

    def on_critical_hit(self, attacker, target, weapon) -> None:
        pass
