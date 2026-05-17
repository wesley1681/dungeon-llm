"""StatusEffect — Modifier with a declarative lifecycle.

Each effect declares how it expires:
  expires_on        — phase that auto-removes it ('self_turn_start',
                      'self_turn_end', 'round_end', 'combat_end', 'never')
  rounds_remaining  — if set, decremented at 'round_end'; removed when <= 0
  save_each         — e.g. 'CON DC14'; at 'self_turn_end', re-roll save to escape

tick_status_effects() is called by GameSession at 4 phases of the combat loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .modifier import Modifier
from .dice import combine_advantage


def parse_save(spec: str) -> tuple[str, int]:
    """Parse 'CON DC14' -> ('CON', 14)."""
    parts = spec.upper().replace("DC", " ").split()
    return parts[0], int(parts[1])


@dataclass
class StatusEffect(Modifier):
    name: str = ""
    expires_on: str = "never"            # self_turn_start|self_turn_end|round_end|combat_end|never
    rounds_remaining: int | None = None  # countdown each round_end
    save_each: str = ""                  # "CON DC14": re-save each self_turn_end
    applied_round: int = 0
    source_id: str = ""                  # caster ID for concentration / cleanup
    metadata: dict = field(default_factory=dict)


class Dodging(StatusEffect):
    """D&D 5e Dodge action. Expires at start of dodger's next turn.
    Forces disadvantage on attacks against the dodger."""
    def __init__(self, applied_round: int = 0):
        super().__init__(
            name="dodging",
            expires_on="self_turn_start",
            applied_round=applied_round,
        )

    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Hidden(StatusEffect):
    """Successful HIDE action. Persists until broken (not auto-expired here)."""
    def __init__(self, applied_round: int = 0):
        super().__init__(name="hidden", expires_on="never", applied_round=applied_round)


def tick_status_effects(char, phase: str, round_num: int) -> None:
    """Apply lifecycle rules for all status effects on `char`.

    Called at 4 phases:
      self_turn_start — start of this character's turn
      self_turn_end   — end of this character's turn
      round_end       — after the initiative table completes a full cycle
      combat_end      — combat has ended

    Rebuilds the list rather than calling list.remove() because StatusEffect is
    a @dataclass with structural equality — two effects with identical fields
    would be indistinguishable to .remove(), causing the wrong instance to be
    dropped if duplicates ever appear.
    """
    from .combat import make_saving_throw  # avoid circular import

    remaining = []
    for fx in char.status_effects:
        # Skip non-StatusEffect entries (legacy strings) — they're not
        # subject to lifecycle rules and survive the tick untouched.
        if not isinstance(fx, StatusEffect):
            remaining.append(fx)
            continue

        # Rule 1: direct phase match → drop
        if fx.expires_on == phase:
            continue
        # Rule 2: countdown at round_end → drop when reaches zero
        if fx.rounds_remaining is not None and phase == "round_end":
            fx.rounds_remaining -= 1
            if fx.rounds_remaining <= 0:
                continue
        # Rule 3: re-save each self_turn_end → drop on success
        if fx.save_each and phase == "self_turn_end":
            stat, dc = parse_save(fx.save_each)
            ok, _ = make_saving_throw(char, stat, dc)
            if ok:
                continue
        remaining.append(fx)
    char.status_effects[:] = remaining
