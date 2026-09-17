import re
import random

# One parser shared by roll() and dice_ev() so a notation can never be rolled
# one way and valued another. Mirrors roll()'s grammar: optional leading count
# (default 1), required dN, optional ±flat. Whitespace tolerant.
_DICE_RE = re.compile(r"^\s*(\d*)d(\d+)\s*([+-]\s*\d+)?\s*$")


def _parse_dice(notation: str):
    """(count, sides, flat) for an 'NdM±K' string, or None if unparseable."""
    m = _DICE_RE.match(notation.strip().lower())
    if not m:
        return None
    count = int(m.group(1)) if m.group(1) else 1
    sides = int(m.group(2))
    flat = int((m.group(3) or "0").replace(" ", ""))
    return count, sides, flat


def dice_ev(notation: str) -> float:
    """Expected value of an 'NdM±K' roll: N*(M+1)/2 + K.

    THE canonical dice→expected-value primitive. Any code that needs the
    average of a dice string (obs `expected_damage`, greedy policy EV) must
    derive it from here so it can never drift from what roll() actually
    produces. Returns 0.0 on empty / unparseable notation (matches the old
    skill._expected_dice contract)."""
    if not notation:
        return 0.0
    parsed = _parse_dice(notation)
    if parsed is None:
        return 0.0
    count, sides, flat = parsed
    return count * (sides + 1) / 2.0 + flat


def cantrip_multiplier(caster_level: int) -> int:
    """5e cantrip dice multiplier by level: 1 / 2 / 3 / 4 at L1 / 5 / 11 / 17.

    Shared by the engine roll (_roll_scaled_cantrip) and the obs EV
    (skill materialize) so the two breakpoint tables can't drift apart."""
    if caster_level >= 17:
        return 4
    if caster_level >= 11:
        return 3
    if caster_level >= 5:
        return 2
    return 1


def roll(notation: str) -> int:
    """Parse and roll dice notation: '2d6+3', '1d20', 'd8', '1d1-1'."""
    parsed = _parse_dice(notation)
    if parsed is None:
        raise ValueError(f"Invalid dice notation: {notation!r}")
    num, sides, modifier = parsed
    return sum(random.randint(1, sides) for _ in range(num)) + modifier


def roll_d20(mode: str = "normal") -> int:
    """Roll 1d20 with D&D 5e advantage/disadvantage semantics.

    mode:
      "normal"       — roll one d20
      "advantage"    — roll two d20, take the higher
      "disadvantage" — roll two d20, take the lower
    """
    if mode == "advantage":
        return max(random.randint(1, 20), random.randint(1, 20))
    if mode == "disadvantage":
        return min(random.randint(1, 20), random.randint(1, 20))
    return random.randint(1, 20)


def combine_advantage(*modes: str) -> str:
    """Combine multiple advantage/disadvantage sources into one final mode.

    D&D 5e rule: advantage and disadvantage don't stack. Any number of each
    just means "one advantage" or "one disadvantage". One of each cancels
    out to normal.
    """
    has_adv = any(m == "advantage"    for m in modes)
    has_dis = any(m == "disadvantage" for m in modes)
    if has_adv and not has_dis:
        return "advantage"
    if has_dis and not has_adv:
        return "disadvantage"
    return "normal"
