import re
import random


def roll(notation: str) -> int:
    """Parse and roll dice notation: '2d6+3', '1d20', 'd8', '1d1-1'."""
    notation = notation.strip().lower()
    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", notation)
    if not match:
        raise ValueError(f"Invalid dice notation: {notation!r}")
    num = int(match.group(1)) if match.group(1) else 1
    sides = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0
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
