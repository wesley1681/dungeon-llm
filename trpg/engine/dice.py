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
