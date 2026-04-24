from .dice import roll
from .character import Character, CombatState
from .world_state import WorldState


def roll_initiative(world_state: WorldState) -> CombatState:
    results = {}
    for name, char in world_state.characters.items():
        if char.is_alive():
            results[name] = roll("1d20") + char.stats.modifier("DEX")
    order = sorted(results, key=lambda n: results[n], reverse=True)
    return CombatState(initiative_order=order)


def resolve_attack(attacker: Character, target: Character) -> tuple[bool, int]:
    """Returns (hit, total_roll). Uses STR modifier + proficiency bonus."""
    attack_roll = roll("1d20")
    total = attack_roll + attacker.stats.modifier("STR") + attacker.proficiency_bonus
    return total >= target.ac, total


def apply_damage(target: Character, damage_notation: str) -> int:
    """Roll damage and apply to target. Returns damage dealt."""
    damage = roll(damage_notation)
    target.hp = max(0, target.hp - damage)
    return damage


def apply_heal(target: Character, dice_notation: str) -> int:
    """Restore HP up to max_hp. Returns amount healed."""
    amount = roll(dice_notation)
    target.hp = min(target.max_hp, target.hp + amount)
    return amount


def make_saving_throw(character: Character, stat: str, dc: int) -> tuple[bool, int]:
    """Returns (success, total). Adds proficiency bonus if stat is in proficiencies."""
    roll_result = roll("1d20")
    modifier = character.stats.modifier(stat)
    if stat in character.proficiencies:
        modifier += character.proficiency_bonus
    total = roll_result + modifier
    return total >= dc, total
