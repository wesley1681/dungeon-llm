"""Spell definitions and central catalog.

`Character.spells` stores spell names (strings); look up the actual Spell
object via SPELLS[name]. Damage / range / save info lives here so spells are
defined once and shared across all casters.

Phase 1 supports only attack_type="save" (AOE saving-throw spells like
fireball). spell_attack and utility variants are reserved for Phase 2.
"""
from dataclasses import dataclass


@dataclass
class Spell:
    name: str
    level: int                       # 1-9: minimum slot needed to cast
    range_m: float                   # distance from caster to AOE center / target
    aoe_radius_m: float = 0.0        # 0 = single target (only the centered creature)
    attack_type: str = "save"        # "save" — Phase 2 will add "spell_attack" / "utility"
    save_ability: str = "DEX"        # ability targets roll for the save
    damage_dice: str = ""            # e.g. "8d6"; "" = no damage (Phase 2 utility)
    damage_type: str = ""            # e.g. "fire", "cold"
    description: str = ""


SPELLS: dict[str, Spell] = {
    "火球術": Spell(
        name="火球術",
        level=3,
        range_m=45.0,
        aoe_radius_m=6.0,
        attack_type="save",
        save_ability="DEX",
        damage_dice="8d6",
        damage_type="火",
        description="向 45m 內一點扔出火球，半徑 6m 內目標 DEX 豁免，成功半傷。",
    ),
}
