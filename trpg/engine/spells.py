"""Spell definitions and central catalog.

`Character.spells` stores spell names (strings); look up the actual Spell
object via SPELLS[name]. Damage / range / save info lives here so spells are
defined once and shared across all casters.

Phase 1 supports attack_type="save" (AOE / single-target saving-throw spells).
spell_attack (attack-roll spells like Firebolt) is reserved for Phase 2.

Spells with `level=0` are cantrips and consume no spell slot.
Spells with `requires_concentration=True` set caster.concentrating_on when
cast (replacing any prior concentration spell).
Spells with `applies_status_on_fail` attach the named StatusEffect to targets
that fail their save.
"""
from dataclasses import dataclass


@dataclass
class Spell:
    name: str
    level: int                       # 0 = cantrip (no slot); 1-9 = required slot
    range_m: float                   # distance from caster to AOE center / target
    aoe_radius_m: float = 0.0        # 0 = single target (only the centered creature)
    attack_type: str = "save"        # "save" — Phase 2 will add "spell_attack" / "utility"
    save_ability: str = "DEX"        # ability targets roll for the save
    damage_dice: str = ""            # e.g. "8d6"; "" = no damage
    damage_type: str = ""            # e.g. "fire", "cold"
    heal_dice: str = ""              # for healing spells (e.g. "1d8+3")
    requires_concentration: bool = False
    applies_status_on_fail: str = "" # status name attached on failed save
    status_rounds: int = 10          # default 1-minute duration if status applies
    # Save outcome rule for damage spells:
    #   False (default) — successful save halves damage (Fireball, Burning Hands)
    #   True            — successful save deals no damage (Sacred Flame, Toll the Dead)
    save_for_no_damage: bool = False
    description: str = ""


SPELLS: dict[str, Spell] = {
    "火球術": Spell(
        name="火球術",
        level=3,
        range_m=45.0,
        aoe_radius_m=6.0,
        save_ability="DEX",
        damage_dice="8d6",
        damage_type="火",
        description="向 45m 內一點扔出火球，半徑 6m 內目標 DEX 豁免，成功半傷。",
    ),
    "神聖光輝": Spell(
        name="神聖光輝",
        level=0,                     # cantrip
        range_m=18.0,
        save_ability="DEX",
        damage_dice="1d8",
        damage_type="光耀",
        save_for_no_damage=True,
        description="單體目標 DEX 豁免，失敗承受 1d8 光耀傷害；成功無傷害。",
    ),
    "定身術": Spell(
        name="定身術",
        level=2,
        range_m=18.0,
        save_ability="WIS",
        applies_status_on_fail="paralyzed",
        status_rounds=10,
        requires_concentration=True,
        description="WIS 豁免，失敗則 paralyzed 10 回合；每回合終可重豁。"
                    "施法者必須維持專注。",
    ),
    "霧步": Spell(
        name="霧步",
        level=2,
        range_m=9.0,
        description="瞬移至 9m 內可見點，無視視線、不被障礙阻擋。"
                    "（霧步透過 MOVE action 執行，不走 SPELL 流程）",
    ),
    "治療術": Spell(
        name="治療術",
        level=1,
        range_m=1.5,
        heal_dice="1d8+3",           # placeholder; real value adds caster's WIS mod
        description="觸碰範圍，治療一個盟友 1d8 + 施法屬性修正 HP。"
                    "（治療術透過 HEAL action 執行，不走 SPELL 流程）",
    ),
}
