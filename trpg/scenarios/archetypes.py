"""Archetype character factories for D&D 5e classes.

Each function returns a fully-configured Character with the correct level-based
passives, known_abilities, archetype_id, weapons, and spell slots for that
class/subclass. Use these to spawn proper characters for RL training instead of
hand-rolling them.

Example
-------
>>> from trpg.scenarios.archetypes import make_battle_master
>>> fighter = make_battle_master("Brom", level=5)
>>> fighter.attacks_per_action
2
>>> fighter.archetype_id
'battle_master'
"""
from __future__ import annotations

from ..engine.character import Character, Stats
from ..engine.items import WEAPON_DEFS
from ..engine.status import Evasion


# ── Helpers ──────────────────────────────────────────────────────────────────


def _proficiency_bonus(level: int) -> int:
    return (level - 1) // 4 + 2


def _sneak_attack_dice(rogue_level: int) -> str:
    """Rogue Sneak Attack scales: 1d6 at L1, +1d6 every 2 levels."""
    dice_count = (rogue_level + 1) // 2
    return f"{dice_count}d6"


# ── Fighter ──────────────────────────────────────────────────────────────────


def make_battle_master(name: str = "Battle Master", level: int = 3) -> Character:
    """Fighter — Battle Master archetype.

    L1: Second Wind, Fighting Style
    L2: Action Surge
    L3: Battle Master maneuvers (Trip / Menacing / Pushing chosen)
    L5: Extra Attack (attacks_per_action = 2)
    """
    c = Character(
        name=name, race="人類", class_="戰士", level=level,
        stats=Stats(STR=16, DEX=12, CON=14, INT=10, WIS=12, CHA=10),
        hp=10 + level * 8, max_hp=10 + level * 8, ac=16,
        weapons=[WEAPON_DEFS["長劍"], WEAPON_DEFS["短弓"]],
        proficiencies=["STR", "CON", "運動", "恐嚇"],
        is_npc=False,
    )
    c.archetype_id = "battle_master"
    c.known_abilities = ["second_wind"]
    if level >= 2: c.known_abilities.append("action_surge")
    if level >= 3:
        c.known_abilities.extend(["trip_attack", "menacing_attack"])
    if level >= 5:
        c.attacks_per_action = 2
    return c


def make_champion(name: str = "Champion", level: int = 3) -> Character:
    """Fighter — Champion archetype.

    L3: Improved Critical (crit_range = 19)
    L5: Extra Attack
    """
    c = Character(
        name=name, race="人類", class_="戰士", level=level,
        stats=Stats(STR=16, DEX=14, CON=14, INT=10, WIS=10, CHA=10),
        hp=10 + level * 8, max_hp=10 + level * 8, ac=16,
        weapons=[WEAPON_DEFS["長劍"], WEAPON_DEFS["短弓"]],
        proficiencies=["STR", "CON"],
        is_npc=False,
    )
    c.archetype_id = "champion"
    c.known_abilities = ["second_wind"]
    if level >= 2: c.known_abilities.append("action_surge")
    if level >= 3:
        c.known_abilities.append("improved_critical")
        c.crit_range = 19
    if level >= 5:
        c.attacks_per_action = 2
    return c


# ── Barbarian ────────────────────────────────────────────────────────────────


def make_totem_bear(name: str = "Totem Warrior", level: int = 3) -> Character:
    """Barbarian — Totem Warrior (Bear) archetype.

    L1: Rage, Unarmored Defense
    L2: Reckless Attack
    L3: Bear Totem (resistance to all damage while raging)
    L5: Extra Attack
    """
    rage_uses = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}.get(level, 4)
    c = Character(
        name=name, race="人類", class_="蠻人", level=level,
        stats=Stats(STR=17, DEX=14, CON=16, INT=8, WIS=10, CHA=10),
        hp=12 + level * 9, max_hp=12 + level * 9, ac=14,
        weapons=[WEAPON_DEFS["長劍"]],
        proficiencies=["STR", "CON"],
        is_npc=False,
    )
    c.archetype_id = "totem_bear"
    c.known_abilities = ["rage"]
    if level >= 2: c.known_abilities.append("reckless_attack")
    if level >= 3: c.known_abilities.append("bear_totem")
    if level >= 5:
        c.attacks_per_action = 2
    c.ability_uses["rage"] = rage_uses
    return c


def make_berserker(name: str = "Berserker", level: int = 3) -> Character:
    """Barbarian — Berserker archetype.

    L3: Frenzy (bonus action attack while raging)
    L5: Extra Attack
    """
    rage_uses = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}.get(level, 4)
    c = Character(
        name=name, race="人類", class_="蠻人", level=level,
        stats=Stats(STR=17, DEX=14, CON=16, INT=8, WIS=10, CHA=10),
        hp=12 + level * 9, max_hp=12 + level * 9, ac=14,
        weapons=[WEAPON_DEFS["長劍"]],
        proficiencies=["STR", "CON"],
        is_npc=False,
    )
    c.archetype_id = "berserker"
    c.known_abilities = ["rage"]
    if level >= 2: c.known_abilities.append("reckless_attack")
    if level >= 3: c.known_abilities.append("berserker_frenzy")
    if level >= 5:
        c.attacks_per_action = 2
    c.ability_uses["rage"] = rage_uses
    return c


# ── Wizard ───────────────────────────────────────────────────────────────────


def _wizard_slots(level: int) -> dict:
    table = {
        1: {1: 2},
        2: {1: 3},
        3: {1: 4, 2: 2},
        4: {1: 4, 2: 3},
        5: {1: 4, 2: 3, 3: 2},
        6: {1: 4, 2: 3, 3: 3},
        7: {1: 4, 2: 3, 3: 3, 4: 1},
        8: {1: 4, 2: 3, 3: 3, 4: 2},
    }
    return table.get(level, table[8]).copy()


def make_evocation_wizard(name: str = "Evoker", level: int = 3) -> Character:
    """Wizard — School of Evocation.

    L1: Spellcasting (Magic Missile, Shield, Burning Hands)
    L2: Sculpt Spells (AOE excludes allies)
    L3: 2nd-level spells (Web, Misty Step, Hold Person)
    L5: 3rd-level spells (Fireball, Counterspell)
    L7: 4th-level (Ice Storm)
    """
    spells = ["魔法飛彈", "燃燒之手"]
    abilities = ["magic_missile", "shield_spell", "burning_hands_ev"]
    if level >= 2:
        spells.append("霧步")
    if level >= 3:
        spells.extend(["蜘蛛網", "定身術"])
        abilities.extend(["web_ev", "misty_step", "hold_person"])
    if level >= 5:
        spells.append("火球術")
        abilities.append("fireball_ev")
    if level >= 7:
        spells.append("冰風暴")
        abilities.append("ice_storm_ev")

    c = Character(
        name=name, race="人類", class_="法師", level=level,
        stats=Stats(STR=8, DEX=14, CON=14, INT=16, WIS=12, CHA=10),
        hp=6 + level * 4, max_hp=6 + level * 4, ac=12,
        weapons=[],
        spells=spells, spellcasting_ability="INT",
        spell_slots=_wizard_slots(level),
        proficiencies=["INT", "WIS", "奧秘", "歷史"],
        reactions=["shield_spell"],
        is_npc=False,
    )
    c.archetype_id = "evocation"
    c.known_abilities = abilities
    if level >= 2:
        c.sculpt_spells = True
    return c


def make_divination_wizard(name: str = "Diviner", level: int = 3,
                            portent_rolls: list | None = None) -> Character:
    """Wizard — School of Divination.

    L2: Portent (2 d20 rolls stored at dawn).
    """
    from trpg.engine.dice import roll as _roll
    if portent_rolls is None:
        portent_rolls = [_roll("1d20"), _roll("1d20")] if level >= 2 else []
    spells = ["魔法飛彈", "燃燒之手"]
    abilities = ["magic_missile", "shield_spell", "burning_hands_div"]
    if level >= 3:
        spells.extend(["蜘蛛網", "定身術", "霧步"])
        abilities.extend(["web_div", "hold_person", "misty_step"])
    if level >= 5:
        spells.append("火球術")
        abilities.append("fireball_div")

    c = Character(
        name=name, race="人類", class_="法師", level=level,
        stats=Stats(STR=8, DEX=14, CON=14, INT=16, WIS=12, CHA=10),
        hp=6 + level * 4, max_hp=6 + level * 4, ac=12,
        weapons=[],
        spells=spells, spellcasting_ability="INT",
        spell_slots=_wizard_slots(level),
        proficiencies=["INT", "WIS", "奧秘"],
        reactions=["shield_spell"],
        is_npc=False,
    )
    c.archetype_id = "divination"
    c.known_abilities = abilities
    if portent_rolls:
        c.portent_dice = list(portent_rolls)
    return c


# ── Cleric ───────────────────────────────────────────────────────────────────


def _cleric_slots(level: int) -> dict:
    return _wizard_slots(level)   # Same progression L1-8


def make_life_cleric(name: str = "Life Cleric", level: int = 3) -> Character:
    """Cleric — Life Domain.

    L1: Cure Wounds, Bless, Healing Word, Guiding Bolt, Sacred Flame
    L2: Channel Divinity (Preserve Life)
    L3: Spiritual Weapon
    L5: Mass Cure Wounds
    """
    spells = ["治療術", "祝福術", "治療語", "神聖光輝"]
    abilities = ["cure_wounds", "bless", "sacred_flame", "healing_word_life"]
    if level >= 2:
        c_div_uses = 1
        abilities.append("channel_divinity_preserve_life")
    if level >= 3:
        spells.append("定身術")
        abilities.append("spiritual_weapon_life")
    if level >= 5:
        abilities.append("mass_cure_wounds_life")

    c = Character(
        name=name, race="人類", class_="牧師", level=level,
        stats=Stats(STR=12, DEX=10, CON=14, INT=10, WIS=16, CHA=12),
        hp=8 + level * 6, max_hp=8 + level * 6, ac=16,
        weapons=[WEAPON_DEFS["長劍"]],
        spells=spells, spellcasting_ability="WIS",
        spell_slots=_cleric_slots(level),
        proficiencies=["WIS", "CHA", "醫療", "宗教"],
        is_npc=False,
    )
    c.archetype_id = "life"
    c.known_abilities = abilities
    if level >= 2:
        c.ability_uses["channel_divinity"] = 1
    return c


def make_war_cleric(name: str = "War Cleric", level: int = 3) -> Character:
    """Cleric — War Domain.

    L1: Guiding Bolt + spells
    L2: Channel Divinity (Guided Strike)
    L6: War Priest (bonus action attack)
    """
    spells = ["治療術", "祝福術", "神聖光輝"]
    abilities = ["cure_wounds", "bless", "sacred_flame", "guiding_bolt_war"]
    if level >= 2:
        abilities.append("channel_divinity_guided_strike")
    if level >= 3:
        spells.append("定身術")
        abilities.append("spiritual_weapon_war")
    if level >= 6:
        abilities.append("war_priest_attack")

    c = Character(
        name=name, race="人類", class_="牧師", level=level,
        stats=Stats(STR=15, DEX=10, CON=14, INT=10, WIS=16, CHA=10),
        hp=8 + level * 6, max_hp=8 + level * 6, ac=18,    # heavy armor
        weapons=[WEAPON_DEFS["長劍"]],
        spells=spells, spellcasting_ability="WIS",
        spell_slots=_cleric_slots(level),
        proficiencies=["WIS", "CHA"],
        is_npc=False,
    )
    c.archetype_id = "war"
    c.known_abilities = abilities
    if level >= 2:
        c.ability_uses["channel_divinity"] = 1
    if level >= 6:
        c.ability_uses["war_priest_attack"] = max(1, c.stats.modifier("WIS"))
    return c


# ── Rogue ────────────────────────────────────────────────────────────────────


def make_assassin(name: str = "Assassin", level: int = 3) -> Character:
    """Rogue — Assassin archetype.

    L1: Sneak Attack (1d6 → +1d6 every 2 levels)
    L2: Cunning Action
    L3: Assassinate (advantage vs surprised + auto-crit)
    L5: Uncanny Dodge
    L7: Evasion
    """
    abilities = []
    if level >= 2:
        abilities.extend(["cunning_action_dash", "cunning_action_disengage",
                          "cunning_action_hide"])
    if level >= 3:
        abilities.append("assassinate")
    if level >= 5:
        abilities.append("uncanny_dodge_rogue")
    if level >= 7:
        abilities.append("evasion_rogue")

    c = Character(
        name=name, race="人類", class_="盜賊", level=level,
        stats=Stats(STR=10, DEX=17, CON=12, INT=14, WIS=12, CHA=10),
        hp=8 + level * 5, max_hp=8 + level * 5, ac=14,
        weapons=[WEAPON_DEFS["短劍"], WEAPON_DEFS["短弓"]],
        proficiencies=["DEX", "INT", "潛行", "開鎖", "察覺", "欺騙"],
        is_npc=False,
    )
    c.archetype_id = "assassin"
    c.known_abilities = abilities
    c.sneak_attack_dice = _sneak_attack_dice(level)
    if level >= 7:
        c.add_status(Evasion(applied_round=0))
    return c


def make_arcane_trickster(name: str = "Arcane Trickster", level: int = 3) -> Character:
    """Rogue — Arcane Trickster archetype.

    L3: Spellcasting (limited wizard spells)
    L5: Uncanny Dodge
    L7: Evasion
    """
    spells = []
    abilities = []
    if level >= 2:
        abilities.extend(["cunning_action_dash_at", "cunning_action_disengage_at", "cunning_action_hide_at"])
    if level >= 3:
        spells.extend(["霧步"])
    if level >= 5:
        abilities.append("uncanny_dodge_at")
    if level >= 7:
        abilities.append("evasion_at")

    c = Character(
        name=name, race="人類", class_="盜賊", level=level,
        stats=Stats(STR=10, DEX=17, CON=12, INT=14, WIS=12, CHA=10),
        hp=8 + level * 5, max_hp=8 + level * 5, ac=14,
        weapons=[WEAPON_DEFS["短劍"], WEAPON_DEFS["短弓"]],
        spells=spells,
        spellcasting_ability="INT" if level >= 3 else "",
        spell_slots={1: 3} if level >= 3 else {},
        proficiencies=["DEX", "INT", "潛行", "察覺"],
        is_npc=False,
    )
    c.archetype_id = "arcane_trickster"
    c.known_abilities = abilities
    c.sneak_attack_dice = _sneak_attack_dice(level)
    if level >= 7:
        c.add_status(Evasion(applied_round=0))
    return c


# ── Paladin ──────────────────────────────────────────────────────────────────


def _paladin_slots(level: int) -> dict:
    table = {
        1: {},
        2: {1: 2},
        3: {1: 3},
        4: {1: 3},
        5: {1: 4, 2: 2},
        6: {1: 4, 2: 2},
        7: {1: 4, 2: 3},
        8: {1: 4, 2: 3},
    }
    return table.get(level, table[8]).copy()


def make_devotion_paladin(name: str = "Devotion Paladin", level: int = 3) -> Character:
    """Paladin — Oath of Devotion.

    L1: Lay on Hands (pool = 5 × level), Divine Sense
    L2: Divine Smite, Spellcasting
    L3: Sacred Weapon, Turn the Unholy (Channel Divinity)
    L5: Extra Attack
    L6: Aura of Protection (+CHA to nearby ally saves)
    """
    spells = []
    abilities = ["lay_on_hands_ability"]
    if level >= 2:
        spells.extend(["祝福術", "治療術"])
        abilities.extend(["divine_smite_dev", "shield_of_faith_dev", "bless"])
    if level >= 3:
        abilities.append("sacred_weapon_dev")
    if level >= 5:
        spells.append("霧步")

    c = Character(
        name=name, race="人類", class_="聖騎士", level=level,
        stats=Stats(STR=16, DEX=10, CON=14, INT=10, WIS=12, CHA=16),
        hp=10 + level * 8, max_hp=10 + level * 8, ac=18,
        weapons=[WEAPON_DEFS["長劍"]],
        spells=spells,
        spellcasting_ability="CHA" if level >= 2 else "",
        spell_slots=_paladin_slots(level),
        proficiencies=["WIS", "CHA", "宗教", "說服"],
        is_npc=False,
    )
    c.archetype_id = "devotion"
    c.known_abilities = abilities
    c.lay_on_hands_pool = 5 * level
    if level >= 2:
        c.ability_uses["channel_divinity"] = 1
    if level >= 5:
        c.attacks_per_action = 2
    if level >= 6:
        c.aura_of_protection_bonus = c.stats.modifier("CHA")
    return c


def make_vengeance_paladin(name: str = "Vengeance Paladin", level: int = 3) -> Character:
    """Paladin — Oath of Vengeance.

    L3: Vow of Enmity (Channel Divinity: advantage on one target)
    L5: Extra Attack
    L6: Aura of Protection
    """
    spells = []
    abilities = ["lay_on_hands_ability_ven"]
    if level >= 2:
        spells.extend(["祝福術"])
        abilities.extend(["divine_smite_ven", "bane_ven"])
    if level >= 3:
        abilities.extend(["vow_of_enmity_ven", "hunters_mark"])
    if level >= 5:
        spells.append("霧步")

    c = Character(
        name=name, race="人類", class_="聖騎士", level=level,
        stats=Stats(STR=16, DEX=10, CON=14, INT=10, WIS=12, CHA=16),
        hp=10 + level * 8, max_hp=10 + level * 8, ac=18,
        weapons=[WEAPON_DEFS["長劍"]],
        spells=spells,
        spellcasting_ability="CHA" if level >= 2 else "",
        spell_slots=_paladin_slots(level),
        proficiencies=["WIS", "CHA"],
        is_npc=False,
    )
    c.archetype_id = "vengeance"
    c.known_abilities = abilities
    c.lay_on_hands_pool = 5 * level
    if level >= 2:
        c.ability_uses["channel_divinity"] = 1
    if level >= 5:
        c.attacks_per_action = 2
    if level >= 6:
        c.aura_of_protection_bonus = c.stats.modifier("CHA")
    return c


# ── Quick lookup ─────────────────────────────────────────────────────────────


ARCHETYPE_FACTORIES = {
    "battle_master":    make_battle_master,
    "champion":         make_champion,
    "totem_bear":       make_totem_bear,
    "berserker":        make_berserker,
    "evocation":        make_evocation_wizard,
    "divination":       make_divination_wizard,
    "life":             make_life_cleric,
    "war":              make_war_cleric,
    "assassin":         make_assassin,
    "arcane_trickster": make_arcane_trickster,
    "devotion":         make_devotion_paladin,
    "vengeance":        make_vengeance_paladin,
}
