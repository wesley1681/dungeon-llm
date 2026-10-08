"""Composable creation of levelled humanoid Characters.

Concrete stats, HP, AC and equipment live on Character.  Race and class
definitions contain only rules, proficiencies and grants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.character import Character, Stats
from ..engine.items import (
    ARMOR_DEFS, WEAPON_DEFS, Armor, Consumable, Shield, Weapon, calculate_ac,
)
from ..engine.progression import GrowthProfile, STAT_NAMES


@dataclass(frozen=True)
class LevelGrant:
    skill_id: str
    min_level: int = 1
    reaction: bool = False


@dataclass(frozen=True)
class RaceDef:
    race_id: str
    display_name: str
    growth_multipliers: dict[str, float] = field(default_factory=dict)
    traits: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassDef:
    class_id: str
    display_name: str
    growth_weights: dict[str, float]
    weapon_proficiencies: tuple[str, ...] = ()
    armor_proficiencies: tuple[str, ...] = ()
    saving_throw_proficiencies: tuple[str, ...] = ()
    skill_proficiencies: tuple[str, ...] = ()
    skills: tuple[LevelGrant, ...] = ()
    spell_ability: str = ""
    spell_slots_table: dict[int, dict[int, int]] | None = None
    spellcasting_min_level: int = 1


@dataclass(frozen=True)
class ArchetypeTrait:
    trait_id: str
    min_level: int = 1
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ArchetypeDef:
    archetype_id: str
    class_id: str
    skills: tuple[LevelGrant, ...] = ()
    traits: tuple[ArchetypeTrait, ...] = ()
    spell_ability: str = ""
    spell_slots_table: dict[int, dict[int, int]] | None = None
    spellcasting_min_level: int = 1


def _weights(**values: float) -> dict[str, float]:
    return {stat: float(values.get(stat, 0.0)) for stat in STAT_NAMES}


RACE_DEFS: dict[str, RaceDef] = {
    # Human is neutral: it preserves the class direction instead of imposing a
    # second direction of its own.
    "human": RaceDef("human", "人類", _weights(
        STR=1, DEX=1, CON=1, INT=1, WIS=1, CHA=1)),
    "dwarf": RaceDef("dwarf", "矮人", _weights(
        STR=1, DEX=1, CON=1.3, INT=1, WIS=1, CHA=1)),
    "goblin": RaceDef("goblin", "地精", _weights(
        STR=0.8, DEX=1.3, CON=1.0, INT=1.0, WIS=1.0, CHA=1.0)),
    "elf": RaceDef("elf", "精靈", _weights(
        STR=0.9, DEX=1.2, CON=0.9, INT=1.1, WIS=1.1, CHA=1.0)),
    "orc": RaceDef("orc", "獸人", _weights(
        STR=1.3, DEX=0.9, CON=1.2, INT=0.8, WIS=1.0, CHA=0.9)),
}


_FULL_CASTER_SLOTS = {
    1: {1: 2}, 2: {1: 3}, 3: {1: 4, 2: 2}, 4: {1: 4, 2: 3},
    5: {1: 4, 2: 3, 3: 2}, 6: {1: 4, 2: 3, 3: 3},
    7: {1: 4, 2: 3, 3: 3, 4: 1}, 8: {1: 4, 2: 3, 3: 3, 4: 2},
}
_HALF_CASTER_SLOTS = {
    1: {}, 2: {1: 2}, 3: {1: 3}, 4: {1: 3},
    5: {1: 4, 2: 2}, 6: {1: 4, 2: 2}, 7: {1: 4, 2: 3}, 8: {1: 4, 2: 3},
}


CLASS_DEFS: dict[str, ClassDef] = {
    "fighter": ClassDef(
        "fighter", "戰士",
        _weights(STR=.35, DEX=.15, CON=.30, INT=.05, WIS=.10, CHA=.05),
        weapon_proficiencies=("simple", "martial"),
        armor_proficiencies=("light", "medium", "heavy", "shield"),
        saving_throw_proficiencies=("STR", "CON"),
        skill_proficiencies=("運動", "恐嚇"),
        skills=(LevelGrant("second_wind"), LevelGrant("action_surge", 2)),
    ),
    "barbarian": ClassDef(
        "barbarian", "蠻人",
        _weights(STR=.35, DEX=.10, CON=.35, INT=.03, WIS=.12, CHA=.05),
        weapon_proficiencies=("simple", "martial"),
        armor_proficiencies=("light", "medium", "shield"),
        saving_throw_proficiencies=("STR", "CON"),
        skill_proficiencies=("運動", "察覺"),
        skills=(LevelGrant("rage"), LevelGrant("reckless_attack", 2)),
    ),
    "rogue": ClassDef(
        "rogue", "盜賊",
        _weights(STR=.05, DEX=.40, CON=.15, INT=.20, WIS=.12, CHA=.08),
        weapon_proficiencies=("simple", "finesse", "shortbow"),
        armor_proficiencies=("light",),
        saving_throw_proficiencies=("DEX", "INT"),
        skill_proficiencies=("潛行", "開鎖", "察覺", "欺騙"),
        skills=(
            LevelGrant("cunning_action_dash", 2),
            LevelGrant("cunning_action_disengage", 2),
            LevelGrant("cunning_action_hide", 2),
        ),
    ),
    "wizard": ClassDef(
        "wizard", "法師",
        _weights(STR=.03, DEX=.15, CON=.15, INT=.42, WIS=.18, CHA=.07),
        weapon_proficiencies=("simple",),
        saving_throw_proficiencies=("INT", "WIS"),
        skill_proficiencies=("奧秘", "歷史"),
        skills=(LevelGrant("firebolt"), LevelGrant("magic_missile")),
        spell_ability="INT", spell_slots_table=_FULL_CASTER_SLOTS,
    ),
    "cleric": ClassDef(
        "cleric", "牧師",
        _weights(STR=.12, DEX=.08, CON=.20, INT=.08, WIS=.40, CHA=.12),
        weapon_proficiencies=("simple",), armor_proficiencies=("light", "medium", "shield"),
        saving_throw_proficiencies=("WIS", "CHA"),
        skill_proficiencies=("醫療", "宗教"),
        skills=(LevelGrant("cure_wounds"), LevelGrant("sacred_flame")),
        spell_ability="WIS", spell_slots_table=_FULL_CASTER_SLOTS,
    ),
    "paladin": ClassDef(
        "paladin", "聖騎士",
        _weights(STR=.30, DEX=.05, CON=.25, INT=.05, WIS=.10, CHA=.25),
        weapon_proficiencies=("simple", "martial"),
        armor_proficiencies=("light", "medium", "heavy", "shield"),
        saving_throw_proficiencies=("WIS", "CHA"),
        skill_proficiencies=("宗教", "說服"),
        skills=(LevelGrant("lay_on_hands_ability"), LevelGrant("divine_smite_dev", 2)),
        spell_ability="CHA", spell_slots_table=_HALF_CASTER_SLOTS,
        spellcasting_min_level=2,
    ),
    "commoner": ClassDef(
        "commoner", "平民",
        _weights(STR=1, DEX=1, CON=1, INT=1, WIS=1, CHA=1),
    ),
    "warrior": ClassDef(
        "warrior", "戰士",
        _weights(STR=.30, DEX=.25, CON=.25, INT=.05, WIS=.10, CHA=.05),
        weapon_proficiencies=("simple", "martial"),
        armor_proficiencies=("light", "medium", "shield"),
        saving_throw_proficiencies=("STR", "CON"),
    ),
    "shaman": ClassDef(
        "shaman", "薩滿",
        _weights(STR=.05, DEX=.10, CON=.20, INT=.10, WIS=.45, CHA=.10),
        weapon_proficiencies=("simple",), armor_proficiencies=("light",),
        saving_throw_proficiencies=("WIS", "CON"),
        skill_proficiencies=("奧秘",),
        skills=(LevelGrant("fire_bolt"),),
        spell_ability="WIS", spell_slots_table=_FULL_CASTER_SLOTS,
    ),
}


_ARCHETYPE_CLASS_IDS = {
    "battle_master": "fighter", "champion": "fighter",
    "totem_bear": "barbarian", "berserker": "barbarian",
    "evocation": "wizard", "divination": "wizard",
    "life": "cleric", "war": "cleric",
    "assassin": "rogue", "arcane_trickster": "rogue",
    "devotion": "paladin", "vengeance": "paladin",
}


def _load_archetype_defs() -> dict[str, ArchetypeDef]:
    """Bridge the existing ability catalog without importing its panel data."""
    from .archetypes import COMBAT_TEMPLATE_DEFS
    definitions: dict[str, ArchetypeDef] = {}
    for archetype_id, class_id in _ARCHETYPE_CLASS_IDS.items():
        source = COMBAT_TEMPLATE_DEFS[archetype_id]
        definitions[archetype_id] = ArchetypeDef(
            archetype_id=archetype_id,
            class_id=class_id,
            skills=tuple(LevelGrant(g.skill_id, g.min_level, g.reaction)
                         for g in source.skills),
            traits=tuple(ArchetypeTrait(t.trait_id, t.min_level, dict(t.params))
                         for t in source.traits),
            spell_ability=source.spell_ability,
            spell_slots_table=source.spell_slots_table,
            spellcasting_min_level=source.spellcasting_min_level,
        )
    return definitions


ARCHETYPE_DEFS: dict[str, ArchetypeDef] = _load_archetype_defs()


def _slots_for_level(table: dict[int, dict[int, int]] | None, level: int) -> dict[int, int]:
    if not table:
        return {}
    eligible = [entry_level for entry_level in table if entry_level <= level]
    return dict(table[max(eligible)]) if eligible else {}


def _resolve_equipment(items: tuple | list) -> list[Armor | Shield | object]:
    return [ARMOR_DEFS[item] if isinstance(item, str) else item for item in items]


def _resolve_weapons(items: tuple | list) -> list[Weapon]:
    return [WEAPON_DEFS[item] if isinstance(item, str) else item for item in items]


def build_character(
    *, name: str, race_id: str, class_id: str, level: int = 1,
    archetype_id: str = "",
    growth_seed: int = 0, growth_points_per_level: int = 2,
    base_stats: Stats | None = None, base_max_hp: int = 8,
    hp_per_level: int = 5, equipment: tuple | list = (),
    weapons: tuple | list = (), consumables: tuple | list = (),
    known_abilities: tuple | list = (), spell_slots: dict | None = None,
    reactions: tuple | list = (), spellcasting_ability: str | None = None,
    level_locked: bool = False,
    is_npc: bool = False, **character_options,
) -> Character:
    """Build one Character; only Character receives concrete panel values."""
    if level < 1:
        raise ValueError("角色等級至少為 1")
    race = RACE_DEFS[race_id]
    class_def = CLASS_DEFS[class_id]
    archetype = ARCHETYPE_DEFS.get(archetype_id) if archetype_id else None
    if archetype_id and archetype is None:
        raise KeyError(f"未知子職業：{archetype_id}")
    if archetype is not None and archetype.class_id != class_id:
        raise ValueError(
            f"{archetype_id} 屬於 {archetype.class_id}，不能套用到 {class_id}"
        )
    stats = Stats(**vars(base_stats)) if base_stats is not None else Stats()
    profile = GrowthProfile(
        class_weights=dict(class_def.growth_weights),
        race_multipliers=dict(race.growth_multipliers),
        points_per_level=growth_points_per_level,
    )
    resolved_equipment = _resolve_equipment(equipment)
    resolved_weapons = _resolve_weapons(weapons)
    saves = list(class_def.saving_throw_proficiencies)
    skills = list(class_def.skill_proficiencies)
    all_grants = list(class_def.skills) + list(archetype.skills if archetype else ())
    grants = [grant for grant in all_grants if level >= grant.min_level]
    ability_ids = list(dict.fromkeys(
        [grant.skill_id for grant in grants] + list(known_abilities)
    ))
    spell_source = archetype if archetype and archetype.spell_ability else class_def
    casting_enabled = level >= spell_source.spellcasting_min_level
    casting_stat = (
        spellcasting_ability if spellcasting_ability is not None
        else spell_source.spell_ability if casting_enabled else ""
    )
    initial_max_hp = max(1, int(base_max_hp) + stats.modifier("CON"))

    c = Character(
        name=name, race=race.display_name, class_=class_def.display_name,
        race_id=race_id, class_id=class_id, level=1,
        stats=stats, hp=initial_max_hp, max_hp=initial_max_hp, ac=10,
        growth_profile=profile, growth_seed=int(growth_seed),
        level_locked=level_locked,
        base_stats=Stats(**vars(stats)), base_max_hp=int(base_max_hp),
        hp_per_level=int(hp_per_level),
        derive_ac_from_equipment=True,
        equipment=resolved_equipment, weapons=resolved_weapons,
        consumables=list(consumables),
        weapon_proficiencies=list(class_def.weapon_proficiencies),
        armor_proficiencies=list(class_def.armor_proficiencies),
        saving_throw_proficiencies=saves,
        skill_proficiencies=skills,
        proficiencies=saves + skills,
        known_abilities=ability_ids,
        reactions=list(dict.fromkeys(
            [grant.skill_id for grant in grants if grant.reaction] + list(reactions)
        )),
        spellcasting_ability=casting_stat,
        spell_slots=(dict(spell_slots) if spell_slots is not None else
                     _slots_for_level(spell_source.spell_slots_table, level)
                     if casting_stat else {}),
        is_npc=is_npc,
        **character_options,
    )
    c.ac = calculate_ac(c)
    c.archetype_id = archetype_id
    if level_locked:
        c.level = level
    elif level > 1:
        c.level_up_to(level)
    if archetype is not None:
        from .archetypes import TRAIT_APPLIERS
        for trait in archetype.traits:
            if level >= trait.min_level:
                TRAIT_APPLIERS[trait.trait_id](c, level, trait.params)
    return c
