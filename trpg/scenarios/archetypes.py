"""Data-driven archetype registry: a class IS a stat block + skills + traits.

A class/monster identity here is pure data (``ClassDef``): a numeric panel
(stats/HP/AC/equipment), a list of level-gated skill grants, and a list of
level-gated trait grants. ONE generic builder (``make_character``) turns any
ClassDef into a Character — adding a new class or monster means adding a data
entry, never a new factory function or an engine branch.

Traits are keyed by MECHANIC, not by class: ``sneak_attack`` / ``portent`` /
``aura_of_protection`` are registered once in ``TRAIT_APPLIERS`` and any
identity that lists them gets the behaviour. The engine already consumes them
through generic Character fields (sculpt_spells, crit_range, ...), so a future
monster bundling e.g. sneak_attack + sculpt_spells needs zero new code.

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

from dataclasses import dataclass, field
from typing import Callable

from ..engine.character import Character, Stats
from ..engine.items import WEAPON_DEFS, Consumable
from ..engine.status import Evasion, Hidden


# ── Grant primitives ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillGrant:
    """A known ability unlocked at ``min_level``.

    ``reaction=True`` additionally registers the skill in
    ``Character.reactions`` (engine auto-triggers it off-turn).
    """
    skill_id: str
    min_level: int = 1
    reaction: bool = False


@dataclass(frozen=True)
class TraitGrant:
    """A passive talent unlocked at ``min_level``.

    ``trait_id`` keys into ``TRAIT_APPLIERS``; ``params`` parameterise the
    mechanic (which die, which stat modifier, ...).
    """
    trait_id: str
    min_level: int = 1
    params: dict = field(default_factory=dict)


# ── Trait appliers — one per MECHANIC, shared across all identities ─────────

TRAIT_APPLIERS: dict[str, Callable[[Character, int, dict], None]] = {}


def _trait(trait_id: str):
    def deco(fn):
        TRAIT_APPLIERS[trait_id] = fn
        return fn
    return deco


@_trait("extra_attack")
def _extra_attack(c: Character, level: int, p: dict) -> None:
    c.attacks_per_action = p.get("attacks", 2)


@_trait("crit_range")
def _crit_range(c: Character, level: int, p: dict) -> None:
    c.crit_range = p["value"]


@_trait("sculpt_spells")
def _sculpt_spells(c: Character, level: int, p: dict) -> None:
    c.sculpt_spells = True


@_trait("sneak_attack")
def _sneak_attack(c: Character, level: int, p: dict) -> None:
    # 1d6 at L1, +1d6 every 2 levels (PHB rogue progression).
    c.sneak_attack_dice = f"{(level + 1) // 2}d{p.get('die', 6)}"


@_trait("lay_on_hands")
def _lay_on_hands(c: Character, level: int, p: dict) -> None:
    c.lay_on_hands_pool = p.get("per_level", 5) * level


@_trait("aura_of_protection")
def _aura_of_protection(c: Character, level: int, p: dict) -> None:
    c.aura_of_protection_bonus = c.stats.modifier(p.get("ability", "CHA"))


@_trait("portent")
def _portent(c: Character, level: int, p: dict) -> None:
    from ..engine.dice import roll as _roll
    c.portent_dice = [_roll("1d20") for _ in range(p.get("count", 2))]


@_trait("uses_flat")
def _uses_flat(c: Character, level: int, p: dict) -> None:
    c.ability_uses[p["skill"]] = p["uses"]


@_trait("uses_from_modifier")
def _uses_from_modifier(c: Character, level: int, p: dict) -> None:
    c.ability_uses[p["skill"]] = max(p.get("minimum", 1),
                                     c.stats.modifier(p["ability"]))


@_trait("uses_table")
def _uses_table(c: Character, level: int, p: dict) -> None:
    c.ability_uses[p["skill"]] = p["table"].get(level, p["default"])


@_trait("starting_status")
def _starting_status(c: Character, level: int, p: dict) -> None:
    c.add_status(p["status"](applied_round=0))


@_trait("damage_table")
def _damage_table(c: Character, level: int, p: dict) -> None:
    """Typed resistance/immunity/vulnerability/absorb table (Wave 1).
    params: {"multipliers": {damage_type: float}} — 0.5 resist, 0.0 immune,
    2.0 vulnerable, negative = absorb. Keys fail loudly on unknown types."""
    from ..engine.damage import validate_damage_type
    for dtype, mult in p["multipliers"].items():
        validate_damage_type(dtype, context=f"damage_table of {c.name}")
        c.damage_multipliers[dtype] = float(mult)


@_trait("condition_immunity")
def _condition_immunity(c: Character, level: int, p: dict) -> None:
    """params: {"conditions": ["poisoned", ...]} — names must exist in the
    status registry so a typo can't silently grant no-op immunity."""
    from ..engine.status import ALL_STATUS_CLASSES
    for name in p["conditions"]:
        if name not in ALL_STATUS_CLASSES:
            raise ValueError(f"condition_immunity: unknown status {name!r}")
        c.condition_immunities.append(name)


@_trait("pack_tactics")
def _pack_tactics(c: Character, level: int, p: dict) -> None:
    c.pack_tactics = True


@_trait("undead_fortitude")
def _undead_fortitude(c: Character, level: int, p: dict) -> None:
    c.undead_fortitude = True


@_trait("recharge")
def _recharge(c: Character, level: int, p: dict) -> None:
    """params: {"skill": skill_id, "on": 5} — 1d6 >= on at turn start
    restores a spent use (5e 'Recharge 5-6')."""
    c.recharge_abilities[p["skill"]] = int(p.get("on", 5))


@_trait("regeneration")
def _regeneration(c: Character, level: int, p: dict) -> None:
    """params: {"amount": 10, "blocked_by": ["火", "強酸"]} — heal `amount`
    at self_turn_start unless a blocking damage type landed since the
    previous turn start (troll; vampire in Wave 3). Types fail loudly."""
    from ..engine.damage import validate_damage_type
    blocked = tuple(
        validate_damage_type(d, context=f"regeneration of {c.name}")
        for d in p.get("blocked_by", ()))
    c.regeneration = {"amount": int(p["amount"]), "blocked_by": blocked}


# ── Wave 3 traits (legendary suite — MONSTER_CATALOG §3 C/D 級) ─────────────

@_trait("legendary_actions")
def _legendary_actions(c: Character, level: int, p: dict) -> None:
    """params: {"per_round": 3, "options": [{"ability": id, "cost": 1} |
    {"weapon": name, "cost": 1}, ...]}. Budget refills at self_turn_start;
    spending happens via combat_policy.run_legendary_actions. Options fail
    loudly: ability ids must be granted to this creature and slot-free
    (legendary casts never consume spell slots — 5e tables only list
    cantrips/naturals); weapon names must resolve somewhere get_weapon can
    find them (own kit or the global WEAPON_DEFS, behir-jaw pattern)."""
    from ..engine.abilities import ABILITY_REGISTRY
    from ..engine.items import WEAPON_DEFS
    options = list(p["options"])
    if not options:
        raise ValueError(f"legendary_actions of {c.name}: empty options")
    for o in options:
        cost = int(o.get("cost", 0))
        if cost < 1:
            raise ValueError(f"legendary_actions of {c.name}: cost must be "
                             f"≥1, got {o!r}")
        if "ability" in o:
            ab = ABILITY_REGISTRY.get(o["ability"])
            if ab is None or o["ability"] not in c.known_abilities:
                raise ValueError(
                    f"legendary_actions of {c.name}: ability {o['ability']!r} "
                    f"not granted to this creature")
            if ab.features.cost_slot_level > 0:
                raise ValueError(
                    f"legendary_actions of {c.name}: {o['ability']!r} costs a "
                    f"spell slot — legendary options must be slot-free")
        elif "weapon" in o:
            if (o["weapon"] not in WEAPON_DEFS
                    and all(w.name != o["weapon"] for w in c.weapons)):
                raise ValueError(
                    f"legendary_actions of {c.name}: unknown weapon "
                    f"{o['weapon']!r}")
        else:
            raise ValueError(f"legendary_actions of {c.name}: option needs "
                             f"'ability' or 'weapon': {o!r}")
    c.legendary_actions_max = int(p.get("per_round", 3))
    c.legendary_actions_remaining = c.legendary_actions_max
    c.legendary_options = options


@_trait("legendary_resistance")
def _legendary_resistance(c: Character, level: int, p: dict) -> None:
    """params: {"uses": 3} — per-combat pool; make_saving_throw turns a failed
    status-consequence save into a success while uses remain (5e X/day,
    encounters are single combats)."""
    c.legendary_resistance_uses = int(p.get("uses", 3))


@_trait("frightful_presence")
def _frightful_presence(c: Character, level: int, p: dict) -> None:
    """params: {"radius_m": 36, "rounds": 10, "dc_stat": "CHA"} — checked at
    each enemy's turn start (tick_aura_damage). DC = 8 + prof + dc_stat mod,
    reproducing the MM numbers from the stat panel like breath weapons do."""
    dc_stat = p.get("dc_stat", "CHA")
    c.frightful_presence = {
        "radius_m": float(p.get("radius_m", 36.0)),
        "rounds":   int(p.get("rounds", 10)),
        "dc":       8 + c.proficiency_bonus + c.stats.modifier(dc_stat),
    }


@_trait("death_throes")
def _death_throes(c: Character, level: int, p: dict) -> None:
    """params: {"damage_dice": "20d6", "damage_type": "火", "radius_m": 9,
    "save_stat": "DEX", "dc_stat": "CON"} — explodes on death (apply_damage
    pops the spec). DC computed from the stat panel (balor: 8+6+6 = 20)."""
    from ..engine.damage import validate_damage_type
    validate_damage_type(p["damage_type"], context=f"death_throes of {c.name}")
    c.death_throes = {
        "damage_dice": p["damage_dice"],
        "damage_type": p["damage_type"],
        "radius_m":    float(p.get("radius_m", 9.0)),
        "save_stat":   p.get("save_stat", "DEX"),
        "dc": 8 + c.proficiency_bonus + c.stats.modifier(p.get("dc_stat", "CON")),
    }


# ── Class definitions ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassDef:
    archetype_id: str
    default_name: str
    class_display: str          # 顯示用基礎職業名（戰士/法師/...）
    role: str                   # front / striker / support — 隊伍取樣用
    stat_block: dict
    hp_base: int                # hp = hp_base + level * hp_per_level
    hp_per_level: int
    ac: int
    weapons: tuple = ()         # WEAPON_DEFS keys
    proficiencies: tuple = ()
    consumables: tuple = ()     # (name, quantity, effect_type, effect_value)
    spell_ability: str = ""     # "" = non-caster
    spell_slots_table: dict | None = None   # level -> slots; 取 ≤level 的最大鍵
    spellcasting_min_level: int = 1
    skills: tuple = ()          # SkillGrant, declaration order = obs slot order
    traits: tuple = ()          # TraitGrant


def _slots_for_level(table: dict | None, level: int) -> dict:
    if not table:
        return {}
    keys = [k for k in table if k <= level]
    if not keys:
        return {}
    return dict(table[max(keys)])


_WIZARD_SLOTS = {
    1: {1: 2},
    2: {1: 3},
    3: {1: 4, 2: 2},
    4: {1: 4, 2: 3},
    5: {1: 4, 2: 3, 3: 2},
    6: {1: 4, 2: 3, 3: 3},
    7: {1: 4, 2: 3, 3: 3, 4: 1},
    8: {1: 4, 2: 3, 3: 3, 4: 2},
}
_CLERIC_SLOTS = _WIZARD_SLOTS   # Same progression L1-8
_PALADIN_SLOTS = {
    1: {},
    2: {1: 2},
    3: {1: 3},
    4: {1: 3},
    5: {1: 4, 2: 2},
    6: {1: 4, 2: 2},
    7: {1: 4, 2: 3},
    8: {1: 4, 2: 3},
}
_RAGE_USES_TABLE = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}


_CLASS_DEF_LIST = [
    # ── Fighter ──────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="battle_master", default_name="Battle Master",
        class_display="戰士", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=12, CHA=10),
        hp_base=10, hp_per_level=8, ac=17,    # 16 chain mail + 1 Defense style
        weapons=("長劍", "短弓"),
        proficiencies=("STR", "CON", "運動", "恐嚇"),
        skills=(
            SkillGrant("second_wind"),
            SkillGrant("action_surge", min_level=2),
            SkillGrant("trip_attack", min_level=3),
            SkillGrant("menacing_attack", min_level=3),
            SkillGrant("distracting_strike", min_level=3),
        ),
        traits=(
            TraitGrant("extra_attack", min_level=5),
        ),
    ),
    ClassDef(
        archetype_id="champion", default_name="Champion",
        class_display="戰士", role="front",
        stat_block=dict(STR=16, DEX=14, CON=14, INT=10, WIS=10, CHA=10),
        hp_base=10, hp_per_level=8, ac=17,    # 16 chain mail + 1 Defense style
        weapons=("長劍", "短弓"),
        proficiencies=("STR", "CON"),
        skills=(
            SkillGrant("second_wind"),
            SkillGrant("action_surge", min_level=2),
            SkillGrant("improved_critical", min_level=3),
        ),
        traits=(
            TraitGrant("crit_range", min_level=3, params={"value": 19}),
            TraitGrant("extra_attack", min_level=5),
        ),
    ),
    # ── Barbarian ────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="totem_bear", default_name="Totem Warrior",
        class_display="蠻人", role="front",
        stat_block=dict(STR=17, DEX=14, CON=16, INT=8, WIS=10, CHA=10),
        hp_base=12, hp_per_level=9, ac=14,
        weapons=("長劍",),
        proficiencies=("STR", "CON"),
        skills=(
            SkillGrant("rage"),
            SkillGrant("reckless_attack", min_level=2),
            SkillGrant("bear_totem", min_level=3),
        ),
        traits=(
            TraitGrant("extra_attack", min_level=5),
            TraitGrant("uses_table", params={
                "skill": "rage", "table": _RAGE_USES_TABLE, "default": 4}),
        ),
    ),
    ClassDef(
        archetype_id="berserker", default_name="Berserker",
        class_display="蠻人", role="front",
        stat_block=dict(STR=17, DEX=14, CON=16, INT=8, WIS=10, CHA=10),
        hp_base=12, hp_per_level=9, ac=14,
        weapons=("長劍",),
        proficiencies=("STR", "CON"),
        skills=(
            SkillGrant("rage"),
            SkillGrant("reckless_attack", min_level=2),
            SkillGrant("berserker_frenzy", min_level=3),
        ),
        traits=(
            TraitGrant("extra_attack", min_level=5),
            TraitGrant("uses_table", params={
                "skill": "rage", "table": _RAGE_USES_TABLE, "default": 4}),
        ),
    ),
    # ── Wizard ───────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="evocation", default_name="Evoker",
        class_display="法師", role="striker",
        stat_block=dict(STR=8, DEX=14, CON=14, INT=16, WIS=12, CHA=10),
        hp_base=6, hp_per_level=4, ac=12,
        proficiencies=("INT", "WIS", "奧秘", "歷史"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=(
            SkillGrant("firebolt"),        # at-will 攻擊戲法（不耗法術位）
            SkillGrant("magic_missile"),
            SkillGrant("shield_spell", reaction=True),
            SkillGrant("burning_hands_ev"),
            SkillGrant("web_ev", min_level=3),
            SkillGrant("misty_step", min_level=3),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("fireball_ev", min_level=5),
            SkillGrant("ice_storm_ev", min_level=7),
        ),
        traits=(
            TraitGrant("sculpt_spells", min_level=2),
        ),
    ),
    ClassDef(
        archetype_id="divination", default_name="Diviner",
        class_display="法師", role="striker",
        stat_block=dict(STR=8, DEX=14, CON=14, INT=16, WIS=12, CHA=10),
        hp_base=6, hp_per_level=4, ac=12,
        proficiencies=("INT", "WIS", "奧秘"),
        spell_ability="INT", spell_slots_table=_WIZARD_SLOTS,
        skills=(
            SkillGrant("firebolt"),        # at-will 攻擊戲法（不耗法術位）
            SkillGrant("magic_missile"),
            SkillGrant("shield_spell", reaction=True),
            SkillGrant("burning_hands_div"),
            SkillGrant("web_div", min_level=3),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("misty_step", min_level=3),
            SkillGrant("fireball_div", min_level=5),
        ),
        traits=(
            TraitGrant("portent", min_level=2, params={"count": 2}),
        ),
    ),
    # ── Cleric ───────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="life", default_name="Life Cleric",
        class_display="牧師", role="support",
        stat_block=dict(STR=12, DEX=10, CON=14, INT=10, WIS=16, CHA=12),
        hp_base=8, hp_per_level=6, ac=16,
        weapons=("長劍",),
        proficiencies=("WIS", "CHA", "醫療", "宗教"),
        spell_ability="WIS", spell_slots_table=_CLERIC_SLOTS,
        skills=(
            SkillGrant("cure_wounds"),
            SkillGrant("bless"),
            SkillGrant("sacred_flame"),
            SkillGrant("healing_word_life"),
            SkillGrant("channel_divinity_preserve_life", min_level=2),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("spiritual_weapon_life", min_level=3),
            SkillGrant("spiritual_weapon_attack_life", min_level=3),
            SkillGrant("mass_cure_wounds_life", min_level=5),
            SkillGrant("spirit_guardians", min_level=5),
        ),
        traits=(
            TraitGrant("uses_flat", min_level=2,
                       params={"skill": "channel_divinity", "uses": 1}),
        ),
    ),
    ClassDef(
        archetype_id="war", default_name="War Cleric",
        class_display="牧師", role="support",
        stat_block=dict(STR=15, DEX=10, CON=14, INT=10, WIS=16, CHA=10),
        hp_base=8, hp_per_level=6, ac=18,    # heavy armor
        weapons=("長劍",),
        proficiencies=("WIS", "CHA"),
        spell_ability="WIS", spell_slots_table=_CLERIC_SLOTS,
        skills=(
            SkillGrant("cure_wounds"),
            SkillGrant("bless"),
            SkillGrant("sacred_flame"),
            SkillGrant("guiding_bolt_war"),
            SkillGrant("war_priest_attack"),
            SkillGrant("channel_divinity_guided_strike", min_level=2),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("spiritual_weapon_war", min_level=3),
            SkillGrant("spiritual_weapon_attack_war", min_level=3),
            SkillGrant("spirit_guardians", min_level=5),
        ),
        traits=(
            TraitGrant("uses_flat", min_level=2,
                       params={"skill": "channel_divinity", "uses": 1}),
            TraitGrant("uses_from_modifier", params={
                "skill": "war_priest_attack", "ability": "WIS", "minimum": 1}),
        ),
    ),
    # ── Rogue ────────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="assassin", default_name="Assassin",
        class_display="盜賊", role="striker",
        stat_block=dict(STR=10, DEX=17, CON=12, INT=14, WIS=12, CHA=10),
        hp_base=8, hp_per_level=5, ac=14,
        weapons=("短劍", "短弓"),
        proficiencies=("DEX", "INT", "潛行", "開鎖", "察覺", "欺騙"),
        consumables=(("箭", 20, "ammo", ""),),
        skills=(
            SkillGrant("cunning_action_dash", min_level=2),
            SkillGrant("cunning_action_disengage", min_level=2),
            SkillGrant("cunning_action_hide", min_level=2),
            SkillGrant("assassinate", min_level=3),
            SkillGrant("uncanny_dodge_rogue", min_level=5, reaction=True),
            SkillGrant("evasion_rogue", min_level=7),
        ),
        traits=(
            TraitGrant("sneak_attack", params={"die": 6}),
            # Ambush flavour: enters combat already Hidden so the L3
            # Assassinate proxy (round 1 + hidden = auto-crit) actually
            # triggers. Hidden is COMBAT_ONLY → clears between encounters.
            TraitGrant("starting_status", params={"status": Hidden}),
            TraitGrant("starting_status", min_level=7,
                       params={"status": Evasion}),
        ),
    ),
    ClassDef(
        archetype_id="arcane_trickster", default_name="Arcane Trickster",
        class_display="盜賊", role="striker",
        stat_block=dict(STR=10, DEX=17, CON=12, INT=14, WIS=12, CHA=10),
        hp_base=8, hp_per_level=5, ac=14,
        weapons=("短劍", "短弓"),
        proficiencies=("DEX", "INT", "潛行", "察覺"),
        consumables=(("箭", 20, "ammo", ""),),
        spell_ability="INT", spell_slots_table={3: {1: 3}},
        spellcasting_min_level=3,
        skills=(
            SkillGrant("cunning_action_dash", min_level=2),
            SkillGrant("cunning_action_disengage", min_level=2),
            SkillGrant("cunning_action_hide", min_level=2),
            SkillGrant("magic_missile", min_level=3),
            SkillGrant("shield_spell", min_level=3, reaction=True),
            SkillGrant("color_spray", min_level=3),
            SkillGrant("sleep", min_level=3),
            SkillGrant("uncanny_dodge_rogue", min_level=5, reaction=True),
            SkillGrant("evasion_rogue", min_level=7),
        ),
        traits=(
            TraitGrant("sneak_attack", params={"die": 6}),
            TraitGrant("starting_status", min_level=7,
                       params={"status": Evasion}),
        ),
    ),
    # ── Paladin ──────────────────────────────────────────────────────────
    ClassDef(
        archetype_id="devotion", default_name="Devotion Paladin",
        class_display="聖騎士", role="front",
        stat_block=dict(STR=16, DEX=10, CON=14, INT=10, WIS=12, CHA=16),
        hp_base=10, hp_per_level=8, ac=18,
        weapons=("長劍",),
        proficiencies=("WIS", "CHA", "宗教", "說服"),
        spell_ability="CHA", spell_slots_table=_PALADIN_SLOTS,
        spellcasting_min_level=2,
        skills=(
            SkillGrant("lay_on_hands_ability"),
            SkillGrant("divine_smite_dev", min_level=2),
            SkillGrant("shield_of_faith_dev", min_level=2),
            SkillGrant("bless", min_level=2),
            SkillGrant("cure_wounds", min_level=2),
            SkillGrant("sacred_weapon_dev", min_level=3),
            SkillGrant("misty_step", min_level=5),
        ),
        traits=(
            TraitGrant("lay_on_hands", params={"per_level": 5}),
            TraitGrant("uses_flat", min_level=2,
                       params={"skill": "channel_divinity", "uses": 1}),
            TraitGrant("extra_attack", min_level=5),
            TraitGrant("aura_of_protection", min_level=6,
                       params={"ability": "CHA"}),
        ),
    ),
    ClassDef(
        archetype_id="vengeance", default_name="Vengeance Paladin",
        class_display="聖騎士", role="front",
        stat_block=dict(STR=16, DEX=10, CON=14, INT=10, WIS=12, CHA=16),
        hp_base=10, hp_per_level=8, ac=18,
        weapons=("長劍",),
        proficiencies=("WIS", "CHA"),
        spell_ability="CHA", spell_slots_table=_PALADIN_SLOTS,
        spellcasting_min_level=2,
        skills=(
            SkillGrant("lay_on_hands_ability_ven"),
            SkillGrant("divine_smite_ven", min_level=2),
            SkillGrant("bane_ven", min_level=2),
            SkillGrant("vow_of_enmity_ven", min_level=3),
            SkillGrant("hunters_mark", min_level=3),
            SkillGrant("misty_step", min_level=5),
        ),
        traits=(
            TraitGrant("lay_on_hands", params={"per_level": 5}),
            TraitGrant("uses_flat", min_level=2,
                       params={"skill": "channel_divinity", "uses": 1}),
            TraitGrant("extra_attack", min_level=5),
            TraitGrant("aura_of_protection", min_level=6,
                       params={"ability": "CHA"}),
        ),
    ),
]

CLASS_DEFS: dict[str, ClassDef] = {cd.archetype_id: cd for cd in _CLASS_DEF_LIST}

# The 12 standard classes, frozen HERE — at module-body time, before any
# runtime registration (monsters, synth identities) can mutate CLASS_DEFS.
# Anything that means "the standard classes" (obs one-hot layout, env
# sampling, eval panels) must snapshot from THIS constant, never from a
# live registry whose content depends on import order.
STANDARD_ARCHETYPES: tuple[str, ...] = tuple(CLASS_DEFS.keys())


# ── Generic builder ──────────────────────────────────────────────────────────


def make_character(archetype_id: str, name: str | None = None, level: int = 3,
                   portent_rolls: list | None = None) -> Character:
    """Build a Character from its ClassDef — the ONLY construction path.

    ``portent_rolls``: explicit stored-d20 override for identities with the
    ``portent`` trait (skips rolling, matching the old divination factory API).
    """
    cd = CLASS_DEFS[archetype_id]
    hp = cd.hp_base + level * cd.hp_per_level
    spell_ability = cd.spell_ability if level >= cd.spellcasting_min_level else ""
    granted = [g for g in cd.skills if level >= g.min_level]

    c = Character(
        name=name or cd.default_name, race="人類", class_=cd.class_display,
        level=level,
        stats=Stats(**cd.stat_block),
        hp=hp, max_hp=hp, ac=cd.ac,
        weapons=[WEAPON_DEFS[w] for w in cd.weapons],
        consumables=[Consumable(name=n, quantity=q, effect_type=t, effect_value=v)
                     for (n, q, t, v) in cd.consumables],
        spellcasting_ability=spell_ability,
        spell_slots=_slots_for_level(cd.spell_slots_table, level)
                    if spell_ability else {},
        proficiencies=list(cd.proficiencies),
        reactions=[g.skill_id for g in granted if g.reaction],
        is_npc=False,
    )
    c.archetype_id = cd.archetype_id
    c.known_abilities = [g.skill_id for g in granted]
    for t in cd.traits:
        if level < t.min_level:
            continue
        if t.trait_id == "portent" and portent_rolls is not None:
            continue   # explicit override below, don't consume RNG
        TRAIT_APPLIERS[t.trait_id](c, level, t.params)
    if portent_rolls:
        c.portent_dice = list(portent_rolls)
    return c


# ── Public registries (API-compatible with the old factory module) ──────────


def _factory(archetype_id: str):
    def make(name: str | None = None, level: int = 3, **overrides) -> Character:
        return make_character(archetype_id, name=name, level=level, **overrides)
    make.__name__ = f"make_{archetype_id}"
    make.__doc__ = f"Build a {archetype_id} Character from CLASS_DEFS."
    return make


ARCHETYPE_FACTORIES = {aid: _factory(aid) for aid in CLASS_DEFS}

# Tactical role of each archetype, used to compose sensible parties
# (1 front + 1 striker + 1 support) for team-mode training/eval.
#   front:   melee line — soaks hits, fights at reach
#   striker: ranged/burst damage and control from the back or flanks
#   support: healing and buffs
ARCHETYPE_ROLES = {aid: cd.role for aid, cd in CLASS_DEFS.items()}

# Backward-compatible named factories.
make_battle_master    = ARCHETYPE_FACTORIES["battle_master"]
make_champion         = ARCHETYPE_FACTORIES["champion"]
make_totem_bear       = ARCHETYPE_FACTORIES["totem_bear"]
make_berserker        = ARCHETYPE_FACTORIES["berserker"]
make_evocation_wizard = ARCHETYPE_FACTORIES["evocation"]
make_divination_wizard = ARCHETYPE_FACTORIES["divination"]
make_life_cleric      = ARCHETYPE_FACTORIES["life"]
make_war_cleric       = ARCHETYPE_FACTORIES["war"]
make_assassin         = ARCHETYPE_FACTORIES["assassin"]
make_arcane_trickster = ARCHETYPE_FACTORIES["arcane_trickster"]
make_devotion_paladin = ARCHETYPE_FACTORIES["devotion"]
make_vengeance_paladin = ARCHETYPE_FACTORIES["vengeance"]
