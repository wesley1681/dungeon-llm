"""Skill — RL-facing abstraction over weapons, spells, and other turn-actions.

The point of this module is the **feature vector**: a fixed-length numeric
encoding of "what this skill does" that an RL policy reads instead of a skill
name. New skills with novel parameter combinations (different range, damage,
save type, status) generalize without retraining; only genuinely new
mechanics (new feature dimension) require schema extension.

Three layers:
  - SkillFeatures        the orthogonal numeric encoding (`.as_vector()` → np)
  - Skill                features + a builder that turns (actor, target,
                         coord) → engine action dict
  - available_skills()   enumerate everything a character can invoke this turn

Status slot indexing in STATUS_SLOTS is append-only — never reorder existing
entries, since trained policies index by position.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

import numpy as np

from .vec2 import Vec2


# ── Schema enums (orderings are frozen — append-only) ────────────────────────

class SaveStat(IntEnum):
    STR = 0
    DEX = 1
    CON = 2
    INT = 3
    WIS = 4
    CHA = 5

N_SAVE_STATS = 6   # save_stat = -1 → no save, all save-stat one-hot bits stay 0


class TargetType(IntEnum):
    SELF = 0
    SINGLE_ENEMY = 1
    SINGLE_ALLY = 2
    POINT = 3        # arbitrary (x, y) on the battlefield
    LINE = 4         # reserved
    CONE = 5         # reserved
    MULTI_ENEMY = 6  # caster picks up to `max_targets` enemies
    MULTI_ALLY = 7   # caster picks up to `max_targets` allies

N_TARGET_TYPES = 8


# Index = slot position in the multihot vector. Append-only.
STATUS_SLOTS: list[str] = [
    "dodging",       # 0
    "hidden",        # 1
    "poisoned",      # 2  reserved (no implementation yet)
    "restrained",    # 3
    "charmed",       # 4
    "frightened",    # 5
    "stunned",       # 6
    "prone",         # 7
    "blinded",       # 8
    "deafened",      # 9
    "grappled",      # 10
    "incapacitated", # 11
    "invisible",     # 12
    "paralyzed",     # 13
    "petrified",     # 14
    "unconscious",   # 15
]
N_STATUS_SLOTS = len(STATUS_SLOTS)   # 16 — leave room by extending the list


# Total feature-vector length (downstream code can introspect this).
SKILL_FEATURE_DIM = (
    23                # scalar fields (12 original + 11 added)
    + N_SAVE_STATS    # save_stat one-hot
    + N_TARGET_TYPES  # target_type one-hot
    + N_STATUS_SLOTS  # applies_status multi-hot
)
# = 23 + 6 + 8 + 16 = 53


def _cantrip_multiplier(caster_level: int) -> float:
    """5e cantrip damage scales at levels 5, 11, 17."""
    if caster_level >= 17:
        return 4.0
    if caster_level >= 11:
        return 3.0
    if caster_level >= 5:
        return 2.0
    return 1.0


# ── SkillFeatures ────────────────────────────────────────────────────────────

@dataclass
class SkillFeatures:
    """Orthogonal numeric encoding. All fields default to 0 / no-op."""

    # Effect magnitudes — zero means "doesn't do this"
    expected_damage:  float = 0.0
    expected_healing: float = 0.0
    status_duration:  float = 0.0   # rounds; 0 = instantaneous / no status

    # Geometry
    range_m:      float = 0.0       # caster → target / AOE centre
    aoe_radius_m: float = 0.0       # 0 = single target / no area

    # Resolution (mutually exclusive in well-formed skills, but the engine
    # supports hybrid attack+save for things like Battle Master maneuvers)
    attack_vs_ac: float = 0.0       # bonus added to d20 attack roll
    save_dc:      float = 0.0       # 0 = no save involved
    save_stat:    int   = -1        # SaveStat enum, or -1 for "no save"
    auto_hit:     bool  = False     # bypass attack roll AND save (e.g. magic missile)

    # Cost (what slots the skill consumes — see consume_resources)
    cost_action:     float = 0.0    # 0 or 1
    cost_bonus:      float = 0.0    # 0 or 1
    cost_reaction:   float = 0.0    # 0 or 1 — triggers on another's action
    cost_movement:   float = 0.0    # metres consumed (0 for non-move skills)
    cost_slot_level: float = 0.0    # spell slot level required (0 = no slot)
    remaining_uses:  float = 1.0    # available right now (slots left, ammo,
                                    # daily uses); 0 means can't use this turn

    # Resource interactions
    requires_concentration: bool = False   # mutex with other concentration spells
    grants_actions:         float = 0.0    # gives the caster N extra actions
                                            # (Action Surge, Haste). Usually 0 or 1.

    # Targeting
    target_type: int = TargetType.SELF
    max_targets: int = 1   # how many entities can be picked when target_type
                           # is MULTI_ENEMY / MULTI_ALLY; otherwise 1
    is_teleport: bool = False   # if True + target_type == POINT, position
                                # change ignores LoS / walls (Misty Step)

    # Multi-hot over STATUS_SLOTS — which conditions this skill applies
    applies_status: tuple[bool, ...] = field(
        default_factory=lambda: (False,) * N_STATUS_SLOTS
    )

    # Conferred numeric modifiers — what the resulting status/buff actually
    # does to the affected creature. Zero = no modifier on that axis.
    # These describe the *effect* of buffs like Bless/Hunter's Mark/Rage and
    # of debuffs that just shift numbers (Bane = −1d4 to rolls = −2.5).
    conferred_attack_mod: float = 0.0   # ±N to target's d20 attack roll
    conferred_ac_mod:     float = 0.0   # ±N to target's AC
    conferred_damage_mod: float = 0.0   # ±N to target's outgoing damage
    conferred_save_mod:   float = 0.0   # ±N to target's saving throws
    damage_resistance:    float = 0.0   # 0..1 fraction of incoming damage
                                         # cancelled (Rage = 0.5)

    def as_vector(self) -> np.ndarray:
        save_oh   = np.zeros(N_SAVE_STATS,   dtype=np.float32)
        target_oh = np.zeros(N_TARGET_TYPES, dtype=np.float32)
        if 0 <= self.save_stat < N_SAVE_STATS:
            save_oh[self.save_stat] = 1.0
        if 0 <= self.target_type < N_TARGET_TYPES:
            target_oh[self.target_type] = 1.0
        return np.concatenate([
            np.array([
                # original 12
                self.expected_damage, self.expected_healing, self.status_duration,
                self.range_m, self.aoe_radius_m,
                self.attack_vs_ac, self.save_dc,
                self.cost_action, self.cost_bonus, self.cost_movement,
                self.cost_slot_level, self.remaining_uses,
                # added 11
                float(self.auto_hit), self.cost_reaction,
                float(self.requires_concentration), self.grants_actions,
                float(self.max_targets), float(self.is_teleport),
                self.conferred_attack_mod, self.conferred_ac_mod,
                self.conferred_damage_mod, self.conferred_save_mod,
                self.damage_resistance,
            ], dtype=np.float32),
            save_oh,
            target_oh,
            np.array(self.applies_status, dtype=np.float32),
        ])

    def materialize(self, char, skill_id: str = "") -> "SkillFeatures":
        """Return a copy with dynamic fields filled from the live character.

        Called before building RL obs vectors so the agent always sees the
        real save DC, damage, and remaining uses — not the static template
        values from the catalog.
        """
        from copy import copy
        f = copy(self)

        # Save DC: 8 + proficiency_bonus + spellcasting_ability_modifier
        if char.spellcasting_ability:
            f.save_dc = float(
                8 + char.proficiency_bonus
                + char.stats.modifier(char.spellcasting_ability)
            )

        # Attack bonus vs AC (weapon attacks use STR by default)
        f.attack_vs_ac = float(char.stats.modifier("STR") + char.proficiency_bonus)

        # Remaining uses as a fraction [0.0, 1.0] for abilities with a pool.
        if skill_id:
            from .abilities import ABILITY_REGISTRY
            ab = ABILITY_REGISTRY.get(skill_id)
            if ab and ab.max_uses > 0:
                current = char.ability_uses.get(skill_id, ab.max_uses)
                f.remaining_uses = float(current) / ab.max_uses

        # Cantrip damage scaling: cost_slot_level == 0.0 identifies cantrips.
        mult = _cantrip_multiplier(char.level)
        if mult > 1.0 and self.cost_slot_level == 0.0 and self.expected_damage > 0:
            f.expected_damage = self.expected_damage * mult

        # Expected healing: templates embed a +3 placeholder spellcasting mod.
        # Replace with the character's actual spellcasting modifier.
        _PLACEHOLDER_MOD = 3.0
        if self.expected_healing > 0 and char.spellcasting_ability:
            actual_spell_mod = float(char.stats.modifier(char.spellcasting_ability))
            f.expected_healing = self.expected_healing + (actual_spell_mod - _PLACEHOLDER_MOD)

        return f


# ── Skill (features + action builder) ────────────────────────────────────────

ActionBuilder = Callable[[str, str | None, "tuple[float, float] | None"], dict | None]


@dataclass
class Skill:
    """A single invokable ability. `features` is what RL sees; `builder` turns
    a high-level choice (actor, optional target entity, optional coord) into
    the engine action dict execute_action consumes."""
    skill_id: str
    display_name: str
    features: SkillFeatures
    builder: ActionBuilder

    def build_action(self, actor_id: str,
                     target_entity_id: str | None = None,
                     target_coord: tuple[float, float] | None = None) -> dict | None:
        """Build an engine action dict. Auto-embeds ``skill_id`` so downstream
        code (BC encoder, engine validator) can identify the source skill
        without reverse-engineering from the dict shape.
        """
        action = self.builder(actor_id, target_entity_id, target_coord)
        if action is not None:
            action["skill_id"] = self.skill_id
        return action


# ── Factory helpers ──────────────────────────────────────────────────────────

_DICE_RE = re.compile(r"(\d+)d(\d+)\s*([+-]\s*\d+)?")


def _expected_dice(dice_str: str) -> float:
    """Expected value of an XdY[+Z] roll. Returns 0.0 on empty / unparseable."""
    if not dice_str:
        return 0.0
    m = _DICE_RE.match(dice_str.replace(" ", ""))
    if not m:
        return 0.0
    n, sides = int(m.group(1)), int(m.group(2))
    flat = int((m.group(3) or "0").replace(" ", ""))
    return n * (sides + 1) / 2.0 + flat


def _save_stat_index(stat: str) -> int:
    try:
        return SaveStat[stat.upper()].value
    except (KeyError, AttributeError):
        return -1


def _melee_damage_mod(char) -> int:
    """STR for melee; max(STR, DEX) for finesse weapons. Mirrors combat.py."""
    return char.stats.modifier("STR")


def from_weapon(weapon, char) -> Skill:
    """Project a Weapon onto a Skill. Damage = dice + ability modifier."""
    is_ranged = weapon.range_type == "遠程"
    is_finesse = "精巧" in weapon.properties
    if is_finesse:
        dmg_mod = max(char.stats.modifier("STR"), char.stats.modifier("DEX"))
    elif is_ranged:
        dmg_mod = char.stats.modifier("DEX")
    else:
        dmg_mod = char.stats.modifier("STR")
    attack_mod = dmg_mod + char.proficiency_bonus

    feats = SkillFeatures(
        expected_damage=max(1.0, _expected_dice(weapon.damage_dice) + dmg_mod),
        range_m=weapon.range_normal,
        attack_vs_ac=float(attack_mod),
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    )

    def builder(actor_id, target_id, coord):
        if not target_id:
            return None
        return {
            "type":     "ATTACK",
            "attacker": actor_id,
            "target":   target_id,
            "weapon":   weapon.name,
            "consumes": ["action"],
        }

    return Skill(
        skill_id=f"weapon:{weapon.name}",
        display_name=weapon.name,
        features=feats,
        builder=builder,
    )


def _status_multihot(*names: str) -> tuple[bool, ...]:
    return tuple(slot in names for slot in STATUS_SLOTS)


def dodge_skill(char) -> Skill:
    feats = SkillFeatures(
        cost_action=1.0,
        target_type=TargetType.SELF,
        applies_status=_status_multihot("dodging"),
        status_duration=1.0,
    )
    return Skill("dodge", "閃避", feats,
                 lambda a, t, c: {"type": "DODGE", "character": a, "consumes": ["action"]})


def hide_skill(char) -> Skill:
    feats = SkillFeatures(
        cost_action=1.0,
        target_type=TargetType.SELF,
        applies_status=_status_multihot("hidden"),
        status_duration=0.0,
    )
    return Skill("hide", "躲藏", feats,
                 lambda a, t, c: {"type": "HIDE", "character": a, "consumes": ["action"]})


def disengage_skill(char) -> Skill:
    feats = SkillFeatures(
        cost_action=1.0,
        target_type=TargetType.SELF,
    )
    return Skill("disengage", "脫身", feats,
                 lambda a, t, c: {"type": "DISENGAGE", "character": a, "consumes": ["action"]})


def move_skill(char, budget_m: float = 9.0) -> Skill:
    feats = SkillFeatures(
        cost_movement=budget_m,
        target_type=TargetType.POINT,
        range_m=budget_m,
    )

    def builder(actor_id, target_id, coord):
        if coord is not None:
            # Accept Vec2 (no __getitem__), tuple, or list uniformly.
            cx, cy = Vec2.coerce(coord).x, Vec2.coerce(coord).y
            return {"type": "MOVE", "character": actor_id,
                    "target_position": [float(cx), float(cy)],
                    "consumes": ["movement"]}
        if target_id:
            return {"type": "MOVE", "character": actor_id, "target": target_id,
                    "consumes": ["movement"]}
        return None

    return Skill("move", "移動", feats, builder)


def end_turn_skill() -> Skill:
    """No-op skill so the policy has an explicit END action."""
    feats = SkillFeatures(target_type=TargetType.SELF)
    return Skill("end", "結束", feats, lambda a, t, c: None)


def _from_ability(ab, char) -> Skill:
    """Build a Skill from a Ability, binding the live character to the
    builder so the Skill.builder signature stays (actor_id, target_id, coord)."""
    materialized = ab.features.materialize(char, ab.skill_id)
    _char = char

    def builder(actor_id: str, target_id, coord):
        if ab.builder is None:
            return None
        return ab.builder(actor_id, target_id, coord, char=_char)

    return Skill(
        skill_id=ab.skill_id,
        display_name=ab.display_name,
        features=materialized,
        builder=builder,
    )


# ── Enumeration ──────────────────────────────────────────────────────────────

def available_skills(char, world_state=None) -> list[Skill]:
    """List everything `char` can invoke this turn. Order is stable across
    calls so the policy's skill_idx stays consistent within an episode.

    Order: [END, MOVE, weapons..., class_abilities..., DODGE, HIDE]

    Ability filtering:
      - engine_ready=True and not is_reaction
      - char.level >= ab.min_level
      - max_uses == 0 (unlimited) or remaining uses > 0
      - "who can use it" is decided entirely by char.known_abilities; the
        ability definition no longer carries class_id / archetype_id.
    """
    from .abilities import ABILITY_REGISTRY

    out: list[Skill] = [end_turn_skill()]

    # MOVE is only listed if the character can actually move (no restrain/grapple)
    can_move = True
    for m in char.iter_modifiers():
        if m.on_speed_multiplier(char) <= 0.0:
            can_move = False
            break
    if can_move:
        out.append(move_skill(char))

    for w in char.weapons:
        # Skip ranged weapons whose ammo type is depleted.
        if w.ammo and not char.has_ammo(w.ammo):
            continue
        out.append(from_weapon(w, char))

    # Class abilities from known_abilities, filtered to what's usable now.
    for skill_id in (char.known_abilities or []):
        ab = ABILITY_REGISTRY.get(skill_id)
        if ab is None:
            continue
        if not ab.engine_ready or ab.is_reaction:
            continue
        if char.level < ab.min_level:
            continue
        if ab.max_uses > 0:
            remaining = char.ability_uses.get(skill_id, ab.max_uses)
            if remaining <= 0:
                continue
        if ab.is_usable is not None and not ab.is_usable(char):
            continue
        # If the ability declares a spell-slot cost in its features, require
        # that exact slot level. Most class abilities hardcode `slot_level` in
        # their builder (no upcast), so a permissive "any level ≥ N" check
        # over-allows the skill when only higher slots remain.
        slot_lvl = int(getattr(ab.features, "cost_slot_level", 0) or 0)
        if slot_lvl > 0 and char.spell_slots.get(slot_lvl, 0) <= 0:
            continue
        # Same concentration filter as spells — re-casting a concentration
        # buff while already concentrating just burns the slot.
        if getattr(ab.features, "requires_concentration", False) and char.concentrating_on:
            continue
        out.append(_from_ability(ab, char))

    out.append(dodge_skill(char))
    out.append(hide_skill(char))
    out.append(disengage_skill(char))
    return out
