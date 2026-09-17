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
from .damage import DAMAGE_TYPES, DAMAGE_TYPE_INDEX, N_DAMAGE_TYPES
from .dice import dice_ev as _dice_ev, cantrip_multiplier as _cantrip_mult


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
# SKILL_DTYPE_START doubles as the frozen pre-dtype dim (53) — checkpoint
# migration (adapt_state_dict_for_skill_dtype) and the model's dtype-slice
# both key off it.
SKILL_DTYPE_START = (
    23                # scalar fields (12 original + 11 added)
    + N_SAVE_STATS    # save_stat one-hot
    + N_TARGET_TYPES  # target_type one-hot
    + N_STATUS_SLOTS  # applies_status multi-hot
)
# = 23 + 6 + 8 + 16 = 53
SKILL_FEATURE_DIM = SKILL_DTYPE_START + N_DAMAGE_TYPES   # + damage-type soft one-hot = 66

# Obs-side magnitude bound on the expected_damage feature (as_vector column 0).
# skill_proj is an UNNORMALISED nn.Linear, so a skill whose expected_damage is
# far outside the range the policy trained on (all trained kits ≤ 66 = a dragon's
# breath) corrupts the pooled skill embedding's direction → the trunk hidden
# state collapses ~3× → the policy falls back to passive turtling. This is the
# 2026-07-02 "OOD-graft → dodge-collapse" root cause (bug_miner, data-proven:
# clipping ONLY this column to ≤100 fully restored play; ‖h‖ 349→103→269).
# CAP sits in the empty (66, 85.5) gap: every ability a trained identity carries
# is ≤ 66, so clamping is a NO-OP on all real/trained scenarios (bit-exact, no
# retrain) and ONLY reins in OOD-grafted boss ultras (swallow 85.5 / breath 91 /
# kraken 149.5 / tarrasque 204) back onto the training manifold. Generic
# magnitude bound — NOT keyed on any skill name. Column index is pinned here
# next to as_vector() so it can't drift from the encoding order.
I_SKILL_EXPECTED_DAMAGE = 0
SKILL_EV_OBS_CAP = 80.0

# Sentinel damage-type token: "this skill deals damage of the wielder's weapon
# type" (maneuvers, frenzy …). Resolved to the actual weapon type by
# SkillFeatures.materialize — an unmaterialized template contributes no bits.
WEAPON_DTYPE = "@weapon"


def _cantrip_multiplier(caster_level: int) -> float:
    """5e cantrip damage scales at levels 5, 11, 17. Delegates to the shared
    dice.cantrip_multiplier so the engine roll and the obs EV use one table."""
    return float(_cantrip_mult(caster_level))


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

    # Damage type(s) of the packets `expected_damage` describes, as a
    # damage-share-weighted soft one-hot over DAMAGE_TYPES. Entries are either
    # a bare type string (share 1.0) or a (type, share) pair for multi-packet
    # skills (a stinger weapon with a poison rider). Shares should sum to 1
    # for damaging skills so that the obs-side dot product against an enemy's
    # typed-resist row reads as "expected damage multiplier − 1". The
    # WEAPON_DTYPE sentinel means "wielder's weapon type" and resolves in
    # materialize(); non-damaging skills leave this empty (all bits 0).
    damage_types: tuple = ()

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

    def iter_damage_types(self):
        """Normalized (type_token, share) pairs from `damage_types`.

        Accepts bare strings (share 1.0) and (type, share) pairs; tokens are
        DAMAGE_TYPES members or the WEAPON_DTYPE sentinel."""
        for entry in (self.damage_types or ()):
            if isinstance(entry, str):
                yield entry, 1.0
            else:
                yield entry[0], float(entry[1])

    def as_vector(self) -> np.ndarray:
        save_oh   = np.zeros(N_SAVE_STATS,   dtype=np.float32)
        target_oh = np.zeros(N_TARGET_TYPES, dtype=np.float32)
        if 0 <= self.save_stat < N_SAVE_STATS:
            save_oh[self.save_stat] = 1.0
        if 0 <= self.target_type < N_TARGET_TYPES:
            target_oh[self.target_type] = 1.0
        # Damage-share-weighted soft one-hot. Unresolved WEAPON_DTYPE (a raw
        # template never run through materialize) contributes nothing.
        dtype_v = np.zeros(N_DAMAGE_TYPES, dtype=np.float32)
        for tok, share in self.iter_damage_types():
            i = DAMAGE_TYPE_INDEX.get(tok, -1)
            if i >= 0:
                dtype_v[i] += share
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
            dtype_v,
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

        ab = None
        if skill_id:
            from .abilities import ABILITY_REGISTRY
            ab = ABILITY_REGISTRY.get(skill_id)

        # Remaining uses as a fraction [0.0, 1.0] for abilities with a pool.
        if ab and ab.max_uses > 0:
            current = char.ability_uses.get(skill_id, ab.max_uses)
            f.remaining_uses = float(current) / ab.max_uses

        # Build the ability's action ONCE with a probe target — the single
        # source for DERIVED expected_damage / damage_types / expected_healing,
        # so none of them can drift from the dice the engine actually rolls.
        # Weapon skills (no registry Ability) keep from_weapon's derived values.
        # NOTE: weapon-riding values (e.g. divine smite = swing + radiant rider)
        # now depend on the wielded weapon — correct, and consistent with how
        # save_dc / attack_vs_ac already materialize per character. Subsumes the
        # old WEAPON_DTYPE sentinel resolution (the action carries the real
        # weapon type) and the +3-placeholder healing adjustment (the HEAL dice
        # already carry the live spellcasting modifier).
        if ab is not None and ab.builder is not None:
            probe = ab.build_action(char.name, "__ev_probe__", Vec2(0.0, 0.0),
                                    char=char)
            base_ed = action_expected_damage(probe, char)
            if base_ed > 0:
                f.expected_damage = base_ed
            shares = action_damage_shares(probe, char)
            if shares:
                f.damage_types = shares
            heal = action_expected_healing(probe, char)
            if heal > 0:
                f.expected_healing = heal

        # Cantrip damage scaling: cost_slot_level == 0.0 identifies cantrips —
        # EXCEPT slot-free monster naturals (breath / swallow / eye rays), which
        # pin Ability.scales_as_cantrip=False. Scales the DERIVED EV (bit-exact
        # with the old hand-typed value for every cantrip, which is flat-mod-free
        # so ×mult equals _roll_scaled_cantrip's dice-count scaling).
        mult = _cantrip_multiplier(char.level)
        if (mult > 1.0 and self.cost_slot_level == 0.0 and f.expected_damage > 0
                and (ab is None or ab.scales_as_cantrip)):
            f.expected_damage = f.expected_damage * mult

        return f


def action_damage_types(action: dict | None, actor) -> set[str]:
    """Ground-truth damage-type set of a BUILT action dict — every typed
    packet the engine will deal when executing it (weapon base, weapon on_hit
    rider dice, divine-smite rider, auto-damage, spell damage). This is the
    engine-data path the dtype cross-check test and EV oracles validate
    against; SkillFeatures.damage_types must stay a subset of it.

    Aura/conferred damage (spirit guardians ticks, hunter's-mark procs) is
    dealt by status machinery, not by the casting action — those skills pin
    their dtype in the catalog and are asserted separately by the test.
    """
    if not action:
        return set()
    t = action.get("type")
    out: set[str] = set()
    if t in ("ATTACK", "MULTI_ATTACK"):
        w = actor.get_weapon(action.get("weapon", ""))
        if w is not None:
            if getattr(w, "damage_type", None):
                out.add(w.damage_type)
            rider = getattr(w, "on_hit", None) or {}
            if rider.get("damage_dice") and rider.get("damage_type"):
                out.add(rider["damage_type"])
        if action.get("divine_smite_slot", 0) > 0:
            out.add("光耀")   # combat._resolve_single_attack smite packet
        # Action-level damage rider (wrathful smite = +1d6 精神 on hit).
        if action.get("rider_damage_dice") and action.get("rider_damage_type"):
            out.add(action["rider_damage_type"])
    elif t == "AUTO_DAMAGE":
        if action.get("damage_type"):
            out.add(action["damage_type"])
    elif t == "MULTI_SPELL_ATTACK":
        if action.get("damage_type"):
            out.add(action["damage_type"])
    elif t == "SPELL":
        from .spells import SPELLS
        sp = SPELLS.get(action.get("spell_name", ""))
        if sp is not None and getattr(sp, "damage_dice", "") and sp.damage_type:
            out.add(sp.damage_type)
    elif t == "EYE_RAYS":
        for spec in action.get("table") or ():
            if spec.get("damage_dice") and spec.get("damage_type"):
                out.add(spec["damage_type"])
    return out


# Expected number of periodic ticks a restrain/swallow rider deals before the
# victim escapes or the source dies — the horizon the catalog's hand-computed
# swallow EV baked in (bite + 3×tick). Named so that EV stays *computed* from
# dice × an explicit horizon rather than a magic constant.
TICK_DAMAGE_HORIZON = 3


def _spell_attack_mod(action: dict, char) -> float:
    """Spellcasting-ability modifier the engine adds to a spell-attack's damage
    when add_spell_mod is set (default True) — mirrors combat._resolve_spell_
    attack_roll (raw = base + spell_mod). Cantrips/rays that pin
    add_spell_mod=False add nothing."""
    if not action.get("add_spell_mod", True):
        return 0.0
    sa = getattr(char, "spellcasting_ability", None)
    return float(char.stats.modifier(sa)) if sa else 0.0


def _weapon_damage_mod(weapon, char) -> int:
    """Ability modifier added to a weapon's damage — mirrors from_weapon /
    combat: STR for melee, DEX for ranged, max(STR,DEX) for finesse."""
    is_ranged = getattr(weapon, "range_type", "") == "遠程"
    is_finesse = "精巧" in getattr(weapon, "properties", ())
    if is_finesse:
        return max(char.stats.modifier("STR"), char.stats.modifier("DEX"))
    if is_ranged:
        return char.stats.modifier("DEX")
    return char.stats.modifier("STR")


def _damage_packets(action: dict, actor) -> "list[tuple[str, float]]":
    """Every (damage_type, expected_value) packet a BUILT action deals — the
    single structural source for both action_expected_damage (sum the EVs) and
    action_damage_shares (normalise the EVs). Mirrors the exact dice the engine
    rolls: weapon base+mod, weapon on_hit rider, action-level rider, divine
    smite, spell / spell-attack / multi-ray / auto-damage dice, eye-ray table,
    restrain/swallow ticks, and conferred aura/proc damage. EV is UNSCALED
    (cantrip level-scaling is layered on later by materialize)."""
    t = action.get("type")
    out: list[tuple[str, float]] = []
    if t in ("ATTACK", "MULTI_ATTACK"):
        w = actor.get_weapon(action.get("weapon", ""))
        if w is not None:
            base = max(1.0, _dice_ev(w.damage_dice) + _weapon_damage_mod(w, actor))
            if getattr(w, "damage_type", None):
                out.append((w.damage_type, base))
            rider = getattr(w, "on_hit", None) or {}
            rev = _dice_ev(rider.get("damage_dice", ""))
            if rev > 0 and rider.get("damage_type"):
                out.append((rider["damage_type"], rev))
        rev = _dice_ev(action.get("rider_damage_dice", ""))
        if rev > 0 and action.get("rider_damage_type"):
            out.append((action["rider_damage_type"], rev))
        slot = int(action.get("divine_smite_slot", 0) or 0)
        if slot > 0:
            out.append(("光耀", _dice_ev(f"{min(5, 1 + slot)}d8")))
        md = action.get("rider_metadata") or {}
        if md.get("tick_damage_dice") and md.get("tick_damage_type"):
            out.append((md["tick_damage_type"],
                        TICK_DAMAGE_HORIZON * _dice_ev(md["tick_damage_dice"])))
    elif t == "SPELL":
        from .spells import SPELLS
        sp = SPELLS.get(action.get("spell_name", ""))
        ev = _dice_ev(getattr(sp, "damage_dice", "")) if sp is not None else 0.0
        if ev > 0 and getattr(sp, "damage_type", ""):
            out.append((sp.damage_type, ev))
    elif t == "SPELL_ATTACK":
        ev = _dice_ev(action.get("damage_dice", "")) + _spell_attack_mod(action, actor)
        if ev > 0 and action.get("damage_type"):
            out.append((action["damage_type"], ev))
    elif t == "MULTI_SPELL_ATTACK":
        rays = action.get("ray_targets") or []
        ev = len(rays) * (_dice_ev(action.get("damage_dice", ""))
                          + _spell_attack_mod(action, actor))
        if ev > 0 and action.get("damage_type"):
            out.append((action["damage_type"], ev))
    elif t == "AUTO_DAMAGE":
        darts = sum(int(tg.get("darts", 1)) for tg in (action.get("targets") or []))
        ev = (darts or 1) * _dice_ev(action.get("damage_per", ""))
        if ev > 0 and action.get("damage_type"):
            out.append((action["damage_type"], ev))
    elif t == "EYE_RAYS":
        table = action.get("table") or ()
        n = int(action.get("n_rays", 1))
        by_type: dict[str, float] = {}
        for s in table:
            dt, dd = s.get("damage_type"), s.get("damage_dice")
            if dt and dd:
                by_type[dt] = by_type.get(dt, 0.0) + _dice_ev(dd) / len(table)
        out.extend((dt, ev * n) for dt, ev in by_type.items())
    # Conferred / aura damage dealt later by STATUS machinery (spirit-guardians
    # aura, hunter's-mark), keyed by skill_id — see CONFERRED_DAMAGE_DICE.
    from .abilities import CONFERRED_DAMAGE_DICE
    conf = CONFERRED_DAMAGE_DICE.get(action.get("skill_id", ""))
    if conf:
        dice, dtype = conf
        out.append((dtype, _dice_ev(dice)))
    return out


def action_expected_damage(action: dict | None, actor) -> float:
    """Ground-truth expected damage of a BUILT action dict — sum of every
    (type, EV) packet from _damage_packets. This is what expected_damage is
    DERIVED from so the RL number can never drift from the dice the engine
    rolls. Returns 0.0 for non-damaging actions."""
    if not action:
        return 0.0
    return sum(ev for _, ev in _damage_packets(action, actor))


def action_damage_shares(action: dict | None, actor) -> tuple:
    """EV-weighted soft one-hot of a BUILT action's damage types — the
    SkillFeatures.damage_types tuple ((type, share), …) DERIVED from the real
    per-packet EVs (so a weapon-riding smite reads as its true weapon+radiant
    split, not a hand-declared single type). Shares sum to 1 for a damaging
    action; () for a non-damaging one. Same packet source as expected_damage,
    so the two can never disagree."""
    if not action:
        return ()
    packets = _damage_packets(action, actor)
    by_type: dict[str, float] = {}
    for dt, ev in packets:
        by_type[dt] = by_type.get(dt, 0.0) + ev
    total = sum(by_type.values())
    if total <= 0:
        return ()
    return tuple((dt, ev / total) for dt, ev in by_type.items())


def action_expected_healing(action: dict | None, actor) -> float:
    """Ground-truth expected healing of a BUILT action — DERIVED from the dice
    or pool the engine actually restores, never a hand-typed constant:
    HEAL / MULTI_HEAL dice → dice.dice_ev (already carries the live spell mod);
    a pool MULTI_HEAL → its pool total; LAY_ON_HANDS → its per-use amount.
    Returns 0.0 for non-healing actions."""
    if not action:
        return 0.0
    t = action.get("type")
    if t in ("HEAL", "MULTI_HEAL"):
        if action.get("dice"):
            return _dice_ev(action["dice"])
        return float(action.get("pool", 0.0) or 0.0)
    if t == "LAY_ON_HANDS":
        return float(action.get("amount", 0.0) or 0.0)
    return 0.0


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

def _expected_dice(dice_str: str) -> float:
    """Expected value of an XdY[+Z] roll. Returns 0.0 on empty / unparseable.
    Thin alias for the canonical dice.dice_ev — kept for existing callers."""
    return _dice_ev(dice_str)


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

    # Damage-type soft one-hot. Base packet = the weapon's type; an on_hit
    # rider with its own typed dice (wyvern stinger poison, fire-touch burn)
    # adds a second packet, weighted by each packet's share of the total
    # expected damage — so the obs-side resist dot product reads as the
    # overall EV multiplier, not a binary tag.
    base_ed = max(1.0, _expected_dice(weapon.damage_dice) + dmg_mod)
    rider = getattr(weapon, "on_hit", None) or {}
    rider_ed = _expected_dice(rider.get("damage_dice", ""))
    rider_dt = rider.get("damage_type", "")
    if rider_ed > 0 and rider_dt:
        tot = base_ed + rider_ed
        dtypes = ((weapon.damage_type, base_ed / tot), (rider_dt, rider_ed / tot))
    else:
        dtypes = (weapon.damage_type,)

    feats = SkillFeatures(
        expected_damage=base_ed,
        range_m=weapon.range_normal,
        attack_vs_ac=float(attack_mod),
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        damage_types=dtypes,
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

def _requires_spellcasting(ab) -> bool:
    """True if ``ab`` is a class spell that execute_action would REJECT for a
    non-spellcaster: it routes through the SPELL handler (its builder emits a
    ``spell_name`` in the SPELLS registry) and carries no innate save DC
    (``save_dc_ability`` unset). Natural abilities that ride the SPELL pipeline
    (dragon breath: save_dc_ability set) and non-spell abilities (weapons/
    maneuvers, not in SPELLS) return False. Cached on the ability — the
    classification is static. Mirrors combat.execute_action's SPELL gate
    (``not caster.spellcasting_ability and not spell.save_dc_ability``)."""
    cached = getattr(ab, "_req_spellcasting", None)
    if cached is not None:
        return cached
    from .spells import SPELLS
    sn = None
    if ab.builder is not None:
        try:
            sn = ab.builder("_", "_", (0.0, 0.0)).get("spell_name")
        except Exception:
            sn = None
    sp = SPELLS.get(sn) if sn else None
    req = sp is not None and not getattr(sp, "save_dc_ability", None)
    try:
        ab._req_spellcasting = req
    except Exception:
        pass
    return req


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
        # A class spell can't be cast by a non-spellcaster — mirror execute_
        # action's SPELL gate so the policy is never OFFERED an action that would
        # ERROR. Without this, the sandbox handing a fighter a wizard cantrip
        # (chill_touch) makes the policy pick it, execute_action returns ERROR,
        # and _run_opponent_turn spends NO resource on the error and re-picks the
        # same greedy action up to the sub-action cap (0 damage + a wasted turn
        # spamming it — the exact GUI bug). No-op for real casters
        # (spellcasting_ability set) and for innate breath weapons.
        if not char.spellcasting_ability and _requires_spellcasting(ab):
            continue
        out.append(_from_ability(ab, char))

    out.append(dodge_skill(char))
    out.append(hide_skill(char))
    out.append(disengage_skill(char))
    return out


def reaction_skills(char, skill_ids) -> list[Skill]:
    """Candidate Skills for a REACTION decision point.

    Returns [DECLINE, <one Skill per legal reaction skill_id>], where DECLINE is
    the end-turn skill in slot 0 (picking it = decline the reaction). ``skill_ids``
    is the engine's already-legality-filtered option list (combat._legal_reactions
    → ReactionContext.options), so this never widens the legal set — it only
    materialises those ids into Skills the policy can observe and pick among.
    Reaction mechanics are fired by the engine on the chosen skill_id (the Skill's
    builder is unused for reactions), so a missing/None builder is fine."""
    from .abilities import ABILITY_REGISTRY
    out: list[Skill] = [end_turn_skill()]
    for sid in skill_ids:
        ab = ABILITY_REGISTRY.get(sid)
        if ab is not None:
            out.append(_from_ability(ab, char))
    return out


def kit_features(char) -> list[tuple[str, "SkillFeatures"]]:
    """The character's full KIT as (skill_id, char-materialized features).

    Source for the obs-v4 capability descriptor (MONSTER_CATALOG §4.1):
    unlike available_skills this does NOT gate on consumable state — no
    uses/slots/ammo/concentration filters — because the descriptor encodes
    appearance-inferable public info ("what can this creature do"), never
    hidden resource state ("what can it do right now"). Level gating stays
    (a L3 wizard genuinely has no fireball). Reactions ARE included (a
    shield-caster's threat profile is kit truth). The universal skills
    (END/MOVE/DODGE/HIDE/DISENGAGE) are excluded — zero information.

    Consumers must not read features.remaining_uses (materialize fills it
    from live resource state).
    """
    from .abilities import ABILITY_REGISTRY
    out: list[tuple[str, SkillFeatures]] = []
    for w in char.weapons:
        out.append((f"weapon:{w.name}", from_weapon(w, char).features))
    for skill_id in (char.known_abilities or []):
        ab = ABILITY_REGISTRY.get(skill_id)
        if ab is None or not ab.engine_ready:
            continue
        if char.level < ab.min_level:
            continue
        out.append((skill_id, ab.features.materialize(char, ab.skill_id)))
    return out
