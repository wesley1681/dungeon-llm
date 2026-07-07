"""Phase 2 RL observation extractors.

Each function reads from WorldState and produces a single numpy array. The
shapes and dtypes match the spec at docs/superpowers/specs/2026-05-19-rl-env-v2-design.md.
"""
from __future__ import annotations
import numpy as np

from ..engine.character import Character
from ..engine.world_state import WorldState
from ..engine.skill import (available_skills, kit_features, SKILL_FEATURE_DIM,
                            STATUS_SLOTS, N_STATUS_SLOTS, N_SAVE_STATS,
                            I_SKILL_EXPECTED_DAMAGE, SKILL_EV_OBS_CAP)
from ..engine.combat import MOVE_BUDGET_M
from ..engine.status import MODIFIER_CLASSES
from ..engine.vec2 import Battlefield, TerrainType, Vec2
from ..scenarios.archetypes import ARCHETYPE_FACTORIES


# ── Schema constants ─────────────────────────────────────────────────────────

N_SKILL_SLOTS = 20            # max skills per turn (pad with zeros)
# Entity slot layout: [self, ally_1..N_ALLY_SLOTS, enemy_1..N_ENEMY_SLOTS].
# v3 (2026-06-11) widened from 2 ally + 3 enemy: a 4th enemy used to be
# INVISIBLE (silently truncated). 3 allies = a full party of four;
# 6 enemies = headroom for monster packs. Everything that maps slot↔char_id
# must use these constants (obs.entities_obs, action._entity_id_at_slot /
# _entity_slot_of, model.apply_entity_mask) — never literal offsets.
N_ALLY_SLOTS = 3
N_ENEMY_SLOTS = 6
N_ENTITY_SLOTS = 1 + N_ALLY_SLOTS + N_ENEMY_SLOTS    # 10
ENEMY_SLOT_START = 1 + N_ALLY_SLOTS                  # first enemy row index
# Stable archetype ordering for the per-entity one-hot. Multi-hot in shape so a
# future multiclass character (paladin/sorcerer) can light up multiple bits;
# single-class characters just have one bit set. Sorted alphabetically so the
# mapping is deterministic across Python versions / dict orderings.
# Snapshots STANDARD_ARCHETYPES (frozen at archetypes.py module body), NOT the
# live ARCHETYPE_FACTORIES — runtime registrations (monsters/synths) must never
# shift obs dims regardless of import order.
from ..scenarios.archetypes import STANDARD_ARCHETYPES
ARCHETYPE_OBS_LIST: tuple[str, ...] = tuple(sorted(STANDARD_ARCHETYPES))
N_ARCHETYPES = len(ARCHETYPE_OBS_LIST)   # 12

# Per-entity status multi-hot. Union of canonical D&D conditions and every
# implemented buff/debuff in the engine registry so adding a new StatusEffect
# subclass (with a MODIFIER_CLASSES entry) auto-extends the obs.
# D&D-fair: these correspond to visible buff/condition icons at the table.
RL_STATUS_NAMES: tuple[str, ...] = tuple(
    sorted(set(STATUS_SLOTS) | set(MODIFIER_CLASSES.keys()))
)
N_RL_STATUS = len(RL_STATUS_NAMES)

# obs v3 per-entity tail (2026-06-11): threat & downed-state scalars, appended
# AFTER is_concentrating so every pre-v3 column index stays valid (model code
# reads cols 1,2 = x,y; 5 = is_alive; 6 = is_self; 7.. = archetype one-hot).
#   level / LEVEL_NORM    threat scale — full-HP L2 and L8 used to be
#                          mathematically identical; also the critic's only
#                          way to see encounter difficulty (cross-difficulty
#                          value miscalibration fix)
#   max_hp / MAXHP_NORM   absolute HP scale — half-dead ogre ≠ half-dead goblin
#   ac / AC_NORM          hit difficulty
#   is_dying              PC at 0 HP on death saves (revive / finish-off target);
#                          1.0 includes stabilised-at-0 (still down, still
#                          needs a pick-up)
#   death successes / 3   1.0 = stabilised (no longer rolling)
#   death failures / 3    pick-up urgency
# These also give future MONSTERS a threat prior even though their archetype
# one-hot is all zeros (they're not in ARCHETYPE_FACTORIES).
#
# obs v4 (2026-06-12, MONSTER_CATALOG §4): LEVEL/MAXHP rescaled for the
# monster band — CR 30 / Tarrasque-676-HP must stay in [0,1]. The new NORMs
# are POWER-OF-2 multiples of the v3 values (20→40 = ×2, 100→800 = ×8) so
# checkpoint migration can compensate by scaling the matching input-weight
# columns — bit-exact in fp32 (exponent shifts only, rounding commutes).
LEVEL_NORM = 40.0   # v3: 20.0 — covers CR 30 natural levels
MAXHP_NORM = 800.0  # v3: 100.0 — covers Tarrasque (676)
AC_NORM = 20.0      # unchanged (monster AC tops out ~25; 1.25 measured harmless)
V3_LEVEL_NORM = 20.0   # frozen era constants for checkpoint migration
V3_MAXHP_NORM = 100.0
N_V3_EXTRA = 6

# obs v4 per-entity capability descriptor (MONSTER_CATALOG §4.1): a fixed-
# length mechanic summary of the entity's KIT, aggregated from char-
# materialized SkillFeatures (engine.skill.kit_features). Solves the enemy-
# identity blind spot: monsters/synths have an all-zero archetype one-hot,
# and without this the policy cannot see whether a stranger heals, casts,
# kites, or paralyzes. Appended AFTER the v3 tail so every pre-v4 column
# index stays valid (the same strict-prefix trick the v3 migration used).
# Public-info principle: aggregated over the full kit, NEVER filtered by
# remaining uses/slots — resource state stays hidden.
#   [0] max_hit_damage / DESC_DMG_NORM   biggest per-action damage option
#                                         (weapon EV × attacks_per_action)
#   [1] max_heal / DESC_HEAL_NORM        can it heal, and how hard
#   [2] max_attack_range / DESC_RANGE_NORM   melee-only vs ranged threat
#   [3] max_aoe_radius / DESC_AOE_NORM   AoE threat
#   [4] best_attack_bonus / DESC_ATK_NORM
#   [5] max_save_dc / DESC_DC_NORM
#   [6..6+N_DAMAGE_TYPES)   typed resist summary: (multiplier − 1) per damage
#                            type — 0 = neutral (and absent rows stay zero),
#                            −0.5 resist, −1 immune, +1 vulnerable, < −1
#                            absorb. Source: char.damage_multipliers (the
#                            damage_table trait) — appearance-inferable lore
#                            (a troll's fire fear, a skeleton's brittleness).
# REMOVED (2026-07-06, obs v8+): the offensive OR-fields is_caster / has_teleport /
# grants_actions / applies_status-union / save-stat-union used to live between the
# 6 max scalars and the resist tail. They are PROVABLY recoverable from the per-
# entity `entity_skills` matrix (mean-pool over the kit's per-skill features:
# cost_slot_level / is_teleport / grants_actions / applies_status / save_stat) —
# measured identical — so they were pure duplication once encode_entity_skills
# exists. Removed to keep the entity row lean. The 6 max scalars stay (a MAX is
# NOT recoverable from the mean-pool encoder); resist_u stays (defensive, NOT in
# any skill feature, and it feeds the typed-resist join). NOTE: this SHRINKS a
# mid-row block, so it deliberately breaks the strict-append checkpoint-migration
# contract — pre-v8 checkpoints no longer load bit-exact (arch-priority decision;
# migration/versioning fix deferred).
DESC_DMG_NORM   = 50.0
DESC_HEAL_NORM  = 30.0
DESC_RANGE_NORM = 30.0   # battlefield size
DESC_AOE_NORM   = 10.0
DESC_ATK_NORM   = 20.0
DESC_DC_NORM    = 30.0
from ..engine.damage import DAMAGE_TYPES, N_DAMAGE_TYPES
N_V4_DESC = 6 + N_DAMAGE_TYPES   # 6 max scalars + typed-resist (OR-fields removed, see above)

# obs v5 (2026-06-13): per-entity PASSIVE-TRAIT descriptor — the behaviour-
# relevant passive traits invisible to BOTH the kit (capability_descriptor) and
# the typed-resist tail. A creature's OWN regeneration / pack tactics change how
# it should fight yet never surface as an action or a damage multiplier, so the
# policy was blind to them (probe_pack_headroom.py: a pack-tactics wolf gains
# +15pp WR purely by flanking, but the champion's flank rate is identical
# trait-on vs trait-off = it cannot see the trait). Appended AFTER the v4
# descriptor (same strict-prefix trick) so every pre-v5 column index stays
# valid. Per-entity on ALL rows: appearance-inferable lore (a troll visibly
# regenerates, a wolf pack visibly flanks), same public-info basis as
# typed_resist — and it lets the policy read an ALLY's pack tactics / an ENEMY's
# regen too. Adding another invisible trait is a strict append here (audit the
# adapter + diag_obsv5_shift.py), NOT a new entity block.
#   [0] pack_tactics             flag — advantage when an ally flanks the target
#   [1] regen_amount/REGEN_NORM  flat self-heal at turn start (0 = none)
#   [2] undead_fortitude         flag — CON save to survive a killing blow
#   [3] legendary_resist/LEGRES_NORM  per-combat auto-save pool (boss)
REGEN_NORM  = 30.0   # troll heals 10/turn; higher-tier regen tops out ~20-30
LEGRES_NORM = 3.0    # 5e legendary creatures: 3/day, single-combat encounters
N_V5_TRAIT = 4

# obs v6 (2026-07-01): per-entity CONDITION-IMMUNITY descriptor — which of the
# STATUS_SLOTS conditions bounce off this creature (undead vs poison/charm,
# elementals vs paralyze/prone/restrained, bosses vs the big control set). The
# engine already models it (character.condition_immunities, applied in combat)
# and the SCRIPTED expert reads it to skip wasting a control ability on an
# immune target (combat_policy._... "target immune -> continue") — but it was
# INVISIBLE to the policy, the exact mirror of the pre-12j typed-resist blind
# spot. A control-caster model sinks hold_person / poison / stun into an immune
# golem and can never learn not to, because immunity is an arbitrary per-
# creature list not derivable from any other obs field. Appended AFTER the v5
# trait tail (same strict-prefix trick) so every pre-v6 column index stays
# valid. Aligned to STATUS_SLOTS — SAME order as the skill row's applies_status
# bits — so the model's skill↔entity join can dot "this ability inflicts C"
# against "the target is immune to C" as ONE shared weight = the zero-shot
# carrier (pure one-hot cannot cover unseen (condition, monster) pairs — the
# orthogonality argument that made the damage-type join necessary in 12j).
N_V6_CIMMUN = N_STATUS_SLOTS   # 16 — one bit per STATUS_SLOTS condition

# obs v7 (2026-07-06): per-entity ABILITY-MODIFIER descriptor — the six D&D
# ability modifiers (STR/DEX/CON/INT/WIS/CHA). Saving throws roll d20 + the
# TARGET's ability modifier (+prof if proficient) — combat.make_saving_throw:
# `stat_mod = character.stats.modifier(stat)` — so a save-or-lose spell's success
# depends on the target's stat, yet it was INVISIBLE to the policy: perturbing an
# enemy's WIS left the entire entity row unchanged (diag). Without it the policy
# cannot prefer the save its target is WEAK at (hold_person WIS vs web DEX) —
# exactly the axis the spell-selection wave wants to test. MODIFIER, not raw
# score: the score's ONLY combat effect is via its modifier, and WIS 10 vs 11
# (both +0, identical save) must read identical — a raw score would add a dead
# low-order bit that is pure noise to the policy. Appended AFTER the v6 cimmun
# tail (same strict-prefix trick) so every pre-v7 column index stays valid. Per-
# entity on ALL rows: a creature's brawn/agility/wits are appearance-inferable
# (public-info basis, same as typed_resist / traits), and it also lets the policy
# read an ally's stats. NOTE: this exposes ABILITY modifiers, not save
# PROFICIENCY — two same-stat creatures that differ only in save proficiency
# still read identical (a separate axis, not added here). Adding a field = append
# one column + bump N_V7_ABILITY + the v6→v7 entity adapter (NEVER insert in the
# middle — that shifts every later checkpoint column).
ABILITY_STATS: tuple[str, ...] = ("STR", "DEX", "CON", "INT", "WIS", "CHA")
N_V7_ABILITY = len(ABILITY_STATS)   # 6
ABILITY_MOD_NORM = 10.0   # 5e mods run ~ -5..+10 (CR30 STR +10); /10 → [-0.5, 1]

ENTITY_DIM = (7 + N_ARCHETYPES + N_RL_STATUS + 1 + N_V3_EXTRA
              + N_V4_DESC + N_V5_TRAIT + N_V6_CIMMUN + N_V7_ABILITY)
# layout: 7 base + archetype multi-hot + status multi-hot + is_concentrating
#         + v3 tail (level, max_hp, ac, is_dying, death_succ, death_fail)
#         + v4 capability descriptor (N_V4_DESC)
#         + v5 passive-trait descriptor (N_V5_TRAIT)
#         + v6 condition-immunity descriptor (N_V6_CIMMUN)
# Named column indices — consumers must use these, never arithmetic from the
# row end (the v4 append broke every negative-offset assumption once already).
I_ENT_ENEMY  = 4    # base-feature col: 1.0 = this row is an enemy of self
ENT_V3_TAIL_START = 7 + N_ARCHETYPES + N_RL_STATUS + 1
I_ENT_LEVEL  = ENT_V3_TAIL_START + 0
I_ENT_MAXHP  = ENT_V3_TAIL_START + 1
I_ENT_AC     = ENT_V3_TAIL_START + 2
I_ENT_DYING  = ENT_V3_TAIL_START + 3
ENT_DESC_START = ENT_V3_TAIL_START + N_V3_EXTRA
# Typed-resist tail of the capability descriptor: (multiplier − 1) per
# DAMAGE_TYPES column. The model's skill↔entity matchup join dots the skill
# row's damage-type bits against this slice — keep both aligned to
# engine.damage.DAMAGE_TYPES (append-only).
I_DESC_RESIST = ENT_DESC_START + 6   # after the 6 max scalars (OR-fields removed, obs v8+)
# v5 passive-trait tail — named indices; consumers use these, never row-end math.
ENT_TRAIT_START = ENT_DESC_START + N_V4_DESC
I_TRAIT_PACK   = ENT_TRAIT_START + 0
I_TRAIT_REGEN  = ENT_TRAIT_START + 1
I_TRAIT_UNDEAD = ENT_TRAIT_START + 2
I_TRAIT_LEGRES = ENT_TRAIT_START + 3
# v6 condition-immunity tail — aligned to STATUS_SLOTS (same order as the skill
# row's applies_status); named start, consumers use this, never row-end math.
ENT_CIMMUN_START = ENT_TRAIT_START + N_V5_TRAIT
I_DESC_CIMMUN = ENT_CIMMUN_START
# v7 ability-modifier tail — named start, consumers use this, never row-end math.
# Order = ABILITY_STATS (STR/DEX/CON/INT/WIS/CHA).
ENT_ABILITY_START = ENT_CIMMUN_START + N_V6_CIMMUN
_ENTITY_DIM_V3 = 7 + N_ARCHETYPES + N_RL_STATUS + 1 + N_V3_EXTRA   # pre-v4 width
_ENTITY_DIM_V4 = (7 + N_ARCHETYPES + N_RL_STATUS + 1 + N_V3_EXTRA
                  + N_V4_DESC)   # pre-v5 width (v4 descriptor, no trait tail)
_ENTITY_DIM_V5 = _ENTITY_DIM_V4 + N_V5_TRAIT   # pre-v6 width (v5 trait, no cimmun)
_ENTITY_DIM_V6 = _ENTITY_DIM_V5 + N_V6_CIMMUN  # pre-v7 width (v6 cimmun, no ability)


def migrate_entities_v6_to_v7(entities):
    """Append the N_V7_ABILITY zero ability-modifier columns to a v6-width entity
    array (strict append, no rescale — v7 added no NORM changes to existing cols).
    Idempotent: returns unchanged if already v7-width."""
    import numpy as _np
    if entities.shape[-1] == ENTITY_DIM:
        return entities
    assert entities.shape[-1] == _ENTITY_DIM_V6, (
        f"expected v6 width {_ENTITY_DIM_V6}, got {entities.shape[-1]}")
    pad = _np.zeros(entities.shape[:-1] + (N_V7_ABILITY,), dtype=_np.float32)
    return _np.concatenate([entities.astype(_np.float32, copy=True), pad],
                           axis=-1)


def migrate_entities_v5_to_v6(entities):
    """Append the N_V6_CIMMUN zero condition-immunity columns to a v5-width
    entity array (strict append, no rescale — v6 added no NORM changes), then
    chain the v6→v7 ability-modifier append so callers land at the live width no
    matter how many tails have since been added. Idempotent: returns unchanged if
    already current-width; passes a v6-width array straight to the v7 step."""
    import numpy as _np
    if entities.shape[-1] == ENTITY_DIM:
        return entities
    if entities.shape[-1] == _ENTITY_DIM_V6:
        return migrate_entities_v6_to_v7(entities)
    assert entities.shape[-1] == _ENTITY_DIM_V5, (
        f"expected v5 width {_ENTITY_DIM_V5}, got {entities.shape[-1]}")
    pad = _np.zeros(entities.shape[:-1] + (N_V6_CIMMUN,), dtype=_np.float32)
    out = _np.concatenate([entities.astype(_np.float32, copy=True), pad],
                          axis=-1)   # now v6 width
    return migrate_entities_v6_to_v7(out)         # → current (v7) width


def migrate_entities_v4_to_v5(entities):
    """Append the N_V5_TRAIT zero passive-trait columns to a v4-width entity
    array (strict append, no rescale — v5 added no NORM changes), then chain the
    v5→v6 condition-immunity append so callers land at the live width no matter
    how many tails have since been added. Idempotent: returns unchanged if
    already current-width; passes a v5-width array straight to the v6 step."""
    import numpy as _np
    if entities.shape[-1] == ENTITY_DIM:
        return entities
    if entities.shape[-1] == _ENTITY_DIM_V6:
        return migrate_entities_v6_to_v7(entities)
    if entities.shape[-1] == _ENTITY_DIM_V5:
        return migrate_entities_v5_to_v6(entities)
    assert entities.shape[-1] == _ENTITY_DIM_V4, (
        f"expected v4 width {_ENTITY_DIM_V4}, got {entities.shape[-1]}")
    pad = _np.zeros(entities.shape[:-1] + (N_V5_TRAIT,), dtype=_np.float32)
    out = _np.concatenate([entities.astype(_np.float32, copy=True), pad],
                          axis=-1)   # now v5 width
    return migrate_entities_v5_to_v6(out)         # → current (v7) width


def migrate_entities_v3_to_v4(entities):
    """Migrate a stored v3-width entity array up to the CURRENT ENTITY_DIM.
    (Historically v3→v4; since obs v5 it also chains the v4→v5 trait-column
    append, so callers replaying v3-era datasets land at the live width no
    matter how many tails have since been added.) The v3 step rescales
    level/max_hp to the v4 norms (the same ×0.5/×0.125 that makes
    adapt_state_dict_for_obs_v4 a product-level no-op) and zero-pads the v4
    descriptor; then migrate_entities_v4_to_v5 appends the trait tail. Idempotent:
    returns unchanged if already current-width; passes a v4-width array straight
    to the v5 step."""
    import numpy as _np
    if entities.shape[-1] == ENTITY_DIM:
        return entities
    if entities.shape[-1] == _ENTITY_DIM_V6:
        return migrate_entities_v6_to_v7(entities)
    if entities.shape[-1] == _ENTITY_DIM_V5:
        return migrate_entities_v5_to_v6(entities)
    if entities.shape[-1] == _ENTITY_DIM_V4:
        return migrate_entities_v4_to_v5(entities)
    assert entities.shape[-1] == _ENTITY_DIM_V3, (
        f"expected v3 width {_ENTITY_DIM_V3}, got {entities.shape[-1]}")
    out = entities.astype(_np.float32, copy=True)
    out[..., I_ENT_LEVEL] *= (V3_LEVEL_NORM / LEVEL_NORM)
    out[..., I_ENT_MAXHP] *= (V3_MAXHP_NORM / MAXHP_NORM)
    pad = _np.zeros(out.shape[:-1] + (N_V4_DESC,), dtype=_np.float32)
    out = _np.concatenate([out, pad], axis=-1)   # now v4 width
    return migrate_entities_v4_to_v5(out)         # → current (v7) width


# obs decision-context (2026-06-13, reaction/legendary wave): a small per-step
# vector telling the policy WHICH decision it is making and the trigger details.
# On a normal own-turn step it is ALL ZEROS — and it is fed into the trunk at the
# END of the feature concat (model._encode), so a zero vector contributes exactly
# 0 through the (zero-padded, on migrated checkpoints) trunk columns: every
# pre-wave checkpoint stays BIT-EXACT on normal turns forever, regardless of how
# the new columns later train. The non-zero vector is produced ONLY at an
# off-turn reaction / legendary decision point (the new neural deciders), so the
# whole feature is invisible to every existing turn-only path. Adding a field =
# append one column + bump N_DECISION_CTX + the decision-ctx migration adapter
# (NEVER insert in the middle — same strict-append rule as the entity tails).
#   [0] is_reaction        this step is a reaction decision (else 0)
#   [1] is_legendary       this step is a legendary-action decision (else 0)
#   [2] trig_attack        reaction trigger: an incoming attack ROLL
#   [3] trig_auto_damage   reaction trigger: an auto-hit (Magic Missile)
#   [4] trig_spell         reaction trigger: an enemy is casting a leveled spell
#   [5] trig_uncanny       reaction trigger: an incoming melee hit (pre-damage)
#   [6] attack_margin      clamp((attack_total − my_eff_ac)/5, 0, 2): how far the
#                          hit clears my AC in 5-AC units — the Shield-flip signal
#                          (a value in [0,1) means +5 AC would turn it into a miss)
#   [7] spell_level        incoming spell level / 9 (Counterspell value)
#   [8] legendary_frac     my remaining legendary budget / LEGRES_NORM
N_DECISION_CTX = 9
(I_DCTX_REACTION, I_DCTX_LEGENDARY, I_DCTX_TRIG_ATTACK, I_DCTX_TRIG_AUTO,
 I_DCTX_TRIG_SPELL, I_DCTX_TRIG_UNCANNY, I_DCTX_ATK_MARGIN,
 I_DCTX_SPELL_LVL, I_DCTX_LEG_FRAC) = range(N_DECISION_CTX)

_DCTX_TRIGGER_SLOT = {
    "attack": I_DCTX_TRIG_ATTACK,
    "auto_damage": I_DCTX_TRIG_AUTO,
    "spell": I_DCTX_TRIG_SPELL,
    "uncanny": I_DCTX_TRIG_UNCANNY,
}


def zero_decision_context() -> np.ndarray:
    """The all-zero decision-context = a normal own-turn step (bit-exact path)."""
    return np.zeros(N_DECISION_CTX, dtype=np.float32)


def reaction_decision_context(trigger: str, *, attack_margin: float = 0.0,
                              spell_level: int = 0) -> np.ndarray:
    """Decision-context vector for a REACTION decision point. ``attack_margin``
    is (attack_total − reactor_effective_ac); ``spell_level`` the incoming spell
    level. Both are normalised here so the engine layer needn't know the norms."""
    v = zero_decision_context()
    v[I_DCTX_REACTION] = 1.0
    slot = _DCTX_TRIGGER_SLOT.get(trigger)
    if slot is not None:
        v[slot] = 1.0
    v[I_DCTX_ATK_MARGIN] = float(np.clip(attack_margin / 5.0, 0.0, 2.0))
    v[I_DCTX_SPELL_LVL] = min(1.0, max(0, spell_level) / 9.0)
    return v


def legendary_decision_context(remaining: int) -> np.ndarray:
    """Decision-context vector for a LEGENDARY-action decision point."""
    v = zero_decision_context()
    v[I_DCTX_LEGENDARY] = 1.0
    v[I_DCTX_LEG_FRAC] = min(1.0, max(0, remaining) / LEGRES_NORM)
    return v


N_GRID = 30                   # battlefield grid resolution (1m cells)
BATTLEFIELD_SIZE_M = 30.0     # matches Battlefield default
GRID_CELL_SIZE_M = BATTLEFIELD_SIZE_M / N_GRID    # 1.0m
N_GRID_CELLS = N_GRID * N_GRID                    # 900

# Entity-grid overlay channels: a parallel 30x30 representation of where
# self/allies/enemies are. Gives the model the same positional info as the
# continuous entity rows BUT pre-discretised to the same cell grid that
# grid_head outputs — so grid_head doesn't have to learn the continuous→
# discrete map with a Linear (which is what bottlenecked the previous arch).
N_ENTITY_GRID_CHANNELS = 3       # [self, ally, enemy]

# Pre-computed distance grids. Each cell carries the normalised Euclidean
# distance to (a) self and (b) the nearest live enemy. CNN can't derive
# all-pairs distance efficiently — its receptive field is local — so this
# is the explicit feature engineering that replaces the CNN's "global
# spatial reasoning" job. grid_head reads these as channels of spatial_feat.
N_DISTANCE_GRID_CHANNELS = 2     # [dist_to_self, dist_to_nearest_enemy]

# Pre-computed per-cell LINE OF SIGHT to the nearest enemy. Whether standing at
# a cell grants LoS depends on walls on the cell->enemy segment — a NON-local
# property the pointwise-linear grid_head cannot derive from a cell's own
# terrain/entity/distance features (probe_los_gridfit: linear acc = baseline
# from those, = 100% once this channel is added). Without it the policy hugs
# the wall face and never flanks (diag_wall_los). 1.0 = LoS, 0.0 = blocked (and
# 1.0 everywhere when no live enemy, mirroring distance_grid's no-enemy rule).
N_LOS_GRID_CHANNELS = 1          # [los_to_nearest_enemy]

# Pre-computed per-cell WEAPON-REACH proximity to the nearest enemy: reach/dist
# capped at 1.0, so a cell WITHIN my weapon reach of the nearest enemy = 1.0 and
# it falls off smoothly outside. 1.5m melee reach is a HARD threshold that a
# linear grid_head over the SMOOTH distance channel cannot represent (proven:
# probe_reach_stall — at a stationary target the head ranks a 1.72m cell ABOVE a
# 0.54m one, so the model closes to ~1.72m, can never step the last 0.22m into
# reach, dodge-loops and NEVER attacks = the user's original "enemy won't come
# fight me" bug). This channel gives the head the explicit in-reach signal it
# was missing, so it can prefer a cell it can actually ATTACK from. No live
# enemy → all-ones (mirrors distance/los no-enemy convention).
N_REACH_GRID_CHANNELS = 1        # [reach_proximity_to_nearest_enemy]

# Pre-computed per-cell MELEE-THREAT flag: 1.0 if a melee enemy could reach this
# cell and attack it on ITS next turn (within one move + a melee swing of any
# live enemy's current position). This is the perception signal a kiting policy
# needs: a ranged attacker should STEP OFF a threatened cell (reposition) and
# STAND on a safe one (end the turn, keep casting). Measured root cause
# (probe_kite_sweep / trace endp): the blind end_head decides reposition-vs-end
# on RESOURCE availability, NOT on whether the current cell is dangerous (post-
# cast endp at a SAFE 11m = 0.335 < a THREATENED 1m = 0.436 — it kites either
# way and corners itself). Distance-to-enemy is in obs but the "can it reach me
# next turn" THRESHOLD is invisible to the linear heads, exactly like the 1.5m
# reach threshold was — so kite-DAgger only flips the GLOBAL move direction
# (retreat-always) instead of learning the conditional. This channel injects the
# threatened/safe boundary so end_head can learn "safe → stop, threatened →
# kite". No live enemy → all-zeros (no threat).
N_THREAT_GRID_CHANNELS = 1       # [melee_threat_at_cell]

# Threat reach = IMMEDIATE melee danger: a cell within a melee reach + one short
# step of an enemy's CURRENT position. Deliberately NOT "enemy move + reach"
# (~10.5m): the +43-WR kite gate triggered only when an enemy was ~adjacent
# (≤2.5m) and that minimal reactive kiting won — a full-move threat radius marks
# nearly the whole field "threatened" so the policy kites every turn and drifts
# to the wall (measured: radius 10.5 → ranged WR still −9). A small radius makes
# end_head fire "stop kiting" as soon as the agent has stepped out of immediate
# melee, so it kites ONCE (far, via the move-head) then casts-and-holds. Fixed
# melee reach (not the enemy's equipped-weapon range) so a ranged enemy doesn't
# paint the whole field threatened — kiting is about escaping MELEE pin.
ENEMY_THREAT_RADIUS_M = 1.5 + 1.5

OBS_KEYS = ("skills", "skill_mask", "entity_skills", "entity_skill_mask",
            "entities", "resources", "terrain",
            "entity_grid", "distance_grid", "los_grid", "reach_grid",
            "threat_grid", "end_features", "decision_context")

# Hand-picked features feeding end_head directly (bypasses the shared encoder
# so end_head doesn't inherit the encoder's drift). 8 scalars + archetype:
#   0: agent hp fraction (low HP — model can learn "near-death → end")
#   1: action remaining       (0/1)
#   2: bonus_action remaining (0/1)
#   3: movement fraction       (0..1)
#   4: # playable non-end-non-move skills (normalised) — main "do I have something useful?"
#   5: any enemy alive (0/1)
#   6: any ally alive (0/1)
#   7: agent is concentrating (0/1)
#   8..8+N_ARCHETYPES-1: agent archetype multi-hot (added 2026-06-09)
# The archetype tail is REQUIRED: verify_endhead.py measured the expert's end
# decision at a fixed (action-spent, bonus-available) bucket varies 4%→100%
# across classes (rogues keep their cunning-action bonus, paladins end). A
# class-blind end_head can only emit one blended P(end) there, which over-ends
# rogues (wasting bonus) and under-ends melee (over-extension damage). Giving
# end_head the archetype lets it make the class-conditional stop decision. It is
# a static identity (no encoder drift) so the decoupling rationale still holds.
END_FEATURES_DIM = 8 + N_ARCHETYPES


def end_features(ws: WorldState, agent_id: str, resources: dict) -> np.ndarray:
    """Compact feature vector for the decoupled end_head.

    Kept small and hand-picked so it doesn't go through any shared parameters
    that other heads' PG gradients can drift. This is the inputs end_head
    learns from — encoder drift cannot leak into the end decision.
    """
    agent = ws.characters[agent_id]
    skills = available_skills(agent, ws)
    n_playable = sum(1 for s in skills
                      if s.skill_id not in ("end", "move"))
    any_enemy = any(
        c.is_alive() and ws.is_party_ally(cid) != ws.is_party_ally(agent_id)
        for cid, c in ws.characters.items()
    )
    any_ally = any(
        cid != agent_id and c.is_alive()
        and ws.is_party_ally(cid) == ws.is_party_ally(agent_id)
        for cid, c in ws.characters.items()
    )
    scalars = np.array([
        agent.hp / max(1, agent.max_hp),
        1.0 if resources.get("action", 0) > 0 else 0.0,
        1.0 if resources.get("bonus_action", 0) > 0 else 0.0,
        max(0.0, min(1.0, resources.get("movement", 0.0) / MOVE_BUDGET_M)),
        min(1.0, n_playable / 8.0),   # normalise; 8 skills = "lots of options"
        1.0 if any_enemy else 0.0,
        1.0 if any_ally else 0.0,
        1.0 if agent.concentrating_on else 0.0,
    ], dtype=np.float32)
    # Archetype multi-hot tail — same encoding as _entity_row so end_head can
    # make a class-conditional stop decision (see END_FEATURES_DIM note).
    arch_oh = np.zeros(N_ARCHETYPES, dtype=np.float32)
    arch_id = getattr(agent, "archetype_id", "") or ""
    for part in arch_id.split(","):
        part = part.strip()
        if part in ARCHETYPE_OBS_LIST:
            arch_oh[ARCHETYPE_OBS_LIST.index(part)] = 1.0
    return np.concatenate([scalars, arch_oh])


def terrain_obs(battlefield: Battlefield) -> np.ndarray:
    """Sample the battlefield at each grid cell centre.

    Returns: float32[N_GRID, N_GRID] with values
       0.0 = open
       0.5 = difficult terrain
       1.0 = obstacle / wall
      -1.0 = dangerous terrain
    """
    grid = np.zeros((N_GRID, N_GRID), dtype=np.float32)
    for ix in range(N_GRID):
        for iy in range(N_GRID):
            cx = (ix + 0.5) * GRID_CELL_SIZE_M
            cy = (iy + 0.5) * GRID_CELL_SIZE_M
            p = Vec2(cx, cy)
            if battlefield.is_blocked(p):
                grid[ix, iy] = 1.0
                continue
            t = battlefield.terrain_at(p)
            if t == TerrainType.DANGEROUS:
                grid[ix, iy] = -1.0
            elif t == TerrainType.DIFFICULT:
                grid[ix, iy] = 0.5
    return grid


def partition_entities(ws: WorldState, agent_id: str) -> tuple[list[str], list[str]]:
    """Partition live characters (excluding self) into (allies, enemies_sorted_by_distance).

    Both lists contain char_ids. Enemy ordering is the slot ordering used by
    entities_obs (slot ENEMY_SLOT_START = enemies[0] = closest).

    DYING characters (PC at 0 HP making death saves) are INCLUDED — they are
    on the field, can be healed back up (revive) or attacked (finished off),
    so the policy must be able to see and target them. Their entity row reads
    hp_frac=0 / is_alive=0, which is also how they're distinguished from
    living rows. Truly dead characters are excluded (rows of zeros).
    """
    self_char = ws.characters[agent_id]
    is_party = ws.is_party_ally(agent_id)
    allies: list[str] = []
    enemies: list[str] = []
    for cid, c in ws.characters.items():
        if cid == agent_id or c.is_dead():
            continue
        other_is_party = ws.is_party_ally(cid)
        if is_party == other_is_party:
            allies.append(cid)
        else:
            enemies.append(cid)
    enemies.sort(key=lambda cid: self_char.position.distance_to(ws.characters[cid].position))
    return allies, enemies


def capability_descriptor(char: Character) -> np.ndarray:
    """Build the N_V4_DESC kit summary for one creature (see constant block).

    Aggregates engine.skill.kit_features — full kit, level-gated, NOT
    resource-gated (remaining_uses is deliberately never read). Cached on
    the character object: the kit is static within an episode, and obs is
    rebuilt every step for up to 10 rows.
    """
    # Cache key must cover EVERY input the descriptor reads, not just the kit
    # shape. damage_multipliers feeds the typed-resist tail (below); it is
    # normally static from creation, but anything that mutates it mid-episode
    # (a doused-troll mechanic, an injected-immunity probe) would otherwise be
    # silently masked by a stale cache keyed only on (level, n_abilities,
    # n_weapons). Include a cheap signature so the descriptor recomputes iff a
    # resistance actually changed — zero overhead in normal play.
    cache_key = (char.level, len(char.known_abilities or ()), len(char.weapons),
                 tuple(sorted((char.damage_multipliers or {}).items())))
    cached = getattr(char, "_cap_desc_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    # Only the 6 MAX scalars survive here; the OR-fields (is_caster/has_tp/grants/
    # applies_status-union/save-stat-union) were removed as provably recoverable
    # from entity_skills (obs v8). A MAX is NOT recoverable from the mean-pool
    # entity-skill encoder, so these stay.
    n_atk = float(getattr(char, "attacks_per_action", 1) or 1)
    max_dmg = max_heal = max_rng = max_aoe = max_atk = max_dc = 0.0
    for sid, f in kit_features(char):
        dmg = f.expected_damage * (n_atk if sid.startswith("weapon:") else 1.0)
        max_dmg = max(max_dmg, dmg)
        max_heal = max(max_heal, f.expected_healing)
        if f.expected_damage > 0:
            max_rng = max(max_rng, f.range_m)
        max_aoe = max(max_aoe, f.aoe_radius_m)
        max_atk = max(max_atk, f.attack_vs_ac)
        max_dc = max(max_dc, f.save_dc)

    resist_u = np.zeros(N_DAMAGE_TYPES, dtype=np.float32)
    for dtype, mult in (char.damage_multipliers or {}).items():
        i = DAMAGE_TYPES.index(dtype) if dtype in DAMAGE_TYPES else -1
        if i >= 0:
            resist_u[i] = float(mult) - 1.0   # 0 = neutral

    desc = np.concatenate([
        np.array([
            max_dmg / DESC_DMG_NORM,
            max_heal / DESC_HEAL_NORM,
            max_rng / DESC_RANGE_NORM,
            max_aoe / DESC_AOE_NORM,
            max_atk / DESC_ATK_NORM,
            max_dc / DESC_DC_NORM,
        ], dtype=np.float32),
        resist_u,
    ])
    char._cap_desc_cache = (cache_key, desc)
    return desc


def trait_descriptor(char: Character) -> np.ndarray:
    """Build the N_V5_TRAIT passive-trait tail for one creature (see the v5
    constant block). These are behaviour-relevant passive traits that never
    appear in the kit (capability_descriptor) or the damage table — read
    straight off the character's trait attrs. Cheap (4 scalars), recomputed per
    row; no cache. Adding a trait here = append one column + bump N_V5_TRAIT +
    the v5 adapter (NEVER insert in the middle — that shifts every later
    checkpoint column)."""
    regen = getattr(char, "regeneration", None) or {}
    return np.array([
        1.0 if getattr(char, "pack_tactics", False) else 0.0,
        float(regen.get("amount", 0) or 0) / REGEN_NORM,
        1.0 if getattr(char, "undead_fortitude", False) else 0.0,
        float(getattr(char, "legendary_resistance_uses", 0) or 0) / LEGRES_NORM,
    ], dtype=np.float32)


def condition_immunity_descriptor(char: Character) -> np.ndarray:
    """Build the N_V6_CIMMUN condition-immunity tail for one creature (see the v6
    constant block). Multi-hot over STATUS_SLOTS: 1.0 = the creature is immune to
    that condition (the engine no-ops an application of it). Aligned to
    STATUS_SLOTS — the SAME order/length as the skill row's applies_status bits —
    so it dots cleanly against them in the model's skill↔entity join. Read
    straight off char.condition_immunities; cheap (16 bits), recomputed per row,
    no cache. Names outside STATUS_SLOTS (engine-only conditions) are skipped —
    they have no aligned skill bit to join against."""
    out = np.zeros(N_V6_CIMMUN, dtype=np.float32)
    for name in (getattr(char, "condition_immunities", None) or ()):
        if name in STATUS_SLOTS:
            out[STATUS_SLOTS.index(name)] = 1.0
    return out


def ability_descriptor(char: Character) -> np.ndarray:
    """Build the N_V7_ABILITY ability-modifier tail for one creature (see the v7
    constant block). The six D&D ability MODIFIERS in ABILITY_STATS order,
    normalised by ABILITY_MOD_NORM. Modifier (not raw score) because the score's
    only combat effect is via its modifier and it is what the saving-throw roll
    adds (combat.make_saving_throw). Cheap (6 scalars), recomputed per row, no
    cache. Adding an axis here = append one column + bump N_V7_ABILITY + the v6→v7
    adapter (NEVER insert in the middle — that shifts every later checkpoint
    column)."""
    return np.array([char.stats.modifier(s) / ABILITY_MOD_NORM
                     for s in ABILITY_STATS], dtype=np.float32)


def _entity_row(char: Character, self_char: Character,
                bf_size_x: float, bf_size_y: float,
                is_self: bool, is_enemy: bool) -> np.ndarray:
    """Build one entity row for the entities observation.

    bf_size_x / bf_size_y are the battlefield dimensions used to normalise
    coordinates to [0, 1]. Distance is normalised by the diagonal so that
    the max possible distance maps to 1.0.

    Layout (ENTITY_DIM = 7 + N_ARCHETYPES + N_RL_STATUS + 1 + N_V3_EXTRA):
        [0..6]  base features (no debuff bit — replaced by status multi-hot)
        [7..7+N_ARCHETYPES-1]              archetype multi-hot
        [7+N_ARCHETYPES..+N_RL_STATUS-1]   status multi-hot (per-status bits;
                                            replaces the old single has_debuff
                                            so timing-dependent buffs like
                                            vow_target / sacred_weapon_buff /
                                            raging are visible to the policy)
        [.. +1]                             is_concentrating
        [ENT_V3_TAIL_START..]               v3 tail — see the constant block:
                                            level, max_hp, ac, is_dying,
                                            death successes, death failures
        [ENT_DESC_START..]                  v4 capability descriptor
    """
    bf_diag = (bf_size_x ** 2 + bf_size_y ** 2) ** 0.5 or 1.0
    base = np.array([
        char.hp / max(1, char.max_hp),
        char.position.x / bf_size_x if bf_size_x else 0.0,
        char.position.y / bf_size_y if bf_size_y else 0.0,
        self_char.position.distance_to(char.position) / bf_diag,
        1.0 if is_enemy else 0.0,
        1.0 if char.is_alive() else 0.0,
        1.0 if is_self else 0.0,
    ], dtype=np.float32)

    # Archetype multi-hot. Multi-hot rather than one-hot so a hypothetical
    # multiclass character (e.g. paladin/sorcerer) can set both bits.
    arch_oh = np.zeros(N_ARCHETYPES, dtype=np.float32)
    arch_id = getattr(char, "archetype_id", "") or ""
    # Accept "vengeance,sorcerer" comma-separated form for future multiclass —
    # split and light each component that we recognise.
    for part in arch_id.split(","):
        part = part.strip()
        if part in ARCHETYPE_OBS_LIST:
            arch_oh[ARCHETYPE_OBS_LIST.index(part)] = 1.0

    status_mh = np.zeros(N_RL_STATUS, dtype=np.float32)
    for i, name in enumerate(RL_STATUS_NAMES):
        if char.has_status(name):
            status_mh[i] = 1.0

    is_concentrating = np.array([1.0 if char.concentrating_on else 0.0],
                                 dtype=np.float32)

    v3_tail = np.array([
        char.level / LEVEL_NORM,
        char.max_hp / MAXHP_NORM,
        char.ac / AC_NORM,
        1.0 if char.is_dying() else 0.0,
        min(1.0, char.death_saves.get("successes", 0) / 3.0),
        min(1.0, char.death_saves.get("failures", 0) / 3.0),
    ], dtype=np.float32)
    return np.concatenate([base, arch_oh, status_mh, is_concentrating, v3_tail,
                           capability_descriptor(char), trait_descriptor(char),
                           condition_immunity_descriptor(char),
                           ability_descriptor(char)])


def entities_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Build the entities observation matrix.

    Row order: [self, ally_1..N_ALLY_SLOTS, enemy_1..N_ENEMY_SLOTS].
    Enemies sorted by distance to self ascending. Missing rows are zero.
    """
    self_char = ws.characters[agent_id]
    # Use the actual battlefield dimensions so non-default sizes still
    # produce normalised coords in [0, 1]. Fall back to the engine default
    # when combat hasn't been set up (e.g. agent observed out of combat).
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    bf_size_x = bf.width if bf else BATTLEFIELD_SIZE_M
    bf_size_y = bf.height if bf else BATTLEFIELD_SIZE_M
    out = np.zeros((N_ENTITY_SLOTS, ENTITY_DIM), dtype=np.float32)

    # Row 0: self
    out[0] = _entity_row(self_char, self_char, bf_size_x, bf_size_y,
                         is_self=True, is_enemy=False)

    ally_ids, enemy_ids = partition_entities(ws, agent_id)

    # Ally rows (truncated at N_ALLY_SLOTS)
    for i, cid in enumerate(ally_ids[:N_ALLY_SLOTS]):
        out[1 + i] = _entity_row(ws.characters[cid], self_char, bf_size_x, bf_size_y,
                                  is_self=False, is_enemy=False)
    # Enemy rows (truncated at N_ENEMY_SLOTS)
    for i, cid in enumerate(enemy_ids[:N_ENEMY_SLOTS]):
        out[ENEMY_SLOT_START + i] = _entity_row(ws.characters[cid], self_char,
                                                 bf_size_x, bf_size_y,
                                                 is_self=False, is_enemy=True)
    return out


def skills_obs(ws: WorldState, agent_id: str,
               skills: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build (skills_matrix, skill_mask).

    skills_matrix : float32[N_SKILL_SLOTS, SKILL_FEATURE_DIM]
    skill_mask    : float32[N_SKILL_SLOTS]    1=valid, 0=padding

    ``skills`` overrides the slot list — used by an off-turn reaction decision
    point (skill.reaction_skills) so the policy sees the reaction candidates in
    the skill slots instead of the normal turn kit. Defaults to available_skills.
    """
    agent = ws.characters[agent_id]
    if skills is None:
        skills = available_skills(agent, ws)
    skill_mat = np.zeros((N_SKILL_SLOTS, SKILL_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(N_SKILL_SLOTS, dtype=np.float32)

    for i, sk in enumerate(skills[:N_SKILL_SLOTS]):
        vec = np.asarray(sk.features.as_vector(), dtype=np.float32)
        # Bound OOD-large expected_damage back onto the training manifold so an
        # unseen ultra-EV skill (a grafted swallow/breath) can't corrupt the
        # unnormalised skill_proj → pooled embedding → trunk-collapse → passive
        # turtling. No-op for every trained kit (all ≤ SKILL_EV_OBS_CAP). See
        # skill.SKILL_EV_OBS_CAP.
        if vec[I_SKILL_EXPECTED_DAMAGE] > SKILL_EV_OBS_CAP:
            vec[I_SKILL_EXPECTED_DAMAGE] = SKILL_EV_OBS_CAP
        skill_mat[i] = vec
        mask[i] = 1.0
    return skill_mat, mask


def _entity_kit_matrix(char) -> tuple[np.ndarray, np.ndarray]:
    """[N_SKILL_SLOTS, SKILL_FEATURE_DIM] + mask for ONE creature's STATIC kit.

    Uses kit_features (full, level-gated, resource-INdependent) — the creature's
    IDENTITY of "what it CAN do", the same 66-d mechanic vectors skills_obs feeds
    for the self, so the model can reuse ONE skill encoder for self and others.
    (available_skills, by contrast, is resource-gated and leaks transient state.)
    move/end pseudo-skills are absent from kit_features = correct (universal, not
    identity). Cached per char (kit static within an episode; key invalidates on a
    mid-episode graft, mirroring capability_descriptor)."""
    key = (char.level, len(char.known_abilities or ()), len(char.weapons))
    cached = getattr(char, "_kit_mat_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    mat = np.zeros((N_SKILL_SLOTS, SKILL_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(N_SKILL_SLOTS, dtype=np.float32)
    for i, (_sid, f) in enumerate(kit_features(char)):
        if i >= N_SKILL_SLOTS:
            break
        vec = np.asarray(f.as_vector(), dtype=np.float32)
        if vec[I_SKILL_EXPECTED_DAMAGE] > SKILL_EV_OBS_CAP:   # same OOD clamp as skills_obs
            vec[I_SKILL_EXPECTED_DAMAGE] = SKILL_EV_OBS_CAP
        mat[i] = vec
        mask[i] = 1.0
    char._kit_mat_cache = (key, (mat, mask))
    return mat, mask


def entity_skills_obs(ws: WorldState, agent_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-entity STATIC kit matrices, SAME slot order as entities_obs.

    Returns (mats[N_ENTITY_SLOTS, N_SKILL_SLOTS, SKILL_FEATURE_DIM],
             mask[N_ENTITY_SLOTS, N_SKILL_SLOTS]). Missing slots are all-zero.
    Slot layout MIRRORS entities_obs exactly (self=0, allies=1.., enemies=
    ENEMY_SLOT_START..) via the same partition_entities, so entity_skills[i]
    always describes the SAME creature as entities[i]. This closes the gap where
    two distinct kits with an identical capability_descriptor (e.g. a raging
    berserker vs a champion, or two different L6 wizards) read identical."""
    mats = np.zeros((N_ENTITY_SLOTS, N_SKILL_SLOTS, SKILL_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((N_ENTITY_SLOTS, N_SKILL_SLOTS), dtype=np.float32)
    self_char = ws.characters[agent_id]
    mats[0], mask[0] = _entity_kit_matrix(self_char)
    ally_ids, enemy_ids = partition_entities(ws, agent_id)
    for i, cid in enumerate(ally_ids[:N_ALLY_SLOTS]):
        mats[1 + i], mask[1 + i] = _entity_kit_matrix(ws.characters[cid])
    for i, cid in enumerate(enemy_ids[:N_ENEMY_SLOTS]):
        mats[ENEMY_SLOT_START + i], mask[ENEMY_SLOT_START + i] = \
            _entity_kit_matrix(ws.characters[cid])
    return mats, mask


def resources_obs(resources: dict, round_number: int) -> np.ndarray:
    """Build the resources observation vector (length 4)."""
    return np.array([
        1.0 if resources.get("action", 0) > 0 else 0.0,
        1.0 if resources.get("bonus_action", 0) > 0 else 0.0,
        max(0.0, min(1.0, resources.get("movement", 0.0) / MOVE_BUDGET_M)),
        min(1.0, round_number / 10.0),
    ], dtype=np.float32)


def entity_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Build the entity-grid overlay.

    Returns: float32[N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — self position   (1.0 at self's cell, 0 elsewhere)
      channel 1 — allies          (sum of ally markers per cell)
      channel 2 — enemies         (sum of enemy markers per cell)

    Each cell value is the count of matching characters whose centre falls in
    that cell — typically 0 or 1, occasionally >1 if two share a cell.
    """
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    cell_size = bf.width / N_GRID if bf else GRID_CELL_SIZE_M
    out = np.zeros((N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)

    self_char = ws.characters[agent_id]
    ally_ids, enemy_ids = partition_entities(ws, agent_id)

    def _mark(ch: int, pos) -> None:
        ix = max(0, min(N_GRID - 1, int(pos.x / cell_size)))
        iy = max(0, min(N_GRID - 1, int(pos.y / cell_size)))
        out[ch, ix, iy] += 1.0

    _mark(0, self_char.position)
    # partition_entities 已排除死者；瀕死（可救/可補刀）照標記在場上。
    for cid in ally_ids:
        _mark(1, ws.characters[cid].position)
    for cid in enemy_ids:
        _mark(2, ws.characters[cid].position)
    return out


def distance_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Per-cell normalised Euclidean distance to self / nearest enemy.

    Returns: float32[N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — distance from each cell to self / diag
      channel 1 — distance from each cell to nearest live enemy / diag
                   (1.0 everywhere if no live enemies)

    These are the features the spatial CNN couldn't reliably derive: all-pairs
    distance is a global quantity, not a local pattern. Computing it explicitly
    here removes the CNN's main bottleneck for grid_head decisions.
    """
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    cell_size = bf.width / N_GRID if bf else GRID_CELL_SIZE_M
    bf_w = bf.width if bf else BATTLEFIELD_SIZE_M
    bf_h = bf.height if bf else BATTLEFIELD_SIZE_M
    diag = (bf_w * bf_w + bf_h * bf_h) ** 0.5 or 1.0

    xs = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    ys = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    self_char = ws.characters[agent_id]
    sx, sy = self_char.position.x, self_char.position.y
    dist_self = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / diag

    _, enemy_ids = partition_entities(ws, agent_id)
    live_enemy_positions = [
        ws.characters[cid].position
        for cid in enemy_ids if ws.characters[cid].is_alive()
    ]
    if live_enemy_positions:
        per_enemy = np.stack([
            np.sqrt((xx - p.x) ** 2 + (yy - p.y) ** 2)
            for p in live_enemy_positions
        ], axis=0)
        dist_enemy = per_enemy.min(axis=0) / diag
    else:
        dist_enemy = np.ones_like(dist_self)

    return np.stack([dist_self, dist_enemy], axis=0).astype(np.float32)


def reach_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Per-cell WEAPON-REACH proximity to the nearest live enemy.

    Returns: float32[N_REACH_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — min(1.0, my_reach / dist(cell, nearest_enemy)). 1.0 for any
                  cell from which I'd be WITHIN weapon reach of the nearest
                  enemy (can attack); falls off smoothly outside reach so the
                  pointwise-linear grid_head has a monotone gradient to climb
                  INTO reach. 1.0 everywhere with no live enemy (mirrors
                  distance/los no-enemy convention).

    The distance channel is smooth and normalised by the field diagonal, so the
    1.5m melee threshold is invisible to a linear head — it cannot tell "0.5m =
    can attack" from "1.7m = cannot". This channel injects exactly that hard
    boundary as a perception feature (the policy still chooses).
    """
    out = np.ones((N_REACH_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)
    self_char = ws.characters[agent_id]
    try:
        reach = float(self_char.get_weapon().range_normal) if self_char.weapons else 1.5
    except Exception:
        reach = 1.5
    if reach <= 0:
        reach = 1.5
    _, enemy_ids = partition_entities(ws, agent_id)
    live = [ws.characters[cid].position
            for cid in enemy_ids if ws.characters[cid].is_alive()]
    if not live:
        return out
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    cell_size = bf.width / N_GRID if bf else GRID_CELL_SIZE_M
    xs = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    ys = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    per_enemy = np.stack([np.sqrt((xx - p.x) ** 2 + (yy - p.y) ** 2) for p in live],
                         axis=0)
    dist_enemy = per_enemy.min(axis=0)
    # BINARY in-reach flag (sharp threshold). A smooth reach/dist barely separates
    # an in-reach cell (1.0m→1.0) from a just-outside one (1.72m→0.87, only 0.13
    # below) — too little contrast for the grid-head to flip onto the attackable
    # cell. The hard 1.0/0.0 step gives the exact in/out boundary the linear head
    # can't synthesise from the smooth distance channel; distance_grid still
    # supplies the gradient to APPROACH from far.
    in_reach = (dist_enemy <= reach + 1e-6).astype(np.float32)
    return in_reach[None, :, :]


def threat_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Per-cell MELEE-THREAT flag for the kiting decision.

    Returns: float32[N_THREAT_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — 1.0 if ANY live enemy could move-and-melee this cell on its
                  next turn (cell within ENEMY_THREAT_RADIUS_M of that enemy's
                  current position); 0.0 if safe. All-zeros with no live enemy.

    Binary, like reach_grid: the linear end/grid heads can't synthesise the
    "enemy can reach me next turn" boundary from the smooth distance channel, so
    they decide reposition-vs-end on resources instead and over-kite. The hard
    flag gives end_head the in/out-of-danger signal directly: stand (end) on a
    safe cell, step off a threatened one. The policy still chooses.
    """
    out = np.zeros((N_THREAT_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)
    _, enemy_ids = partition_entities(ws, agent_id)
    live = [ws.characters[cid].position
            for cid in enemy_ids if ws.characters[cid].is_alive()]
    if not live:
        return out
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    cell_size = bf.width / N_GRID if bf else GRID_CELL_SIZE_M
    xs = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    ys = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    per_enemy = np.stack([np.sqrt((xx - p.x) ** 2 + (yy - p.y) ** 2) for p in live],
                         axis=0)
    dist_enemy = per_enemy.min(axis=0)
    threatened = (dist_enemy <= ENEMY_THREAT_RADIUS_M + 1e-6).astype(np.float32)
    return threatened[None, :, :]


def los_grid_obs(ws: WorldState, agent_id: str) -> np.ndarray:
    """Per-cell line-of-sight PROXIMITY to the nearest live enemy.

    Returns: float32[N_LOS_GRID_CHANNELS, N_GRID, N_GRID]
      channel 0 — 1.0 if a creature standing at the cell centre would have an
                  unobstructed line to the nearest live enemy; otherwise a
                  graded value 1/(1+d) where d is the GEODESIC (walk-around-the-
                  wall, 8-connected) cell distance from this cell to the nearest
                  line-of-sight cell. Walls / unreachable cells → ~0. 1.0
                  everywhere with no live enemy / no battlefield (mirrors
                  distance_grid's no-enemy convention).

    Why graded, not binary: the pointwise grid_head scores each cell from its
    OWN features, so a binary has-LoS flag lets it prefer cells that ALREADY
    see the enemy but gives NO gradient to WALK toward one when no reachable
    cell sees yet — the agent flanks to the wall corner then stalls (measured:
    latEnd plateaus at the pillar edge, self-play freeze unchanged). The
    geodesic proximity rises monotonically along a walkable path to a sightline
    cell, so the linear head can climb it AROUND the wall = a multi-step flank.
    In open layouts every cell has LoS → proximity is 1.0 everywhere = a
    constant channel = bit-exact with the all-ones fast path (open play, hence
    symmetric team combat, is untouched). It is a PERCEPTION feature, the same
    kind as the existing distance-to-enemy channel — the policy still chooses;
    this only lets it see where the flank is.
    """
    out = np.ones((N_LOS_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32)
    bf = ws.combat.battlefield if (ws.combat and ws.combat.battlefield) else None
    if bf is None:
        return out
    wall = np.asarray(bf.cells) == int(TerrainType.BLOCKED)   # [ny, nx]
    # Fast path: no walls → no sight-blockers → LoS everywhere (the all-ones
    # default). The common open/difficult/lava layouts hit this every step,
    # avoiding the ray cast entirely.
    if not wall.any():
        return out
    _, enemy_ids = partition_entities(ws, agent_id)
    live = [ws.characters[cid].position
            for cid in enemy_ids if ws.characters[cid].is_alive()]
    if not live:
        return out

    cell_size = bf.width / N_GRID
    res = bf.grid_resolution
    ny, nx = wall.shape
    coord = (np.arange(N_GRID, dtype=np.float32) + 0.5) * cell_size
    cx, cy = np.meshgrid(coord, coord, indexing="ij")        # [N_GRID, N_GRID]

    # Vectorised ray cast: sample every cell->enemy segment at once. The same
    # half-resolution step and endpoint-exclusion as Battlefield.has_line_of_
    # sight; a single fixed K (sized to the diagonal) over-samples short rays
    # harmlessly. Per-enemy blocked grid, then combine by each cell's nearest.
    diag = (bf.width ** 2 + bf.height ** 2) ** 0.5
    K = max(2, int(np.ceil(diag / (res * 0.5))))
    ex = np.array([p.x for p in live], dtype=np.float32)
    ey = np.array([p.y for p in live], dtype=np.float32)
    d2 = np.stack([(cx - px) ** 2 + (cy - py) ** 2 for px, py in zip(ex, ey)])
    nearest = d2.argmin(axis=0)                              # [N_GRID, N_GRID]
    los = np.ones((N_GRID, N_GRID), dtype=bool)
    ts = (np.arange(1, K, dtype=np.float32) / K)             # exclude endpoints
    for ei in range(len(live)):
        blocked = np.zeros((N_GRID, N_GRID), dtype=bool)
        for t in ts:
            px = cx + (ex[ei] - cx) * t
            py = cy + (ey[ei] - cy) * t
            ix = np.clip((px / res).astype(np.int32), 0, nx - 1)
            iy = np.clip((py / res).astype(np.int32), 0, ny - 1)
            blocked |= wall[iy, ix]
        sel = nearest == ei
        los[sel] = ~blocked[sel]

    # Graded proximity = 1/(1+geodesic distance to the nearest LoS cell). A
    # multi-source 8-connected BFS from the LoS cells over the walkable grid
    # (wall cells are non-traversable, distance ∞). Min-plus relaxation: cheap
    # on 30×30, converges in ≤ a few × N iterations. The result is a field that
    # peaks (=1) on sightline cells and decays AROUND walls, so a pointwise
    # argmax over reachable cells steps along a path toward the flank.
    ix_g = np.clip((cx / res).astype(np.int32), 0, nx - 1)
    iy_g = np.clip((cy / res).astype(np.int32), 0, ny - 1)
    wall_cell = wall[iy_g, ix_g]                             # [N_GRID, N_GRID]
    INF = np.float32(1e9)
    dist = np.where(los & ~wall_cell, np.float32(0.0), INF).astype(np.float32)
    dist[wall_cell] = INF
    for _ in range(N_GRID * 2):
        padded = np.pad(dist, 1, constant_values=INF)
        best = dist.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neigh = padded[1 + dx:1 + dx + N_GRID, 1 + dy:1 + dy + N_GRID]
                best = np.minimum(best, neigh + np.float32(1.0))
        best[wall_cell] = INF
        if np.array_equal(best, dist):
            break
        dist = best
    prox = (1.0 / (1.0 + dist)).astype(np.float32)
    prox[wall_cell] = 0.0
    out[0] = prox
    return out


def build_obs(ws: WorldState, agent_id: str, resources: dict,
              decision_context: np.ndarray | None = None,
              skills: list | None = None) -> dict:
    """Assemble the full Phase 2 Dict observation.

    ``decision_context`` defaults to the all-zero vector = a normal own-turn
    step. An off-turn reaction / legendary decision point passes the vector from
    reaction_decision_context() / legendary_decision_context() so the policy can
    tell the decision apart and read the trigger details. ``skills`` overrides
    the skill-slot list for those off-turn points (reaction/legendary candidates)."""
    skills_mat, skill_mask = skills_obs(ws, agent_id, skills=skills)
    ent_skills, ent_skill_mask = entity_skills_obs(ws, agent_id)
    round_num = ws.combat.round_number if ws.combat else 0
    return {
        "skills":           skills_mat,
        "skill_mask":       skill_mask,
        "entity_skills":      ent_skills,        # obs v8: per-entity STATIC kit matrices
        "entity_skill_mask":  ent_skill_mask,    # (only read by encode_entity_skills nets)
        "entities":         entities_obs(ws, agent_id),
        "resources":        resources_obs(resources, round_num),
        "terrain":          terrain_obs(ws.combat.battlefield),
        "entity_grid":      entity_grid_obs(ws, agent_id),
        "distance_grid":    distance_grid_obs(ws, agent_id),
        "los_grid":         los_grid_obs(ws, agent_id),
        "reach_grid":       reach_grid_obs(ws, agent_id),
        "threat_grid":      threat_grid_obs(ws, agent_id),
        "end_features":     end_features(ws, agent_id, resources),
        "decision_context": (zero_decision_context() if decision_context is None
                             else decision_context.astype(np.float32)),
    }
