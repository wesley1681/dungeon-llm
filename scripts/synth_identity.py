"""Random legal identity synthesis from the CLASS_DEFS component pools.

A synthesized identity = a standard class's numeric chassis (panel, HP, AC,
weapons) + a random bundle of skill grants drawn from the union of every
standard class's kit + whatever traits those skills depend on (plus a few
random standalone passives). Everything is sampled from the data registry —
no skill, class, or trait is named in this module.

Legality rules (all derived from registry data, not hand-listed):
  - If any sampled ability costs a spell slot (features.cost_slot_level > 0),
    the identity gets a casting ability + slots table from one of the sampled
    spells' donor classes.
  - A trait is LINKED to the kit when its params["skill"] or its own trait_id
    is a prefix of a sampled skill_id (covers shared-pool mechanics like
    channel_divinity_* and lay_on_hands_*); linked traits are always included.
  - Standalone passives (no skill linkage) are each included with prob 0.3.
  - The identity must be able to deal damage at the lowest training level:
    a weapon, or a granted ability with expected_damage > 0. Rejection-sample
    otherwise.

Synthesized defs are registered under rotating ids (synth_0..synth_N-1) so the
registry never grows unboundedly; the obs one-hot stays all-zero for them
(N_ARCHETYPES is frozen by importing trpg.rl.obs first) — the monster
condition, exactly like the chimeras.
"""
from __future__ import annotations

import random

import trpg.rl.obs  # noqa: F401  (freeze N_ARCHETYPES before registry mutation)

from trpg.scenarios.archetypes import (
    CLASS_DEFS, ARCHETYPE_FACTORIES, ARCHETYPE_ROLES, STANDARD_ARCHETYPES,
    ClassDef, SkillGrant, TraitGrant, _factory, make_character,
)
from trpg.engine.abilities import ABILITY_REGISTRY

# The 12 standard ids — the frozen module-body constant, not a local
# snapshot, so a process that registers monsters before importing this
# module still gets the true standard panel.
STANDARD_IDS: tuple[str, ...] = STANDARD_ARCHETYPES

N_SYNTH_SLOTS = 64          # rotating registry ids synth_0 .. synth_63


def _component_pools():
    """(skill_pool, trait_pool) from the standard defs.

    skill_pool: skill_id -> list[(SkillGrant, donor ClassDef)]
    trait_pool: list[(TraitGrant, donor ClassDef)] (deduped by id+params)
    """
    skill_pool: dict[str, list] = {}
    trait_pool: list = []
    seen_traits = set()
    for aid in STANDARD_IDS:
        cd = CLASS_DEFS[aid]
        for g in cd.skills:
            skill_pool.setdefault(g.skill_id, []).append((g, cd))
        for t in cd.traits:
            key = (t.trait_id, t.min_level, tuple(sorted(
                (k, str(v)) for k, v in t.params.items())))
            if key not in seen_traits:
                seen_traits.add(key)
                trait_pool.append((t, cd))
    return skill_pool, trait_pool


_SKILL_POOL, _TRAIT_POOL = _component_pools()


def _trait_links_skill(t: TraitGrant, skill_ids: set[str]) -> bool:
    """A trait is linked when its skill param or its own id prefixes a
    sampled skill (channel_divinity -> channel_divinity_preserve_life,
    lay_on_hands -> lay_on_hands_ability, rage -> rage, ...)."""
    keys = [t.trait_id]
    if "skill" in t.params:
        keys.append(t.params["skill"])
    return any(sid == k or sid.startswith(k) for k in keys for sid in skill_ids)


_PROBE_CHAR = None


def _ability_deals_damage(skill_id: str) -> bool:
    """Does this ability deal damage? expected_damage is now DERIVED on
    materialize (0 on the static features), so probe it with a generic capable
    character (has weapons + stats) and sum the built action's damage packets."""
    global _PROBE_CHAR
    ab = ABILITY_REGISTRY.get(skill_id)
    if ab is None or ab.builder is None:
        return False
    if _PROBE_CHAR is None:
        _PROBE_CHAR = make_character("champion", level=20)
    from trpg.engine.skill import action_expected_damage
    from trpg.engine.vec2 import Vec2
    act = ab.build_action(_PROBE_CHAR.name, "foe", Vec2(0.0, 0.0), char=_PROBE_CHAR)
    return action_expected_damage(act, _PROBE_CHAR) > 0


def _can_fight(cd: ClassDef, level: int) -> bool:
    if cd.weapons:
        return True
    for g in cd.skills:
        if g.min_level > level or g.reaction:
            continue
        if _ability_deals_damage(g.skill_id):
            return True
    return False


def synth_class_def(rng: random.Random, ident_id: str,
                    min_skills: int = 4, max_skills: int = 8) -> ClassDef:
    """Sample one legal synthesized ClassDef (rejection-sampled)."""
    for _ in range(64):
        chassis = CLASS_DEFS[rng.choice(STANDARD_IDS)]
        k = rng.randint(min_skills, max_skills)
        sampled_ids = rng.sample(list(_SKILL_POOL.keys()),
                                 min(k, len(_SKILL_POOL)))
        grants = []
        donors = []
        for sid in sampled_ids:
            g, donor = rng.choice(_SKILL_POOL[sid])
            grants.append(g)
            donors.append(donor)
        # obs slot order = declaration order; keep it deterministic per sample
        grants.sort(key=lambda g: (g.min_level, g.skill_id))
        skill_ids = {g.skill_id for g in grants}

        # Spellcasting: required iff any sampled ability costs a slot.
        needs_slots = any(
            (ab := ABILITY_REGISTRY.get(g.skill_id)) is not None
            and ab.features.cost_slot_level > 0
            for g in grants)
        spell_ability, slots_table, cast_min = "", None, 1
        if needs_slots:
            caster_donors = [d for g, d in zip(grants, donors)
                             if d.spell_slots_table
                             and (ab := ABILITY_REGISTRY.get(g.skill_id))
                             and ab.features.cost_slot_level > 0]
            src = rng.choice(caster_donors) if caster_donors else None
            if src is None:
                continue   # slot-costing skill with no caster donor — resample
            slots_table = src.spell_slots_table
            spell_ability = chassis.spell_ability or src.spell_ability
            cast_min = 1   # synths cast as soon as the skill unlocks

        all_pool_skills = set(_SKILL_POOL)
        traits = []
        for t, _ in _TRAIT_POOL:
            if _trait_links_skill(t, skill_ids):
                traits.append(t)               # dependency satisfied → include
            elif _trait_links_skill(t, all_pool_skills):
                continue                       # links a skill we didn't sample
            elif rng.random() < 0.3:
                traits.append(t)               # standalone passive

        cd = ClassDef(
            archetype_id=ident_id, default_name=ident_id,
            class_display="合成", role=chassis.role,
            stat_block=dict(chassis.stat_block),
            hp_base=chassis.hp_base, hp_per_level=chassis.hp_per_level,
            ac=chassis.ac, weapons=chassis.weapons,
            proficiencies=chassis.proficiencies,
            consumables=chassis.consumables,
            spell_ability=spell_ability, spell_slots_table=slots_table,
            spellcasting_min_level=cast_min,
            skills=tuple(grants), traits=tuple(traits),
        )
        if not _can_fight(cd, level=3):
            continue
        # Must build cleanly across the level range the env samples.
        prev = CLASS_DEFS.get(ident_id)
        CLASS_DEFS[ident_id] = cd
        try:
            for lvl in (3, 8):
                c = make_character(ident_id, level=lvl)
                assert c.known_abilities or c.weapons
        except Exception:
            if prev is None:
                CLASS_DEFS.pop(ident_id, None)
            else:
                CLASS_DEFS[ident_id] = prev
            continue
        return cd
    raise RuntimeError("synth_class_def: rejection sampling failed 64 times")


def register_synth(cd: ClassDef) -> str:
    """(Re-)register a synthesized def under its id; returns the id."""
    CLASS_DEFS[cd.archetype_id] = cd
    ARCHETYPE_FACTORIES[cd.archetype_id] = _factory(cd.archetype_id)
    ARCHETYPE_ROLES[cd.archetype_id] = cd.role
    return cd.archetype_id


def fresh_synth(rng: random.Random, slot: int) -> str:
    """Sample + register a fresh identity into rotating slot `slot`."""
    ident = f"synth_{slot % N_SYNTH_SLOTS}"
    cd = synth_class_def(rng, ident)
    register_synth(cd)
    return ident
