"""Three chimera (縫合怪) identities for the zero-shot generalization probe.

Each one is a ClassDef that breaks a kit/panel correlation every training
identity respects — composed purely from existing skills and traits, which is
exactly what the data-driven archetype registry is for. They are registered
at RUNTIME (never imported by trpg.*), so N_ARCHETYPES / obs dims are
untouched: a chimera's archetype one-hot is simply all-zero, the same input
a future monster would produce.

What each one probes:
  chimera_omni        the user's spec — rage + heal + fireball + misty_step +
                      hold_person + action_surge. Bonus-action arbitration
                      (rage vs healing_word) and stance arbitration
                      (rage-melee vs fireball-kite) were demonstrated by NO
                      teacher.
  chimera_gish        tanky panel (AC18, aura, lay-on-hands, extra attack)
                      with a wizard's nukes (fireball/misty_step/shield).
                      The panel says "front line", the kit says "back line" —
                      the correlation every teacher obeyed is broken.
  chimera_trickhealer rogue chassis (hidden start, sneak attack, cunning
                      actions) carrying cleric support (healing word, cure,
                      sacred flame, hold person). Bonus action is contested
                      three ways: cunning dash/hide vs healing_word.
"""
from __future__ import annotations

# Import order matters: pull in obs FIRST so N_ARCHETYPES is frozen from the
# 12 standard classes before we mutate the registry.
import trpg.rl.obs  # noqa: F401  (freeze N_ARCHETYPES)

from trpg.scenarios.archetypes import (
    CLASS_DEFS, ARCHETYPE_FACTORIES, ARCHETYPE_ROLES,
    ClassDef, SkillGrant, TraitGrant, _factory, _WIZARD_SLOTS, _RAGE_USES_TABLE,
)
from trpg.engine.status import Hidden


CHIMERA_DEFS = [
    ClassDef(
        archetype_id="chimera_omni", default_name="Omni Chimera",
        class_display="縫合怪", role="front",
        stat_block=dict(STR=16, DEX=12, CON=14, INT=10, WIS=16, CHA=10),
        hp_base=10, hp_per_level=7, ac=16,
        weapons=("長劍",),
        proficiencies=("STR", "CON"),
        spell_ability="WIS", spell_slots_table=_WIZARD_SLOTS,
        skills=(
            SkillGrant("rage"),
            SkillGrant("cure_wounds"),
            SkillGrant("healing_word_life"),
            SkillGrant("action_surge", min_level=2),
            SkillGrant("misty_step", min_level=3),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("fireball_ev", min_level=5),
        ),
        traits=(
            TraitGrant("uses_table", params={
                "skill": "rage", "table": _RAGE_USES_TABLE, "default": 4}),
            TraitGrant("extra_attack", min_level=5),
        ),
    ),
    ClassDef(
        archetype_id="chimera_gish", default_name="Gish Chimera",
        class_display="縫合怪", role="front",
        stat_block=dict(STR=16, DEX=10, CON=14, INT=10, WIS=12, CHA=16),
        hp_base=10, hp_per_level=8, ac=18,
        weapons=("長劍",),
        proficiencies=("WIS", "CHA"),
        spell_ability="CHA", spell_slots_table=_WIZARD_SLOTS,
        skills=(
            SkillGrant("lay_on_hands_ability"),
            SkillGrant("divine_smite_dev", min_level=2),
            SkillGrant("shield_spell", min_level=2, reaction=True),
            SkillGrant("misty_step", min_level=3),
            SkillGrant("fireball_ev", min_level=5),
        ),
        traits=(
            TraitGrant("lay_on_hands", params={"per_level": 5}),
            TraitGrant("extra_attack", min_level=5),
            TraitGrant("aura_of_protection", min_level=6,
                       params={"ability": "CHA"}),
        ),
    ),
    ClassDef(
        archetype_id="chimera_trickhealer", default_name="Trickhealer Chimera",
        class_display="縫合怪", role="striker",
        stat_block=dict(STR=10, DEX=17, CON=12, INT=10, WIS=16, CHA=10),
        hp_base=8, hp_per_level=5, ac=14,
        weapons=("短劍", "短弓"),
        proficiencies=("DEX", "WIS", "潛行", "察覺"),
        consumables=(("箭", 20, "ammo", ""),),
        spell_ability="WIS", spell_slots_table=_WIZARD_SLOTS,
        skills=(
            SkillGrant("sacred_flame"),
            SkillGrant("cure_wounds"),
            SkillGrant("healing_word_life"),
            SkillGrant("cunning_action_dash", min_level=2),
            SkillGrant("cunning_action_disengage", min_level=2),
            SkillGrant("cunning_action_hide", min_level=2),
            SkillGrant("assassinate", min_level=3),
            SkillGrant("hold_person", min_level=3),
            SkillGrant("uncanny_dodge_rogue", min_level=5, reaction=True),
        ),
        traits=(
            TraitGrant("sneak_attack", params={"die": 6}),
            TraitGrant("starting_status", params={"status": Hidden}),
        ),
    ),
]

CHIMERA_IDS = tuple(cd.archetype_id for cd in CHIMERA_DEFS)


def register_chimeras() -> tuple[str, ...]:
    """Add the chimeras to the live registries (idempotent).

    Must be called AFTER trpg.rl.obs is imported (this module's own import
    guarantees it) so the obs schema keeps N_ARCHETYPES = 12 and chimeras get
    the all-zero one-hot — the monster condition.
    """
    for cd in CHIMERA_DEFS:
        CLASS_DEFS[cd.archetype_id] = cd
        ARCHETYPE_FACTORIES[cd.archetype_id] = _factory(cd.archetype_id)
        ARCHETYPE_ROLES[cd.archetype_id] = cd.role
    return CHIMERA_IDS
