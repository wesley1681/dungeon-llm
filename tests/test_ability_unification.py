"""Regression tests for the ability registry unification refactor.

These tests capture behavior that MUST hold both before and after the refactor:
  1. Healing skills produce HEAL actions (not SPELL actions).
  2. No archetype's available_skills returns duplicate display_names
     (catches the dual-path 治療術 + cure_wounds bug class).
  3. Every skill returned by available_skills has a builder that returns
     either None or a valid action dict (never raises).
"""
from __future__ import annotations
import pytest

from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import (
    make_life_cleric, make_war_cleric, make_devotion_paladin,
    make_vengeance_paladin, make_evocation_wizard, make_divination_wizard,
    make_berserker, make_totem_bear, make_battle_master, make_champion,
    make_assassin, make_arcane_trickster,
)


ARCHETYPE_BUILDERS = [
    make_life_cleric, make_war_cleric, make_devotion_paladin,
    make_vengeance_paladin, make_evocation_wizard, make_divination_wizard,
    make_berserker, make_totem_bear, make_battle_master, make_champion,
    make_assassin, make_arcane_trickster,
]


def _action_types_for_display_name(builder_fn, display_name, level=3):
    caster = builder_fn("Caster", level=level)
    skills = available_skills(caster, world_state=None)
    types = set()
    for s in skills:
        if s.display_name != display_name:
            continue
        a = s.builder("caster", "caster", None)
        if isinstance(a, dict):
            types.add(a.get("type"))
    return types, len(skills)


def test_cure_wounds_builder_produces_HEAL_action():
    """At least one skill named 治療術 must produce a HEAL action.
    Pre-refactor: may also include a SPELL entry (broken duplicate).
    Post-refactor: only HEAL remains."""
    types, _ = _action_types_for_display_name(make_life_cleric, "治療術")
    assert "HEAL" in types, f"治療術 must include a HEAL-producing entry, got {types}"


def test_healing_word_builder_produces_HEAL_action():
    types, _ = _action_types_for_display_name(make_life_cleric, "治療語")
    assert "HEAL" in types, f"治療語 must include a HEAL-producing entry, got {types}"


@pytest.mark.parametrize("builder", ARCHETYPE_BUILDERS)
def test_no_duplicate_display_names_in_available_skills(builder):
    """After unification there must be exactly one entry per spell.
    Currently this test FAILS for wizards/clerics/paladins because the
    spells list and abilities list both register the same spell. The
    refactor makes this test pass for everyone."""
    caster = builder("Caster", level=5)
    skills = available_skills(caster, world_state=None)
    names = [s.display_name for s in skills]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, (
        f"{builder.__name__} has duplicate skill names: {duplicates}"
    )


@pytest.mark.parametrize("builder", ARCHETYPE_BUILDERS)
def test_every_skill_builder_returns_action_dict_or_none(builder):
    """Every skill returned by available_skills must have a builder that
    returns either None or a dict — never raises and never returns garbage."""
    from trpg.engine.vec2 import Vec2
    caster = builder("Caster", level=5)
    skills = available_skills(caster, world_state=None)
    for sk in skills:
        action = sk.builder("caster", "enemy", Vec2(1.0, 0.0))
        assert action is None or isinstance(action, dict), (
            f"{builder.__name__}/{sk.skill_id} builder returned {type(action)}"
        )
