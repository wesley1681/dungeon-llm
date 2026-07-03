"""Regression: a non-spellcaster must NOT be offered a class spell it cannot
cast (2026-07-01 GUI bug).

The sandbox lets you hand any character any skill. Handing a fighter a wizard
cantrip (chill_touch) used to leave it in available_skills even though
execute_action rejects it ("不是施法者"). The policy then picked it every
sub-action, execute_action ERRORed, _run_opponent_turn spent NO resource on the
error and re-picked it up to the sub-action cap → the enemy spammed a 0-damage
cantrip and never did anything (combat_log.txt). available_skills now mirrors
execute_action's SPELL gate. Innate abilities (breath weapons: save_dc_ability)
and real casters are unaffected.
"""
import dataclasses
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                       ARCHETYPE_ROLES, SkillGrant, _factory)
from trpg.engine.skill import available_skills, _requires_spellcasting
from trpg.engine.abilities import ABILITY_REGISTRY


def _mk_bm_with(*skill_ids):
    base = CLASS_DEFS["battle_master"]
    aid = "bm_test_" + "_".join(skill_ids)
    cd = dataclasses.replace(
        base, archetype_id=aid,
        skills=tuple(base.skills) + tuple(SkillGrant(s, min_level=1) for s in skill_ids))
    CLASS_DEFS[aid] = cd
    ARCHETYPE_FACTORIES[aid] = _factory(aid)
    ARCHETYPE_ROLES[aid] = cd.role
    ch = ARCHETYPE_FACTORIES[aid](f"x_{aid}")
    ch.level = 8
    return ch


def test_noncaster_not_offered_class_cantrip():
    """A fighter handed chill_touch (class cantrip, no innate DC) must NOT see
    it in available_skills — it would ERROR at execution."""
    ch = _mk_bm_with("chill_touch")
    assert not ch.spellcasting_ability          # fighter = non-caster
    ids = [s.skill_id for s in available_skills(ch)]
    assert "chill_touch" not in ids, ids


def test_noncaster_keeps_innate_breath_weapon():
    """cold_breath rides the SPELL pipeline but carries its own save DC
    (save_dc_ability) — a non-caster CAN use it, so it stays available."""
    ch = _mk_bm_with("cold_breath")
    ids = [s.skill_id for s in available_skills(ch)]
    assert "cold_breath" in ids, ids


def test_caster_keeps_class_spells():
    """A real caster (evocation wizard, spellcasting_ability=INT) still gets its
    class spells — the gate is a no-op for spellcasters."""
    evo = ARCHETYPE_FACTORIES["evocation"](f"x_evo")
    evo.level = 8
    assert evo.spellcasting_ability
    ids = [s.skill_id for s in available_skills(evo)]
    assert any(s in ids for s in ("fireball_ev", "hold_person", "burning_hands_ev")), ids


def test_requires_spellcasting_classification():
    """The helper distinguishes class spells from innate abilities and
    non-spells."""
    assert _requires_spellcasting(ABILITY_REGISTRY["chill_touch"]) is True
    assert _requires_spellcasting(ABILITY_REGISTRY["cold_breath"]) is False   # innate DC
    assert _requires_spellcasting(ABILITY_REGISTRY["action_surge"]) is False  # not a spell
