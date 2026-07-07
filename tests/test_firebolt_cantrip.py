"""火焰箭 (Fire Bolt) — the wizard's at-will attack cantrip.

Both wizard subclasses (evocation / divination) used to have a 100% slot-gated
kit: once every spell slot was spent they had no ranged damage option at all,
only a STR-8 melee basic attack (the user's play-test: "打完火球、魔法飛彈就啥事
都不能幹"). Clerics already solved this with 神聖光輝 (sacred_flame). This pins
that firebolt is a real cantrip (no slot), reaches both wizards from level 1, and
stays available after every slot is gone.
"""
from unittest.mock import patch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.combat import execute_action
from trpg.engine.skill import available_skills, _requires_spellcasting
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.engine.spells import SPELLS
from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES


# ── the ability itself is a fire cantrip ──────────────────────────────────────

def test_firebolt_is_a_registered_fire_cantrip():
    ab = ABILITY_REGISTRY["firebolt"]
    assert ab.engine_ready
    f = ab.features
    assert f.cost_slot_level == 0.0                  # cantrip → no slot
    assert f.cost_action == 1.0                       # costs the action
    assert "火" in [t for t, _ in f.iter_damage_types()]
    spell = SPELLS[ab.display_name]
    assert spell.level == 0 and spell.scales_as_cantrip   # scales with level
    assert spell.damage_type == "火"


def test_firebolt_requires_spellcasting_like_other_class_cantrips():
    # class cantrip (no innate save_dc_ability) → gated to real casters, exactly
    # like chill_touch / sacred_flame. A non-caster must NOT be offered it.
    assert _requires_spellcasting(ABILITY_REGISTRY["firebolt"]) is True


# ── both wizards get it, and it is NOT slot-gated ─────────────────────────────

def _wizard(archetype_id, level=1):
    w = ARCHETYPE_FACTORIES[archetype_id](f"x_{archetype_id}")
    w.level = level
    w.position = Vec2(5, 5)
    return w


def _dummy(hp=100, ac=10):
    e = Character(name="E", race="", class_="", level=1, stats=Stats(),
                  hp=hp, max_hp=hp, ac=ac, is_npc=True, attitude=0)
    e.position = Vec2(6, 5)
    return e


def _world(caster, enemy):
    ws = WorldState(characters={"c": caster, "e0": enemy}, scene="",
                    dungeon_map=None)
    ws.party_ids = ["c"]
    ws.combat = CombatState(initiative_order=["c", "e0"])
    return ws


def test_both_wizards_have_firebolt_from_level_1():
    for arch in ("evocation", "divination"):
        wiz = _wizard(arch, level=1)
        ws = _world(wiz, _dummy())
        ids = [s.skill_id for s in available_skills(wiz, ws)]
        assert "firebolt" in ids, (arch, ids)


def test_firebolt_survives_when_all_slots_are_spent():
    """The whole point: with every slot at 0 the evoker still has an at-will
    ranged option (firebolt), whereas its slotted kit disappears."""
    wiz = _wizard("evocation", level=5)
    ws = _world(wiz, _dummy())
    for lvl in list(wiz.spell_slots):
        wiz.spell_slots[lvl] = 0
    ids = [s.skill_id for s in available_skills(wiz, ws)]
    assert "firebolt" in ids, ids                     # cantrip stays
    assert "fireball_ev" not in ids                   # slotted spell gone


def test_firebolt_deals_fire_and_spends_no_slot():
    wiz = _wizard("evocation", level=5)
    slots_before = dict(wiz.spell_slots)
    enemy = _dummy(hp=100, ac=10)
    ws = _world(wiz, enemy)
    built = ABILITY_REGISTRY["firebolt"].build_action("c", "e0", None, char=wiz)
    # save fails → full cantrip damage lands
    with patch("trpg.engine.combat.make_saving_throw", return_value=(False, 1)):
        res = execute_action(built, ws)
    assert res["type"] != "ERROR", res
    assert enemy.hp < 100                              # took fire damage
    assert dict(wiz.spell_slots) == slots_before       # cantrip: no slot spent
