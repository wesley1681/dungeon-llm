from dataclasses import replace

import pytest

from trpg.engine.character import Stats
from trpg.scenarios.character_builder import (
    CLASS_DEFS, RACE_DEFS, build_character,
)
from trpg.scenarios.monsters import MONSTER_DEFS, make_monster


def test_class_direction_and_race_multiplier_are_normalized():
    fighter = CLASS_DEFS["fighter"]
    dwarf = RACE_DEFS["dwarf"]
    char = build_character(
        name="dwarf", race_id="dwarf", class_id="fighter",
        level=1,
    )
    weights = char.growth_profile.normalized_weights()

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["CON"] == pytest.approx((.30 * 1.3) / 1.09)
    assert weights["STR"] == pytest.approx(.35 / 1.09)
    assert fighter.growth_weights["STR"] == .35
    assert dwarf.growth_multipliers["CON"] == 1.3


def test_direct_generation_matches_stepwise_level_up():
    direct = build_character(
        name="direct", race_id="human", class_id="rogue",
        level=6, growth_seed=443,
    )
    stepwise = build_character(
        name="stepwise", race_id="human", class_id="rogue",
        level=1, growth_seed=443,
    )
    for level in range(2, 7):
        stepwise.level_up_to(level)

    assert direct.stats == stepwise.stats
    assert direct.max_hp == stepwise.max_hp
    assert direct.ac == stepwise.ac
    assert direct.growth_history == stepwise.growth_history
    assert len(direct.growth_history) == 5 * 2


def test_class_def_owns_rules_not_concrete_panel():
    fighter = CLASS_DEFS["fighter"]
    for forbidden in ("stats", "stat_block", "hp", "max_hp", "hp_base", "ac"):
        assert not hasattr(fighter, forbidden)


def test_archetype_adds_features_without_owning_panel():
    champion = build_character(
        name="champion", race_id="human", class_id="fighter",
        archetype_id="champion", level=5, growth_seed=12,
        weapons=["長劍"],
    )
    assert champion.archetype_id == "champion"
    assert champion.crit_range == 19
    assert champion.attacks_per_action == 2


def test_character_owns_vitality_and_equipment_driven_ac():
    char = build_character(
        name="tank", race_id="human", class_id="fighter",
        level=3, growth_seed=5, base_stats=Stats(DEX=14, CON=14),
        base_max_hp=10, hp_per_level=6,
        equipment=["鏈甲", "盾牌"],
    )
    assert char.max_hp > 10
    assert char.ac == 18
    assert char.hp_per_level == 6

    low_dex = build_character(
        name="heavy", race_id="human", class_id="fighter",
        base_stats=Stats(DEX=8), equipment=["鏈甲"],
    )
    assert low_dex.ac == 16


def test_monster_can_advance_from_its_catalog_panel():
    natural = make_monster("ogre")
    grown = make_monster("ogre", level=4, growth_seed=9)

    assert natural.level == 2
    assert grown.level == 4
    assert len(grown.growth_history) == 4
    assert grown.max_hp > natural.max_hp
    assert grown.ac == natural.ac == 11


def test_fixed_panel_is_an_optional_capability(monkeypatch):
    locked = replace(MONSTER_DEFS["ogre"], level_locked=True)
    monkeypatch.setitem(MONSTER_DEFS, "ogre", locked)

    with pytest.raises(ValueError, match="固定面板"):
        make_monster("ogre", level=3)
