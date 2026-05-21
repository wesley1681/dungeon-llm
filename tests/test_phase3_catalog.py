import pytest
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.engine.spells import SPELLS
from trpg.engine.status import MODIFIER_CLASSES


def _check_registered(skill_ids: list, class_id: str, archetype_id: str,
                       min_level: int = None):
    for sid in skill_ids:
        ab = ABILITY_REGISTRY.get(sid)
        assert ab is not None, f"'{sid}' missing from ABILITY_REGISTRY"
        assert ab.class_id == class_id, f"{sid}: expected class_id='{class_id}', got '{ab.class_id}'"
        assert ab.archetype_id == archetype_id, (
            f"{sid}: expected archetype_id='{archetype_id}', got '{ab.archetype_id}'"
        )
        if min_level is not None:
            assert ab.min_level == min_level, (
                f"{sid}: expected min_level={min_level}, got {ab.min_level}"
            )


def test_new_status_classes_registered():
    for name in ["blurred", "baned", "sacred_weapon_buff"]:
        assert name in MODIFIER_CLASSES, f"'{name}' missing from MODIFIER_CLASSES"


def test_new_spells_registered():
    for spell_name in ["燃燒之手", "蜘蛛網", "冰風暴", "定怪術", "治療語"]:
        assert spell_name in SPELLS, f"'{spell_name}' missing from SPELLS"


def test_fighter_battle_master_catalog():
    _check_registered(
        ["menacing_attack", "precision_attack", "pushing_attack"],
        class_id="fighter", archetype_id="battle_master", min_level=3,
    )
    assert ABILITY_REGISTRY["menacing_attack"].engine_ready is True
    assert ABILITY_REGISTRY["precision_attack"].engine_ready is False
    assert ABILITY_REGISTRY["pushing_attack"].engine_ready is False


def test_fighter_champion_catalog():
    _check_registered(["improved_critical"], class_id="fighter", archetype_id="champion", min_level=3)
    assert ABILITY_REGISTRY["improved_critical"].engine_ready is False


def test_barbarian_catalog():
    _check_registered(["bear_totem", "frenzy_attack"], class_id="barbarian", archetype_id="totem_bear")
    _check_registered(["berserker_frenzy"], class_id="barbarian", archetype_id="berserker")
    assert ABILITY_REGISTRY["bear_totem"].engine_ready is False
    assert ABILITY_REGISTRY["frenzy_attack"].engine_ready is True
    assert ABILITY_REGISTRY["berserker_frenzy"].engine_ready is True


def test_wizard_evocation_catalog():
    _check_registered(
        ["burning_hands_ev", "scorching_ray_ev", "fireball_ev",
         "web_ev", "ice_storm_ev", "hold_monster_ev"],
        class_id="wizard", archetype_id="evocation",
    )
    assert ABILITY_REGISTRY["fireball_ev"].engine_ready is True
    assert ABILITY_REGISTRY["web_ev"].engine_ready is True
    assert ABILITY_REGISTRY["scorching_ray_ev"].engine_ready is False


def test_wizard_divination_catalog():
    _check_registered(
        ["burning_hands_div", "web_div", "fireball_div", "hold_monster_div"],
        class_id="wizard", archetype_id="divination",
    )


def test_cleric_life_catalog():
    _check_registered(
        ["healing_word_life", "guiding_bolt_life", "spiritual_weapon_life",
         "channel_divinity_preserve_life", "mass_cure_wounds_life"],
        class_id="cleric", archetype_id="life",
    )
    assert ABILITY_REGISTRY["healing_word_life"].engine_ready is True
    assert ABILITY_REGISTRY["guiding_bolt_life"].engine_ready is False
    assert ABILITY_REGISTRY["spiritual_weapon_life"].engine_ready is False
    assert ABILITY_REGISTRY["mass_cure_wounds_life"].engine_ready is False


def test_cleric_war_catalog():
    _check_registered(
        ["guiding_bolt_war", "channel_divinity_guided_strike",
         "spiritual_weapon_war", "war_priest_attack"],
        class_id="cleric", archetype_id="war",
    )
    assert ABILITY_REGISTRY["war_priest_attack"].engine_ready is True
    assert ABILITY_REGISTRY["channel_divinity_guided_strike"].engine_ready is False


def test_rogue_assassin_catalog():
    _check_registered(
        ["cunning_action_dash", "cunning_action_disengage", "cunning_action_hide",
         "assassinate", "uncanny_dodge_rogue", "evasion_rogue"],
        class_id="rogue", archetype_id="assassin",
    )
    assert ABILITY_REGISTRY["cunning_action_dash"].engine_ready is True
    assert ABILITY_REGISTRY["assassinate"].engine_ready is False
    assert ABILITY_REGISTRY["evasion_rogue"].engine_ready is True


def test_rogue_arcane_trickster_catalog():
    _check_registered(
        ["cunning_action_dash_at", "cunning_action_disengage_at",
         "uncanny_dodge_at", "evasion_at"],
        class_id="rogue", archetype_id="arcane_trickster",
    )


def test_paladin_devotion_catalog():
    _check_registered(
        ["divine_smite_dev", "lay_on_hands_ability", "shield_of_faith_dev",
         "sacred_weapon_dev", "wrathful_smite_dev"],
        class_id="paladin", archetype_id="devotion",
    )
    assert ABILITY_REGISTRY["divine_smite_dev"].engine_ready is True
    assert ABILITY_REGISTRY["shield_of_faith_dev"].engine_ready is True
    assert ABILITY_REGISTRY["sacred_weapon_dev"].engine_ready is True


def test_paladin_vengeance_catalog():
    _check_registered(
        ["divine_smite_ven", "bane_ven", "vow_of_enmity_ven", "lay_on_hands_ability_ven"],
        class_id="paladin", archetype_id="vengeance",
    )
    assert ABILITY_REGISTRY["bane_ven"].engine_ready is True
    assert ABILITY_REGISTRY["vow_of_enmity_ven"].engine_ready is True


from unittest.mock import patch
from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.items import WEAPON_DEFS
from trpg.engine.combat import execute_action


def test_full_catalog_count():
    """Total ABILITY_REGISTRY should be >= 45 after Phase 3."""
    assert len(ABILITY_REGISTRY) >= 45, (
        f"Expected >=45 abilities, got {len(ABILITY_REGISTRY)}"
    )


def test_divine_smite_applies_bonus_damage():
    pal = Character(name="P", race="", class_="聖騎士", level=3,
                    stats=Stats(STR=16), hp=25, max_hp=25, ac=18, is_npc=False,
                    weapons=[WEAPON_DEFS["長劍"]], spell_slots={1: 2})
    enemy = Character(name="E", race="", class_="", level=1,
                      stats=Stats(), hp=100, max_hp=100, ac=10,
                      is_npc=True, attitude=0)
    pal.position = Vec2(5, 5)
    enemy.position = Vec2(6, 5)
    ws = WorldState(characters={"p": pal, "e": enemy}, scene="", dungeon_map=None)
    ws.party_ids = ["p"]
    ws.combat = CombatState(initiative_order=["p", "e"])
    with patch("trpg.engine.combat.roll_d20", return_value=15), \
         patch("trpg.engine.combat.roll", side_effect=[5, 7, 4]):
        # rolls: 1d8 weapon (5), then 2d8 smite: 1d8 each (7, 4) = 11 total
        res = execute_action({
            "type": "ATTACK", "skill_id": "test_handwritten", "attacker": "p", "target": "e",
            "weapon": "長劍", "divine_smite_slot": 1, "consumes": ["action"],
        }, ws)
    assert res["hit"] is True
    assert res.get("divine_smite_damage", 0) == 11
    assert pal.spell_slots[1] == 1   # one slot consumed from 2


def test_divine_smite_does_not_fire_on_miss():
    pal = Character(name="P", race="", class_="聖騎士", level=3,
                    stats=Stats(STR=16), hp=25, max_hp=25, ac=18, is_npc=False,
                    weapons=[WEAPON_DEFS["長劍"]], spell_slots={1: 2})
    enemy = Character(name="E", race="", class_="", level=1,
                      stats=Stats(), hp=100, max_hp=100, ac=20,
                      is_npc=True, attitude=0)
    pal.position = Vec2(5, 5)
    enemy.position = Vec2(6, 5)
    ws = WorldState(characters={"p": pal, "e": enemy}, scene="", dungeon_map=None)
    ws.party_ids = ["p"]
    ws.combat = CombatState(initiative_order=["p", "e"])
    with patch("trpg.engine.combat.roll_d20", return_value=5):
        res = execute_action({
            "type": "ATTACK", "skill_id": "test_handwritten", "attacker": "p", "target": "e",
            "weapon": "長劍", "divine_smite_slot": 1, "consumes": ["action"],
        }, ws)
    assert res["hit"] is False
    assert res.get("divine_smite_damage", 0) == 0
    assert pal.spell_slots[1] == 2   # no slot consumed on miss
