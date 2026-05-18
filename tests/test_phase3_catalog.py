import pytest
from trpg.engine.abilities import CLASS_ABILITIES
from trpg.engine.spells import SPELLS
from trpg.engine.status import MODIFIER_CLASSES


def _check_registered(skill_ids: list, class_id: str, archetype_id: str,
                       min_level: int = None):
    for sid in skill_ids:
        ab = CLASS_ABILITIES.get(sid)
        assert ab is not None, f"'{sid}' missing from CLASS_ABILITIES"
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
    assert CLASS_ABILITIES["menacing_attack"].engine_ready is True
    assert CLASS_ABILITIES["precision_attack"].engine_ready is False
    assert CLASS_ABILITIES["pushing_attack"].engine_ready is False


def test_fighter_champion_catalog():
    _check_registered(["improved_critical"], class_id="fighter", archetype_id="champion", min_level=3)
    assert CLASS_ABILITIES["improved_critical"].engine_ready is False


def test_barbarian_catalog():
    _check_registered(["bear_totem", "frenzy_attack"], class_id="barbarian", archetype_id="totem_bear")
    _check_registered(["berserker_frenzy"], class_id="barbarian", archetype_id="berserker")
    assert CLASS_ABILITIES["bear_totem"].engine_ready is False
    assert CLASS_ABILITIES["frenzy_attack"].engine_ready is True
    assert CLASS_ABILITIES["berserker_frenzy"].engine_ready is True


def test_wizard_evocation_catalog():
    _check_registered(
        ["burning_hands_ev", "scorching_ray_ev", "fireball_ev",
         "web_ev", "ice_storm_ev", "hold_monster_ev"],
        class_id="wizard", archetype_id="evocation",
    )
    assert CLASS_ABILITIES["fireball_ev"].engine_ready is True
    assert CLASS_ABILITIES["web_ev"].engine_ready is True
    assert CLASS_ABILITIES["scorching_ray_ev"].engine_ready is False


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
    assert CLASS_ABILITIES["healing_word_life"].engine_ready is True
    assert CLASS_ABILITIES["guiding_bolt_life"].engine_ready is False
    assert CLASS_ABILITIES["spiritual_weapon_life"].engine_ready is False
    assert CLASS_ABILITIES["mass_cure_wounds_life"].engine_ready is False


def test_cleric_war_catalog():
    _check_registered(
        ["guiding_bolt_war", "channel_divinity_guided_strike",
         "spiritual_weapon_war", "war_priest_attack"],
        class_id="cleric", archetype_id="war",
    )
    assert CLASS_ABILITIES["war_priest_attack"].engine_ready is True
    assert CLASS_ABILITIES["channel_divinity_guided_strike"].engine_ready is False


def test_rogue_assassin_catalog():
    _check_registered(
        ["cunning_action_dash", "cunning_action_disengage", "cunning_action_hide",
         "assassinate", "uncanny_dodge_rogue", "evasion_rogue"],
        class_id="rogue", archetype_id="assassin",
    )
    assert CLASS_ABILITIES["cunning_action_dash"].engine_ready is True
    assert CLASS_ABILITIES["assassinate"].engine_ready is False
    assert CLASS_ABILITIES["evasion_rogue"].engine_ready is True


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
    assert CLASS_ABILITIES["divine_smite_dev"].engine_ready is True
    assert CLASS_ABILITIES["shield_of_faith_dev"].engine_ready is True
    assert CLASS_ABILITIES["sacred_weapon_dev"].engine_ready is True


def test_paladin_vengeance_catalog():
    _check_registered(
        ["divine_smite_ven", "bane_ven", "vow_of_enmity_ven", "lay_on_hands_ability_ven"],
        class_id="paladin", archetype_id="vengeance",
    )
    assert CLASS_ABILITIES["bane_ven"].engine_ready is True
    assert CLASS_ABILITIES["vow_of_enmity_ven"].engine_ready is True
