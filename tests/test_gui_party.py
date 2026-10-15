from trpg.engine.character import Character, Stats
from trpg.engine.items import Armor, Consumable, Weapon
from trpg.engine.status import Dodging
from trpg.engine.world_state import WorldState
from trpg.gui import character_panel_data, party_panel_data


def _character(character_id: str, name: str) -> Character:
    return Character(
        id=character_id,
        name=name,
        race="人類",
        class_="戰士",
        level=3,
        stats=Stats(STR=14, DEX=12, CON=13, INT=8, WIS=10, CHA=9),
        hp=17,
        max_hp=24,
        ac=16,
        weapons=[Weapon("長劍", "1d8", "斬擊", "近戰")],
        consumables=[Consumable("治療藥水", 2, "heal", "2d4+2")],
        equipment=[Armor("鏈甲", 16, 0)],
        gear=["繩索"],
        skill_proficiencies=["運動", "察覺"],
        saving_throw_proficiencies=["STR", "CON"],
        known_abilities=["second_wind"],
        reactions=["shield_spell"],
        ability_uses={"second_wind": 1},
        spell_slots={1: 2},
        spellcasting_ability="WIS",
    )


def test_character_panel_data_contains_requested_details():
    character = _character("hero", "艾拉")
    character.status_effects.append(Dodging())

    data = character_panel_data(character)

    assert data["name"] == "艾拉"
    assert (data["level"], data["hp"], data["max_hp"], data["ac"]) == (3, 17, 24, 16)
    assert data["status"] == "閃避"
    assert data["stats"][0] == {
        "id": "STR", "label": "力量", "value": 14, "modifier": 2,
    }
    assert data["skills"] == ["運動", "察覺"]
    assert data["abilities"] == ["二度氣息（剩 1 次）", "法盾"]
    assert data["spell_slots"] == ["施法屬性：感知 WIS", "1 環 × 2"]
    assert data["weapons"] == ["長劍"]
    assert data["consumables"] == ["治療藥水 × 2"]
    assert data["equipment"] == ["鏈甲"]
    assert data["gear"] == ["繩索"]


def test_party_panel_data_follows_live_party_order_and_values():
    first = _character("first", "甲")
    second = _character("second", "乙")
    outsider = _character("outsider", "丙")
    state = WorldState(
        characters={"first": first, "second": second, "outsider": outsider},
        scene="測試",
        party_ids=["second", "missing", "first"],
    )

    data = party_panel_data(state)
    assert [member["id"] for member in data] == ["second", "first"]

    second.hp = 4
    second.consumables[0].quantity = 1
    changed = party_panel_data(state)
    assert changed[0]["hp"] == 4
    assert changed[0]["consumables"] == ["治療藥水 × 1"]

    state.party_ids.append("outsider")
    joined = party_panel_data(state)
    assert [member["id"] for member in joined] == ["second", "first", "outsider"]
