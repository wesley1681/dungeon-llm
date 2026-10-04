from trpg.engine.character import Character, Stats
from trpg.engine.dungeon_map import DungeonMap, Room
from trpg.scenarios.dungeon import (
    _configure_character_roster,
    build_npc_agents,
    build_player_agents,
    build_world_state,
)


def _character(char_id: str, *, controller: str, npc: bool = False, **kwargs):
    return Character(
        id=char_id,
        name=char_id,
        race="測試",
        class_="測試",
        level=1,
        stats=Stats(),
        hp=10,
        max_hp=10,
        ac=10,
        controller_type=controller,
        is_npc=npc,
        **kwargs,
    )


def test_world_indexes_are_derived_from_character_configuration():
    world = build_world_state()

    assert all(char_id == character.id for char_id, character in world.characters.items())
    assert all(character.enabled for character in world.characters.values())
    assert world.pc_ids == ["thor", "kaine"]
    assert world.party_ids == ["kaine", "thor"]
    assert world.dungeon_map.rooms["entrance"].npc_ids == ["civilian"]
    assert world.dungeon_map.rooms["guard_room"].npc_ids == [
        "goblin_1", "goblin_2", "goblin_shaman",
    ]
    assert world.dungeon_map.rooms["boss_chamber"].npc_ids == [
        "goblin_3", "goblin_boss",
    ]


def test_disabled_character_is_removed_from_every_derived_index():
    dungeon_map = DungeonMap(
        rooms={"room": Room("room", "房間", "")},
        current_room_id="room",
    )
    human = _character(
        "human", controller="human", starts_in_party=True, turn_order=20,
    )
    disabled_npc = _character(
        "disabled", controller="npc", npc=True,
        starting_room_id="room", enabled=False,
    )

    characters, pc_ids, party_ids = _configure_character_roster(
        [human, disabled_npc], dungeon_map,
    )

    assert set(characters) == {"human"}
    assert pc_ids == ["human"]
    assert party_ids == ["human"]
    assert dungeon_map.current_room.npc_ids == []


def test_agent_factories_follow_character_controller_type():
    world = build_world_state()

    npc_agents = build_npc_agents(
        world, model="stub", base_url="stub", backend="stub",
    )
    player_agents = build_player_agents(
        world, model="stub", base_url="stub", backend="stub",
    )

    assert set(npc_agents) == {
        "civilian", "goblin_1", "goblin_2", "goblin_3",
        "goblin_boss", "goblin_shaman",
    }
    assert set(player_agents) == {"thor"}
    assert player_agents["thor"].character is world.characters["thor"]
