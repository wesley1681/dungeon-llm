from unittest.mock import patch

from trpg.engine.character import Character, Stats
from trpg.engine.dungeon_map import DungeonMap, Room
from trpg.engine.items import Chest, Consumable, Hideout, Weapon
from trpg.engine.world_state import WorldState
from trpg.llm.action_executor import (
    build_action_candidates,
    execute_action_command,
    execute_action_commands,
)


def _character(name, *, npc=False, hp=10):
    return Character(
        name=name,
        race="人類",
        class_="冒險者",
        level=1,
        stats=Stats(STR=14, DEX=14, INT=12),
        hp=hp,
        max_hp=10,
        ac=10,
        is_npc=npc,
    )


def _world():
    hero = _character("英雄", hp=5)
    ally = _character("盟友")
    guard = _character("守衛", npc=True)
    scout = _character("斥候", npc=True)
    hero.weapons.append(Weapon("短劍", "1d6", "穿刺", "近戰"))
    hero.consumables.append(Consumable("治療藥水", 3, "heal", "1d4"))
    hero.gear.append("繩索")
    chest = Chest("鐵箱", ["寶石"], lock_dc=12)
    room = Room(
        id="hall",
        name="大廳",
        description="",
        exits={"east": "yard"},
        npc_ids=["guard", "scout"],
        loot=["火把", chest],
        hideouts=[Hideout("暗格", ["戒指"], find_dc=10)],
    )
    yard = Room(id="yard", name="庭院", description="", exits={"west": "hall"})
    return WorldState(
        characters={"hero": hero, "ally": ally, "guard": guard, "scout": scout},
        scene="",
        dungeon_map=DungeonMap(
            rooms={"hall": room, "yard": yard}, current_room_id="hall",
        ),
        pc_ids=["hero", "ally"],
        party_ids=["hero", "ally"],
    )


def test_talk_and_attack_execute_with_canonical_target_ids():
    ws = _world()

    talk = execute_action_command(
        {"action": "talk", "actor_id": "hero", "target_id": "guard"}, ws,
    )
    assert "開始" in talk
    assert ws.pending_conversation == "guard"

    with patch("trpg.engine.combat.roll", return_value=10):
        attack = execute_action_command(
            {"action": "attack", "actor_id": "hero", "target_id": "guard"}, ws,
        )
    assert "發起攻擊" in attack
    assert ws.pending_conversation == ""
    assert ws.characters["guard"].attitude == 0
    assert ws.combat.active
    assert {"hero", "ally", "guard"}.issubset(ws.combat.initiative_order)


def test_npc_attack_and_flee_use_structured_commands():
    ws = _world()

    with patch("trpg.engine.combat.roll", return_value=10):
        attack = execute_action_command(
            {"action": "attack", "actor_id": "guard", "target_id": "hero"}, ws,
        )
    assert "發起攻擊" in attack
    assert ws.characters["guard"].attitude == 0
    assert {"hero", "ally", "guard"}.issubset(ws.combat.initiative_order)

    ws.party_ids.append("guard")
    ws.combat_party_ids.append("guard")
    ws.follow_targets["guard"] = "hero"
    flee = execute_action_command(
        {"action": "flee", "actor_id": "guard"}, ws,
    )
    assert "逃跑" in flee
    assert "guard" not in ws.dungeon_map.current_room.npc_ids
    assert "guard" not in ws.combat.initiative_order
    assert "guard" not in ws.party_ids
    assert "guard" not in ws.combat_party_ids
    assert "guard" not in ws.follow_targets


def test_give_transfers_complete_consumable_stack_and_gear():
    ws = _world()

    potion_result = execute_action_command(
        {
            "action": "give",
            "actor_id": "hero",
            "target_id": "guard",
            "item": {"kind": "consumable", "name": "治療藥水"},
        },
        ws,
    )
    assert "全部" in potion_result
    assert ws.characters["hero"].get_consumable("治療藥水") is None
    assert ws.characters["guard"].get_consumable("治療藥水").quantity == 3

    execute_action_command(
        {
            "action": "give",
            "actor_id": "hero",
            "target_id": "guard",
            "item": {"kind": "gear", "name": "繩索"},
        },
        ws,
    )
    assert "繩索" not in ws.characters["hero"].gear
    assert "繩索" in ws.characters["guard"].gear


def test_follow_and_join_party_have_different_combat_membership():
    ws = _world()

    execute_action_command(
        {"action": "follow", "actor_id": "scout", "target_id": "hero"}, ws,
    )
    execute_action_command(
        {"action": "join_party", "actor_id": "guard", "target_id": "ally"}, ws,
    )

    assert "scout" in ws.party_ids
    assert "scout" not in ws.combat_party_ids
    assert "guard" in ws.party_ids
    assert "guard" in ws.combat_party_ids
    assert ws.follow_targets == {"scout": "hero", "guard": "ally"}


def test_travel_moves_both_follow_types_but_only_joined_npc_fights():
    ws = _world()
    execute_action_command(
        {"action": "follow", "actor_id": "scout", "target_id": "hero"}, ws,
    )
    execute_action_command(
        {"action": "join_party", "actor_id": "guard", "target_id": "hero"}, ws,
    )
    enemy = _character("強盜", npc=True)
    enemy.attitude = 0
    ws.characters["bandit"] = enemy
    ws.dungeon_map.rooms["yard"].npc_ids.append("bandit")

    with patch("trpg.engine.combat.roll", return_value=10):
        result = execute_action_command(
            {"action": "travel", "actor_id": "hero", "direction": "east"}, ws,
        )

    assert "庭院" in result
    assert "scout" in ws.dungeon_map.current_room.npc_ids
    assert "guard" in ws.dungeon_map.current_room.npc_ids
    assert "guard" in ws.combat.initiative_order
    assert "scout" not in ws.combat.initiative_order


def test_pickup_and_use_apply_item_rules():
    ws = _world()
    candidates = build_action_candidates(ws, "hero")
    torch = next(item for item in candidates["pickup_items"] if item["name"] == "火把")

    execute_action_command(
        {"action": "pickup", "actor_id": "hero", "item": torch}, ws,
    )
    assert "火把" in ws.characters["hero"].gear

    with patch("trpg.engine.combat.roll", return_value=3):
        result = execute_action_command(
            {
                "action": "use",
                "actor_id": "hero",
                "target_id": "hero",
                "item": {"kind": "consumable", "name": "治療藥水"},
            },
            ws,
        )
    assert "恢復" in result
    assert ws.characters["hero"].hp == 8
    assert ws.characters["hero"].get_consumable("治療藥水").quantity == 2


def test_unlock_allows_only_strength_or_dexterity_with_same_dc():
    for method, stat in (("strength", "STR"), ("dexterity", "DEX")):
        ws = _world()
        target = build_action_candidates(ws, "hero")["locked_objects"][0]
        with patch(
            "trpg.engine.combat.make_saving_throw", return_value=(True, 15),
        ) as saving_throw:
            result = execute_action_command(
                {
                    "action": "unlock",
                    "actor_id": "hero",
                    "target": target,
                    "method": method,
                },
                ws,
            )
        saving_throw.assert_called_once_with(ws.characters["hero"], stat, 12)
        assert "成功" in result

    ws = _world()
    result = execute_action_command(
        {
            "action": "unlock",
            "actor_id": "hero",
            "target": {"name": "鐵箱"},
            "method": "intelligence",
        },
        ws,
    )
    assert result.startswith("無效")


def test_search_does_not_require_a_target_and_reveals_hideouts():
    ws = _world()
    with patch("trpg.engine.combat.make_saving_throw", return_value=(True, 12)):
        result = execute_action_command(
            {"action": "search", "actor_id": "hero"}, ws,
        )

    assert "暗格" in result
    assert ws.dungeon_map.current_room.hideouts[0].discovered


def test_batch_executor_logs_success_and_separates_validation_errors():
    ws = _world()
    ok, errors = execute_action_commands(
        [
            {"action": "talk", "actor_id": "hero", "target_id": "guard"},
            {"action": "unlock", "actor_id": "hero", "target": {}, "method": "wisdom"},
        ],
        ws,
        log_to_narrative=True,
    )

    assert len(ok) == 1
    assert len(errors) == 1
    assert ws.narrative_log[-1]["text"] == ok[0]
