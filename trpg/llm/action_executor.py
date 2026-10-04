from __future__ import annotations

from typing import Any

from ..engine import combat
from ..engine.items import Chest, Consumable, Weapon
from ..engine.world_state import WorldState


_ERROR_PREFIXES = ("無效", "找不到", "無法", "沒有", "戰鬥中")


def _character_id(ws: WorldState, character: object) -> str | None:
    return next((cid for cid, value in ws.characters.items() if value is character), None)


def _present_character_ids(ws: WorldState) -> list[str]:
    """Canonical ids for living characters in the current scene."""
    result: list[str] = []
    if ws.dungeon_map:
        result.extend(ws.dungeon_map.current_room.npc_ids)
    result.extend(
        cid for cid, char in ws.characters.items()
        if not char.is_npc
    )
    return list(dict.fromkeys(
        cid for cid in result
        if cid in ws.characters and ws.characters[cid].is_alive()
    ))


def _same_room(ws: WorldState, first_id: str, second_id: str) -> bool:
    present = set(_present_character_ids(ws))
    return first_id in present and second_id in present


def _combat_allies_in_room(ws: WorldState) -> list[str]:
    present = set(_present_character_ids(ws))
    ids = list(ws.pc_ids) + list(ws.combat_party_ids)
    return list(dict.fromkeys(
        cid for cid in ids
        if cid in present
        and cid in ws.characters
        and ws.characters[cid].is_alive()
    ))


def _item_name(item: object) -> str:
    return item.name if hasattr(item, "name") else str(item)


def _matching_index(items: list, item_ref: dict[str, Any]) -> int | None:
    """Prefer the selected index, then fall back to its stable display name."""
    name = item_ref.get("name", "")
    index = item_ref.get("index")
    if isinstance(index, int) and 0 <= index < len(items):
        if _item_name(items[index]) == name:
            return index
    return next(
        (position for position, item in enumerate(items) if _item_name(item) == name),
        None,
    )


def build_action_candidates(ws: WorldState, actor_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return serializable candidates that Jev may select for one actor."""
    actor = ws.characters.get(actor_id)
    if actor is None:
        return {
            "room_characters": [],
            "room_npcs": [],
            "room_player_characters": [],
            "inventory": [],
            "pickup_items": [],
            "locked_objects": [],
        }

    present = [cid for cid in _present_character_ids(ws) if cid != actor_id]
    room_characters = [
        {"id": cid, "name": ws.characters[cid].name}
        for cid in present
    ]
    room_npcs = [
        item for item in room_characters
        if ws.characters[item["id"]].is_npc
    ]
    room_players = [
        item for item in room_characters
        if item["id"] in ws.pc_ids and not ws.characters[item["id"]].is_npc
    ]

    inventory: list[dict[str, Any]] = []
    for index, item in enumerate(actor.weapons):
        inventory.append({
            "kind": "weapon", "index": index, "name": item.name,
        })
    for index, item in enumerate(actor.consumables):
        inventory.append({
            "kind": "consumable", "index": index, "name": item.name,
            "quantity": item.quantity,
        })
    for index, item in enumerate(actor.gear):
        inventory.append({
            "kind": "gear", "index": index, "name": str(item),
        })

    pickup_items: list[dict[str, Any]] = []
    locked_objects: list[dict[str, Any]] = []
    if ws.dungeon_map:
        room = ws.dungeon_map.current_room
        for index, item in enumerate(room.loot):
            if isinstance(item, Chest):
                if item.lock_dc > 0:
                    locked_objects.append({
                        "source": "room_chest", "index": index, "name": item.name,
                    })
                if item.opened:
                    for inner_index, inner in enumerate(item.loot):
                        pickup_items.append({
                            "source": "chest", "container_index": index,
                            "index": inner_index, "name": _item_name(inner),
                        })
                continue
            pickup_items.append({
                "source": "room", "index": index, "name": _item_name(item),
            })
        for hideout_index, hideout in enumerate(room.hideouts):
            if not hideout.discovered:
                continue
            for inner_index, inner in enumerate(hideout.contents):
                pickup_items.append({
                    "source": "hideout", "container_index": hideout_index,
                    "index": inner_index, "name": _item_name(inner),
                })

    return {
        "room_characters": room_characters,
        "room_npcs": room_npcs,
        "room_player_characters": room_players,
        "inventory": inventory,
        "pickup_items": pickup_items,
        "locked_objects": locked_objects,
    }


def _transfer_item(giver, recipient, item_ref: dict[str, Any]) -> str:
    kind = item_ref.get("kind")
    name = item_ref.get("name", "")

    if kind == "weapon":
        index = _matching_index(giver.weapons, item_ref)
        if index is not None:
            item = giver.weapons.pop(index)
            recipient.weapons.append(item)
            return f"{giver.name} 把 {item.name} 交給 {recipient.name}"
    elif kind == "consumable":
        index = _matching_index(giver.consumables, item_ref)
        if index is not None:
            item = giver.consumables[index]
            if item.quantity > 0:
                giver.consumables.pop(index)
                existing = recipient.get_consumable(item.name)
                if existing:
                    existing.quantity += item.quantity
                else:
                    recipient.consumables.append(item)
                return f"{giver.name} 把 {item.name}×{item.quantity} 全部交給 {recipient.name}"
    elif kind == "gear":
        index = _matching_index(giver.gear, item_ref)
        if index is not None:
            item = giver.gear.pop(index)
            recipient.gear.append(item)
            return f"{giver.name} 把 {item} 交給 {recipient.name}"
    return f"找不到可給予物品：{giver.name} 身上沒有「{name}」"


def _take_pickup_item(ws: WorldState, item_ref: dict[str, Any]):
    if not ws.dungeon_map:
        return None
    room = ws.dungeon_map.current_room
    source = item_ref.get("source")
    name = item_ref.get("name", "")

    if source == "room":
        index = _matching_index(room.loot, item_ref)
        if index is not None and not isinstance(room.loot[index], Chest):
            return room.loot.pop(index)
    elif source == "chest":
        container_index = item_ref.get("container_index")
        containers = list(enumerate(room.loot))
        if isinstance(container_index, int) and 0 <= container_index < len(room.loot):
            containers.sort(key=lambda pair: pair[0] != container_index)
        for _, chest in containers:
            if not isinstance(chest, Chest) or not chest.opened:
                continue
            index = _matching_index(chest.loot, item_ref)
            if index is not None:
                return chest.loot.pop(index)
    elif source == "hideout":
        container_index = item_ref.get("container_index")
        containers = list(enumerate(room.hideouts))
        if isinstance(container_index, int) and 0 <= container_index < len(room.hideouts):
            containers.sort(key=lambda pair: pair[0] != container_index)
        for _, hideout in containers:
            if not hideout.discovered:
                continue
            index = _matching_index(hideout.contents, item_ref)
            if index is not None:
                return hideout.contents.pop(index)
    return None


def _give_picked_item(actor, item: object) -> str:
    if isinstance(item, Weapon):
        actor.weapons.append(item)
        return f"{actor.name} 拾取了 {item.name}"
    if isinstance(item, Consumable):
        existing = actor.get_consumable(item.name)
        if existing:
            existing.quantity += item.quantity
        else:
            actor.consumables.append(item)
        return f"{actor.name} 拾取了 {item.name}×{item.quantity}"
    actor.gear.append(str(item))
    return f"{actor.name} 拾取了 {item}"


def _move_party(ws: WorldState, direction: str) -> str:
    if ws.combat and ws.combat.active:
        return "戰鬥中無法移動"
    if not ws.dungeon_map:
        return "沒有地圖"
    if direction not in {"north", "south", "east", "west"}:
        return f"無效移動方向：{direction}"

    old_room = ws.dungeon_map.current_room
    room = ws.dungeon_map.move(direction)
    if room is None:
        valid = "、".join(old_room.exits) or "無"
        return f"無法往{direction}走（有效出口：{valid}）"

    for npc_id in list(ws.party_ids):
        char = ws.characters.get(npc_id)
        if not char or not char.is_npc or not char.is_alive():
            continue
        if npc_id in old_room.npc_ids:
            old_room.npc_ids.remove(npc_id)
        if npc_id not in room.npc_ids:
            room.npc_ids.append(npc_id)

    if ws.pending_conversation and ws.pending_conversation not in room.npc_ids:
        ws.pending_conversation = ""

    attackers = [
        npc_id for npc_id in room.npc_ids
        if npc_id in ws.characters
        and ws.characters[npc_id].is_alive()
        and ws.characters[npc_id].attitude == 0
        and npc_id not in ws.party_ids
    ]
    if attackers and not room.cleared:
        ws.combat = combat.roll_initiative(
            ws, attackers + _combat_allies_in_room(ws)
        )
        names = "、".join(ws.characters[cid].name for cid in attackers)
        return f"移動至：{room.name}。發現敵人：{names}！"
    return f"移動至：{room.name}"


def execute_action_command(command: dict[str, Any], ws: WorldState) -> str:
    """Validate and execute one structured action command."""
    action = command.get("action", "")
    actor_id = command.get("actor_id", "")
    actor = ws.characters.get(actor_id)
    if actor is None:
        return f"找不到行動者：{actor_id}"
    if not actor.is_alive():
        return f"無法行動：{actor.name} 已經倒下"

    if action == "talk":
        target_id = command.get("target_id", "")
        target = ws.characters.get(target_id)
        if target is None:
            return f"找不到角色：{target_id}"
        if not target.is_npc or not _same_room(ws, actor_id, target_id):
            return f"無法與 {target.name} 交談：目標不在當前房間"
        ws.pending_conversation = target_id
        return f"開始與 {target.name} 對話"

    if action == "give":
        target_id = command.get("target_id", "")
        target = ws.characters.get(target_id)
        if target is None:
            return f"找不到角色：{target_id}"
        if not _same_room(ws, actor_id, target_id):
            return f"無法交付：{actor.name} 和 {target.name} 不在同一房間"
        return _transfer_item(actor, target, command.get("item", {}))

    if action in {"follow", "join_party"}:
        target_id = command.get("target_id", "")
        target = ws.characters.get(target_id)
        if target is None or target_id not in ws.pc_ids or target.is_npc:
            return f"找不到可跟隨的玩家角色：{target_id}"
        if not _same_room(ws, actor_id, target_id):
            return f"無法跟隨：{actor.name} 和 {target.name} 不在同一房間"
        if actor_id not in ws.party_ids:
            ws.party_ids.append(actor_id)
        ws.follow_targets[actor_id] = target_id
        if action == "join_party":
            if actor_id not in ws.combat_party_ids and actor_id not in ws.pc_ids:
                ws.combat_party_ids.append(actor_id)
            return f"{actor.name} 加入 {target.name} 的隊伍，將跟隨移動並參與戰鬥"
        if actor_id in ws.combat_party_ids:
            ws.combat_party_ids.remove(actor_id)
        return f"{actor.name} 開始跟隨 {target.name}，但不參與戰鬥"

    if action == "travel":
        return _move_party(ws, command.get("direction", ""))

    if action == "pickup":
        if not ws.dungeon_map:
            return "沒有地圖"
        item_ref = command.get("item", {})
        item = _take_pickup_item(ws, item_ref)
        if item is None:
            return f"找不到可拾取物品：{item_ref.get('name', '')}"
        return _give_picked_item(actor, item)

    if action == "use":
        item_ref = command.get("item", {})
        if item_ref.get("kind") != "consumable":
            return f"無法使用物品：{item_ref.get('name', '')} 沒有已定義的使用效果"
        item = actor.get_consumable(item_ref.get("name", ""))
        if item is None or item.quantity <= 0:
            return f"找不到可使用物品：{item_ref.get('name', '')}"
        target_id = command.get("target_id") or actor_id
        target = ws.characters.get(target_id)
        if target is None or not _same_room(ws, actor_id, target_id):
            return f"找不到物品作用目標：{target_id}"
        actor.consume(item.name)
        result = f"{actor.name} 使用了 {item.name}"
        if item.effect_type == "heal" and item.effect_value:
            healed = combat.apply_heal(target, item.effect_value)
            result += f"，{target.name} 恢復 {healed} HP"
        return result

    if action == "unlock":
        if not ws.dungeon_map:
            return "沒有地圖"
        method = command.get("method", "")
        stat = {"strength": "STR", "dexterity": "DEX"}.get(method)
        if stat is None:
            return f"無效開鎖屬性：{method}"
        target_ref = command.get("target", {})
        target_name = target_ref.get("name", "")
        room_loot = ws.dungeon_map.current_room.loot
        target_index = target_ref.get("index")
        candidates = list(enumerate(room_loot))
        if isinstance(target_index, int) and 0 <= target_index < len(room_loot):
            candidates.sort(key=lambda pair: pair[0] != target_index)
        for _, item in candidates:
            if not isinstance(item, Chest) or item.name != target_name or item.lock_dc <= 0:
                continue
            dc = item.lock_dc
            success, total = combat.make_saving_throw(actor, stat, dc)
            if not success:
                return f"{actor.name} 開鎖失敗（{stat} {total} vs DC{dc}）——{item.name} 依然緊閉"
            item.lock_dc = 0
            item.opened = True
            return f"{actor.name} 成功解鎖 {item.name}（{stat} {total} vs DC{dc}）"
        return f"找不到可解鎖的物件：{target_name}"

    if action == "search":
        if not ws.dungeon_map:
            return "沒有地圖"
        room = ws.dungeon_map.current_room
        undiscovered = [item for item in room.hideouts if not item.discovered]
        if not undiscovered:
            return f"{actor.name} 仔細搜索，沒有發現任何特別之物"
        _, total = combat.make_saving_throw(
            actor, "INT", min(item.find_dc for item in undiscovered)
        )
        revealed = []
        for hideout in undiscovered:
            if total >= hideout.find_dc:
                hideout.discovered = True
                revealed.append(hideout.description)
        if revealed:
            return f"{actor.name} 搜索（INT {total}）→ 發現：{'；'.join(revealed)}"
        return f"{actor.name} 搜索（INT {total}）→ 沒看出名堂"

    if action == "attack":
        target_id = command.get("target_id", "")
        target = ws.characters.get(target_id)
        if target is None:
            return f"找不到角色：{target_id}"
        if not target.is_npc or not _same_room(ws, actor_id, target_id):
            return f"無法攻擊 {target.name}：目標不在當前房間"
        target.attitude = 0
        if target_id in ws.party_ids:
            ws.party_ids.remove(target_id)
        if target_id in ws.combat_party_ids:
            ws.combat_party_ids.remove(target_id)
        ws.follow_targets.pop(target_id, None)
        if ws.dungeon_map:
            ws.dungeon_map.current_room.cleared = False
        ws.pending_conversation = ""
        ws.combat = combat.roll_initiative(
            ws, [target_id] + _combat_allies_in_room(ws)
        )
        order = "、".join(
            ws.characters[cid].name for cid in ws.combat.initiative_order
            if cid in ws.characters
        )
        return f"{actor.name} 向 {target.name} 發起攻擊！先攻順序：{order}"

    return f"無效行動：{action}"


def execute_action_commands(
    commands: list[dict[str, Any]],
    ws: WorldState,
    *,
    log_to_narrative: bool = False,
) -> tuple[list[str], list[str]]:
    """Execute commands in order and separate successful results from errors."""
    ok: list[str] = []
    errors: list[str] = []
    seen_once: set[str] = set()
    for command in commands:
        action = command.get("action", "")
        if action in {"travel", "talk", "attack"} and action in seen_once:
            continue
        try:
            result = execute_action_command(command, ws)
        except Exception as exc:  # Keep a malformed command from aborting the turn.
            result = f"無效行動 {action}：{exc}"
        if result.startswith(_ERROR_PREFIXES):
            errors.append(result)
        else:
            ok.append(result)
            if log_to_narrative:
                ws.log_event("system", result)
            if action in {"travel", "talk", "attack"}:
                seen_once.add(action)
    return ok, errors
