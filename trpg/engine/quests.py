from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world_state import WorldState


@dataclass
class Quest:
    """A side quest. Status lifecycle:
       inactive → active (player accepted) → completed (objective met) → turned_in (rewards given)

    objective: dict describing completion condition. Supported types:
      {"type": "collect", "item": "月光草", "count": 3}
      {"type": "kill", "target": "goblin_boss"}      # char_id
      {"type": "reach", "room": "boss_chamber"}      # room_id

    reward: dict describing what to give on turn-in.
      {"items": [Consumable(...), ...],  # given to recipient_id
       "info":  ["你聽到頭目怕火..."],     # added to giver_id's secrets
       "attitude_delta": 1,               # change to giver_id's attitude
       "recipient_id": "aria"}            # who receives items (default "aria")
    """
    id: str
    title: str
    description: str            # one-liner shown in UI
    giver_id: str               # NPC that offers and receives turn-in
    objective: dict
    reward: dict = field(default_factory=dict)
    status: str = "inactive"    # inactive / active / completed / turned_in


def _count_item(ws: WorldState, item_name: str) -> int:
    total = 0
    for c in ws.characters.values():
        if c.is_npc:
            continue
        for cons in c.consumables:
            if cons.name == item_name:
                total += cons.quantity
    return total


def is_objective_met(quest: Quest, ws: WorldState) -> bool:
    obj = quest.objective
    t = obj.get("type")
    if t == "collect":
        return _count_item(ws, obj["item"]) >= obj["count"]
    if t == "kill":
        target = ws.characters.get(obj["target"])
        return target is not None and not target.is_alive()
    if t == "reach":
        return ws.dungeon_map is not None and ws.dungeon_map.current_room.id == obj["room"]
    return False


def check_quest_progress(ws: WorldState) -> list[Quest]:
    """Auto-advance active quests whose objectives are now met.

    Returns the list of quests that transitioned active → completed this call.
    """
    just_completed: list[Quest] = []
    for q in ws.quests.values():
        if q.status == "active" and is_objective_met(q, ws):
            q.status = "completed"
            just_completed.append(q)
    return just_completed


def objective_progress_str(quest: Quest, ws: WorldState) -> str:
    """Short status like '2/3' for collect quests, '' otherwise."""
    obj = quest.objective
    if obj.get("type") == "collect":
        return f"{min(_count_item(ws, obj['item']), obj['count'])}/{obj['count']}"
    return ""


def apply_reward(quest: Quest, ws: WorldState) -> list[str]:
    """Distribute quest reward. Returns list of human-readable lines for narration."""
    lines: list[str] = []
    reward = quest.reward or {}

    items = reward.get("items") or []
    recipient_id = reward.get("recipient_id", "aria")
    recipient = ws.characters.get(recipient_id)
    if items and recipient:
        from .items import Weapon, Consumable
        for item in items:
            if isinstance(item, Weapon):
                recipient.weapons.append(item)
                lines.append(f"{recipient.name} 獲得 {item.name}")
            elif isinstance(item, Consumable):
                existing = recipient.get_consumable(item.name)
                if existing:
                    existing.quantity += item.quantity
                else:
                    recipient.consumables.append(item)
                lines.append(f"{recipient.name} 獲得 {item.name}×{item.quantity}")
            else:
                recipient.gear.append(str(item))
                lines.append(f"{recipient.name} 獲得 {item}")

    info = reward.get("info") or []
    giver = ws.characters.get(quest.giver_id)
    if info and giver:
        lines.append(f"{giver.name} 告訴你新情報")

    delta = reward.get("attitude_delta", 0)
    if delta and giver and giver.is_npc:
        old = giver.attitude
        giver.attitude = max(0, min(4, old + delta))
        if giver.attitude != old:
            lines.append(f"{giver.name} 的態度變化（{old} → {giver.attitude}）")

    return lines
