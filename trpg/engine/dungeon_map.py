from dataclasses import dataclass, field


@dataclass
class Room:
    id: str
    name: str
    description: str
    exits: dict = field(default_factory=dict)       # {"north": "room_id", ...}
    npc_ids: list = field(default_factory=list)     # all NPC IDs in this room (hostile or not)
    loot: list = field(default_factory=list)        # Weapon / Consumable / Chest / str — always-visible
    hideouts: list = field(default_factory=list)    # list[Hideout] — concealed placements; revealed by SEARCH
    visited: bool = False
    cleared: bool = False                           # True when no living hostile NPCs remain

    def alive_hostile_npcs(self, characters: dict) -> dict:
        """Return {id: char} for living hostile (attitude==0) NPCs in this room."""
        return {
            nid: characters[nid]
            for nid in self.npc_ids
            if nid in characters
               and characters[nid].is_alive()
               and characters[nid].attitude == 0
        }

    # Backwards-compatible alias: previously "enemies" meant attitude==0 hostile NPCs
    def alive_enemies(self, characters: dict) -> dict:
        return self.alive_hostile_npcs(characters)

    def has_hidden(self) -> bool:
        """True if any hideout is still undiscovered."""
        return any(not h.discovered for h in self.hideouts)

    def loot_names(self) -> list[str]:
        """Flat list of every currently-pickupable item name.
        Includes: room.loot top-level (except chests themselves),
                  opened chest contents,
                  discovered hideout contents."""
        from .items import Chest
        names = []
        for item in self.loot:
            if isinstance(item, Chest):
                if item.opened:
                    for c in item.loot:
                        names.append(c.name if hasattr(c, "name") else str(c))
                else:
                    names.append(item.visible_name())   # locked chest counts as a "thing"
            else:
                names.append(item.name if hasattr(item, "name") else str(item))
        for h in self.hideouts:
            if h.discovered:
                for c in h.contents:
                    names.append(c.name if hasattr(c, "name") else str(c))
        return names

    def loot_state(self) -> str:
        """Structured multi-line display: obvious items + opened-chest contents +
        discovered-hideout contents. Undiscovered hideouts are NOT shown."""
        from .items import Chest

        def _label(it):
            if hasattr(it, "quantity") and it.quantity > 1:
                return f"{it.name}×{it.quantity}"
            return it.name if hasattr(it, "name") else str(it)

        lines = []
        for item in self.loot:
            if isinstance(item, Chest):
                if item.opened:
                    inner = "、".join(_label(c) for c in item.loot) or "（空）"
                    lines.append(f"- {item.name}（已開啟）內：{inner}")
                else:
                    lines.append(f"- {item.visible_name()}")
            else:
                lines.append(f"- {_label(item)}")
        for h in self.hideouts:
            if h.discovered and h.contents:
                inner = "、".join(_label(c) for c in h.contents)
                lines.append(f"- {h.description}：{inner}")

        if not lines:
            return "無"
        return "\n".join(lines)


@dataclass
class DungeonMap:
    rooms: dict = field(default_factory=dict)   # {room_id: Room}
    current_room_id: str = ""

    @property
    def current_room(self) -> Room:
        return self.rooms[self.current_room_id]

    def move(self, direction: str) -> "Room | None":
        """Move in direction. Returns new Room or None if exit doesn't exist."""
        next_id = self.current_room.exits.get(direction.lower())
        if next_id and next_id in self.rooms:
            self.current_room_id = next_id
            self.current_room.visited = True
            return self.current_room
        return None
