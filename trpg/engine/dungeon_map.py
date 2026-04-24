from dataclasses import dataclass, field


@dataclass
class Room:
    id: str
    name: str
    description: str
    exits: dict = field(default_factory=dict)       # {"north": "room_id", ...}
    enemy_ids: list = field(default_factory=list)   # character IDs in this room
    loot: list = field(default_factory=list)        # Weapon / Consumable / str (quest item)
    visited: bool = False
    cleared: bool = False                           # True when no living enemies remain

    def alive_enemies(self, characters: dict) -> dict:
        """Return {id: char} for living enemies in this room."""
        return {
            eid: characters[eid]
            for eid in self.enemy_ids
            if eid in characters and characters[eid].is_alive()
        }

    def loot_names(self) -> list[str]:
        return [item.name if hasattr(item, "name") else str(item) for item in self.loot]


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
