from dataclasses import dataclass, field
from typing import Optional
from .character import Character, CombatState


@dataclass
class WorldState:
    characters: dict          # str → Character (both PCs and NPCs)
    scene: str
    combat: Optional[CombatState] = None
    dungeon_map: Optional[object] = None   # DungeonMap — Optional import to avoid circular
    event_log: list = field(default_factory=list)
    scenario_name: str = ""
