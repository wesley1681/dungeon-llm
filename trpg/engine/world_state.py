from dataclasses import dataclass, field
from typing import Optional
from .character import Character, CombatState


@dataclass
class WorldState:
    characters: dict          # str → Character (both PCs and NPCs)
    scene: str
    combat: Optional[CombatState] = None
    event_log: list = field(default_factory=list)
    scenario_name: str = ""
