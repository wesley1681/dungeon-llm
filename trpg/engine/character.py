from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stats:
    STR: int = 10
    DEX: int = 10
    CON: int = 10
    INT: int = 10
    WIS: int = 10
    CHA: int = 10

    def modifier(self, stat: str) -> int:
        return (getattr(self, stat) - 10) // 2


@dataclass
class Character:
    name: str
    race: str
    class_: str
    level: int
    stats: Stats
    hp: int
    max_hp: int
    ac: int
    spell_slots: dict = field(default_factory=dict)
    inventory: list = field(default_factory=list)
    status_effects: list = field(default_factory=list)
    proficiencies: list = field(default_factory=list)
    is_npc: bool = False

    @property
    def proficiency_bonus(self) -> int:
        return (self.level - 1) // 4 + 2

    def is_alive(self) -> bool:
        return self.hp > 0


@dataclass
class CombatState:
    initiative_order: list
    current_turn_index: int = 0
    round_number: int = 1
    active: bool = True
