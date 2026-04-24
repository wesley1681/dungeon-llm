from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .items import Weapon, Consumable


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
    weapons: list = field(default_factory=list)       # list[Weapon]
    consumables: list = field(default_factory=list)   # list[Consumable]
    gear: list = field(default_factory=list)          # list[str] — 非戰鬥道具
    status_effects: list = field(default_factory=list)
    proficiencies: list = field(default_factory=list)
    is_npc: bool = False

    @property
    def proficiency_bonus(self) -> int:
        return (self.level - 1) // 4 + 2

    def is_alive(self) -> bool:
        return self.hp > 0

    # ── Inventory helpers ──────────────────────────────────────────────────────

    @property
    def inventory(self) -> list[str]:
        """Display-only flat list of all carried items."""
        items = [w.name for w in self.weapons]
        for c in self.consumables:
            items.append(f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name)
        items.extend(self.gear)
        return items

    def get_weapon(self, name: str = "") -> Weapon:
        """Return weapon by name, or first weapon, or bare-hand fallback."""
        from .items import WEAPON_DEFS
        if name:
            for w in self.weapons:
                if w.name == name:
                    return w
            if name in WEAPON_DEFS:
                return WEAPON_DEFS[name]
        return self.weapons[0] if self.weapons else WEAPON_DEFS["無武器"]

    def has_ammo(self, ammo: str) -> bool:
        return any(c.name == ammo and c.quantity > 0 for c in self.consumables)

    def consume(self, name: str) -> bool:
        """Use one unit of a consumable. Returns True if successful."""
        for c in self.consumables:
            if c.name == name and c.quantity > 0:
                c.quantity -= 1
                return True
        return False

    def get_consumable(self, name: str):
        """Return the Consumable object with the given name, or None."""
        for c in self.consumables:
            if c.name == name:
                return c
        return None


@dataclass
class CombatState:
    initiative_order: list
    current_turn_index: int = 0
    round_number: int = 1
    active: bool = True
