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
    spells: list = field(default_factory=list)        # list[str] — spell names; look up via engine.spells.SPELLS
    spellcasting_ability: str = ""                    # "INT"/"WIS"/"CHA"; "" = non-caster
    weapons: list = field(default_factory=list)       # list[Weapon]
    consumables: list = field(default_factory=list)   # list[Consumable]
    gear: list = field(default_factory=list)          # list[str] — 非戰鬥道具
    equipment: list = field(default_factory=list)     # 穿戴中的裝備（Modifier-capable）
    status_effects: list = field(default_factory=list)
    proficiencies: list = field(default_factory=list)
    is_npc: bool = False
    attitude: int = 2   # 0=敵意 1=戒備 2=中立 3=友好 4=信任（僅 NPC 使用）
    hostile_reaction: str = "attack"   # 敵意時的反應："attack" 開戰 / "flee" 逃跑
    position: float = 0.0   # 戰鬥中的位置（公尺）：0 = 我方原點、正向 = 敵方那側。戰鬥開始時 reset。

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

    # ── Status effect helpers ──────────────────────────────────────────────────

    def has_status(self, name: str) -> bool:
        from .status import StatusEffect
        for fx in self.status_effects:
            if isinstance(fx, StatusEffect):
                if fx.name == name:
                    return True
            elif fx == name:          # legacy 字串相容（過渡期）
                return True
        return False

    def add_status(self, fx) -> None:
        """Add a StatusEffect (idempotent on name)."""
        if self.has_status(fx.name):
            return
        self.status_effects.append(fx)

    def remove_status(self, name: str) -> None:
        from .status import StatusEffect
        self.status_effects = [
            fx for fx in self.status_effects
            if not (
                (isinstance(fx, StatusEffect) and fx.name == name)
                or fx == name
            )
        ]

    def iter_modifiers(self):
        """Yield all Modifier-implementing objects attached to this character."""
        from .modifier import Modifier
        for w in self.weapons:
            if isinstance(w, Modifier):
                yield w
        for eq in self.equipment:
            if isinstance(eq, Modifier):
                yield eq
        for fx in self.status_effects:
            if isinstance(fx, Modifier):
                yield fx


@dataclass
class CombatState:
    initiative_order: list
    current_turn_index: int = 0
    round_number: int = 1
    active: bool = True
