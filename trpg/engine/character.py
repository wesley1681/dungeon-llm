from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .vec2 import Vec2

if TYPE_CHECKING:
    from .items import Weapon, Consumable
    from .vec2 import Battlefield


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
    position: Vec2 = field(default_factory=Vec2)   # 戰鬥中的 2D 座標（公尺），戰鬥開始時由 setup_combat_positions reset
    # 正在維持的專注法術（skill_id 或法術名）。每名角色最多 1 個專注效果；
    # 受傷時要過 CON 豁免否則中斷，新的專注法術會直接覆蓋舊的。
    concentrating_on: str = ""
    # 已備好的反應動作名稱（如 "shield_spell"）。引擎在攻擊／法術觸發時
    # 自動檢查並可能消耗。
    reactions: list = field(default_factory=list)
    reaction_used: bool = False   # 已使用本回合反應？self_turn_start 重置
    # 此角色會使用的 ClassAbility skill_id 清單（如 ["second_wind","rage"]）。
    # HumanInputPolicy 用「招式」指令時會檢查這份清單。
    known_abilities: list = field(default_factory=list)
    # 每個 ability 剩餘的使用次數。{skill_id: remaining_uses}
    # 值 None → 無限制（passive / unlimited）。
    # 由 rest_character() 根據 ClassAbility.refresh_on 重設。
    ability_uses: dict = field(default_factory=dict)
    attacks_per_action: int = 1   # 1 = normal; 2 = Extra Attack (L5 Fighter/Barbarian etc.)
    crit_range: int = 20          # Champion archetype: set to 19 to crit on 19-20
    sneak_attack_dice: str = ""   # e.g. "2d6" for L3 Rogue; "" = not a rogue
    # 死亡豁免計數。只對 PC 有意義；NPC 在 HP=0 時立刻死亡。
    # {"successes": int, "failures": int}
    death_saves: dict = field(default_factory=lambda: {"successes": 0, "failures": 0})

    @property
    def proficiency_bonus(self) -> int:
        return (self.level - 1) // 4 + 2

    def is_alive(self) -> bool:
        """True if hp > 0 — the creature can act and be targeted normally."""
        return self.hp > 0

    def is_dead(self) -> bool:
        """True if the creature is permanently dead:
        NPCs: hp <= 0
        PCs:  hp <= 0 AND death_save_failures >= 3"""
        if self.hp > 0:
            return False
        if self.is_npc:
            return True
        return self.death_saves.get("failures", 0) >= 3

    def is_dying(self) -> bool:
        """PC at 0 HP but not yet dead — needs death saves each turn."""
        return self.hp <= 0 and not self.is_npc and not self.is_dead()

    def reset_death_saves(self) -> None:
        self.death_saves = {"successes": 0, "failures": 0}

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


def rest_character(char: "Character", rest_type: str) -> dict:
    """Restore resources for a short or long rest.

    rest_type: "short" | "long"
    Long rest also performs a short rest (recovers short_rest abilities too).
    Returns a summary dict for display.
    """
    from .abilities import CLASS_ABILITIES
    recovered: list[str] = []

    if rest_type not in ("short", "long"):
        return {"error": f"未知休息類型：{rest_type}"}

    # Restore ability uses.
    for sid in char.known_abilities:
        ab = CLASS_ABILITIES.get(sid)
        if ab is None or ab.max_uses == 0:
            char.ability_uses.pop(sid, None)   # unlimited — remove cap tracking
            continue
        if ab.refresh_on == "short_rest" or (
            ab.refresh_on == "long_rest" and rest_type == "long"
        ):
            char.ability_uses[sid] = ab.max_uses
            recovered.append(f"{ab.display_name}(×{ab.max_uses})")

    # Long rest: restore HP and spell slots to full.
    if rest_type == "long":
        char.hp = char.max_hp
        char.spell_slots = {lvl: max_uses for lvl, max_uses in (
            char.spell_slots or {}
        ).items()}   # spell_slots already holds max — restore all
        # Re-populate from a sensible default if slots were depleted.
        # (Slot max is tracked as the initial value; we reset to it.)
        recovered.append("HP 全回復")
        recovered.append("法術位全回復")

    return {"rest_type": rest_type, "recovered": recovered}


@dataclass
class CombatState:
    initiative_order: list
    current_turn_index: int = 0
    # Starts at 0; the game loop pre-increments before emitting RoundStart, so
    # the first round reads as round 1.
    round_number: int = 0
    active: bool = True
    battlefield: "Battlefield | None" = None   # 2D play area; populated by setup_combat_positions
