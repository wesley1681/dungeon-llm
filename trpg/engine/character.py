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
    # 5e 規則：一回合只能施一個「有環法術」（非戲法）。戲法 + 任意一個有環
    # 法術可以共存，兩個有環則不行（例如 misty_step + fireball 違規）。
    # 每回合在 self_turn_start 重置；execute_action 在消耗法術位前檢查。
    leveled_spell_cast_this_turn: bool = False
    # 此角色會使用的 Ability skill_id 清單（如 ["second_wind","rage"]）。
    # HumanInputPolicy 用「招式」指令時會檢查這份清單。
    known_abilities: list = field(default_factory=list)
    # 每個 ability 剩餘的使用次數。{skill_id: remaining_uses}
    # 值 None → 無限制（passive / unlimited）。
    # 由 rest_character() 根據 Ability.refresh_on 重設。
    ability_uses: dict = field(default_factory=dict)
    attacks_per_action: int = 1   # 1 = normal; 2 = Extra Attack (L5 Fighter/Barbarian etc.)
    crit_range: int = 20          # Champion archetype: set to 19 to crit on 19-20
    archetype_id: str = ""        # subclass label matching Ability.archetype_id;
                                  # "" = base class or unspecified. Used to filter
                                  # which known_abilities appear in RL observations.
    sneak_attack_dice: str = ""   # e.g. "2d6" for L3 Rogue; "" = not a rogue
    lay_on_hands_pool: int = 0    # Paladin healing pool = 5 × level; 0 = non-paladin
    sculpt_spells: bool = False   # Evocation Wizard L2: AOE spells skip chosen allies
    aura_of_protection_bonus: int = 0  # Paladin L6: CHA mod added to all nearby ally saves
    portent_dice: list = field(default_factory=list)  # stored d20 values (Divination Wizard)
    pending_portent: int | None = None                 # overrides target's next save roll
    # ── Wave 1 monster mechanics (all data-driven via TRAIT_APPLIERS) ────────
    # {damage_type: multiplier} applied in apply_damage AFTER modifiers.
    # 0.5 = resistance, 0.0 = immunity, 2.0 = vulnerability, negative = absorb
    # (damage heals instead — Iron Golem fire). Keys validated vs DAMAGE_TYPES.
    damage_multipliers: dict = field(default_factory=dict)
    # Status names this creature can never receive (add_status drops them).
    condition_immunities: list = field(default_factory=list)
    # {skill_id: threshold} — at self_turn_start, if the ability is spent,
    # roll 1d6 >= threshold to restore one use (dragon breath recharge 5-6).
    recharge_abilities: dict = field(default_factory=dict)
    pack_tactics: bool = False     # advantage when an ally is adjacent to target
    undead_fortitude: bool = False  # Zombie: CON save to drop to 1 HP instead of 0
    # ── Wave 2 monster mechanics ─────────────────────────────────────────────
    # {"amount": 10, "blocked_by": ("火", "強酸")} — heal at self_turn_start
    # unless a blocking damage type landed since the previous turn start.
    regeneration: dict | None = None
    # Damage types received since this creature's last turn start (consumed by
    # regeneration suppression; cleared each self_turn_start in tick_status_effects).
    recent_damage_types: set = field(default_factory=set)
    # ── Wave 3 monster mechanics (all data-driven via TRAIT_APPLIERS) ────────
    # Legendary actions (5e): a budget refilled at self_turn_start, spent ONE
    # option at a time at the end of OTHER creatures' turns
    # (combat_policy.run_legendary_actions). Options: {"ability": skill_id} or
    # {"weapon": name} plus {"cost": int}. None of these fields enter obs —
    # remaining counts are hidden resource state (MONSTER_CATALOG §4.1).
    legendary_actions_max: int = 0
    legendary_actions_remaining: int = 0
    legendary_options: list = field(default_factory=list)
    # Legendary resistance: auto-succeed a failed save that would leave a
    # status on this creature (per-combat pool; make_saving_throw consumes).
    legendary_resistance_uses: int = 0
    # Frightful presence aura spec {"radius_m", "dc", "rounds"} — checked at
    # each enemy's turn start (tick_aura_damage); a successful save grants
    # immunity for the rest of the combat (frightful_immune_to, victim-side).
    frightful_presence: dict | None = None
    frightful_immune_to: set = field(default_factory=set)
    # Death throes {"damage_dice","damage_type","radius_m","save_stat","dc"} —
    # detonated by apply_damage when this NPC dies (popped for re-entrancy).
    death_throes: dict | None = None
    # Lair actions (5e): once per round at "initiative count 20" a creature in
    # its lair triggers ONE environmental effect — chosen from lair_options
    # ({"ability": skill_id, "ev": float}). Fired by combat_policy.run_lair_actions
    # (piggybacked on run_legendary_actions, self-limited via lair_acted_round).
    # Like legendary actions, NONE of these enter obs (hidden resource state).
    lair_options: list = field(default_factory=list)
    lair_acted_round: int = 0
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
        """Add a StatusEffect (idempotent on name; condition immunities win —
        a zombie can never become poisoned no matter the source)."""
        if fx.name in self.condition_immunities:
            return
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

    def is_incapacitated(self) -> bool:
        """Cannot take actions (paralyzed/asleep/stunned/petrified). The ONE
        predicate shared by execute_action's refusal and the drivers' turn
        skips — extend status.INCAPACITATING_STATUSES, never the call sites."""
        from .status import INCAPACITATING_STATUSES
        return any(self.has_status(s) for s in INCAPACITATING_STATUSES)

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
    from .abilities import ABILITY_REGISTRY
    recovered: list[str] = []

    if rest_type not in ("short", "long"):
        return {"error": f"未知休息類型：{rest_type}"}

    # Restore ability uses.
    for sid in char.known_abilities:
        ab = ABILITY_REGISTRY.get(sid)
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
