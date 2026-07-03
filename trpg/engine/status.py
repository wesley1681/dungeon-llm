"""StatusEffect — Modifier with a declarative lifecycle.

Each effect declares how it expires:
  expires_on        — phase that auto-removes it ('self_turn_start',
                      'self_turn_end', 'round_end', 'combat_end', 'never')
  rounds_remaining  — if set, decremented at 'round_end'; removed when <= 0
  save_each         — e.g. 'CON DC14'; at 'self_turn_end', re-roll save to escape

tick_status_effects() is called by GameSession at 4 phases of the combat loop.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .modifier import Modifier
from .dice import combine_advantage


def parse_save(spec: str) -> tuple[str, int]:
    """Parse 'CON DC14' -> ('CON', 14)."""
    parts = spec.upper().replace("DC", " ").split()
    return parts[0], int(parts[1])


@dataclass
class StatusEffect(Modifier):
    name: str = ""
    expires_on: str = "never"            # self_turn_start|self_turn_end|round_end|combat_end|never
    rounds_remaining: int | None = None  # countdown each round_end
    save_each: str = ""                  # "CON DC14": re-save each self_turn_end
    applied_round: int = 0
    source_id: str = ""                  # caster ID for concentration / cleanup
    metadata: dict = field(default_factory=dict)
    # Whether this effect helps ("buff"), hurts ("debuff"), or is tactical/
    # neutral ("neutral"). Read by observation extractors (e.g. RL obs's
    # has_debuff flag) and any other code that needs to classify effects
    # without maintaining a separate name allowlist. Subclasses override the
    # default by setting kind="buff" / "neutral" in their __init__.
    kind: str = "debuff"


class Dodging(StatusEffect):
    """D&D 5e Dodge action. Expires at start of dodger's next turn.
    Forces disadvantage on attacks against the dodger."""
    def __init__(self, applied_round: int = 0):
        super().__init__(
            name="dodging",
            expires_on="self_turn_start",
            applied_round=applied_round,
            kind="buff",
        )

    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Hidden(StatusEffect):
    """Successful HIDE action. Grants advantage on the next attack the holder
    makes; engine removes the status after that attack resolves (5e RAW: making
    an attack reveals you)."""
    def __init__(self, applied_round: int = 0):
        super().__init__(name="hidden", expires_on="never",
                         applied_round=applied_round, kind="buff")
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")


class Reckless(StatusEffect):
    """Reckless Attack rider. Lasts until the attacker's next turn — until
    then, attacks against them have advantage. Tactical self-imposed effect
    (trades vulnerability for advantage on this turn), so classed neutral."""
    def __init__(self, applied_round: int = 0):
        super().__init__(name="reckless", expires_on="self_turn_start",
                         applied_round=applied_round, kind="neutral")
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")


class Blessed(StatusEffect):
    """Bless target — +1d4 to attack rolls and saving throws.
    Default 10 rounds (1 minute); the caster's concentration ends it sooner."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(
            name="blessed", expires_on="never",
            rounds_remaining=10, applied_round=applied_round,
            source_id=source_id, kind="buff",
        )
    def on_outgoing_attack_total(self, attacker, target, weapon, total: int) -> int:
        from .dice import roll
        return total + roll("1d4")
    def on_saving_throw(self, char, stat: str, modifier: int) -> int:
        from .dice import roll
        return modifier + roll("1d4")


class Shielded(StatusEffect):
    """Shield spell — +5 AC until the start of the defender's next turn.
    Auto-attached by the engine when a Shield reaction fires."""
    def __init__(self, applied_round: int = 0):
        super().__init__(name="shielded", expires_on="self_turn_start",
                         applied_round=applied_round, kind="buff")
    def on_compute_ac(self, char, current_ac: int) -> int:
        return current_ac + 5


class Raging(StatusEffect):
    """Barbarian Rage. +2 to physical-type outgoing damage; physical-type
    incoming damage is halved (resistance). 10 rounds in our simplified model
    (5e has more nuanced end-conditions tied to dealing/taking damage)."""
    _PHYSICAL = ("斬擊", "穿刺", "鈍擊", "slashing", "piercing", "bludgeoning")
    def __init__(self, applied_round: int = 0):
        super().__init__(
            name="raging", expires_on="never",
            rounds_remaining=10, applied_round=applied_round,
            kind="buff",
        )
    def on_outgoing_damage(self, attacker, target, amount: int, dtype: str) -> int:
        return amount + 2 if dtype in self._PHYSICAL else amount
    def on_incoming_damage(self, target, attacker, amount: int, dtype: str) -> int:
        # Bear Totem barbarian (metadata flag): resist ALL damage except
        # psychic. Plain rage resists only physical types.
        if self.metadata.get("all_types"):
            return amount // 2 if dtype not in ("精神", "psychic") else amount
        return amount // 2 if dtype in self._PHYSICAL else amount


# ── Condition effects (5e rule book) ─────────────────────────────────────────
# All of these are applied via the SPELL handler's `applies_status_on_fail`
# or directly via APPLY_MOD. Duration/source tracked on the base dataclass.

class Prone(StatusEffect):
    """Fallen. Melee attacks against prone targets have advantage; ranged have
    disadvantage. The prone creature's own attack rolls are at disadvantage.
    Standing up costs half the creature's normal speed."""
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="prone", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        if weapon and weapon.range_type == "近戰":
            return combine_advantage(mode, "advantage")
        return combine_advantage(mode, "disadvantage")
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")
    def on_speed_multiplier(self, char) -> float:
        return 0.5   # standing up costs half movement; implemented as half speed


class Paralyzed(StatusEffect):
    """Cannot move, speak, or take actions. Auto-fails STR/DEX saves.
    Attacks vs the paralyzed target have advantage. Melee hits within 5ft
    (1.5m) are auto-crits (handled in ATTACK handler via this hook)."""
    _AUTO_FAIL = ("STR", "DEX")
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="paralyzed", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_auto_fail_save(self, char, stat: str) -> bool:
        return stat.upper() in self._AUTO_FAIL
    def on_speed_multiplier(self, char) -> float:
        return 0.0


class Stunned(StatusEffect):
    """Incapacitated, cannot move, can only speak falteringly. Auto-fails STR/DEX
    saves. Attacks vs the stunned target have advantage."""
    _AUTO_FAIL = ("STR", "DEX")
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="stunned", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_auto_fail_save(self, char, stat: str) -> bool:
        return stat.upper() in self._AUTO_FAIL
    def on_speed_multiplier(self, char) -> float:
        return 0.0


class Poisoned(StatusEffect):
    """Disadvantage on attack rolls and ability checks."""
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="poisoned", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Restrained(StatusEffect):
    """Speed = 0. Attack rolls against have advantage; own attack rolls
    have disadvantage. DEX saves at disadvantage (modelled as -5)."""
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="restrained", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")
    def on_saving_throw(self, char, stat: str, modifier: int) -> int:
        return modifier - 5 if stat.upper() == "DEX" else modifier
    def on_speed_multiplier(self, char) -> float:
        return 0.0


class Frightened(StatusEffect):
    """Cannot willingly move toward the source of its fear. Disadvantage on
    attack rolls while the source is in its line of sight. (Source tracking
    kept simple: disadvantage applies unconditionally in our model since
    checking LoS to the source would require source_id lookup.)"""
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="frightened", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Charmed(StatusEffect):
    """Cannot attack the charmer; charmer has advantage on social ability
    checks against the charmed creature. Attack restriction enforced in the
    ATTACK handler by checking source_id vs the target."""
    def __init__(self, applied_round=0, source_id=""):
        super().__init__(name="charmed", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    # The actual attack-block lives in combat.py's ATTACK handler:
    # if the ATTACKER has charmed with source_id == target_id → reject
    # (the charmed one cannot strike its charmer; the reverse is legal).


class Evasion(StatusEffect):
    """Rogue L7 passive. DEX saves vs area effects:
    success = 0 damage (instead of normal half-damage).
    failure = half damage (instead of full damage).
    """
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="evasion", expires_on="never",
                         applied_round=applied_round, source_id=source_id,
                         kind="buff")

    def on_incoming_save_damage(self, char, stat: str, success: bool,
                                amount: int) -> int:
        if success:
            return 0
        return amount // 2


class HuntersMark(StatusEffect):
    """Ranger Hunter's Mark. Attached to the marked creature.
    The attacker whose source_id matches deals +1d6 extra damage on each hit."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="hunters_mark", expires_on="never",
                         applied_round=applied_round, source_id=source_id)


class Blurred(StatusEffect):
    """Blur spell (L2 concentration). All attacks against this creature have disadvantage."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="blurred", expires_on="never",
                         applied_round=applied_round, source_id=source_id,
                         kind="buff")
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Baned(StatusEffect):
    """Bane spell (L1 concentration). Subtract 1d4 from attack rolls and saving throws."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="baned", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_outgoing_attack_total(self, attacker, target, weapon, total: int) -> int:
        from .dice import roll
        return total - roll("1d4")
    def on_saving_throw(self, char, stat: str, modifier: int) -> int:
        from .dice import roll
        return modifier - roll("1d4")


class SacredWeaponBuff(StatusEffect):
    """Sacred Weapon (Channel Divinity, Devotion Paladin). +3 to attack rolls for 1 minute.
    (Approximates CHA +3 for a typical L6 Devotion Paladin with CHA 16.)"""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="sacred_weapon_buff", expires_on="never",
                         rounds_remaining=10, applied_round=applied_round,
                         source_id=source_id, kind="buff")
    def on_outgoing_attack_total(self, attacker, target, weapon, total: int) -> int:
        return total + 3


class ShieldOfFaith(StatusEffect):
    """Shield of Faith spell (L1 concentration). Target gains +2 AC for 10 minutes
    while the caster maintains concentration."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="shield_of_faith", expires_on="never",
                         rounds_remaining=100,    # ~10 min in 6-second rounds
                         applied_round=applied_round, source_id=source_id,
                         kind="buff")
    def on_compute_ac(self, char, current_ac: int) -> int:
        return current_ac + 2


class Distracted(StatusEffect):
    """Battle Master Distracting Strike rider. The target's next incoming
    attack (from anyone other than the BM) is at advantage. Engine model:
    advantage on all incoming attacks for 1 round (round_end ticks it down)."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="distracted", expires_on="never",
                         rounds_remaining=1,
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")


class Blinded(StatusEffect):
    """Cannot see — attacks against the blinded creature have advantage, and
    the blinded creature's attacks have disadvantage. Engine-modeled durations
    are short (1-2 rounds) because the original D&D 5e effects span longer."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="blinded", expires_on="never",
                         rounds_remaining=2, applied_round=applied_round,
                         source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")


class Asleep(StatusEffect):
    """D&D Sleep / unconscious: cannot move/act, auto-fails STR/DEX saves,
    melee attacks within 1.5m are auto-crits, all attacks against have
    advantage. Identical mechanics to Paralyzed in this engine. Wakes in
    `rounds_remaining` rounds or on taking damage (handled by apply_damage)."""
    _AUTO_FAIL = ("STR", "DEX")
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="asleep", expires_on="never",
                         rounds_remaining=2,
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_auto_fail_save(self, char, stat: str) -> bool:
        return stat.upper() in self._AUTO_FAIL
    def on_speed_multiplier(self, char) -> float:
        return 0.0


class SpiritualWeaponActive(StatusEffect):
    """Spiritual Weapon (Cleric L2). Caster has a floating force weapon for
    10 rounds; each turn they can spend a bonus action to make a melee spell
    attack via the spiritual_weapon_attack ability. Non-concentration."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(
            name="spiritual_weapon_active", expires_on="never",
            rounds_remaining=10, applied_round=applied_round,
            source_id=source_id, kind="buff",
        )


class SpiritGuardiansActive(StatusEffect):
    """Spirit Guardians (Cleric L3, concentration). 4.5m radius aura around the
    caster; at each enemy's turn start, if within range, they take 3d8 radiant
    (WIS save halves). Damage is applied by tick_status_effects via the engine
    on_turn_start hook. Source_id holds the caster's char_id for save DC
    lookup."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(
            name="spirit_guardians_active", expires_on="never",
            rounds_remaining=10, applied_round=applied_round,
            source_id=source_id, kind="buff",
        )


class VowTarget(StatusEffect):
    """Vow of Enmity target. The paladin (source_id) attacks this creature with
    advantage. Handled in _resolve_single_attack (no hook here — needs
    attacker identity from world_state)."""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="vow_target", expires_on="never",
                         rounds_remaining=10, applied_round=applied_round,
                         source_id=source_id)


# ── Wave 2 conditions (MONSTER_CATALOG §3 C 級) ──────────────────────────────

class Petrified(StatusEffect):
    """石化（5e 簡化）：等同麻痺（不能行動、STR/DEX 豁免自動失敗、被攻擊
    有優勢）＋全傷害類型抗性（減半）。與 5e 的差異：不變物件、毒免疫略；
    近戰命中不自動暴擊（5e 的 auto-crit 只屬 paralyzed/unconscious）。
    永久（rounds=None）；戰鬥結束時由 COMBAT_ONLY 清理（戰役仁慈規則）。
    Stage-2 of the two-stage gaze/ray escalation (basilisk, beholder)."""
    _AUTO_FAIL = ("STR", "DEX")
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="petrified", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_auto_fail_save(self, char, stat: str) -> bool:
        return stat.upper() in self._AUTO_FAIL
    def on_speed_multiplier(self, char) -> float:
        return 0.0
    def on_incoming_damage(self, target, attacker, amount: int, dtype: str) -> int:
        return amount // 2


class Swallowed(StatusEffect):
    """吞噬（Behir／Kraken／Tarrasque 系，簡化版）：束縛＋目盲的合成效果
    （被攻擊優勢、自身攻擊劣勢、速度 0、DEX 豁免 −5）＋體內持續傷害。
    DoT 由 tick_aura_damage 的通用 status-tick 掃描執行，參數放 metadata：
      tick_damage_dice / tick_damage_type   每回合開始的體內傷害
      ends_if_source_dead = True            吞噬者死亡 → 被吐出（效果移除）
    逃脫：save_each（簡化 5e 的「對吞噬者造成 30+ 傷害則吐出」）。"""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="swallowed", expires_on="never",
                         applied_round=applied_round, source_id=source_id)
    def on_incoming_attack(self, defender, attacker, weapon, mode: str) -> str:
        return combine_advantage(mode, "advantage")
    def on_outgoing_attack(self, attacker, target, weapon, mode: str) -> str:
        return combine_advantage(mode, "disadvantage")
    def on_saving_throw(self, char, stat: str, modifier: int) -> int:
        return modifier - 5 if stat.upper() == "DEX" else modifier
    def on_speed_multiplier(self, char) -> float:
        return 0.0


class Slowed(StatusEffect):
    """緩速（眼魔緩速射線；Wave 3 石魔像緩速 AoE 同款）：速度減半、AC −2。
    5e slow 還有「每回合一動作、無反應」——本引擎以此兩項近似。"""
    def __init__(self, applied_round: int = 0, source_id: str = ""):
        super().__init__(name="slowed", expires_on="never",
                         rounds_remaining=10, applied_round=applied_round,
                         source_id=source_id)
    def on_speed_multiplier(self, char) -> float:
        return 0.5
    def on_compute_ac(self, char, current_ac: int) -> int:
        return current_ac - 2


# ── Registry ─────────────────────────────────────────────────────────────────
# Looked up by APPLY_MOD action handler. Append-only when adding new buff /
# debuff statuses that need to be applied through the generic action.

MODIFIER_CLASSES: dict[str, type] = {
    "dodging":      Dodging,
    "hidden":       Hidden,
    "reckless":     Reckless,
    "blessed":      Blessed,
    "raging":       Raging,
    "shielded":     Shielded,
    "prone":        Prone,
    "paralyzed":    Paralyzed,
    "asleep":       Asleep,
    "blinded":      Blinded,
    "distracted":   Distracted,
    "stunned":      Stunned,
    "poisoned":     Poisoned,
    "restrained":   Restrained,
    "frightened":   Frightened,
    "charmed":      Charmed,
    "evasion":            Evasion,
    "hunters_mark":       HuntersMark,
    "blurred":            Blurred,
    "baned":              Baned,
    "sacred_weapon_buff":      SacredWeaponBuff,
    "shield_of_faith":         ShieldOfFaith,
    "vow_target":              VowTarget,
    "spiritual_weapon_active": SpiritualWeaponActive,
    "spirit_guardians_active": SpiritGuardiansActive,
    # "petrified" 在 STATUS_SLOTS 既有保留位 → 加入本表不改變 RL_STATUS_NAMES
    # 聯集（obs 維度凍結），且 obs 立即可見此狀態。
    "petrified":               Petrified,
}

# 引擎可施加、但「名字還不在 obs 狀態詞彙」的狀態。RL_STATUS_NAMES（= obs
# ENTITY_DIM 的一部分）取自 STATUS_SLOTS ∪ MODIFIER_CLASSES.keys()——往
# MODIFIER_CLASSES 加新名字會位移所有 checkpoint 的 obs 維度。新狀態先放
# 這裡（引擎機制照常運作、obs 暫不可見），下一次 obs 手術時再整批收編。
ENGINE_ONLY_STATUS_CLASSES: dict[str, type] = {
    "swallowed": Swallowed,
    "slowed":    Slowed,
}

# 施加類狀態的單一查找入口（rider／SPELL／射線表共用）。
ALL_STATUS_CLASSES: dict[str, type] = {
    **MODIFIER_CLASSES, **ENGINE_ONLY_STATUS_CLASSES,
}

# 不能行動／不能反應的狀態（5e incapacitated 家族——本引擎建模的子集）。
# execute_action 的行動拒絕與 env 的回合跳過必須吃同一份清單。
INCAPACITATING_STATUSES: tuple[str, ...] = (
    "paralyzed", "asleep", "stunned", "petrified",
)

# Statuses that exist only for the duration of one combat — buffs/debuffs the
# engine attaches via the action system. They get cleared in bulk when combat
# ends so the next encounter doesn't start with stale `raging`, `blessed` etc.
# Persistent conditions (e.g. poisoned, charmed, frightened from external
# sources) should be omitted here so they survive combat transitions.
COMBAT_ONLY_STATUSES: frozenset[str] = frozenset({
    "dodging", "hidden", "reckless", "blessed", "raging", "shielded",
    "prone", "paralyzed", "asleep", "blinded", "distracted",
    "stunned", "poisoned", "restrained",
    "frightened", "charmed", "disengaging", "evasion", "hunters_mark",
    "blurred", "baned", "sacred_weapon_buff",
    "shield_of_faith", "vow_target",
    "spiritual_weapon_active", "spirit_guardians_active",
    "petrified", "swallowed", "slowed",
})


def tick_status_effects(char, phase: str, round_num: int) -> None:
    """Apply lifecycle rules for all status effects on `char`.

    Called at 4 phases:
      self_turn_start — start of this character's turn
      self_turn_end   — end of this character's turn
      round_end       — after the initiative table completes a full cycle
      combat_end      — combat has ended

    Rebuilds the list rather than calling list.remove() because StatusEffect is
    a @dataclass with structural equality — two effects with identical fields
    would be indistinguishable to .remove(), causing the wrong instance to be
    dropped if duplicates ever appear.
    """
    from .combat import make_saving_throw  # avoid circular import

    # Breath-weapon recharge (Wave 1): at the start of the creature's turn,
    # each spent recharge ability rolls 1d6 — at or above its threshold the
    # use comes back (dragon breath "Recharge 5-6"). Lives here because this
    # function is the ONE turn-start chokepoint every driver already calls
    # (game.py / env_v2 / bc_collect / sandbox) — no per-driver wiring.
    if phase == "self_turn_start" and char.recharge_abilities:
        from .dice import roll as _roll
        for sid, threshold in char.recharge_abilities.items():
            if char.ability_uses.get(sid, 1) <= 0 and _roll("1d6") >= threshold:
                char.ability_uses[sid] = 1

    # Regeneration (Wave 2): at the creature's turn start, heal a flat amount
    # unless it took any of the suppressing damage types since its previous
    # turn start (troll: fire/acid). Same chokepoint as breath recharge — every
    # driver already calls this phase; no dice involved, so fights without a
    # regenerator consume zero extra RNG (baseline-safe). The damage window
    # set is cleared here for EVERY creature so it always means "damage taken
    # since my last turn started".
    if phase == "self_turn_start":
        regen = getattr(char, "regeneration", None)
        if (regen and char.is_alive() and char.hp < char.max_hp
                and not (set(regen["blocked_by"]) & char.recent_damage_types)):
            char.hp = min(char.max_hp, char.hp + regen["amount"])
        if char.recent_damage_types:
            char.recent_damage_types.clear()

    # Legendary actions (Wave 3): the budget refills at the creature's own
    # turn start (5e RAW). Spending happens at the end of OTHER creatures'
    # turns via combat_policy.run_legendary_actions — this is only the regain,
    # riding the same zero-wiring chokepoint as recharge/regeneration.
    if phase == "self_turn_start" and char.legendary_actions_max:
        char.legendary_actions_remaining = char.legendary_actions_max

    # Combat-end blanket cleanup: any combat-only buff/debuff is cleared so
    # the next encounter starts fresh.
    if phase == "combat_end":
        char.status_effects[:] = [
            fx for fx in char.status_effects
            if not (isinstance(fx, StatusEffect) and fx.name in COMBAT_ONLY_STATUSES)
        ]
        return

    remaining = []
    for fx in char.status_effects:
        # Skip non-StatusEffect entries (legacy strings) — they're not
        # subject to lifecycle rules and survive the tick untouched.
        if not isinstance(fx, StatusEffect):
            remaining.append(fx)
            continue

        # Rule 1: direct phase match → drop
        if fx.expires_on == phase:
            continue
        # Rule 2: countdown at round_end → drop when reaches zero
        if fx.rounds_remaining is not None and phase == "round_end":
            fx.rounds_remaining -= 1
            if fx.rounds_remaining <= 0:
                continue
        # Rule 3: re-save each self_turn_end → drop on success. Failing keeps
        # a status on the creature, so legendary resistance may burn a use
        # here (a held/restrained boss buying its action economy back).
        if fx.save_each and phase == "self_turn_end":
            stat, dc = parse_save(fx.save_each)
            ok, _ = make_saving_throw(char, stat, dc,
                                      fail_applies_status=True)
            if ok:
                continue
        remaining.append(fx)
    char.status_effects[:] = remaining
