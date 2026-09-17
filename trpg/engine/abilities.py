"""Class ability catalog — D&D 5e flavoured.

Phase 1 deliverable: each ability is encoded as a SkillFeatures vector so we
can stress-test the schema with real content. Engine-side execution is not
universally wired up yet:

  engine_ready=True   the action dict returned by `build_action()` is fully
                       executable by execute_action()
  engine_ready=False  the ability is encoded for the RL observation layer
                       but the engine doesn't yet know how to execute it
                       (auto-hit damage, multi-target buffs, reactions,
                       teleport, attack-plus-save hybrids, action-economy
                       meta-effects). Each entry's `engine_todo` field
                       lists exactly what's missing.

Adding a new ability:
  - Pick the closest match below as a template
  - If your ability needs a feature dim that doesn't exist, extend
    SkillFeatures in skill.py first (append-only — never reorder)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from .skill import (
    SkillFeatures, TargetType, SaveStat, STATUS_SLOTS, N_STATUS_SLOTS,
)


def _status_multihot(*names: str) -> tuple[bool, ...]:
    return tuple(slot in names for slot in STATUS_SLOTS)


def _can_move(char) -> bool:
    """True iff the character isn't reduced to zero movement (restrain, grapple)."""
    return all(m.on_speed_multiplier(char) > 0 for m in char.iter_modifiers())


def _csv_targets(target, fallback=None) -> list:
    """Parse a multi-target selection into a list of entity ids.

    The action layer passes MULTI_* targets as a comma-separated string
    ("kaine,thor") — the magic-missile / bless convention — or already as a
    list. Empty selection falls back to ``fallback`` (e.g. heal yourself when
    no ally is named) so a multi-target builder always yields ≥1 target."""
    if isinstance(target, str):
        ids = [t.strip() for t in target.split(",") if t.strip()]
    elif target:
        ids = list(target)
    else:
        ids = []
    if not ids and fallback is not None:
        ids = [fallback]
    return ids


@dataclass
class Ability:
    """Static description of an executable ability (class feature or spell).

    The character's ``known_abilities`` list decides who has access — this
    definition only describes what the ability does and how the engine
    executes it.
    """
    skill_id: str
    display_name: str
    description: str
    features: SkillFeatures
    engine_ready: bool = False
    engine_todo: str = ""     # what the engine still needs for this to execute
    # Optional builder: (actor_id, target_entity_id, target_coord) -> action dict.
    # For reactions (is_reaction=True), the builder is irrelevant — the engine
    # checks Character.reactions and auto-fires on the right trigger event.
    builder: Callable | None = None
    is_reaction: bool = False   # if True, listed for RL observation but
                                # NEVER offered as an active-turn action choice
    min_level: int = 1          # character level at which this ability is available
    # Rest recovery: "short_rest" | "long_rest" | "never"
    # Informs rest_character() how to replenish uses_remaining.
    refresh_on: str = "short_rest"
    # Maximum uses per refresh period (0 = unlimited / passive).
    max_uses: int = 0
    # Optional gate for resource pools that aren't tracked by ability_uses
    # (e.g. lay_on_hands_pool). Returns True when the ability can still be used.
    is_usable: Callable | None = None
    # Target-state preconditions (Wave 2, data-driven — behir swallow needs a
    # restrained target and must not re-swallow; kraken/tarrasque reuse).
    # Policies read these to gate the option; builders embed the same fields
    # in the action dict so the ATTACK handler re-validates.
    requires_target_status: str = ""
    blocked_by_target_status: str = ""
    # Slot-free DAMAGING abilities are cantrip-scaled by materialize() unless
    # this is False (monster naturals: breath weapons, swallow, eye rays —
    # their dice never scale with level). Mirrors Spell.scales_as_cantrip.
    scales_as_cantrip: bool = True

    def build_action(self, actor_id: str,
                     target_entity_id: str | None = None,
                     target_coord=None,
                     char=None) -> dict | None:
        """Build an engine action dict. Auto-embeds ``skill_id`` so downstream
        code can identify the source ability without reverse-engineering.
        """
        if self.builder is None:
            return None
        action = self.builder(actor_id, target_entity_id, target_coord, char=char)
        if action is not None:
            action["skill_id"] = self.skill_id
        return action


# ── Catalog ──────────────────────────────────────────────────────────────────

ABILITY_REGISTRY: dict[str, Ability] = {}


def _register(ab: Ability) -> Ability:
    ABILITY_REGISTRY[ab.skill_id] = ab
    return ab


# Conferred / aura damage — the ONE place the (dice, damage_type) live for
# damage dealt by STATUS machinery rather than by the casting action itself
# (spirit-guardians aura tick, hunter's-mark on-hit proc). Both the engine roll
# (combat.py) and the obs expected_damage / damage_types
# (skill.action_expected_damage / action_damage_shares) read this map, keyed by
# the ability's skill_id, so the RL numbers can never drift from the dice the
# engine actually rolls. The catalog convention counts a single application.
CONFERRED_DAMAGE_DICE: dict[str, tuple[str, str]] = {
    "spirit_guardians": ("3d8", "光耀"),
    "hunters_mark":     ("1d6", "穿刺"),
}


# ── Fighter (戰士) ───────────────────────────────────────────────────────────

_register(Ability(
    skill_id="second_wind",
    display_name="二度氣息",
    description="bonus action 回復 1d10 + 戰士等級 HP，每短休 1 次。",
    refresh_on="short_rest", max_uses=1,
    features=SkillFeatures(
        cost_bonus=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": actor,
        "dice": f"1d10+{int(char.level) if char else 3}", "range_m": 0.0,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="action_surge",
    display_name="動作激增",
    description="本回合多獲得 1 個動作，每短休 1 次。",
    refresh_on="short_rest", max_uses=1,
    features=SkillFeatures(
        grants_actions=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
    ),
    engine_ready=True,
    min_level=2, builder=lambda actor, target, coord, char=None: {
        "type": "ACTION_SURGE", "character": actor,
    },
))

_register(Ability(
    skill_id="trip_attack",
    display_name="絆倒攻擊",
    description="武器攻擊 + 命中後 STR 豁免，失敗則 prone。消耗 1 個戰技骰。",
    refresh_on="short_rest", max_uses=4,
    features=SkillFeatures(
        attack_vs_ac=5.0,
        save_dc=14.0,
        save_stat=SaveStat.STR,
        cost_action=1.0,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("prone"),
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=3, builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "rider_save_dc": 14, "rider_save_stat": "STR", "rider_status": "prone",
        "consumes": ["action"],
    },
    engine_todo="each maneuver tracks its own 4-use pool "
                "(max_uses) rather than a shared superiority-die pool — a "
                "deliberate from-the-engine simplification, no schema impact.",
))


# ── Wizard (法師) ────────────────────────────────────────────────────────────

_register(Ability(
    skill_id="magic_missile",
    display_name="魔法飛彈",
    description="3 道力場箭自動命中所選敵人（最多 3 個），每箭 1d4+1。",
    features=SkillFeatures(
        auto_hit=True,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.MULTI_ENEMY,
        max_targets=3,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: (
        lambda targets: {
            "type": "AUTO_DAMAGE", "attacker": actor,
            "targets": [
                {"id": t, "darts": 1 + (max(0, 3 - len(targets)) if i == 0 else 0)}
                for i, t in enumerate(targets)
            ],
            "damage_per": "1d4+1", "damage_type": "力場",
            "range_m": 36.0, "slot_level": 1,
            "consumes": ["action"],
        }
    )([t.strip() for t in (target or "").split(",") if t.strip()] or [target]),
))

_register(Ability(
    skill_id="shield_spell",
    display_name="法盾",
    description="REACTION：被攻擊命中或被魔法飛彈鎖定時，+5 AC 直到下回合。"
                "由引擎自動觸發 — 角色將 'shield_spell' 加入 Character.reactions "
                "且有 1+ 環法術位即可使用。",
    features=SkillFeatures(
        cost_reaction=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SELF,
        conferred_ac_mod=5.0,
        status_duration=1.0,
    ),
    engine_ready=True,
    is_reaction=True,
    min_level=1, engine_todo="smart-fire only triggers when +5 AC would flip a hit to miss; "
                "Magic Missile is always blocked. No 'always cast' option yet.",
))

_register(Ability(
    skill_id="counterspell",
    display_name="法術反制",
    description="REACTION：敵人施放有環法術時，消耗 1 個 ≥3 環法術位反制之。"
                "由引擎在施法事件觸發（角色將 'counterspell' 加入 Character.reactions "
                "且有 3+ 環法術位即可使用）；is_reaction=True，永不在主回合列為主動選項。",
    features=SkillFeatures(
        cost_reaction=1.0,
        cost_slot_level=3.0,
        range_m=18.0,                       # 60 ft
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    is_reaction=True,
    min_level=5,                            # earliest a caster has a 3rd-level slot
))

_register(Ability(
    skill_id="color_spray",
    display_name="彩光繽紛",
    description="4.5m 範圍敵人 CON 豁免失敗則 blinded 2 回合。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.CON,
        range_m=4.5,
        aoe_radius_m=3.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.POINT,
        applies_status=_status_multihot("blinded"),
        status_duration=2.0,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "彩光繽紛",
        "target_position": [coord.x, coord.y] if coord is not None
                            else (target if isinstance(target, list) else None),
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="sleep",
    display_name="睡眠術",
    description="6m 範圍敵人 WIS 豁免失敗則 asleep 2 回合（受傷即醒）。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.WIS,
        range_m=27.0,
        aoe_radius_m=6.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.POINT,
        applies_status=_status_multihot("asleep"),
        status_duration=2.0,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "睡眠術",
        "target_position": [coord.x, coord.y] if coord is not None
                            else (target if isinstance(target, list) else None),
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="hold_person",
    display_name="定身術",
    description="WIS 豁免，失敗則 paralyzed，專注、最多 10 回合，每回合可重豁。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.WIS,
        range_m=18.0,
        cost_action=1.0,
        cost_slot_level=2.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("paralyzed"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定身術",
        "target": target, "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="misty_step",
    display_name="霧步",
    description="bonus action 瞬移最多 9m，無視視線與障礙。",
    features=SkillFeatures(
        range_m=9.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.POINT,
        is_teleport=True,
    ),
    engine_ready=True,
    min_level=3, builder=lambda actor, target, coord, char=None: {
        "type": "MOVE", "character": actor,
        "target_position": list(coord) if coord else [0.0, 0.0],
        "teleport": True, "range_m": 9.0, "slot_level": 2,
        "consumes": ["bonus_action"],
    },
))


# ── Cleric (牧師) ────────────────────────────────────────────────────────────

_register(Ability(
    skill_id="cure_wounds",
    display_name="治療術",
    description="觸碰範圍治療一個盟友 1d8 + WIS 修正。",
    features=SkillFeatures(
        range_m=1.5,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": target,
        # 1d8 + spellcasting modifier (was a hardcoded +3 placeholder that made
        # the engine roll under/over-heal for non-+3 casters — mirror the live
        # mod like healing_word / mass_cure_wounds so engine AND obs agree).
        "dice": "1d8{:+d}".format(
            char.stats.modifier(char.spellcasting_ability)
            if (char and char.spellcasting_ability) else 3),
        "range_m": 1.5, "slot_level": 1,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="sacred_flame",
    display_name="神聖光輝",
    description="單一目標 DEX 豁免，失敗則 1d8 光耀傷害。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=18.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "神聖光輝",
        "target": target, "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="bless",
    display_name="祝福術",
    description="9m 內最多 3 個盟友，攻擊與豁免 +1d4，專注、最多 10 回合。",
    features=SkillFeatures(
        range_m=9.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.MULTI_ALLY,
        max_targets=3,
        conferred_attack_mod=2.5,
        conferred_save_mod=2.5,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=1, # `target` is a comma-separated list of ally ids ("kaine,thor"); the
    # builder splits it for the engine's multi-target dispatch.
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "blessed", "spell_name": "祝福術",
        "targets": (target.split(",") if isinstance(target, str) else list(target or [])),
        "max_targets": 3, "range_m": 9.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["action"],
    },
))


# ── Barbarian (蠻人) ─────────────────────────────────────────────────────────

_register(Ability(
    skill_id="rage",
    display_name="狂暴",
    description="bonus action：melee 傷害 +2、物理傷害抗性、CON 豁免優勢，10 回合。",
    refresh_on="long_rest", max_uses=3,
    features=SkillFeatures(
        cost_bonus=1.0,
        remaining_uses=3.0,
        target_type=TargetType.SELF,
        conferred_damage_mod=2.0,
        damage_resistance=0.5,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=1, builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "raging", "spell_name": "狂暴",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        # Totem (Bear) barbarians resist ALL damage (except psychic) while
        # raging — a passive that upgrades THIS rage, so the flag rides the
        # rage cast rather than being a separate castable (5e RAW).
        "mod_metadata": ({"all_types": True}
                         if (char and "bear_totem" in (char.known_abilities or []))
                         else {}),
        "consumes": ["bonus_action"],
    },
    engine_todo="CON-save-advantage not modelled (uses now tracked via max_uses)",
))

_register(Ability(
    skill_id="reckless_attack",
    display_name="魯莽攻擊",
    refresh_on="never", max_uses=0,
    description="declared on a melee weapon attack — this attack has advantage, "
                "and all incoming attacks have advantage until your next turn.",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        conferred_attack_mod=5.0,      # advantage ≈ +5 statistically
        conferred_ac_mod=-5.0,         # incoming attacks have advantage
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=2, builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target,
        "weapon": "", "reckless": True,
        "consumes": ["action"],
    },
))

# ── Shared resource sentinels ─────────────────────────────────────────────────
# Meta-abilities representing shared resource pools. Listed in known_abilities
# so rest_character() resets them via the normal refresh_on mechanism.

_register(Ability(
    skill_id="channel_divinity",
    display_name="引導神力",
    description="每短休 1 次的神力引導資源池（牧師 L2）。",
    features=SkillFeatures(target_type=TargetType.SELF),
    engine_ready=False,
    engine_todo="資源池佔位（非可施放動作）。簡化：各引導神力技能"
                "（guided_strike／preserve_life／sacred_weapon）各自獨立計次"
                "（max_uses），而非共用單一池——刻意從寬，無 schema 影響。",
    refresh_on="short_rest", max_uses=1,
    min_level=2, ))

_register(Ability(
    skill_id="hunters_mark",
    display_name="獵人印記",
    description="bonus action：標記一個敵人，每次命中 +1d6 傷害，專注，消耗 1 環。",
    features=SkillFeatures(
        range_m=27.0,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        # Engine deals the +1d6 proc as 穿刺 (combat._resolve_single_attack),
        # regardless of the marked-by weapon — mirror the engine, not 5e RAW.
    ),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "hunters_mark", "spell_name": "獵人印記",
        "targets": [target], "max_targets": 1, "range_m": 27.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["bonus_action"],
    },
))


# ── Fighter: Battle Master additional maneuvers ───────────────────────────────

_register(Ability(
    skill_id="distracting_strike",
    display_name="擾敵打擊",
    description="武器攻擊。命中時消耗 1 個戰技骰（每短休 4 次），目標下次受擊有優勢"
                "（1 回合）。簡化版：所有來源的下一個攻擊都會獲得優勢。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_action=1.0,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("distracted"),
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="short_rest", max_uses=4,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "rider_status": "distracted",
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="menacing_attack",
    display_name="威嚇攻擊",
    description="武器攻擊 + WIS 豁免，失敗則 frightened 1 回合。消耗 1 個戰技骰。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        save_dc=14.0,
        save_stat=SaveStat.STR,
        cost_action=1.0,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("frightened"),
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="short_rest", max_uses=4,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "rider_save_dc": 14, "rider_save_stat": "WIS", "rider_status": "frightened",
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="precision_attack",
    display_name="精準攻擊",
    description="使用動作後，消耗 1 個戰技骰（d8），將骰值加到剛才的攻擊骰。",
    features=SkillFeatures(
        attack_vs_ac=4.5,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=False,
    engine_todo="被動代理：命中判定後若仍 miss，_try_attack_roll_boost 自動消耗"
                "一個戰技骰（+1d8）嘗試翻成命中（貪心預設；模型化控制延後至整合波）。",
    min_level=3, refresh_on="short_rest", max_uses=4,
))

_register(Ability(
    skill_id="pushing_attack",
    display_name="推擊攻擊",
    description="武器攻擊 + STR 豁免，失敗則推開目標 4.5m。消耗 1 個戰技骰。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        save_dc=14.0,
        save_stat=SaveStat.STR,
        cost_action=1.0,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=3, refresh_on="short_rest", max_uses=4,
    # Weapon swing + STR save; on a failed save the target is shoved 4.5m
    # straight back (clamped to the battlefield). The superiority-die pool is
    # tracked as this ability's max_uses (deducted by execute_action). Matches
    # the sibling maneuvers' model: weapon EV only, no +1d8 die damage.
    builder=lambda actor, target, coord, char=None: ({
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "push_distance_m": 4.5,
        "push_save_dc": (8 + char.proficiency_bonus + char.stats.modifier("STR"))
                        if char else 14,
        "push_save_stat": "STR",
        "consumes": ["action"],
    } if target else None),
))

_register(Ability(
    skill_id="improved_critical",
    display_name="強化暴擊",
    description="暴擊範圍擴大：d20=19 或 20 均視為暴擊。角色創建時設 crit_range=19。",
    features=SkillFeatures(target_type=TargetType.SELF),
    engine_ready=False,
    engine_todo="被動特性：角色創建時將 Character.crit_range 設為 19，無需 builder。",
    min_level=3, ))


# ── Barbarian: Totem Warrior (Bear) ──────────────────────────────────────────

_register(Ability(
    skill_id="bear_totem",
    display_name="熊圖騰",
    description="狂暴時對所有傷害類型（除心靈傷害）獲得抗性。",
    features=SkillFeatures(
        damage_resistance=0.5,
        target_type=TargetType.SELF,
        remaining_uses=3.0,
        status_duration=10.0,
    ),
    engine_ready=False,
    engine_todo="被動特性：知曉 bear_totem 的野蠻人，其 rage 自動對所有傷害"
                "（除精神）減半——由 rage builder 注入 all_types 旗標實現，無主動 builder。",
    min_level=3, refresh_on="long_rest", max_uses=3,
))

_register(Ability(
    skill_id="frenzy_attack",
    display_name="狂戰攻擊",
    description="狂暴時每回合可用額外動作進行 1 次近戰武器攻擊。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="berserker_frenzy",
    display_name="狂戰狂暴攻擊",
    description="狂暴時每回合可用額外動作進行 1 次近戰武器攻擊（長休後消除疲憊）。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=True,
    engine_todo="刻意簡化：疲憊（Exhaustion）不建模——5e 疲憊在 rage 結束後才施加、"
                "且只能長休消除，對單場戰鬥（RL episode 內）幾乎無影響；無 schema 衝擊。",
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "consumes": ["bonus_action"],
    },
))


# ── Wizard: Evocation ─────────────────────────────────────────────────────────

_register(Ability(
    skill_id="burning_hands_ev",
    display_name="燃燒之手",
    description="3d6 火焰 AOE，4.5m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=4.5,
        aoe_radius_m=4.5,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "燃燒之手",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="scorching_ray_ev",
    display_name="烈焰射線",
    description="3 道射線，每道各自進行攻擊骰，命中各造成 2d6 火焰傷害。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=27.0,
        cost_action=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.MULTI_ENEMY,
        max_targets=3,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    # MULTI_ENEMY targets arrive as a comma-separated id string (the magic
    # missile convention). Three rays cycle over the picked targets — one
    # target → all 3 rays at it, three targets → one ray each. Flat 2d6 火 per
    # ray (no spellcasting mod), one L2 slot for the casting.
    builder=lambda actor, target, coord, char=None: ((lambda picks: {
        "type": "MULTI_SPELL_ATTACK", "caster": actor,
        "ray_targets": [picks[i % len(picks)] for i in range(3)],
        "damage_dice": "2d6", "damage_type": "火",
        "range_m": 27.0, "slot_level": 2, "add_spell_mod": False,
        "spell_name": "烈焰射線", "consumes": ["action"],
    } if picks else None)(
        [t.strip() for t in (target or "").split(",") if t.strip()])),
))

_register(Ability(
    skill_id="fireball_ev",
    display_name="火球術",
    description="8d6 火焰 AOE，6m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=45.0,
        aoe_radius_m=6.0,
        cost_action=1.0,
        cost_slot_level=3.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=5, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火球術",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="fire_bolt",
    display_name="火焰箭",
    description="戲法：36m 內單體法術攻擊骰，命中造成 1d10 火焰傷害（不耗法術位）。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=36.0,                  # 120 ft
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,   # cantrip: any level, unlimited
    # Single-target spell attack (vs AC) — same engine path as guiding_bolt_life,
    # but slot_level=0 so no spell slot is spent (cantrip). Flat 1d10 (this
    # codebase does not level-scale cantrip dice; the shaman is fixed at L3).
    builder=lambda actor, target, coord, char=None: ({
        "type": "SPELL_ATTACK", "caster": actor, "target": target,
        "spell_name": "火焰箭",
        "damage_dice": "1d10", "damage_type": "火",
        "range_m": 36.0, "slot_level": 0, "add_spell_mod": False,
        "consumes": ["action"],
    } if target else None),
))

_register(Ability(
    skill_id="web_ev",
    display_name="蜘蛛網",
    description="4.5m 半徑 AOE，DEX 豁免失敗則 restrained，專注。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=18.0,
        aoe_radius_m=4.5,
        cost_action=1.0,
        cost_slot_level=2.0,
        requires_concentration=True,
        target_type=TargetType.POINT,
        applies_status=_status_multihot("restrained"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "蜘蛛網",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="ice_storm_ev",
    display_name="冰風暴",
    description="4m 半徑 AOE，DEX 豁免，失敗 2d8 冰冷傷害，成功半傷。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=36.0,
        aoe_radius_m=4.0,
        cost_action=1.0,
        cost_slot_level=4.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=7, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "冰風暴",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="hold_monster_ev",
    display_name="定怪術",
    description="WIS 豁免失敗則 paralyzed，適用任何生物，專注，最多 10 回合。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.WIS,
        range_m=18.0,
        cost_action=1.0,
        cost_slot_level=5.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("paralyzed"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=8, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定怪術",
        "target": target, "consumes": ["action"],
    },
))


# ── Wizard: Divination ────────────────────────────────────────────────────────

_register(Ability(
    skill_id="burning_hands_div",
    display_name="燃燒之手",
    description="3d6 火焰 AOE，4.5m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        save_dc=13.0, save_stat=SaveStat.DEX,
        range_m=4.5, aoe_radius_m=4.5,
        cost_action=1.0, cost_slot_level=1.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "燃燒之手",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="web_div",
    display_name="蜘蛛網",
    description="4.5m 半徑 AOE，DEX 豁免失敗則 restrained，專注。",
    features=SkillFeatures(
        save_dc=13.0, save_stat=SaveStat.DEX,
        range_m=18.0, aoe_radius_m=4.5,
        cost_action=1.0, cost_slot_level=2.0,
        requires_concentration=True,
        target_type=TargetType.POINT,
        applies_status=_status_multihot("restrained"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "蜘蛛網",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="fireball_div",
    display_name="火球術",
    description="8d6 火焰 AOE，6m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        save_dc=13.0, save_stat=SaveStat.DEX,
        range_m=45.0, aoe_radius_m=6.0,
        cost_action=1.0, cost_slot_level=3.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=5, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火球術",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="hold_monster_div",
    display_name="定怪術",
    description="WIS 豁免失敗則 paralyzed，任何生物，專注。",
    features=SkillFeatures(
        save_dc=13.0, save_stat=SaveStat.WIS,
        range_m=18.0,
        cost_action=1.0, cost_slot_level=5.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("paralyzed"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=8, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定怪術",
        "target": target, "consumes": ["action"],
    },
))


# ── Cleric: Life Domain ───────────────────────────────────────────────────────

_register(Ability(
    skill_id="healing_word_life",
    display_name="治療語",
    description="bonus action 遠距治療一個盟友 1d4 + WIS 修正 HP，射程 18m。",
    features=SkillFeatures(
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": target,
        "dice": "1d4{:+d}".format(
            char.stats.modifier(char.spellcasting_ability)
            if char and char.spellcasting_ability else 3),
        "range_m": 18.0, "slot_level": 1,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="guiding_bolt_life",
    display_name="引導光彈",
    description="攻擊骰，命中造成 4d6 光耀傷害，下一個攻擊者擲優勢。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: ({
        "type": "SPELL_ATTACK", "caster": actor, "target": target,
        "spell_name": "引導光彈",
        "damage_dice": "4d6", "damage_type": "光耀",
        "range_m": 36.0, "slot_level": 1, "add_spell_mod": False,
        # 命中→目標「distracted」：下一個攻擊者擲優勢（複用既有狀態，
        # obs 詞彙不變）。5e RAW 是「下一次攻擊」，本引擎近似為 1 回合。
        "on_hit_status": "distracted",
        "consumes": ["action"],
    } if target else None),
))

_register(Ability(
    skill_id="spiritual_weapon_life",
    display_name="精神武器：召喚",
    description="bonus action 召喚光能武器 10 回合，非專注。施法後每回合可用 bonus "
                "action 觸發 spiritual_weapon_attack_life 攻擊。",
    features=SkillFeatures(
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.SELF,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "spiritual_weapon_active", "spell_name": "精神武器",
        "targets": [actor],
        "max_targets": 1, "range_m": 0.0,
        "slot_level": 2, "requires_concentration": False,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="spiritual_weapon_attack_life",
    display_name="精神武器：攻擊",
    description="bonus action 用召喚的光能武器作法術攻擊，1d8+WIS 力場傷害。"
                "需要 spiritual_weapon_active 狀態。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=18.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    is_usable=lambda char: char.has_status("spiritual_weapon_active"),
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL_ATTACK", "caster": actor, "target": target,
        "spell_name": "精神武器",
        "damage_dice": "1d8", "damage_type": "力場",
        "range_m": 18.0,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="channel_divinity_preserve_life",
    display_name="引導神力：守護生命",
    description="消耗引導神力：30ft 內治療總量 5×牧師等級 HP，分配給多個目標。",
    features=SkillFeatures(
        range_m=9.0,
        cost_action=1.0,
        target_type=TargetType.MULTI_ALLY,
        max_targets=6,
        remaining_uses=1.0,
    ),
    engine_ready=True,
    min_level=2, refresh_on="short_rest", max_uses=1,
    # Pool = 5 × cleric level, distributed most-wounded-first, capped at half
    # max HP each (5e Preserve Life). No spell slot — the channel_divinity use
    # is deducted by execute_action (max_uses=1).
    builder=lambda actor, target, coord, char=None: {
        "type": "MULTI_HEAL", "caster": actor,
        "targets": _csv_targets(target, fallback=actor),
        "pool": 5 * (char.level if char else 1), "cap_half": True,
        "range_m": 9.0, "spell_name": "守護生命",
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="mass_cure_wounds_life",
    display_name="群體治療術",
    description="9m 內最多 6 個生物各回復 3d8 + WIS HP。",
    features=SkillFeatures(
        range_m=9.0,
        cost_action=1.0,
        cost_slot_level=5.0,
        target_type=TargetType.MULTI_ALLY,
        max_targets=6,
    ),
    engine_ready=True,
    min_level=8, refresh_on="never", max_uses=0,
    # Each chosen ally heals 3d8 + WIS; one L5 slot for the whole casting.
    builder=lambda actor, target, coord, char=None: {
        "type": "MULTI_HEAL", "caster": actor,
        "targets": _csv_targets(target, fallback=actor),
        "dice": "3d8{:+d}".format(
            char.stats.modifier(char.spellcasting_ability)
            if (char and char.spellcasting_ability) else 3),
        "slot_level": 5, "range_m": 9.0, "spell_name": "群體治療術",
        "consumes": ["action"],
    },
))


# ── Cleric: War Domain ────────────────────────────────────────────────────────

_register(Ability(
    skill_id="guiding_bolt_war",
    display_name="引導光彈",
    description="攻擊骰，命中造成 4d6 光耀傷害，下一個攻擊者擲優勢。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: ({
        "type": "SPELL_ATTACK", "caster": actor, "target": target,
        "spell_name": "引導光彈",
        "damage_dice": "4d6", "damage_type": "光耀",
        "range_m": 36.0, "slot_level": 1, "add_spell_mod": False,
        "on_hit_status": "distracted",
        "consumes": ["action"],
    } if target else None),
))

_register(Ability(
    skill_id="channel_divinity_guided_strike",
    display_name="引導神力：引導打擊",
    description="看到攻擊骰後，消耗引導神力，在骰值上 +10。",
    features=SkillFeatures(
        attack_vs_ac=10.0,
        cost_reaction=1.0,
        target_type=TargetType.SELF,
        remaining_uses=1.0,
    ),
    engine_ready=False,
    engine_todo="被動代理：命中判定後若仍 miss，_try_attack_roll_boost 自動消耗"
                "引導神力（+10）翻成命中（貪心預設；模型化控制延後至整合波）。",
    min_level=2, refresh_on="short_rest", max_uses=1,
))

_register(Ability(
    skill_id="spiritual_weapon_war",
    display_name="精神武器：召喚",
    description="bonus action 召喚光能武器 10 回合，非專注。施法後每回合可用 bonus "
                "action 觸發 spiritual_weapon_attack_war 攻擊。",
    features=SkillFeatures(
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.SELF,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "spiritual_weapon_active", "spell_name": "精神武器",
        "targets": [actor],
        "max_targets": 1, "range_m": 0.0,
        "slot_level": 2, "requires_concentration": False,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="spirit_guardians",
    display_name="靈體守護",
    description="action 召喚 4.5m 護衛靈光，10 回合。專注，敵人於其回合開始或進入"
                "範圍時 WIS 豁免，失敗 3d8 光耀傷害，成功半傷。",
    features=SkillFeatures(
        aoe_radius_m=4.5,
        cost_action=1.0,
        cost_slot_level=3.0,
        requires_concentration=True,
        target_type=TargetType.SELF,
        status_duration=10.0,
        # Aura ticks deal 3d8 光耀 via tick_status_effects (combat.py), not
        # via this action — dtype pinned here, asserted by test_skill_dtype.
    ),
    engine_ready=True,
    min_level=5, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "spirit_guardians_active", "spell_name": "靈體守護",
        "targets": [actor],
        "max_targets": 1, "range_m": 0.0,
        "slot_level": 3, "requires_concentration": True,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="spiritual_weapon_attack_war",
    display_name="精神武器：攻擊",
    description="bonus action 用召喚的光能武器作法術攻擊，1d8+WIS 力場傷害。"
                "需要 spiritual_weapon_active 狀態。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        range_m=18.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=3, refresh_on="never", max_uses=0,
    is_usable=lambda char: char.has_status("spiritual_weapon_active"),
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL_ATTACK", "caster": actor, "target": target,
        "spell_name": "精神武器",
        "damage_dice": "1d8", "damage_type": "力場",
        "range_m": 18.0,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="war_priest_attack",
    display_name="戰爭祭司攻擊",
    description="使用動作攻擊後，可額外用 bonus action 再攻擊一次，WIS 次數每長休重置。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        remaining_uses=3.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="long_rest", max_uses=3,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "consumes": ["bonus_action"],
    },
))


# ── Rogue: Assassin ───────────────────────────────────────────────────────────

_register(Ability(
    skill_id="cunning_action_dash",
    display_name="狡猾動作：衝刺",
    description="bonus action：本回合移動距離翻倍。",
    features=SkillFeatures(cost_bonus=1.0, range_m=9.0,
                           target_type=TargetType.POINT),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    is_usable=_can_move,
    builder=lambda actor, target, coord, char=None: {
        "type": "MOVE", "character": actor,
        "target_position": [float(coord.x), float(coord.y)] if coord else [0.0, 0.0],
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="cunning_action_disengage",
    display_name="狡猾動作：脫身",
    description="bonus action：本回合移動不觸發藉機攻擊。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "DISENGAGE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="cunning_action_hide",
    display_name="狡猾動作：躲藏",
    description="bonus action：嘗試躲藏（DEX DC12）。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF,
                           applies_status=_status_multihot("hidden")),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "HIDE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="assassinate",
    display_name="刺殺",
    description="被動：第一回合且攻擊者處於 hidden 狀態時，命中視為暴擊"
                "（D&D 5e 驚訝目標 → 必爆代理機制）。",
    features=SkillFeatures(attack_vs_ac=5.0, target_type=TargetType.SINGLE_ENEMY),
    engine_ready=False,
    engine_todo="被動特性：在 _resolve_single_attack 檢查 round_num==1 且 "
                "attacker.has_status('hidden') 觸發 auto_crit，無 builder。",
    min_level=3, refresh_on="never", max_uses=0,
))

_register(Ability(
    skill_id="uncanny_dodge_rogue",
    display_name="閃避直覺",
    description="REACTION：被攻擊命中時消耗反應將傷害減半。"
                "角色將 'uncanny_dodge_rogue' 加入 Character.reactions 即可使用。",
    features=SkillFeatures(cost_reaction=1.0, target_type=TargetType.SELF,
                            damage_resistance=0.5),
    engine_ready=True,
    is_reaction=True,
    min_level=5, refresh_on="never", max_uses=0,
))

_register(Ability(
    skill_id="evasion_rogue",
    display_name="閃避",
    description="DEX 豁免成功→0 傷，失敗→半傷。由 Evasion 狀態效果實現。",
    features=SkillFeatures(target_type=TargetType.SELF, damage_resistance=0.5),
    engine_ready=True,
    engine_todo="被動特性：角色創建時將 Evasion() 加入 status_effects，無需主動觸發。",
    min_level=7, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "evasion", "spell_name": "閃避",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": [],
    },
))


# ── Paladin: Oath of Devotion ─────────────────────────────────────────────────

_register(Ability(
    skill_id="divine_smite_dev",
    display_name="神聖打擊",
    description="命中後消耗 1 環法術位，每環 +2d8 光耀傷害（max 5d8）。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        # expected_damage now materializes from the FULL action (weapon swing +
        # 2d8 光耀 smite), so it varies with the wielded weapon. The dtype stays
        # pinned to the smite's characteristic 光耀 (a subset of the action's
        # real types — weapon base + 光耀; test_skill_dtype checks the subset).
    ),
    engine_ready=True,
    engine_todo="傷害固定 2d8（1 環）；之後可按 slot 等級縮放。",
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "divine_smite_slot": 1,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="lay_on_hands_ability",
    display_name="聖療之手",
    description="觸碰治療：從 5×等級 HP 的資源池中恢復指定量。",
    features=SkillFeatures(
        range_m=1.5,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="long_rest", max_uses=0,
    is_usable=lambda char: getattr(char, "lay_on_hands_pool", 0) > 0,
    builder=lambda actor, target, coord, char=None: {
        "type": "LAY_ON_HANDS", "caster": actor, "target": target,
        "amount": min(5, getattr(char, "lay_on_hands_pool", 5)) if char else 5,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="shield_of_faith_dev",
    display_name="信仰護盾",
    description="專注：目標 +2 AC，持續 10 分鐘。",
    features=SkillFeatures(
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ALLY,
        conferred_ac_mod=2.0,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "shield_of_faith", "spell_name": "信仰護盾",
        "targets": [target], "max_targets": 1, "range_m": 18.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="sacred_weapon_dev",
    display_name="神聖武器",
    description="引導神力：1 分鐘內武器攻擊 +CHA 修正（近似 +3）。",
    features=SkillFeatures(
        conferred_attack_mod=3.0,
        cost_action=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="short_rest", max_uses=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "sacred_weapon_buff", "spell_name": "神聖武器",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="wrathful_smite_dev",
    display_name="憤怒打擊",
    description="bonus action 揮擊：命中造成武器傷害 + 1d6 精神，WIS 豁免失敗則"
                "frightened，消耗 1 環法術位且需專注。",
    features=SkillFeatures(
        # Bundled single-strike model (per the original engine_todo): the
        # bonus action IS the swing. EV = weapon swing (≈7.5) + 1d6 精神 rider
        # (3.5). damage_types splits the two packets so the obs-side typed
        # resist dot product reads the true blended multiplier.
        save_dc=13.0,
        save_stat=SaveStat.WIS,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("frightened"),
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: ({
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "spell_slot_cost": 1, "concentration": True, "spell_name": "憤怒打擊",
        "rider_damage_dice": "1d6", "rider_damage_type": "精神",
        "rider_status": "frightened",
        "rider_save_dc": (8 + char.proficiency_bonus
                          + char.stats.modifier(char.spellcasting_ability))
                         if (char and char.spellcasting_ability) else 13,
        "rider_save_stat": "WIS",
        "consumes": ["bonus_action"],
    } if target else None),
))


# ── Paladin: Oath of Vengeance ────────────────────────────────────────────────

_register(Ability(
    skill_id="divine_smite_ven",
    display_name="神聖打擊",
    description="命中後消耗法術位，每環 +2d8 光耀傷害。",
    features=SkillFeatures(
        attack_vs_ac=5.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        # expected_damage now materializes from the FULL action (weapon swing +
        # 2d8 光耀 smite), so it varies with the wielded weapon. The dtype stays
        # pinned to the smite's characteristic 光耀 (a subset of the action's
        # real types — weapon base + 光耀; test_skill_dtype checks the subset).
    ),
    engine_ready=True,
    engine_todo="固定 1 環（2d8）；之後可擴展為按 slot 等級縮放。",
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "",
        "divine_smite_slot": 1,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="bane_ven",
    display_name="詛咒術",
    description="最多 3 個目標，CHA 豁免失敗則攻擊骰和豁免 -1d4，專注。",
    features=SkillFeatures(
        save_dc=13.0,
        save_stat=SaveStat.CHA,
        range_m=9.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.MULTI_ENEMY,
        max_targets=3,
        conferred_attack_mod=-2.5,
        conferred_save_mod=-2.5,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=2, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "baned", "spell_name": "詛咒術",
        "targets": (target.split(",") if isinstance(target, str) else list(target or [])),
        "max_targets": 3, "range_m": 9.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="vow_of_enmity_ven",
    display_name="仇敵誓言",
    description="引導神力：聖騎士對選定目標的攻擊擲優勢，持續 1 分鐘。",
    features=SkillFeatures(
        conferred_attack_mod=5.0,
        cost_bonus=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, refresh_on="short_rest", max_uses=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "vow_target", "spell_name": "仇敵誓言",
        "targets": [target], "max_targets": 1, "range_m": 18.0,
        "consumes": ["bonus_action"],
    },
))

_register(Ability(
    skill_id="lay_on_hands_ability_ven",
    display_name="聖療之手",
    description="觸碰治療，5×等級 HP 資源池。",
    features=SkillFeatures(
        range_m=1.5,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="long_rest", max_uses=0,
    is_usable=lambda char: getattr(char, "lay_on_hands_pool", 0) > 0,
    builder=lambda actor, target, coord, char=None: {
        "type": "LAY_ON_HANDS", "caster": actor, "target": target,
        "amount": min(5, getattr(char, "lay_on_hands_pool", 5)) if char else 5,
        "consumes": ["action"],
    },
))


# ── 怪物天然能力（Wave 1，MONSTER_CATALOG §3）──────────────────────────────────

_register(Ability(
    skill_id="fire_breath",
    display_name="火焰吐息",
    description="9m 錐形（簡化 AOE）火焰，DEX 豁免半傷；充能 5-6（recharge trait）。",
    features=SkillFeatures(
        save_dc=17.0,                # 幼紅龍：8 + 熟練4 + CON5（引擎按 CON 實算）
        save_stat=SaveStat.DEX,
        range_m=9.0,
        aoe_radius_m=4.5,
        cost_action=1.0,
        target_type=TargetType.CONE,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1, refresh_on="never", max_uses=1,   # uses_flat + recharge 管次數
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火焰吐息",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))


# ── 怪物天然能力（Wave 2，MONSTER_CATALOG §3 C 級）────────────────────────────

_register(Ability(
    skill_id="lightning_breath",
    display_name="閃電吐息",
    description="6m 直線（寬 1.5m）閃電，DEX 豁免半傷；充能 5-6（recharge trait）。",
    features=SkillFeatures(
        save_dc=16.0,                # 貝希爾：8 + 熟練4 + CON4（引擎按 CON 實算）
        save_stat=SaveStat.DEX,
        range_m=6.0,                 # = line_length（瞄準點需在線上）
        aoe_radius_m=0.75,           # 線寬/2 — policy 友軍檢查與描述子用
        cost_action=1.0,
        target_type=TargetType.LINE,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1, refresh_on="never", max_uses=1,   # uses_flat + recharge 管次數
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "閃電吐息",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="lightning_bolt",
    display_name="閃電束",
    description="30m 直線（寬 1.5m）閃電，DEX 豁免，失敗 8d6，成功半傷。",
    features=SkillFeatures(
        save_dc=15.0,                # 模板值；materialize 按施法者實算
        save_stat=SaveStat.DEX,
        range_m=30.0,
        aoe_radius_m=0.75,           # 線寬/2
        cost_action=1.0,
        cost_slot_level=3.0,
        target_type=TargetType.LINE,
    ),
    engine_ready=True,
    min_level=5,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "閃電束",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="petrifying_gaze",
    display_name="石化凝視",
    description="9m 內單體 CON 豁免：失敗束縛（每回合末可重豁）；"
                "已束縛者再失敗則石化（永久）。",
    features=SkillFeatures(
        save_dc=12.0,                # 石化蜥蜴：8 + 熟練2 + CON2（引擎按 CON 實算）
        save_stat=SaveStat.CON,
        range_m=9.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("restrained", "petrified"),
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "石化凝視",
        "target": target,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="swallow",
    display_name="吞噬",
    description="對已束縛目標的單次撕咬；命中則吞入體內（束縛＋目盲＋"
                "每回合 6d6 強酸），吞噬者死亡時吐出；STR 豁免可掙脫。",
    features=SkillFeatures(
        # 貪心 policy 的每動作估值：單咬 3d10+6 (22.5) + 3 回合體內酸 (63)。
        # 引擎單回合永遠不會打出這個數字——descriptor 的 max_hit 由吐息
        # (66+) 主導，不受此慣例影響；status-dealt 的強酸份額在 dtype
        # 交叉驗證測試以 STATUS_DEALT_DTYPES 釘住（spirit_guardians 同例）。
        range_m=3.0,                 # 巨顎觸及 10ft
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("restrained", "blinded"),  # 吞噬的可見組件
        status_duration=10.0,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    requires_target_status="restrained",
    blocked_by_target_status="swallowed",
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target,
        "weapon": "貝希爾巨顎", "n_attacks": 1,
        "requires_target_status": "restrained",
        "blocked_by_target_status": "swallowed",
        "rider_status": "swallowed",
        "rider_save_each": "STR DC16",
        "rider_rounds": 10,
        "rider_metadata": {"tick_damage_dice": "6d6",
                           "tick_damage_type": "強酸",
                           "ends_if_source_dead": True},
        "consumes": ["action"],
    },
))


def _summon_ability(skill_id: str, display: str, monster_id: str, count: int,
                    *, max_uses: int = 1, min_level: int = 1) -> None:
    """Register a SUMMON ability: brings ``count`` × ``monster_id`` into the
    fight on the summoner's side (engine SUMMON handler). Data-driven so the
    summon wave can attach summon kits to summoner monsters without touching
    the engine. The summons ride the existing roster/obs (truncated past the
    slot budget like any pack) — no new obs dimension."""
    _register(Ability(
        skill_id=skill_id,
        display_name=display,
        description=f"召喚 {count} 隻 {monster_id} 加入戰鬥，與召喚者同陣營。",
        features=SkillFeatures(
            cost_action=1.0,
            remaining_uses=float(max_uses),
            target_type=TargetType.SELF,
        ),
        engine_ready=True,
        min_level=min_level, refresh_on="never", max_uses=max_uses,
        builder=lambda actor, target, coord, char=None: {
            "type": "SUMMON", "summoner": actor,
            "monster_id": monster_id, "count": count,
            "consumes": ["action"],
        },
    ))


# Concrete summon ability proving the build→execute path. Not yet on any
# monster sheet (the summon wave attaches summon kits to summoner monsters);
# registering it keeps the mechanism live and testable without altering any
# existing roster's balance.
_summon_ability("summon_wolf_pack", "召喚狼群", "wolf", 2, max_uses=1)


# Concrete lair action proving the lair-decision path (combat_policy.run_lair_
# actions). Environmental auto-hit damage on the nearest enemy — ignores the
# action economy (consumes nothing) and range/LoS (it's the lair itself). Not
# attached to any boss yet (the lair wave assigns lair_options per boss); the
# registry entry keeps the mechanism live and testable. engine_ready=True but
# it never enters a normal turn kit (no creature lists it in known_abilities).
_register(Ability(
    skill_id="lair_crushing_rocks",
    display_name="巢穴：落石",
    description="巢穴動作：落石砸向最近的敵人，自動命中造成 2d6 鈍擊傷害。",
    features=SkillFeatures(
        auto_hit=True,
        range_m=100.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: ({
        "type": "AUTO_DAMAGE", "attacker": actor,
        "targets": [{"id": target, "darts": 1}],
        "damage_per": "2d6", "damage_type": "鈍擊",
        "range_m": 100.0, "consumes": [],
    } if target else None),
))


# 眼魔射線效果表（5e MM 十射線；緩速/念力/石化按引擎狀態池近似——
# slowed＝半速−2AC、念力＝束縛 1 回合、石化＝束縛→petrified 兩段式）。
# 純資料：EYE_RAYS handler 與怪物名完全解耦。
EYE_RAY_TABLE: tuple[dict, ...] = (
    {"name": "魅惑射線", "save_stat": "WIS", "status": "charmed",    "rounds": 10},
    {"name": "麻痺射線", "save_stat": "CON", "status": "paralyzed",  "rounds": 10,
     "save_each": True},
    {"name": "恐懼射線", "save_stat": "WIS", "status": "frightened", "rounds": 10,
     "save_each": True},
    {"name": "緩速射線", "save_stat": "DEX", "status": "slowed",     "rounds": 10,
     "save_each": True},
    {"name": "衰弱射線", "save_stat": "CON", "damage_dice": "8d8",
     "damage_type": "黯蝕", "save_half": True},
    {"name": "念力射線", "save_stat": "STR", "status": "restrained", "rounds": 1},
    {"name": "沉睡射線", "save_stat": "WIS", "status": "asleep",     "rounds": 10},
    {"name": "石化射線", "save_stat": "DEX", "status": "restrained", "rounds": 10,
     "save_each": True, "escalates_to": "petrified"},
    {"name": "解離射線", "save_stat": "DEX", "damage_dice": "10d8",
     "damage_type": "力場"},
    {"name": "死亡射線", "save_stat": "DEX", "damage_dice": "10d10",
     "damage_type": "黯蝕"},
)

# Fail-loud data validation at import: a typo'd damage type or status name in
# the table must explode here, not silently no-op mid-fight.
def _validate_ray_table() -> None:
    from .damage import validate_damage_type
    from .status import ALL_STATUS_CLASSES
    for spec in EYE_RAY_TABLE:
        if spec.get("damage_type"):
            validate_damage_type(spec["damage_type"], context=f"eye ray {spec['name']}")
        for key in ("status", "escalates_to"):
            name = spec.get(key)
            if name and name not in ALL_STATUS_CLASSES:
                raise ValueError(f"eye ray {spec['name']}: unknown status {name!r}")
_validate_ray_table()

# ── 怪物天然能力（Wave 3，MONSTER_CATALOG §3 傳奇套件）────────────────────────

def _breath(skill_id: str, spell_name: str, dc: float,
            save: "SaveStat", rng: float, radius: float) -> None:
    """Register one breath-weapon ability riding the SPELL pipeline —
    fire_breath (Wave 1) pattern with per-dragon numbers. expected_damage and
    damage_types are derived by materialize from the SPELLS entry, not passed in."""
    _register(Ability(
        skill_id=skill_id,
        display_name=spell_name,
        description=f"{rng:.0f}m 錐形（簡化 AOE），豁免半傷；充能 5-6。",
        features=SkillFeatures(
            save_dc=dc,              # 模板值；引擎按 CON 實算
            save_stat=save,
            range_m=rng,
            aoe_radius_m=radius,
            cost_action=1.0,
            target_type=TargetType.CONE,
        ),
        engine_ready=True,
        scales_as_cantrip=False,
        min_level=1, refresh_on="never", max_uses=1,   # uses_flat + recharge
        builder=lambda actor, target, coord, char=None, _sn=spell_name: {
            "type": "SPELL", "caster": actor, "spell_name": _sn,
            "target_position": list(coord) if coord else [0.0, 0.0],
            "consumes": ["action"],
        },
    ))


_breath("cold_breath", "寒冰吐息", 19.0, SaveStat.CON,
        18.0, 9.0)                               # 成年白龍 12d8 冰
_breath("fire_breath_adult", "火焰吐息（成龍）", 21.0, SaveStat.DEX,
        18.0, 9.0)                               # 成年紅龍 18d6 火
_breath("fire_breath_ancient", "火焰吐息（古龍）", 24.0, SaveStat.DEX,
        27.0, 13.5)                              # 遠古紅龍 26d6 火

_register(Ability(
    skill_id="wing_attack",
    display_name="龍翼拍擊",
    description="傳奇行動（2 點）：以自身為中心 3m，DEX 豁免，失敗 2d6+8 "
                "鈍擊並倒地，成功無事。之後龍可飛行半速（未建模）。",
    features=SkillFeatures(
        save_dc=22.0,                # 模板值；引擎按 STR 實算（白19/紅22/古25）
        save_stat=SaveStat.DEX,
        range_m=3.0,                 # 自心 nova：敵在 3m 內才有意義
        aoe_radius_m=3.0,
        cost_action=1.0,
        target_type=TargetType.POINT,
        applies_status=_status_multihot("prone"),
        status_duration=10.0,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "龍翼拍擊",
        # 自心 nova：忽略瞄準座標，以自身位置為圓心。
        "target_position": ([char.position.x, char.position.y]
                            if char is not None else [0.0, 0.0]),
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="chill_touch",
    display_name="寒冰之觸",
    description="戲法：36m 內單體 DEX 豁免，失敗 1d8 黯蝕（隨等級縮放）。",
    features=SkillFeatures(
        save_dc=15.0,                # 模板值
        save_stat=SaveStat.DEX,
        range_m=36.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "寒冰之觸",
        "target": target,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="firebolt",
    display_name="火焰箭",
    description="戲法：36m 內單體 DEX 豁免，失敗 1d10 火焰（隨等級縮放）。",
    features=SkillFeatures(
        save_dc=15.0,                # 模板值；materialize 依施法者實算
        save_stat=SaveStat.DEX,
        range_m=36.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火焰箭",
        "target": target,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="disrupt_life",
    display_name="生命擾亂",
    description="傳奇行動（3 點）：以巫妖為中心 6m 內 CON 豁免，"
                "失敗 6d6 黯蝕，成功半傷。",
    features=SkillFeatures(
        save_dc=20.0,                # 巫妖：8 + 熟練7 + INT5（引擎實算）
        save_stat=SaveStat.CON,
        range_m=6.0,                 # 自心 nova：敵在 6m 內才有意義
        aoe_radius_m=6.0,
        cost_action=1.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "生命擾亂",
        "target_position": ([char.position.x, char.position.y]
                            if char is not None else [0.0, 0.0]),
        "consumes": ["action"],
    },
))

# 海妖閃電風暴：3 道同型落雷（distinct_rays=False 的單款射線表——
# EYE_RAYS handler 與眼魔共用，重複目標合法、單款表零擲表 RNG）。
KRAKEN_BOLT_TABLE: tuple[dict, ...] = (
    {"name": "落雷", "save_stat": "DEX", "damage_dice": "4d10",
     "damage_type": "閃電", "save_half": True},
)

_register(Ability(
    skill_id="lightning_storm",
    display_name="閃電風暴",
    description="召來 3 道落雷，各自打擊 36m 內隨機可見敵人；"
                "DEX 豁免，失敗 4d10 閃電，成功半傷。",
    features=SkillFeatures(
        save_dc=22.0,                # 海妖：8 + 熟練7 + CON7（引擎實算）
        save_stat=SaveStat.DEX,
        range_m=36.0,
        cost_action=1.0,
        target_type=TargetType.MULTI_ENEMY,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "EYE_RAYS", "caster": actor,
        "n_rays": 3, "range_m": 36.0, "dc_stat": "CON",
        "table": KRAKEN_BOLT_TABLE, "distinct_rays": False,
        "consumes": ["action"],
    },
))


def _swallow_variant(skill_id: str, display: str, *, weapon: str,
                     tick_dice: str, escape: str, reach: float) -> None:
    """Register a swallow-chain ability (behir Wave 2 pattern): single bite
    vs a restrained target → swallowed (restrained+blinded+酸 tick,
    regurgitated on swallower death). expected_damage / damage_types are
    derived by materialize from the bite weapon + tick_dice, not passed in."""
    _register(Ability(
        skill_id=skill_id,
        display_name=display,
        description="對已束縛目標的單次撕咬；命中則吞入體內（束縛＋目盲＋"
                    f"每回合 {tick_dice} 強酸），吞噬者死亡時吐出；"
                    "STR 豁免可掙脫。",
        features=SkillFeatures(
            range_m=reach,
            cost_action=1.0,
            target_type=TargetType.SINGLE_ENEMY,
            applies_status=_status_multihot("restrained", "blinded"),
            status_duration=10.0,
        ),
        engine_ready=True,
        scales_as_cantrip=False,
        min_level=1,
        requires_target_status="restrained",
        blocked_by_target_status="swallowed",
        builder=lambda actor, target, coord, char=None, _w=weapon, _e=escape, \
                _td=tick_dice: {
            "type": "ATTACK", "attacker": actor, "target": target,
            "weapon": _w, "n_attacks": 1,
            "requires_target_status": "restrained",
            "blocked_by_target_status": "swallowed",
            "rider_status": "swallowed",
            "rider_save_each": _e,
            "rider_rounds": 10,
            "rider_metadata": {"tick_damage_dice": _td,
                               "tick_damage_type": "強酸",
                               "ends_if_source_dead": True},
            "consumes": ["action"],
        },
    ))


# 海妖：咬 3d8+10＋體內 12d6/回合；逃脫 DC18（觸手擒抱 DC 同值）
_swallow_variant("kraken_swallow", "海妖吞噬", weapon="海妖巨口",
                 tick_dice="12d6", escape="STR DC18", reach=1.5)
# 泰拉斯克：咬 4d12+10＋體內 16d6/回合；逃脫 DC20
_swallow_variant("tarrasque_swallow", "泰拉斯克吞噬", weapon="泰拉斯克巨顎",
                 tick_dice="16d6", escape="STR DC20", reach=3.0)

_register(Ability(
    skill_id="eye_ray_single",
    display_name="眼魔射線（傳奇）",
    description="傳奇行動（1 點）：自十種效果表隨機射出 1 道射線。",
    features=SkillFeatures(
        save_dc=16.0,
        save_stat=SaveStat.DEX,
        range_m=36.0,
        cost_action=1.0,
        target_type=TargetType.MULTI_ENEMY,
        applies_status=_status_multihot(
            "charmed", "paralyzed", "frightened", "asleep", "restrained",
            "petrified"),
        status_duration=10.0,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "EYE_RAYS", "caster": actor,
        "n_rays": 1, "range_m": 36.0, "dc_stat": "INT",
        "table": EYE_RAY_TABLE,
        "consumes": ["action"],
    },
))

_register(Ability(
    skill_id="eye_rays",
    display_name="眼魔射線",
    description="每回合自十種效果表隨機射出 3 道互異射線，各自瞄準視野內"
                "隨機敵人（DC = 8 + 熟練 + INT）。",
    features=SkillFeatures(
        # 三道射線的期望總傷：傷害射線 (36+45+55)/10 × 3 = 40.8
        save_dc=16.0,                # 眼魔：8 + 熟練5 + INT3（引擎實算）
        save_stat=SaveStat.DEX,      # 眾數豁免（4/10 道）
        range_m=36.0,
        cost_action=1.0,
        target_type=TargetType.MULTI_ENEMY,
        applies_status=_status_multihot(
            "charmed", "paralyzed", "frightened", "asleep", "restrained",
            "petrified"),
        status_duration=10.0,
    ),
    engine_ready=True,
    scales_as_cantrip=False,
    min_level=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "EYE_RAYS", "caster": actor,
        "n_rays": 3, "range_m": 36.0, "dc_stat": "INT",
        "table": EYE_RAY_TABLE,
        "consumes": ["action"],
    },
))
