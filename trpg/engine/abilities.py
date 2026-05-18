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


@dataclass
class ClassAbility:
    """Static description of a class ability."""
    skill_id: str
    display_name: str
    class_id: str             # "fighter", "wizard", "cleric", "barbarian", "bard"
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
    archetype_id: str = ""      # "" = base class; subclass label for archetype abilities
    # Rest recovery: "short_rest" | "long_rest" | "never"
    # Informs rest_character() how to replenish uses_remaining.
    refresh_on: str = "short_rest"
    # Maximum uses per refresh period (0 = unlimited / passive).
    max_uses: int = 0


# ── Catalog ──────────────────────────────────────────────────────────────────

CLASS_ABILITIES: dict[str, ClassAbility] = {}


def _register(ab: ClassAbility) -> ClassAbility:
    CLASS_ABILITIES[ab.skill_id] = ab
    return ab


# ── Fighter (戰士) ───────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="second_wind",
    display_name="二度氣息",
    class_id="fighter",
    description="bonus action 回復 1d10 + 戰士等級 HP，每短休 1 次。",
    refresh_on="short_rest", max_uses=1,
    features=SkillFeatures(
        expected_healing=10.0,    # ~5.5 + level-3 placeholder
        cost_bonus=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
    ),
    engine_ready=True,
    min_level=1, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": actor,
        "dice": f"1d10+{char.level if char else 3}", "range_m": 0.0,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="action_surge",
    display_name="動作激增",
    class_id="fighter",
    description="本回合多獲得 1 個動作，每短休 1 次。",
    refresh_on="short_rest", max_uses=1,
    features=SkillFeatures(
        grants_actions=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
    ),
    engine_ready=True,
    min_level=2, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "ACTION_SURGE", "character": actor,
    },
    engine_todo="short-rest uses_per pool not yet tracked; agent can spam",
))

_register(ClassAbility(
    skill_id="trip_attack",
    display_name="絆倒攻擊",
    class_id="fighter",
    description="武器攻擊 + 命中後 STR 豁免，失敗則 prone。消耗 1 個戰技骰。",
    refresh_on="short_rest", max_uses=4,
    features=SkillFeatures(
        expected_damage=7.5,
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
    min_level=3, archetype_id="battle_master",
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "rider_save_dc": 14, "rider_save_stat": "STR", "rider_status": "prone",
        "consumes": ["action"],
    },
    engine_todo="weapon name hard-coded to 長劍; superiority die pool not tracked",
))


# ── Wizard (法師) ────────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="magic_missile",
    display_name="魔法飛彈",
    class_id="wizard",
    description="3 道力場箭自動命中所選敵人（最多 3 個），每箭 1d4+1。",
    features=SkillFeatures(
        expected_damage=10.5,
        auto_hit=True,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.MULTI_ENEMY,
        max_targets=3,
    ),
    engine_ready=True,
    min_level=1, archetype_id="",
    builder=lambda actor, target, coord, char=None: (
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

_register(ClassAbility(
    skill_id="shield_spell",
    display_name="法盾",
    class_id="wizard",
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
    min_level=1, archetype_id="",
    engine_todo="smart-fire only triggers when +5 AC would flip a hit to miss; "
                "Magic Missile is always blocked. No 'always cast' option yet.",
))

_register(ClassAbility(
    skill_id="hold_person",
    display_name="定身術",
    class_id="wizard",
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
    min_level=3, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定身術",
        "target": target, "consumes": ["action"],
    },
    engine_todo="paralyzed status is attached as a label; the actual D&D "
                "paralyzed effects (incoming attacks have advantage + crit on "
                "hit within 1.5m) are not yet hooked into resolve_attack",
))

_register(ClassAbility(
    skill_id="misty_step",
    display_name="霧步",
    class_id="wizard",
    description="bonus action 瞬移最多 9m，無視視線與障礙。",
    features=SkillFeatures(
        range_m=9.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.POINT,
        is_teleport=True,
    ),
    engine_ready=True,
    min_level=3, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "MOVE", "character": actor,
        "target_position": list(coord) if coord else [0.0, 0.0],
        "teleport": True, "range_m": 9.0, "slot_level": 2,
        "consumes": ["bonus_action"],
    },
))


# ── Cleric (牧師) ────────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="cure_wounds",
    display_name="治療術",
    class_id="cleric",
    description="觸碰範圍治療一個盟友 1d8 + WIS 修正。",
    features=SkillFeatures(
        expected_healing=7.5,         # 1d8 + ~3 (WIS mod)
        range_m=1.5,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": target,
        "dice": "1d8+3", "range_m": 1.5, "slot_level": 1,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="sacred_flame",
    display_name="神聖光輝",
    class_id="cleric",
    description="單一目標 DEX 豁免，失敗則 1d8 光耀傷害。",
    features=SkillFeatures(
        expected_damage=4.5,
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=18.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=1, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "神聖光輝",
        "target": target, "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="bless",
    display_name="祝福術",
    class_id="cleric",
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
    min_level=1, archetype_id="",
    # `target` is a comma-separated list of ally ids ("aria,thor"); the
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

_register(ClassAbility(
    skill_id="rage",
    display_name="狂暴",
    class_id="barbarian",
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
    min_level=1, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "raging", "spell_name": "狂暴",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": ["bonus_action"],
    },
    engine_todo="rage daily uses not tracked; CON-save-advantage not modelled",
))

_register(ClassAbility(
    skill_id="reckless_attack",
    display_name="魯莽攻擊",
    refresh_on="never", max_uses=0,
    class_id="barbarian",
    description="declared on a melee weapon attack — this attack has advantage, "
                "and all incoming attacks have advantage until your next turn.",
    features=SkillFeatures(
        expected_damage=7.5,           # 1d8 + STR mod (uses the long sword)
        attack_vs_ac=5.0,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        conferred_attack_mod=5.0,      # advantage ≈ +5 statistically
        conferred_ac_mod=-5.0,         # incoming attacks have advantage
        status_duration=1.0,
    ),
    engine_ready=True,
    min_level=2, archetype_id="",
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target,
        "weapon": "長劍", "reckless": True,
        "consumes": ["action"],
    },
    engine_todo="weapon hard-coded to 長劍; works correctly for any melee "
                "weapon since the reckless flag is weapon-agnostic",
))

# ── Shared resource sentinels ─────────────────────────────────────────────────
# Meta-abilities representing shared resource pools. Listed in known_abilities
# so rest_character() resets them via the normal refresh_on mechanism.

_register(ClassAbility(
    skill_id="channel_divinity",
    display_name="引導神力",
    class_id="cleric",
    description="每短休 1 次的神力引導資源池（牧師 L2）。",
    features=SkillFeatures(target_type=TargetType.SELF),
    engine_ready=False,
    refresh_on="short_rest", max_uses=1,
    min_level=2, archetype_id="",
))
