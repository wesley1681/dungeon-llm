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
    # Optional gate for resource pools that aren't tracked by ability_uses
    # (e.g. lay_on_hands_pool). Returns True when the ability can still be used.
    is_usable: Callable | None = None

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

_register(ClassAbility(
    skill_id="hunters_mark",
    display_name="獵人印記",
    class_id="ranger",
    description="bonus action：標記一個敵人，每次命中 +1d6 傷害，專注，消耗 1 環。",
    features=SkillFeatures(
        expected_damage=3.5,
        range_m=27.0,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=2, archetype_id="",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "hunters_mark", "spell_name": "獵人印記",
        "targets": [target], "max_targets": 1, "range_m": 27.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["bonus_action"],
    },
))


# ── Fighter: Battle Master additional maneuvers ───────────────────────────────

_register(ClassAbility(
    skill_id="menacing_attack",
    display_name="威嚇攻擊",
    class_id="fighter",
    description="武器攻擊 + WIS 豁免，失敗則 frightened 1 回合。消耗 1 個戰技骰。",
    features=SkillFeatures(
        expected_damage=7.5,
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
    min_level=3, archetype_id="battle_master",
    refresh_on="short_rest", max_uses=4,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "rider_save_dc": 14, "rider_save_stat": "WIS", "rider_status": "frightened",
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="precision_attack",
    display_name="精準攻擊",
    class_id="fighter",
    description="使用動作後，消耗 1 個戰技骰（d8），將骰值加到剛才的攻擊骰。",
    features=SkillFeatures(
        attack_vs_ac=4.5,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=False,
    engine_todo="需要攻擊後插入骰值的機制：引擎目前在擲骰前決定全部，無法事後加骰。",
    min_level=3, archetype_id="battle_master",
    refresh_on="short_rest", max_uses=4,
))

_register(ClassAbility(
    skill_id="pushing_attack",
    display_name="推擊攻擊",
    class_id="fighter",
    description="武器攻擊 + STR 豁免，失敗則推開目標 4.5m。消耗 1 個戰技骰。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        save_dc=14.0,
        save_stat=SaveStat.STR,
        cost_action=1.0,
        remaining_uses=4.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=False,
    engine_todo="需要擊退位移機制（將目標向外移動 4.5m），目前 MOVE 只能主動移動。",
    min_level=3, archetype_id="battle_master",
    refresh_on="short_rest", max_uses=4,
))

_register(ClassAbility(
    skill_id="improved_critical",
    display_name="強化暴擊",
    class_id="fighter",
    description="暴擊範圍擴大：d20=19 或 20 均視為暴擊。角色創建時設 crit_range=19。",
    features=SkillFeatures(target_type=TargetType.SELF),
    engine_ready=False,
    engine_todo="被動特性：角色創建時將 Character.crit_range 設為 19，無需 builder。",
    min_level=3, archetype_id="champion",
))


# ── Barbarian: Totem Warrior (Bear) ──────────────────────────────────────────

_register(ClassAbility(
    skill_id="bear_totem",
    display_name="熊圖騰",
    class_id="barbarian",
    description="狂暴時對所有傷害類型（除心靈傷害）獲得抗性。",
    features=SkillFeatures(
        damage_resistance=0.5,
        target_type=TargetType.SELF,
        remaining_uses=3.0,
        status_duration=10.0,
    ),
    engine_ready=False,
    engine_todo="需要擴展 Raging modifier：目前只對物理類型減半，熊圖騰需要覆蓋全部類型。",
    min_level=3, archetype_id="totem_bear",
    refresh_on="long_rest", max_uses=3,
))

_register(ClassAbility(
    skill_id="frenzy_attack",
    display_name="狂戰攻擊",
    class_id="barbarian",
    description="狂暴時每回合可用額外動作進行 1 次近戰武器攻擊。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=3, archetype_id="totem_bear",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="berserker_frenzy",
    display_name="狂戰狂暴攻擊",
    class_id="barbarian",
    description="狂暴時每回合可用額外動作進行 1 次近戰武器攻擊（長休後消除疲憊）。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=True,
    engine_todo="疲憊（Exhaustion）尚未建模；目前忽略疲憊代價。",
    min_level=3, archetype_id="berserker",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "consumes": ["bonus_action"],
    },
))


# ── Wizard: Evocation ─────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="burning_hands_ev",
    display_name="燃燒之手",
    class_id="wizard",
    description="3d6 火焰 AOE，4.5m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        expected_damage=10.5,
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=4.5,
        aoe_radius_m=4.5,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=1, archetype_id="evocation",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "燃燒之手",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="scorching_ray_ev",
    display_name="烈焰射線",
    class_id="wizard",
    description="3 道射線，每道各自進行攻擊骰，命中各造成 2d6 火焰傷害。",
    features=SkillFeatures(
        expected_damage=21.0,
        attack_vs_ac=5.0,
        range_m=27.0,
        cost_action=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.MULTI_ENEMY,
        max_targets=3,
    ),
    engine_ready=False,
    engine_todo="需要多條射線各自獨立進行攻擊骰的機制。",
    min_level=3, archetype_id="evocation",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="fireball_ev",
    display_name="火球術",
    class_id="wizard",
    description="8d6 火焰 AOE，6m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        expected_damage=28.0,
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=45.0,
        aoe_radius_m=6.0,
        cost_action=1.0,
        cost_slot_level=3.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=5, archetype_id="evocation",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火球術",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="web_ev",
    display_name="蜘蛛網",
    class_id="wizard",
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
    min_level=3, archetype_id="evocation",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "蜘蛛網",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="ice_storm_ev",
    display_name="冰風暴",
    class_id="wizard",
    description="4m 半徑 AOE，DEX 豁免，失敗 2d8 冰冷傷害，成功半傷。",
    features=SkillFeatures(
        expected_damage=9.0,
        save_dc=13.0,
        save_stat=SaveStat.DEX,
        range_m=36.0,
        aoe_radius_m=4.0,
        cost_action=1.0,
        cost_slot_level=4.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=7, archetype_id="evocation",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "冰風暴",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="hold_monster_ev",
    display_name="定怪術",
    class_id="wizard",
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
    min_level=8, archetype_id="evocation",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定怪術",
        "target": target, "consumes": ["action"],
    },
))


# ── Wizard: Divination ────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="burning_hands_div",
    display_name="燃燒之手",
    class_id="wizard",
    description="3d6 火焰 AOE，4.5m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        expected_damage=10.5,
        save_dc=13.0, save_stat=SaveStat.DEX,
        range_m=4.5, aoe_radius_m=4.5,
        cost_action=1.0, cost_slot_level=1.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=1, archetype_id="divination",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "燃燒之手",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="web_div",
    display_name="蜘蛛網",
    class_id="wizard",
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
    min_level=3, archetype_id="divination",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "蜘蛛網",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="fireball_div",
    display_name="火球術",
    class_id="wizard",
    description="8d6 火焰 AOE，6m 半徑，DEX 豁免，成功半傷。",
    features=SkillFeatures(
        expected_damage=28.0,
        save_dc=13.0, save_stat=SaveStat.DEX,
        range_m=45.0, aoe_radius_m=6.0,
        cost_action=1.0, cost_slot_level=3.0,
        target_type=TargetType.POINT,
    ),
    engine_ready=True,
    min_level=5, archetype_id="divination",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "火球術",
        "target_position": list(coord) if coord else [0.0, 0.0],
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="hold_monster_div",
    display_name="定怪術",
    class_id="wizard",
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
    min_level=8, archetype_id="divination",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "SPELL", "caster": actor, "spell_name": "定怪術",
        "target": target, "consumes": ["action"],
    },
))


# ── Cleric: Life Domain ───────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="healing_word_life",
    display_name="治療語",
    class_id="cleric",
    description="bonus action 遠距治療一個盟友 1d4 + WIS 修正 HP，射程 18m。",
    features=SkillFeatures(
        expected_healing=5.5,
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, archetype_id="life",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "HEAL", "caster": actor, "target": target,
        "dice": f"1d4+{char.stats.modifier(char.spellcasting_ability) if char and char.spellcasting_ability else 3}",
        "range_m": 18.0, "slot_level": 1,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="guiding_bolt_life",
    display_name="引導光彈",
    class_id="cleric",
    description="攻擊骰，命中造成 4d6 光耀傷害，下一個攻擊者擲優勢。",
    features=SkillFeatures(
        expected_damage=14.0,
        attack_vs_ac=5.0,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=False,
    engine_todo="需要法術攻擊骰（spell_attack 類型）；命中後附加優勢狀態尚未建模。",
    min_level=1, archetype_id="life",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="spiritual_weapon_life",
    display_name="精神武器",
    class_id="cleric",
    description="bonus action 召喚光能武器，每回合 bonus action 攻擊 1d8+WIS，非專注。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=False,
    engine_todo="需要持久性召喚物機制：每回合 bonus action 攻擊，非專注。",
    min_level=3, archetype_id="life",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="channel_divinity_preserve_life",
    display_name="引導神力：守護生命",
    class_id="cleric",
    description="消耗引導神力：30ft 內治療總量 5×牧師等級 HP，分配給多個目標。",
    features=SkillFeatures(
        expected_healing=15.0,
        range_m=9.0,
        cost_action=1.0,
        target_type=TargetType.MULTI_ALLY,
        max_targets=6,
        remaining_uses=1.0,
    ),
    engine_ready=False,
    engine_todo="需要多目標分配治療機制，且需扣除 channel_divinity 資源。",
    min_level=2, archetype_id="life",
    refresh_on="short_rest", max_uses=1,
))

_register(ClassAbility(
    skill_id="mass_cure_wounds_life",
    display_name="群體治療術",
    class_id="cleric",
    description="9m 內最多 6 個生物各回復 3d8 + WIS HP。",
    features=SkillFeatures(
        expected_healing=16.5,
        range_m=9.0,
        cost_action=1.0,
        cost_slot_level=5.0,
        target_type=TargetType.MULTI_ALLY,
        max_targets=6,
    ),
    engine_ready=False,
    engine_todo="需要多目標治療機制，目前 HEAL 只能單一目標。",
    min_level=8, archetype_id="life",
    refresh_on="never", max_uses=0,
))


# ── Cleric: War Domain ────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="guiding_bolt_war",
    display_name="引導光彈",
    class_id="cleric",
    description="攻擊骰，命中造成 4d6 光耀傷害，下一個攻擊者擲優勢。",
    features=SkillFeatures(
        expected_damage=14.0,
        attack_vs_ac=5.0,
        range_m=36.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=False,
    engine_todo="同 guiding_bolt_life：需要法術攻擊骰機制。",
    min_level=1, archetype_id="war",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="channel_divinity_guided_strike",
    display_name="引導神力：引導打擊",
    class_id="cleric",
    description="看到攻擊骰後，消耗引導神力，在骰值上 +10。",
    features=SkillFeatures(
        attack_vs_ac=10.0,
        cost_reaction=1.0,
        target_type=TargetType.SELF,
        remaining_uses=1.0,
    ),
    engine_ready=False,
    engine_todo="需要攻擊後插入加值的機制，且需扣除 channel_divinity 資源。",
    min_level=2, archetype_id="war",
    refresh_on="short_rest", max_uses=1,
))

_register(ClassAbility(
    skill_id="spiritual_weapon_war",
    display_name="精神武器",
    class_id="cleric",
    description="bonus action 召喚光能武器攻擊，非專注。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        range_m=18.0,
        cost_bonus=1.0,
        cost_slot_level=2.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=False,
    engine_todo="同 spiritual_weapon_life：需要持久召喚物機制。",
    min_level=3, archetype_id="war",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="war_priest_attack",
    display_name="戰爭祭司攻擊",
    class_id="cleric",
    description="使用動作攻擊後，可額外用 bonus action 再攻擊一次，WIS 次數每長休重置。",
    features=SkillFeatures(
        expected_damage=7.5,
        attack_vs_ac=5.0,
        cost_bonus=1.0,
        remaining_uses=3.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    min_level=6, archetype_id="war",
    refresh_on="long_rest", max_uses=3,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "consumes": ["bonus_action"],
    },
))


# ── Rogue: Assassin ───────────────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="cunning_action_dash",
    display_name="狡猾動作：衝刺",
    class_id="rogue",
    description="bonus action：本回合移動距離翻倍。",
    features=SkillFeatures(cost_bonus=1.0, range_m=9.0,
                           target_type=TargetType.POINT),
    engine_ready=True,
    min_level=2, archetype_id="assassin",
    refresh_on="never", max_uses=0,
    is_usable=_can_move,
    builder=lambda actor, target, coord, char=None: {
        "type": "MOVE", "character": actor,
        "target_position": [float(coord.x), float(coord.y)] if coord else [0.0, 0.0],
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="cunning_action_disengage",
    display_name="狡猾動作：脫身",
    class_id="rogue",
    description="bonus action：本回合移動不觸發藉機攻擊。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF),
    engine_ready=True,
    min_level=2, archetype_id="assassin",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "DISENGAGE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="cunning_action_hide",
    display_name="狡猾動作：躲藏",
    class_id="rogue",
    description="bonus action：嘗試躲藏（DEX DC12）。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF),
    engine_ready=True,
    min_level=2, archetype_id="assassin",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "HIDE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="assassinate",
    display_name="刺殺",
    class_id="rogue",
    description="對驚訝的敵人攻擊擲優勢，並自動視為暴擊。",
    features=SkillFeatures(attack_vs_ac=5.0, target_type=TargetType.SINGLE_ENEMY),
    engine_ready=False,
    engine_todo="需要驚訝狀態（surprised）追蹤：第一回合對方尚未行動視為驚訝。",
    min_level=3, archetype_id="assassin",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="uncanny_dodge_rogue",
    display_name="閃避直覺",
    class_id="rogue",
    description="REACTION：攻擊命中時，消耗反應將傷害減半。",
    features=SkillFeatures(cost_reaction=1.0, target_type=TargetType.SELF, damage_resistance=0.5),
    engine_ready=False,
    engine_todo="需要在攻擊命中後、傷害結算前觸發反應的機制。",
    min_level=5, archetype_id="assassin",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="evasion_rogue",
    display_name="閃避",
    class_id="rogue",
    description="DEX 豁免成功→0 傷，失敗→半傷。由 Evasion 狀態效果實現。",
    features=SkillFeatures(target_type=TargetType.SELF, damage_resistance=0.5),
    engine_ready=True,
    engine_todo="被動特性：角色創建時將 Evasion() 加入 status_effects，無需主動觸發。",
    min_level=7, archetype_id="assassin",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "evasion", "spell_name": "閃避",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": [],
    },
))


# ── Rogue: Arcane Trickster ───────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="cunning_action_dash_at",
    display_name="狡猾動作：衝刺",
    class_id="rogue",
    description="bonus action：本回合移動距離翻倍。",
    features=SkillFeatures(cost_bonus=1.0, range_m=9.0,
                           target_type=TargetType.POINT),
    engine_ready=True,
    min_level=2, archetype_id="arcane_trickster",
    refresh_on="never", max_uses=0,
    is_usable=_can_move,
    builder=lambda actor, target, coord, char=None: {
        "type": "MOVE", "character": actor,
        "target_position": [float(coord.x), float(coord.y)] if coord else [0.0, 0.0],
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="cunning_action_disengage_at",
    display_name="狡猾動作：脫身",
    class_id="rogue",
    description="bonus action：本回合移動不觸發藉機攻擊。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF),
    engine_ready=True,
    min_level=2, archetype_id="arcane_trickster",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "DISENGAGE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="cunning_action_hide_at",
    display_name="狡猾動作：躲藏",
    class_id="rogue",
    description="bonus action：進行躲藏（Stealth 對抗對手 Perception），成功則獲得 hidden 狀態。",
    features=SkillFeatures(cost_bonus=1.0, target_type=TargetType.SELF),
    engine_ready=True,
    min_level=2, archetype_id="arcane_trickster",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "HIDE", "character": actor,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="uncanny_dodge_at",
    display_name="閃避直覺",
    class_id="rogue",
    description="REACTION：攻擊命中時傷害減半。",
    features=SkillFeatures(cost_reaction=1.0, target_type=TargetType.SELF, damage_resistance=0.5),
    engine_ready=False,
    engine_todo="同 uncanny_dodge_rogue：需要命中後插入反應的路徑。",
    min_level=5, archetype_id="arcane_trickster",
    refresh_on="never", max_uses=0,
))

_register(ClassAbility(
    skill_id="evasion_at",
    display_name="閃避",
    class_id="rogue",
    description="DEX 豁免成功→0 傷，失敗→半傷。",
    features=SkillFeatures(target_type=TargetType.SELF, damage_resistance=0.5),
    engine_ready=True,
    engine_todo="被動：角色創建時加 Evasion() 到 status_effects。",
    min_level=7, archetype_id="arcane_trickster",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "evasion", "spell_name": "閃避",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": [],
    },
))


# ── Paladin: Oath of Devotion ─────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="divine_smite_dev",
    display_name="神聖打擊",
    class_id="paladin",
    description="命中後消耗 1 環法術位，每環 +2d8 光耀傷害（max 5d8）。",
    features=SkillFeatures(
        expected_damage=9.0,
        attack_vs_ac=5.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    engine_todo="傷害固定 2d8（1 環）；之後可按 slot 等級縮放。",
    min_level=2, archetype_id="devotion",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "divine_smite_slot": 1,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="lay_on_hands_ability",
    display_name="聖療之手",
    class_id="paladin",
    description="觸碰治療：從 5×等級 HP 的資源池中恢復指定量。",
    features=SkillFeatures(
        expected_healing=15.0,
        range_m=1.5,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, archetype_id="devotion",
    refresh_on="long_rest", max_uses=0,
    is_usable=lambda char: getattr(char, "lay_on_hands_pool", 0) > 0,
    builder=lambda actor, target, coord, char=None: {
        "type": "LAY_ON_HANDS", "caster": actor, "target": target,
        "amount": min(5, getattr(char, "lay_on_hands_pool", 5)) if char else 5,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="shield_of_faith_dev",
    display_name="信仰護盾",
    class_id="paladin",
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
    min_level=2, archetype_id="devotion",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "shield_of_faith", "spell_name": "信仰護盾",
        "targets": [target], "max_targets": 1, "range_m": 18.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="sacred_weapon_dev",
    display_name="神聖武器",
    class_id="paladin",
    description="引導神力：1 分鐘內武器攻擊 +CHA 修正（近似 +3）。",
    features=SkillFeatures(
        conferred_attack_mod=3.0,
        cost_action=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SELF,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, archetype_id="devotion",
    refresh_on="short_rest", max_uses=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "sacred_weapon_buff", "spell_name": "神聖武器",
        "targets": [actor], "max_targets": 1, "range_m": 0.0,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="wrathful_smite_dev",
    display_name="憤怒打擊",
    class_id="paladin",
    description="命中後額外 1d6 精神傷害，WIS 豁免失敗則 frightened，專注。",
    features=SkillFeatures(
        expected_damage=3.5,
        save_dc=13.0,
        save_stat=SaveStat.WIS,
        cost_bonus=1.0,
        cost_slot_level=1.0,
        requires_concentration=True,
        target_type=TargetType.SINGLE_ENEMY,
        applies_status=_status_multihot("frightened"),
        status_duration=1.0,
    ),
    engine_ready=False,
    engine_todo="需要命中後觸發附加傷害+豁免的 bonus_action 打擊機制。",
    min_level=2, archetype_id="devotion",
    refresh_on="never", max_uses=0,
))


# ── Paladin: Oath of Vengeance ────────────────────────────────────────────────

_register(ClassAbility(
    skill_id="divine_smite_ven",
    display_name="神聖打擊",
    class_id="paladin",
    description="命中後消耗法術位，每環 +2d8 光耀傷害。",
    features=SkillFeatures(
        expected_damage=9.0,
        attack_vs_ac=5.0,
        cost_action=1.0,
        cost_slot_level=1.0,
        target_type=TargetType.SINGLE_ENEMY,
    ),
    engine_ready=True,
    engine_todo="固定 1 環（2d8）；之後可擴展為按 slot 等級縮放。",
    min_level=2, archetype_id="vengeance",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "ATTACK", "attacker": actor, "target": target, "weapon": "長劍",
        "divine_smite_slot": 1,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="bane_ven",
    display_name="詛咒術",
    class_id="paladin",
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
    min_level=2, archetype_id="vengeance",
    refresh_on="never", max_uses=0,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "baned", "spell_name": "詛咒術",
        "targets": (target.split(",") if isinstance(target, str) else list(target or [])),
        "max_targets": 3, "range_m": 9.0,
        "slot_level": 1, "requires_concentration": True,
        "consumes": ["action"],
    },
))

_register(ClassAbility(
    skill_id="vow_of_enmity_ven",
    display_name="仇敵誓言",
    class_id="paladin",
    description="引導神力：聖騎士對選定目標的攻擊擲優勢，持續 1 分鐘。",
    features=SkillFeatures(
        conferred_attack_mod=5.0,
        cost_bonus=1.0,
        remaining_uses=1.0,
        target_type=TargetType.SINGLE_ENEMY,
        status_duration=10.0,
    ),
    engine_ready=True,
    min_level=3, archetype_id="vengeance",
    refresh_on="short_rest", max_uses=1,
    builder=lambda actor, target, coord, char=None: {
        "type": "APPLY_MOD", "caster": actor,
        "modifier": "vow_target", "spell_name": "仇敵誓言",
        "targets": [target], "max_targets": 1, "range_m": 18.0,
        "consumes": ["bonus_action"],
    },
))

_register(ClassAbility(
    skill_id="lay_on_hands_ability_ven",
    display_name="聖療之手",
    class_id="paladin",
    description="觸碰治療，5×等級 HP 資源池。",
    features=SkillFeatures(
        expected_healing=15.0,
        range_m=1.5,
        cost_action=1.0,
        target_type=TargetType.SINGLE_ALLY,
    ),
    engine_ready=True,
    min_level=1, archetype_id="vengeance",
    refresh_on="long_rest", max_uses=0,
    is_usable=lambda char: getattr(char, "lay_on_hands_pool", 0) > 0,
    builder=lambda actor, target, coord, char=None: {
        "type": "LAY_ON_HANDS", "caster": actor, "target": target,
        "amount": min(5, getattr(char, "lay_on_hands_pool", 5)) if char else 5,
        "consumes": ["action"],
    },
))
