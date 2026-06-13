"""Wave 0 monster registry: a monster IS a ClassDef (MONSTER_CATALOG.md §5).

Every monster here is PURE ASSEMBLY — chassis numbers + existing abilities +
existing traits. Zero new engine mechanics (that's the Wave 0 contract).
Statblocks follow the 5e MM: because the engine computes to-hit as
``d20 + stat_mod + proficiency_bonus`` and prof uses the same (level-1)//4+2
table as 5e CR, setting ``natural_level = max(1, round(cr))`` reproduces MM
attack bonuses and save DCs from the stat panel alone.

Known approximations (documented, accepted for Wave 0):
  - Multiattack mixes (beak+claw) use one natural weapon × attacks_per_action;
    expected damage stays within ~15% of MM.
  - Orc "Aggressive" ≈ cunning_action_dash; Goblin "Nimble Escape" =
    cunning_action_disengage/hide (exact mechanic).
  - Bandit's light crossbow → 短弓 proxy (1d6 vs 1d8).
  - Mage/Archmage kits draw from the in-repo spell registry (L1-L8 catalog),
    so the Archmage is the "部分" build flagged in MONSTER_CATALOG §2.
  - Tail spikes / rocks are ammo-less (fights are short; MM ammo counts moot).
  Wave 2:
  - Troll multiattack (bite+2 claws) = 巨魔之爪×3; no 0-HP re-regeneration
    (engine monsters die at 0 — the fire/acid decision lives in the per-turn
    regen suppression instead).
  - Behir multiattack (bite+constrict) = 緊勒×2; constrict auto-grapples on
    hit (no contest roll), STR DC16 save_each = MM escape DC; swallow is a
    single bite vs a restrained target; 體內酸傷 ticks at the VICTIM's turn
    start (MM: behir's turn) — same per-round cadence; escape by save instead
    of the 30-damage regurgitate rule.
  - Basilisk gaze is an active action (MM: passive start-of-turn trigger);
    two-stage restrained→petrified per failed save while restrained.
  - Beholder is grounded (no flight), antimagic cone deferred (catalog §3 E);
    slow ray ≈ half speed −2 AC; telekinetic ray ≈ restrained 1 round;
    charm/sleep rays don't model the "harmed → effect breaks" clause.
  Wave 3 (legendary suite):
  - Dragon multiattack (bite+2 claws) = 龍之爪牙 2d8×3 (within ~6% of the MM
    mix for white/red/ancient); the bite's fire rider (red) is not modelled —
    the breath carries the fire threat. 龍尾 2d8+STR reproduces all three
    tail attacks exactly. Wing attack uses one shared spell entry at the red
    dragon's 2d6+8 (white +6 / ancient +10 differ by ±2); the follow-up
    half-speed fly is not modelled (grounded dragons).
  - Frightful presence is a victim-turn-start aura (MM: rider on multiattack)
    — save cadence ≈ once/round until first success, immune for the combat
    thereafter; escaping the frightened status via the end-of-turn re-save
    does NOT grant immunity (next turn-start check rolls again). No LoS gate.
  - Legendary resistance burns ONLY on saves whose failure leaves a status
    (action-economy protection, the MM tactical intent) — never to halve
    damage; concentration saves are also outside it. Pool = per combat.
  - Legendary actions: one option per trigger, EV-per-cost greedy, no
    movement spent (out-of-reach options simply don't fire); dragon "Detect"
    and tarrasque "Move" options are dropped (perception/positioning no-ops
    in this engine). Lich's legendary cantrip = 寒冰之觸 (save-based proxy
    for the spell-attack cantrip, 神聖光輝 precedent).
  - Tarrasque multiattack (bite+2 claws+horns+tail) = 泰拉斯克巨顎×4 with the
    bite's grapple rider on every swing (MM: bite only); reflective carapace
    deferred (catalog §3 E). Kraken tentacle reach 9m, grapple-on-hit like
    behir; Fling and ink cloud are not modelled; lightning storm = 3 bolts
    at random visible enemies (MM: kraken chooses), DC 22 (CON-derived; MM
    lists 23). Kraken/lich/tarrasque physical immunity (nonmagical) → 0.5
    resistance proxy, the werewolf precedent.
  - Balor whip (reach + pull) is not modelled — sword-only multiattack; the
    forced-move pull primitive is deferred until a second user shows up
    (design vigilance ≥2 rule).

HP uses ``hp_base = MM HP, hp_per_level = 0`` — a monster's HP is its
statblock, whatever level the env passes. ``natural_level`` is what callers
should pass as the monster's level (prof/DC fidelity + obs threat scale);
the CR→「等效對位等級」mapping is measured separately (scripts/calibrate_cr.py)
and lives in MONSTER_CATALOG, not here.

Registration protocol (MUST mirror scripts/synth_identity.py): importing this
module has NO side effects; call ``register_monsters()`` which first imports
``trpg.rl.obs`` to freeze N_ARCHETYPES, so the obs one-hot stays 12-wide and
every monster reads as all-zero — the monster condition the blind student
already handles.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..engine.items import Weapon
from .archetypes import (
    CLASS_DEFS, ARCHETYPE_FACTORIES, ARCHETYPE_ROLES,
    ClassDef, SkillGrant, TraitGrant, _factory, make_character,
)


@dataclass(frozen=True)
class MonsterDef(ClassDef):
    """A ClassDef plus the monster-book metadata the classes don't need."""
    cr: float = 0.0
    natural_level: int = 1   # max(1, round(cr)) — reproduces MM prof/DC


# ── Natural weapons (merged into WEAPON_DEFS on register) ────────────────────

MONSTER_WEAPONS: dict[str, Weapon] = {
    "棍棒":     Weapon("棍棒",     "1d4",  "鈍擊", "近戰", range_normal=1.5),
    "巨斧":     Weapon("巨斧",     "1d12", "斬擊", "近戰", range_normal=1.5),
    "巨棒":     Weapon("巨棒",     "2d8",  "鈍擊", "近戰", range_normal=1.5),
    "巨人巨棒": Weapon("巨人巨棒", "3d8",  "鈍擊", "近戰", range_normal=3.0),
    "巨型戰斧": Weapon("巨型戰斧", "2d8",  "斬擊", "近戰", range_normal=1.5),
    "熊爪":     Weapon("熊爪",     "2d8",  "斬擊", "近戰", range_normal=1.5),
    "利爪":     Weapon("利爪",     "1d6",  "斬擊", "近戰", range_normal=1.5),
    "尾刺":     Weapon("尾刺",     "1d8",  "穿刺", "遠程",
                       range_normal=30, range_long=60),
    "擲岩":     Weapon("擲岩",     "3d10", "鈍擊", "遠程",
                       range_normal=18, range_long=72),
    # ── Wave 1：帶 on-hit rider 的天然武器 ────────────────────────────────
    "狼牙":     Weapon("狼牙",     "2d4",  "穿刺", "近戰", range_normal=1.5,
                       properties=["精巧"],
                       on_hit={"status": "prone", "save_stat": "STR",
                               "save_dc": 11}),
    "巨狼牙":   Weapon("巨狼牙",   "2d6",  "穿刺", "近戰", range_normal=1.5,
                       on_hit={"status": "prone", "save_stat": "STR",
                               "save_dc": 13}),
    "鬼爪":     Weapon("鬼爪",     "2d4",  "斬擊", "近戰", range_normal=1.5,
                       properties=["精巧"],
                       on_hit={"status": "paralyzed", "save_stat": "CON",
                               "save_dc": 10, "save_each": "CON DC10",
                               "rounds": 10}),
    "暗影之觸": Weapon("暗影之觸", "2d6",  "黯蝕", "近戰", range_normal=1.5,
                       properties=["精巧"],
                       on_hit={"drain_stat": ("STR", "1d4")}),
    "生命吸取": Weapon("生命吸取", "1d6",  "黯蝕", "近戰", range_normal=1.5,
                       on_hit={"drain_max_hp": True}),
    "石爪":     Weapon("石爪",     "1d6",  "斬擊", "近戰", range_normal=1.5),
    "重擊":     Weapon("重擊",     "1d6",  "鈍擊", "近戰", range_normal=1.5),
    "飛龍螫刺": Weapon("飛龍螫刺", "2d6",  "穿刺", "近戰", range_normal=3.0,
                       on_hit={"damage_dice": "7d6", "damage_type": "毒",
                               "save_stat": "CON", "save_dc": 15,
                               "save_half": True}),
    "火焰之觸": Weapon("火焰之觸", "2d6",  "火",   "近戰", range_normal=1.5,
                       properties=["精巧"],
                       on_hit={"damage_dice": "1d10", "damage_type": "火"}),
    "龍之爪牙": Weapon("龍之爪牙", "2d8",  "斬擊", "近戰", range_normal=3.0),
    # ── Wave 2 ────────────────────────────────────────────────────────────
    "巨魔之爪": Weapon("巨魔之爪", "2d6",  "斬擊", "近戰", range_normal=1.5),
    # 緊勒：2d10+6 鈍擊 ＋ 2d10+6 斬擊 rider（MM 雙包）＋無豁免擒抱
    # （簡化 5e 擒抱對抗檢定；STR DC16 每回合末掙脫 = MM 逃脫 DC）。
    "貝希爾緊勒": Weapon("貝希爾緊勒", "2d10", "鈍擊", "近戰", range_normal=1.5,
                       on_hit={"damage_dice": "2d10+6", "damage_type": "斬擊",
                               "status": "restrained",
                               "save_each": "STR DC16", "rounds": 10}),
    # 巨顎不進 behir.weapons（緊勒才是 multiattack 主體）——只作為吞噬
    # ability 的攻擊載具，由 get_weapon 的 WEAPON_DEFS 全域回退取得。
    "貝希爾巨顎": Weapon("貝希爾巨顎", "3d10", "穿刺", "近戰", range_normal=3.0),
    "蜥蜴巨口": Weapon("蜥蜴巨口", "2d6",  "穿刺", "近戰", range_normal=1.5,
                       on_hit={"damage_dice": "2d6", "damage_type": "毒"}),
    "眼魔巨口": Weapon("眼魔巨口", "4d6",  "穿刺", "近戰", range_normal=1.5),
    # ── Wave 3：傳奇套件 ──────────────────────────────────────────────────
    # 龍尾：傳奇行動共用（武器骰不含屬性調整 → 白 +6／紅 +8／古 +10 由
    # STR 自動補上，三龍各自精確對 MM）。不入 weapons 清單（傳奇行動專屬，
    # 多重攻擊主體是爪牙）——behir 巨顎同模式，經 get_weapon 全域回退取得。
    "龍尾":       Weapon("龍尾", "2d8", "鈍擊", "近戰", range_normal=4.5),
    # 麻痺之觸：MM 為法術攻擊 +12（INT）——武器管線最接近的是精巧（DEX3+熟練7
    # = +10）；DC 18 = MM 字面值。
    "麻痺之觸":   Weapon("麻痺之觸", "3d6", "冰", "近戰", range_normal=1.5,
                       properties=["精巧"],
                       on_hit={"status": "paralyzed", "save_stat": "CON",
                               "save_dc": 18, "save_each": "CON DC18",
                               "rounds": 10}),
    # 海妖觸手：命中即擒（無對抗檢定，STR DC18 每回合末掙脫＝MM 逃脫 DC）
    "海妖觸手":   Weapon("海妖觸手", "3d6", "鈍擊", "近戰", range_normal=9.0,
                       on_hit={"status": "restrained",
                               "save_each": "STR DC18", "rounds": 10}),
    "海妖巨口":   Weapon("海妖巨口", "3d8", "穿刺", "近戰", range_normal=1.5),
    "泰拉斯克巨顎": Weapon("泰拉斯克巨顎", "4d12", "穿刺", "近戰",
                       range_normal=3.0,
                       on_hit={"status": "restrained",
                               "save_each": "STR DC20", "rounds": 10}),
    "泰拉斯克巨爪": Weapon("泰拉斯克巨爪", "4d8", "斬擊", "近戰",
                       range_normal=4.5),
    "炎魔火劍":   Weapon("炎魔火劍", "3d8", "斬擊", "近戰", range_normal=3.0,
                       on_hit={"damage_dice": "3d6", "damage_type": "閃電"}),
}

# 物理三型非魔法武器抗性（多數靈體/魔像/元素），單一定義避免逐怪重打
_PHYS_RESIST = {"斬擊": 0.5, "穿刺": 0.5, "鈍擊": 0.5}


# ── Wave 0 monster definitions (5e MM statblocks) ────────────────────────────

_MONSTER_DEF_LIST = [
    MonsterDef(
        archetype_id="commoner", default_name="平民",
        class_display="怪物", role="front", cr=0.0, natural_level=1,
        stat_block=dict(STR=10, DEX=10, CON=10, INT=10, WIS=10, CHA=10),
        hp_base=4, hp_per_level=0, ac=10,
        weapons=("棍棒",),
    ),
    MonsterDef(
        archetype_id="bandit", default_name="強盜",
        class_display="怪物", role="striker", cr=0.125, natural_level=1,
        stat_block=dict(STR=11, DEX=12, CON=12, INT=10, WIS=10, CHA=10),
        hp_base=11, hp_per_level=0, ac=12,
        weapons=("彎刀", "短弓"),
        consumables=(("箭", 20, "ammo", ""),),
    ),
    MonsterDef(
        archetype_id="goblin", default_name="哥布林",
        class_display="怪物", role="striker", cr=0.25, natural_level=1,
        stat_block=dict(STR=8, DEX=14, CON=10, INT=10, WIS=8, CHA=8),
        hp_base=7, hp_per_level=0, ac=15,
        weapons=("彎刀", "短弓"),
        consumables=(("箭", 20, "ammo", ""),),
        # Nimble Escape — the exact rogue mechanic, granted at L1.
        skills=(
            SkillGrant("cunning_action_disengage"),
            SkillGrant("cunning_action_hide"),
        ),
    ),
    MonsterDef(
        archetype_id="orc", default_name="獸人",
        class_display="怪物", role="front", cr=0.5, natural_level=1,
        stat_block=dict(STR=16, DEX=12, CON=16, INT=7, WIS=11, CHA=10),
        hp_base=15, hp_per_level=0, ac=13,
        weapons=("巨斧",),
        # Aggressive (bonus move toward enemy) ≈ bonus-action dash.
        skills=(SkillGrant("cunning_action_dash"),),
    ),
    MonsterDef(
        archetype_id="ogre", default_name="食人魔",
        class_display="怪物", role="front", cr=2.0, natural_level=2,
        stat_block=dict(STR=19, DEX=8, CON=16, INT=5, WIS=7, CHA=7),
        hp_base=59, hp_per_level=0, ac=11,
        weapons=("巨棒",),
    ),
    MonsterDef(
        archetype_id="owlbear", default_name="梟熊",
        class_display="怪物", role="front", cr=3.0, natural_level=3,
        stat_block=dict(STR=20, DEX=12, CON=17, INT=3, WIS=12, CHA=7),
        hp_base=59, hp_per_level=0, ac=13,
        weapons=("熊爪",),
        traits=(TraitGrant("extra_attack", params={"attacks": 2}),),
    ),
    MonsterDef(
        archetype_id="manticore", default_name="蠍尾獅",
        class_display="怪物", role="striker", cr=3.0, natural_level=3,
        stat_block=dict(STR=17, DEX=16, CON=17, INT=7, WIS=12, CHA=8),
        hp_base=68, hp_per_level=0, ac=14,
        weapons=("利爪", "尾刺"),
        traits=(TraitGrant("extra_attack", params={"attacks": 3}),),
    ),
    MonsterDef(
        archetype_id="ettin", default_name="雙頭巨人",
        class_display="怪物", role="front", cr=4.0, natural_level=4,
        stat_block=dict(STR=21, DEX=8, CON=17, INT=6, WIS=10, CHA=8),
        hp_base=85, hp_per_level=0, ac=12,
        weapons=("巨型戰斧",),
        traits=(TraitGrant("extra_attack", params={"attacks": 2}),),
    ),
    MonsterDef(
        archetype_id="hill_giant", default_name="丘陵巨人",
        class_display="怪物", role="front", cr=5.0, natural_level=5,
        stat_block=dict(STR=21, DEX=8, CON=19, INT=5, WIS=9, CHA=6),
        hp_base=105, hp_per_level=0, ac=13,
        weapons=("巨人巨棒", "擲岩"),
        traits=(TraitGrant("extra_attack", params={"attacks": 2}),),
    ),
    MonsterDef(
        archetype_id="mage_npc", default_name="法師",
        class_display="怪物", role="striker", cr=6.0, natural_level=6,
        stat_block=dict(STR=9, DEX=14, CON=11, INT=17, WIS=12, CHA=11),
        hp_base=40, hp_per_level=0, ac=12,
        weapons=("匕首",),
        spell_ability="INT",
        # Fixed full panel at any level — a monster's slots are its statblock.
        spell_slots_table={1: {1: 4, 2: 3, 3: 3, 4: 1}},
        skills=(
            SkillGrant("magic_missile"),
            SkillGrant("shield_spell", reaction=True),
            SkillGrant("burning_hands_ev"),
            SkillGrant("misty_step"),
            SkillGrant("hold_person"),
            SkillGrant("fireball_ev"),
            SkillGrant("ice_storm_ev"),
        ),
    ),
    MonsterDef(
        archetype_id="archmage", default_name="大法師",
        class_display="怪物", role="striker", cr=12.0, natural_level=12,
        stat_block=dict(STR=10, DEX=14, CON=12, INT=20, WIS=15, CHA=16),
        hp_base=99, hp_per_level=0, ac=12,
        weapons=("匕首",),
        spell_ability="INT",
        spell_slots_table={1: {1: 4, 2: 3, 3: 3, 4: 3}},
        skills=(
            SkillGrant("magic_missile"),
            SkillGrant("shield_spell", reaction=True),
            SkillGrant("misty_step"),
            SkillGrant("hold_person"),
            SkillGrant("fireball_ev"),
            SkillGrant("ice_storm_ev"),
            # Wave 2: LINE 語義的第二個活體用戶（補高環法術之一）
            SkillGrant("lightning_bolt"),
        ),
    ),
    # ── Wave 1（B 級原語落地，MONSTER_CATALOG §5）──────────────────────────
    MonsterDef(
        archetype_id="kobold", default_name="狗頭人",
        class_display="怪物", role="striker", cr=0.125, natural_level=1,
        stat_block=dict(STR=7, DEX=15, CON=9, INT=8, WIS=7, CHA=8),
        hp_base=5, hp_per_level=0, ac=12,
        weapons=("匕首",),
        traits=(TraitGrant("pack_tactics"),),
    ),
    MonsterDef(
        archetype_id="wolf", default_name="狼",
        class_display="怪物", role="front", cr=0.25, natural_level=1,
        stat_block=dict(STR=12, DEX=15, CON=12, INT=3, WIS=12, CHA=6),
        hp_base=11, hp_per_level=0, ac=13,
        weapons=("狼牙",),
        traits=(TraitGrant("pack_tactics"),),
    ),
    MonsterDef(
        archetype_id="dire_wolf", default_name="恐狼",
        class_display="怪物", role="front", cr=1.0, natural_level=1,
        stat_block=dict(STR=17, DEX=15, CON=15, INT=3, WIS=12, CHA=7),
        hp_base=37, hp_per_level=0, ac=14,
        weapons=("巨狼牙",),
        traits=(TraitGrant("pack_tactics"),),
    ),
    MonsterDef(
        archetype_id="skeleton", default_name="骷髏",
        class_display="怪物", role="striker", cr=0.25, natural_level=1,
        stat_block=dict(STR=10, DEX=14, CON=15, INT=6, WIS=8, CHA=5),
        hp_base=13, hp_per_level=0, ac=13,
        weapons=("短劍", "短弓"),
        consumables=(("箭", 20, "ammo", ""),),
        traits=(
            TraitGrant("damage_table",
                       params={"multipliers": {"鈍擊": 2.0, "毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
        ),
    ),
    MonsterDef(
        archetype_id="zombie", default_name="殭屍",
        class_display="怪物", role="front", cr=0.25, natural_level=1,
        stat_block=dict(STR=13, DEX=6, CON=16, INT=3, WIS=6, CHA=5),
        hp_base=22, hp_per_level=0, ac=8,
        weapons=("重擊",),
        traits=(
            TraitGrant("undead_fortitude"),
            TraitGrant("damage_table", params={"multipliers": {"毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
        ),
    ),
    MonsterDef(
        archetype_id="ghoul", default_name="食屍鬼",
        class_display="怪物", role="front", cr=1.0, natural_level=1,
        stat_block=dict(STR=13, DEX=15, CON=10, INT=7, WIS=10, CHA=6),
        hp_base=22, hp_per_level=0, ac=12,
        weapons=("鬼爪",),
        traits=(
            TraitGrant("damage_table", params={"multipliers": {"毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned", "charmed"]}),
        ),
    ),
    MonsterDef(
        archetype_id="shadow", default_name="黑影",
        class_display="怪物", role="striker", cr=0.5, natural_level=1,
        stat_block=dict(STR=6, DEX=14, CON=13, INT=6, WIS=10, CHA=8),
        hp_base=16, hp_per_level=0, ac=12,
        weapons=("暗影之觸",),
        traits=(
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "火": 0.5, "冰": 0.5, "閃電": 0.5,
                "雷鳴": 0.5, "強酸": 0.5,
                "黯蝕": 0.0, "毒": 0.0, "光耀": 2.0}}),
            TraitGrant("condition_immunity", params={"conditions": [
                "poisoned", "frightened", "paralyzed", "prone", "restrained"]}),
        ),
    ),
    MonsterDef(
        archetype_id="wight", default_name="屍妖",
        class_display="怪物", role="front", cr=3.0, natural_level=3,
        stat_block=dict(STR=15, DEX=14, CON=16, INT=10, WIS=13, CHA=15),
        hp_base=45, hp_per_level=0, ac=14,
        weapons=("生命吸取",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "黯蝕": 0.5, "毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
        ),
    ),
    MonsterDef(
        archetype_id="gargoyle", default_name="石像鬼",
        class_display="怪物", role="front", cr=2.0, natural_level=2,
        stat_block=dict(STR=15, DEX=11, CON=16, INT=6, WIS=11, CHA=7),
        hp_base=52, hp_per_level=0, ac=15,
        weapons=("石爪",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
        ),
    ),
    MonsterDef(
        archetype_id="wyvern", default_name="雙足飛龍",
        class_display="怪物", role="front", cr=6.0, natural_level=6,
        stat_block=dict(STR=19, DEX=10, CON=16, INT=5, WIS=12, CHA=6),
        hp_base=110, hp_per_level=0, ac=13,
        weapons=("飛龍螫刺",),
        traits=(TraitGrant("extra_attack", params={"attacks": 2}),),
    ),
    MonsterDef(
        archetype_id="fire_elemental", default_name="火元素",
        class_display="怪物", role="front", cr=5.0, natural_level=5,
        stat_block=dict(STR=10, DEX=17, CON=16, INT=6, WIS=10, CHA=7),
        hp_base=102, hp_per_level=0, ac=13,
        weapons=("火焰之觸",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "火": 0.0, "毒": 0.0}}),
            TraitGrant("condition_immunity", params={"conditions": [
                "poisoned", "paralyzed", "prone", "restrained"]}),
        ),
    ),
    MonsterDef(
        archetype_id="young_red_dragon", default_name="幼紅龍",
        class_display="怪物", role="front", cr=10.0, natural_level=10,
        stat_block=dict(STR=23, DEX=10, CON=21, INT=14, WIS=11, CHA=19),
        hp_base=178, hp_per_level=0, ac=18,
        weapons=("龍之爪牙",),
        skills=(SkillGrant("fire_breath"),),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("uses_flat", params={"skill": "fire_breath", "uses": 1}),
            TraitGrant("recharge", params={"skill": "fire_breath", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"火": 0.0}}),
        ),
    ),
    # ── Wave 2（C 級原語落地，MONSTER_CATALOG §5）──────────────────────────
    MonsterDef(
        archetype_id="troll", default_name="巨魔",
        class_display="怪物", role="front", cr=5.0, natural_level=5,
        stat_block=dict(STR=18, DEX=13, CON=20, INT=7, WIS=9, CHA=7),
        hp_base=84, hp_per_level=0, ac=15,
        weapons=("巨魔之爪",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("regeneration",
                       params={"amount": 10, "blocked_by": ["火", "強酸"]}),
        ),
    ),
    MonsterDef(
        archetype_id="basilisk", default_name="石化蜥蜴",
        class_display="怪物", role="front", cr=3.0, natural_level=3,
        stat_block=dict(STR=16, DEX=8, CON=15, INT=2, WIS=8, CHA=7),
        hp_base=52, hp_per_level=0, ac=15,
        weapons=("蜥蜴巨口",),
        skills=(SkillGrant("petrifying_gaze"),),
    ),
    MonsterDef(
        archetype_id="behir", default_name="貝希爾",
        class_display="怪物", role="front", cr=11.0, natural_level=11,
        stat_block=dict(STR=23, DEX=16, CON=18, INT=7, WIS=14, CHA=12),
        hp_base=168, hp_per_level=0, ac=17,
        weapons=("貝希爾緊勒",),
        skills=(SkillGrant("lightning_breath"), SkillGrant("swallow")),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("uses_flat",
                       params={"skill": "lightning_breath", "uses": 1}),
            TraitGrant("recharge",
                       params={"skill": "lightning_breath", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"閃電": 0.0}}),
        ),
    ),
    MonsterDef(
        archetype_id="beholder", default_name="眼魔",
        class_display="怪物", role="striker", cr=13.0, natural_level=13,
        stat_block=dict(STR=10, DEX=14, CON=18, INT=17, WIS=15, CHA=17),
        hp_base=180, hp_per_level=0, ac=18,
        weapons=("眼魔巨口",),
        # Wave 3 retrofit: the beholder IS legendary in the MM — 3 extra eye
        # rays per round, one at the end of each other creature's turn.
        skills=(SkillGrant("eye_rays"), SkillGrant("eye_ray_single")),
        traits=(
            TraitGrant("condition_immunity", params={"conditions": ["prone"]}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"ability": "eye_ray_single", "cost": 1}]}),
        ),
    ),
    # ── Wave 3（D 級傳奇套件，MONSTER_CATALOG §5）──────────────────────────
    MonsterDef(
        archetype_id="adult_white_dragon", default_name="成年白龍",
        class_display="怪物", role="front", cr=13.0, natural_level=13,
        stat_block=dict(STR=22, DEX=10, CON=22, INT=8, WIS=12, CHA=12),
        hp_base=200, hp_per_level=0, ac=18,
        weapons=("龍之爪牙",),
        skills=(SkillGrant("cold_breath"), SkillGrant("wing_attack")),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("uses_flat", params={"skill": "cold_breath", "uses": 1}),
            TraitGrant("recharge", params={"skill": "cold_breath", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"冰": 0.0}}),
            TraitGrant("frightful_presence",
                       params={"radius_m": 36.0, "dc_stat": "CHA"}),   # DC 14
            TraitGrant("legendary_resistance", params={"uses": 3}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"weapon": "龍尾", "cost": 1},
                            {"ability": "wing_attack", "cost": 2}]}),
        ),
    ),
    MonsterDef(
        archetype_id="adult_red_dragon", default_name="成年紅龍",
        class_display="怪物", role="front", cr=17.0, natural_level=17,
        stat_block=dict(STR=27, DEX=10, CON=25, INT=16, WIS=13, CHA=21),
        hp_base=256, hp_per_level=0, ac=19,
        weapons=("龍之爪牙",),
        skills=(SkillGrant("fire_breath_adult"), SkillGrant("wing_attack")),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("uses_flat",
                       params={"skill": "fire_breath_adult", "uses": 1}),
            TraitGrant("recharge",
                       params={"skill": "fire_breath_adult", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"火": 0.0}}),
            TraitGrant("frightful_presence",
                       params={"radius_m": 36.0, "dc_stat": "CHA"}),   # DC 19
            TraitGrant("legendary_resistance", params={"uses": 3}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"weapon": "龍尾", "cost": 1},
                            {"ability": "wing_attack", "cost": 2}]}),
        ),
    ),
    MonsterDef(
        archetype_id="ancient_red_dragon", default_name="遠古紅龍",
        class_display="怪物", role="front", cr=24.0, natural_level=24,
        stat_block=dict(STR=30, DEX=10, CON=29, INT=18, WIS=15, CHA=23),
        hp_base=546, hp_per_level=0, ac=22,
        weapons=("龍之爪牙",),
        skills=(SkillGrant("fire_breath_ancient"), SkillGrant("wing_attack")),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("uses_flat",
                       params={"skill": "fire_breath_ancient", "uses": 1}),
            TraitGrant("recharge",
                       params={"skill": "fire_breath_ancient", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"火": 0.0}}),
            TraitGrant("frightful_presence",
                       params={"radius_m": 36.0, "dc_stat": "CHA"}),   # DC 21
            TraitGrant("legendary_resistance", params={"uses": 3}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"weapon": "龍尾", "cost": 1},
                            {"ability": "wing_attack", "cost": 2}]}),
        ),
    ),
    MonsterDef(
        archetype_id="lich", default_name="巫妖",
        class_display="怪物", role="striker", cr=21.0, natural_level=21,
        stat_block=dict(STR=11, DEX=16, CON=16, INT=20, WIS=14, CHA=16),
        hp_base=135, hp_per_level=0, ac=17,
        weapons=("麻痺之觸",),
        spell_ability="INT",         # 法術 DC 20 = 8 + 熟練7 + INT5
        spell_slots_table={1: {1: 4, 2: 3, 3: 3, 4: 3}},
        skills=(
            SkillGrant("magic_missile"),
            SkillGrant("shield_spell", reaction=True),
            SkillGrant("misty_step"),
            SkillGrant("hold_person"),
            SkillGrant("fireball_ev"),
            SkillGrant("ice_storm_ev"),
            SkillGrant("lightning_bolt"),
            SkillGrant("chill_touch"),
            SkillGrant("disrupt_life"),
        ),
        traits=(
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "冰": 0.5, "閃電": 0.5, "黯蝕": 0.5,
                "毒": 0.0}}),
            TraitGrant("condition_immunity", params={"conditions": [
                "poisoned", "charmed", "frightened", "paralyzed"]}),
            TraitGrant("legendary_resistance", params={"uses": 3}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"ability": "chill_touch", "cost": 1},
                            {"weapon": "麻痺之觸", "cost": 2},
                            {"ability": "disrupt_life", "cost": 3}]}),
        ),
    ),
    MonsterDef(
        archetype_id="kraken", default_name="海妖",
        class_display="怪物", role="front", cr=23.0, natural_level=23,
        stat_block=dict(STR=30, DEX=11, CON=25, INT=22, WIS=18, CHA=20),
        hp_base=472, hp_per_level=0, ac=18,
        weapons=("海妖觸手",),
        skills=(SkillGrant("lightning_storm"), SkillGrant("kraken_swallow")),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "閃電": 0.0}}),
            TraitGrant("condition_immunity", params={"conditions": [
                "frightened", "paralyzed"]}),
            # MM：海妖沒有傳奇抗性（傳奇行動三選：觸手／落雷；墨雲緩議）
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"weapon": "海妖觸手", "cost": 1},
                            {"ability": "lightning_storm", "cost": 2}]}),
        ),
    ),
    MonsterDef(
        archetype_id="balor", default_name="炎魔",
        class_display="怪物", role="front", cr=19.0, natural_level=19,
        stat_block=dict(STR=26, DEX=15, CON=22, INT=20, WIS=16, CHA=22),
        hp_base=262, hp_per_level=0, ac=19,
        weapons=("炎魔火劍",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "火": 0.0, "毒": 0.0,
                "冰": 0.5, "閃電": 0.5}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
            TraitGrant("death_throes", params={
                "damage_dice": "20d6", "damage_type": "火",
                "radius_m": 9.0, "save_stat": "DEX",
                "dc_stat": "CON"}),                          # DC 20 = 8+6+6
        ),
    ),
    MonsterDef(
        archetype_id="tarrasque", default_name="塔拉斯克",
        class_display="怪物", role="front", cr=30.0, natural_level=30,
        stat_block=dict(STR=30, DEX=11, CON=30, INT=3, WIS=11, CHA=11),
        hp_base=676, hp_per_level=0, ac=25,
        weapons=("泰拉斯克巨顎",),
        skills=(SkillGrant("tarrasque_swallow"),),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 4}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "火": 0.0, "毒": 0.0}}),
            TraitGrant("condition_immunity", params={"conditions": [
                "charmed", "frightened", "paralyzed", "poisoned"]}),
            TraitGrant("frightful_presence",
                       params={"radius_m": 36.0, "dc_stat": "CHA"}),   # DC 17
            TraitGrant("legendary_resistance", params={"uses": 3}),
            TraitGrant("legendary_actions", params={
                "per_round": 3,
                "options": [{"weapon": "泰拉斯克巨爪", "cost": 1},
                            {"ability": "tarrasque_swallow", "cost": 2}]}),
        ),
    ),
]

MONSTER_DEFS: dict[str, MonsterDef] = {
    md.archetype_id: md for md in _MONSTER_DEF_LIST
}


# ── 1v1 equivalent-level table (MEASURED, not designed) ──────────────────────
# 50% win-rate crossing of a standard class vs the monster, n=120/level, from
# scripts/calibrate_cr.py → eval_results/calibrate_cr_wave1_10g.txt. This is the
# FAIR-PAIRING level: a class at equiv_level fights the monster (at its
# natural_level) to a ~coin-flip. Monsters with NO crossing through L8 are 1vN
# party content (CR≥4 brutes/elementals/dragons) — marked inf and excluded from
# every 1v1 opponent pool (in 1v1 the agent loses regardless of play quality, so
# there is no gradient toward good descriptor-reading — see project memory).
import math as _math

EQUIV_LEVEL_1V1: dict[str, float] = {
    "commoner": 0.5, "bandit": 0.5, "kobold": 0.5, "goblin": 0.5,
    "wolf": 0.5, "skeleton": 0.5, "zombie": 0.5, "orc": 0.5, "ghoul": 0.5,
    "shadow": 2.9, "dire_wolf": 4.1, "ogre": 4.2, "wight": 4.6,
    "gargoyle": 4.7, "manticore": 4.9, "owlbear": 6.6,
    # mage_npc re-measured under GenericMonsterPolicy v1 (control gating):
    # 6.1 vs the pre-wave 6.0 — within cross-run dice drift, v0-equivalent.
    "mage_npc": 6.1,
    "ettin": _math.inf, "hill_giant": _math.inf, "fire_elemental": _math.inf,
    "wyvern": _math.inf, "young_red_dragon": _math.inf, "archmage": _math.inf,
    # Wave 2 (calibrate_cr_wave2_10g.txt, n=120/level): basilisk's petrify
    # punches a CR3 chassis up to gargoyle weight; troll regeneration is the
    # resist-table force-multiplier law again (10/turn ≈ doubles effective HP
    # vs a single attacker → 1vN content like the brutes).
    "troll": _math.inf, "basilisk": 4.4,
    "behir": _math.inf, "beholder": _math.inf,
    # Wave 3 legendary bosses: CR 13-30 with legendary actions — strictly
    # stronger than the already-inf behir/beholder band. Spot-checked vs L8
    # experts (calibrate_boss_spotcheck) rather than full 8-level sweeps;
    # they are 1vN party content by construction (see calibrate_boss.py).
    "adult_white_dragon": _math.inf, "adult_red_dragon": _math.inf,
    "ancient_red_dragon": _math.inf, "lich": _math.inf,
    "kraken": _math.inf, "balor": _math.inf, "tarrasque": _math.inf,
}
# Fail-loud: a monster added without a calibrated entry must not silently fall
# through into (or out of) a 1v1 pool. Adding a Wave 2/3 monster forces either a
# measured equiv_level or an explicit inf here.
assert set(EQUIV_LEVEL_1V1) == set(MONSTER_DEFS), (
    "EQUIV_LEVEL_1V1 out of sync with MONSTER_DEFS: missing="
    f"{set(MONSTER_DEFS) - set(EQUIV_LEVEL_1V1)} extra="
    f"{set(EQUIV_LEVEL_1V1) - set(MONSTER_DEFS)}")


def onev1_viable_monsters(max_level: float = 8.0) -> list[str]:
    """Monster ids whose 1v1 equivalent level falls in the trainable band
    (default ≤ L8). Excludes 1vN party content."""
    return [m for m in MONSTER_DEFS if EQUIV_LEVEL_1V1[m] <= max_level]


# ── 1vN (3-person party) equivalent-level table (MEASURED, not designed) ─────
# 50% crossing of the calibration party (battle_master+life+evocation, n=30 per
# level) vs the monster at natural_level — scripts/calibrate_boss.py →
# eval_results/calibrate_boss_wave3_30g.txt, MONSTER_CATALOG §6.8. Only the
# monsters that were 1v1-inf were swept; CR≥10 crosses nowhere through 3×L8
# (engine classes cap at L8) and is marked inf = content for larger parties /
# higher-level registries. Same fail-loud contract as EQUIV_LEVEL_1V1, scoped
# to the swept set.
PARTY3_EQUIV_LEVEL: dict[str, float] = {
    "ettin": 2.6, "hill_giant": 3.2, "fire_elemental": 4.3, "troll": 4.5,
    "wyvern": 4.3, "archmage": 4.4,
    "young_red_dragon": _math.inf, "behir": _math.inf, "beholder": _math.inf,
    "adult_white_dragon": _math.inf, "adult_red_dragon": _math.inf,
    "ancient_red_dragon": _math.inf, "lich": _math.inf, "kraken": _math.inf,
    "balor": _math.inf, "tarrasque": _math.inf,
}
assert all(_math.isinf(EQUIV_LEVEL_1V1[m]) for m in PARTY3_EQUIV_LEVEL), (
    "PARTY3_EQUIV_LEVEL must cover exactly the 1v1-inf band (a 1v1-banded "
    "monster has no party calibration)")


def party3_boss_monsters(max_level: float = 8.0) -> list[str]:
    """Monster ids playable as the BOSS seat of a 1v3 fight at a fair party
    level (party-equiv ≤ L8). The legendary CR≥10 set is out of band."""
    return [m for m in PARTY3_EQUIV_LEVEL
            if PARTY3_EQUIV_LEVEL[m] <= max_level]


# ── Registration (explicit, obs-freeze-safe) ─────────────────────────────────

_registered = False


def register_monsters() -> tuple[str, ...]:
    """Register all monsters into the live registries. Idempotent.

    Order matters: trpg.rl.obs is imported FIRST so ARCHETYPE_OBS_LIST /
    N_ARCHETYPES freeze on the 12 standard classes — monsters must read as
    all-zero in the archetype one-hot (the monster condition), and a
    pre-freeze registration would silently shift every obs dim and invalidate
    every checkpoint.
    """
    global _registered
    import trpg.rl.obs  # noqa: F401  (freeze N_ARCHETYPES before mutation)
    from ..engine.items import WEAPON_DEFS
    from ..engine.combat_policy import ARCHETYPE_POLICIES, GenericMonsterPolicy

    for wname, w in MONSTER_WEAPONS.items():
        WEAPON_DEFS.setdefault(wname, w)
    for mid, md in MONSTER_DEFS.items():
        CLASS_DEFS[mid] = md
        ARCHETYPE_FACTORIES[mid] = _factory(mid)
        ARCHETYPE_ROLES[mid] = md.role
        ARCHETYPE_POLICIES[mid] = GenericMonsterPolicy
    _registered = True
    return tuple(MONSTER_DEFS)


def make_monster(monster_id: str, name: str | None = None):
    """Build a monster Character at its natural level (registers on demand)."""
    if not _registered:
        register_monsters()
    return make_character(monster_id, name=name,
                          level=MONSTER_DEFS[monster_id].natural_level)
