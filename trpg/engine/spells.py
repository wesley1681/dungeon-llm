"""Spell definitions — pure reference data, not directly executable.

SPELLS is a registry of spell properties (damage_dice, save_ability, aoe_radius,
etc.) keyed by display name. Engine handlers (SPELL action handler, _spells_str
in combat.py) look up these properties at execution time.

To make a spell castable, register a Ability in `abilities.py` whose
display_name matches the SPELLS entry and whose builder produces the right
action type (SPELL for damage/save spells, HEAL for heal_dice spells,
APPLY_MOD for pure buff/debuff spells). Characters reference the ability by
its skill_id in `known_abilities`.

Phase 1 supports attack_type="save" (AOE / single-target saving-throw spells).
spell_attack (attack-roll spells like Firebolt) is reserved for Phase 2.

Spells with `level=0` are cantrips and consume no spell slot.
Spells with `requires_concentration=True` set caster.concentrating_on when
cast (replacing any prior concentration spell).
Spells with `applies_status_on_fail` attach the named StatusEffect to targets
that fail their save.
"""
from dataclasses import dataclass


@dataclass
class Spell:
    name: str
    level: int                       # 0 = cantrip (no slot); 1-9 = required slot
    range_m: float                   # distance from caster to AOE center / target
    aoe_radius_m: float = 0.0        # 0 = single target (only the centered creature)
    attack_type: str = "save"        # "save" — Phase 2 will add "spell_attack" / "utility"
    save_ability: str = "DEX"        # ability targets roll for the save
    damage_dice: str = ""            # e.g. "8d6"; "" = no damage
    damage_type: str = ""            # e.g. "fire", "cold"
    heal_dice: str = ""              # for healing spells (e.g. "1d8+3")
    requires_concentration: bool = False
    applies_status_on_fail: str = "" # status name attached on failed save
    status_rounds: int = 10          # default 1-minute duration if status applies
    # Save outcome rule for damage spells:
    #   False (default) — successful save halves damage (Fireball, Burning Hands)
    #   True            — successful save deals no damage (Sacred Flame, Toll the Dead)
    save_for_no_damage: bool = False
    # Whether applies_status_on_fail attaches a save_each retry. When True
    # (default, matches hold_person / hold_monster), the target re-rolls the
    # initial save at each self_turn_end and escapes on success. When False
    # (sleep, color_spray), the status persists for `status_rounds` regardless
    # of saves — closer to D&D 5e RAW for those spells.
    save_each_on_status: bool = True
    # level=0 normally means "cantrip" → damage scales with caster level.
    # Natural abilities that ride the SPELL pipeline (dragon breath: no slot,
    # uncounterable, but NOT a cantrip) set this False to opt out.
    scales_as_cantrip: bool = True
    # Save DC stat override: "" = caster's spellcasting ability (spells);
    # breath weapons use the creature's CON (5e: DC 8 + prof + CON).
    save_dc_ability: str = ""
    # AoE shape. "circle" (default) = within aoe_radius_m of the centre.
    # "line" = within line_width_m/2 of the caster→endpoint segment, endpoint
    # line_length_m along the aim direction (behir breath, lightning bolt);
    # range_m should equal line_length_m so aim points stay on the line.
    aoe_shape: str = "circle"
    line_length_m: float = 0.0
    line_width_m: float = 1.5
    # Two-stage status escalation (basilisk gaze / beholder petrify ray): a
    # failed save while applies_status_on_fail is ALREADY on the target
    # applies this terminal status instead — permanent, no retry save.
    escalates_to: str = ""
    # Self-centred novas (Wave 3: dragon wing attack, lich disrupt life) —
    # "each creature within X of the caster" never hits the caster. Default
    # False keeps the historical circle-AoE behaviour bit-identical.
    excludes_caster: bool = False
    description: str = ""


SPELLS: dict[str, Spell] = {
    "火球術": Spell(
        name="火球術",
        level=3,
        range_m=45.0,
        aoe_radius_m=6.0,
        save_ability="DEX",
        damage_dice="8d6",
        damage_type="火",
        description="向 45m 內一點扔出火球，半徑 6m 內目標 DEX 豁免，成功半傷。",
    ),
    "神聖光輝": Spell(
        name="神聖光輝",
        level=0,                     # cantrip
        range_m=18.0,
        save_ability="DEX",
        damage_dice="1d8",
        damage_type="光耀",
        save_for_no_damage=True,
        description="單體目標 DEX 豁免，失敗承受 1d8 光耀傷害；成功無傷害。",
    ),
    "定身術": Spell(
        name="定身術",
        level=2,
        range_m=18.0,
        save_ability="WIS",
        applies_status_on_fail="paralyzed",
        status_rounds=10,
        requires_concentration=True,
        description="WIS 豁免，失敗則 paralyzed 10 回合；每回合終可重豁。"
                    "施法者必須維持專注。",
    ),
    "霧步": Spell(
        name="霧步",
        level=2,
        range_m=9.0,
        description="瞬移至 9m 內可見點，無視視線、不被障礙阻擋。"
                    "（霧步透過 MOVE action 執行，不走 SPELL 流程）",
    ),
    "火焰吐息": Spell(
        name="火焰吐息",
        level=0,                     # 天然能力：不耗法術位、不可被反制
        scales_as_cantrip=False,     # 但絕不是戲法——傷害不隨等級縮放
        save_dc_ability="CON",       # 5e 吐息 DC = 8 + 熟練 + CON
        range_m=9.0,
        aoe_radius_m=4.5,            # 30ft 錐形 → 簡化為錐口半徑的 AOE
        save_ability="DEX",
        damage_dice="16d6",
        damage_type="火",
        description="幼紅龍火焰吐息：9m 錐形（簡化 AOE），DEX 豁免，"
                    "失敗 16d6 火焰，成功半傷。充能 5-6。",
    ),
    "閃電吐息": Spell(
        name="閃電吐息",
        level=0,                     # 天然能力（貝希爾）：免法術位、不可反制
        scales_as_cantrip=False,
        save_dc_ability="CON",       # 5e：DC 16 = 8 + 熟練4 + CON4
        range_m=6.0,                 # 20ft 直線 → 瞄準點需在線長內
        aoe_shape="line",
        line_length_m=6.0,
        line_width_m=1.5,            # 5ft 寬
        save_ability="DEX",
        damage_dice="12d10",
        damage_type="閃電",
        description="貝希爾閃電吐息：6m 直線（寬 1.5m），DEX 豁免，"
                    "失敗 12d10 閃電，成功半傷。充能 5-6。",
    ),
    "閃電束": Spell(
        name="閃電束",
        level=3,
        range_m=30.0,                # 100ft 直線
        aoe_shape="line",
        line_length_m=30.0,
        line_width_m=1.5,
        save_ability="DEX",
        damage_dice="8d6",
        damage_type="閃電",
        description="自施法者射出 30m 長、1.5m 寬的閃電，沿線目標 DEX 豁免，"
                    "失敗 8d6 閃電，成功半傷。",
    ),
    "石化凝視": Spell(
        name="石化凝視",
        level=0,                     # 天然能力（石化蜥蜴）
        scales_as_cantrip=False,
        save_dc_ability="CON",       # 5e：DC 12 = 8 + 熟練2 + CON2
        range_m=9.0,                 # 30ft 目光所及
        save_ability="CON",
        applies_status_on_fail="restrained",
        escalates_to="petrified",    # 已束縛者再失敗 → 石化（永久）
        status_rounds=10,
        save_each_on_status=True,    # 束縛階段每回合末可重豁逃脫
        description="石化凝視（簡化為主動動作）：CON 豁免，失敗先束縛；"
                    "已束縛者再次失敗則石化（永久、傷害減半、不能行動）。",
    ),
    # ── Wave 3：傳奇怪物天然能力（SPELL 管線資料欄，零特例分支）─────────────
    "寒冰吐息": Spell(
        name="寒冰吐息",
        level=0,                     # 天然能力（成年白龍）
        scales_as_cantrip=False,
        save_dc_ability="CON",       # 白龍 DC 19 = 8 + 熟練5 + CON6
        range_m=18.0,                # 60ft 錐形 → 簡化 AOE（幼紅龍同慣例）
        aoe_radius_m=9.0,
        save_ability="CON",
        damage_dice="12d8",
        damage_type="冰",
        description="成年白龍寒冰吐息：18m 錐形（簡化 AOE），CON 豁免，"
                    "失敗 12d8 寒冰，成功半傷。充能 5-6。",
    ),
    "火焰吐息（成龍）": Spell(
        name="火焰吐息（成龍）",
        level=0,                     # 天然能力（成年紅龍）
        scales_as_cantrip=False,
        save_dc_ability="CON",       # 紅龍 DC 21 = 8 + 熟練6 + CON7
        range_m=18.0,
        aoe_radius_m=9.0,
        save_ability="DEX",
        damage_dice="18d6",
        damage_type="火",
        description="成年紅龍火焰吐息：18m 錐形（簡化 AOE），DEX 豁免，"
                    "失敗 18d6 火焰，成功半傷。充能 5-6。",
    ),
    "火焰吐息（古龍）": Spell(
        name="火焰吐息（古龍）",
        level=0,                     # 天然能力（遠古紅龍）
        scales_as_cantrip=False,
        save_dc_ability="CON",       # 古龍 DC 24 = 8 + 熟練7 + CON9
        range_m=27.0,                # 90ft 錐形
        aoe_radius_m=13.5,
        save_ability="DEX",
        damage_dice="26d6",
        damage_type="火",
        description="遠古紅龍火焰吐息：27m 錐形（簡化 AOE），DEX 豁免，"
                    "失敗 26d6 火焰，成功半傷。充能 5-6。",
    ),
    "龍翼拍擊": Spell(
        name="龍翼拍擊",
        level=0,                     # 天然能力（傳奇行動，成年龍共用）
        scales_as_cantrip=False,
        save_dc_ability="STR",       # 5e 翼擊 DC = 8 + 熟練 + STR（按龍實算）
        range_m=0.0,                 # 自心 nova：builder 以自身為圓心
        aoe_radius_m=3.0,            # 10ft
        excludes_caster=True,
        save_ability="DEX",
        damage_dice="2d6+8",         # 共用條目取成年紅龍值（白 +6／古 +10 略差）
        damage_type="鈍擊",
        save_for_no_damage=True,
        applies_status_on_fail="prone",
        save_each_on_status=False,   # 倒地不靠豁免解除（起身即可）
        description="龍翼拍擊（傳奇行動）：以自身為中心 3m，DEX 豁免，"
                    "失敗 2d6+8 鈍擊並倒地，成功無事。",
    ),
    "生命擾亂": Spell(
        name="生命擾亂",
        level=0,                     # 天然能力（巫妖傳奇行動）
        scales_as_cantrip=False,
        range_m=0.0,                 # 自心 nova
        aoe_radius_m=6.0,            # 20ft
        excludes_caster=True,
        save_ability="CON",
        damage_dice="6d6",
        damage_type="黯蝕",
        description="生命擾亂（傳奇行動 ×3）：以巫妖為中心 6m 內 CON 豁免，"
                    "失敗 6d6 黯蝕，成功半傷。",
    ),
    "寒冰之觸": Spell(
        name="寒冰之觸",
        level=0,                     # 真戲法：傷害隨施法者等級縮放
        range_m=36.0,
        save_ability="DEX",
        damage_dice="1d8",
        damage_type="黯蝕",
        save_for_no_damage=True,
        description="戲法：36m 內單體 DEX 豁免，失敗 1d8 黯蝕（隨等級縮放；"
                    "巫妖 L21 ×4）。攻擊骰型戲法的豁免近似（神聖光輝同例）。",
    ),
    "燃燒之手": Spell(
        name="燃燒之手",
        level=1,
        range_m=4.5,
        aoe_radius_m=4.5,
        save_ability="DEX",
        damage_dice="3d6",
        damage_type="火",
        description="半徑 4.5m 錐形，DEX 豁免，失敗承受 3d6 火焰傷害，成功半傷。",
    ),
    "彩光繽紛": Spell(
        name="彩光繽紛",
        level=1,
        range_m=4.5,
        aoe_radius_m=3.0,
        save_ability="CON",
        damage_dice="",
        damage_type="",
        applies_status_on_fail="blinded",
        status_rounds=1,
        save_each_on_status=False,
        description="4.5m 內目標 CON 豁免失敗則 blinded 1 回合（短錐效果簡化為小 AOE）。",
    ),
    "睡眠術": Spell(
        name="睡眠術",
        level=1,
        range_m=27.0,
        aoe_radius_m=6.0,
        save_ability="WIS",
        damage_dice="",
        damage_type="",
        applies_status_on_fail="asleep",
        status_rounds=2,
        save_each_on_status=False,
        description="6m 半徑 AOE，WIS 豁免失敗則 asleep 2 回合（或受傷立即清醒）。"
                    "原版無豁免按 HP 上限結算；引擎用 WIS 豁免代之以平衡。",
    ),
    "蜘蛛網": Spell(
        name="蜘蛛網",
        level=2,
        range_m=18.0,
        aoe_radius_m=4.5,
        save_ability="DEX",
        damage_dice="",
        damage_type="",
        requires_concentration=True,
        applies_status_on_fail="restrained",
        status_rounds=10,
        description="4.5m 半徑 AOE，DEX 豁免失敗則 restrained，專注。",
    ),
    "冰風暴": Spell(
        name="冰風暴",
        level=4,
        range_m=36.0,
        aoe_radius_m=4.0,
        save_ability="DEX",
        damage_dice="2d8",
        damage_type="冰",
        description="4m 半徑 AOE，DEX 豁免，失敗 2d8 冰冷傷害，成功半傷。",
    ),
    "定怪術": Spell(
        name="定怪術",
        level=5,
        range_m=18.0,
        save_ability="WIS",
        applies_status_on_fail="paralyzed",
        status_rounds=10,
        requires_concentration=True,
        description="WIS 豁免失敗則 paralyzed，適用任何生物，專注。",
    ),
    "治療語": Spell(
        name="治療語",
        level=1,
        range_m=18.0,
        heal_dice="1d4+3",
        description="遠距治療一個生物 1d4 + 施法屬性修正 HP（透過 HEAL action 執行）。",
    ),
    "治療術": Spell(
        name="治療術",
        level=1,
        range_m=1.5,
        heal_dice="1d8+3",           # placeholder; real value adds caster's WIS mod
        description="觸碰範圍，治療一個盟友 1d8 + 施法屬性修正 HP。"
                    "（治療術透過 HEAL action 執行，不走 SPELL 流程）",
    ),
}
