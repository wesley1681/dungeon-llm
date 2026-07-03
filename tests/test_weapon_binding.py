"""ATTACK-builder 武器綁定：技能必須用施放者的真武器，不得憑空造武器。

Bug（bug_miner NvM 波挖出，mine|0|0195）：9 個 ATTACK 型 builder 硬寫
``"weapon": "長劍"``。施放者沒帶長劍時 ``get_weapon("長劍")`` 掉到全域
WEAPON_DEFS 憑空造一把 reach 1.5m 的幻影長劍——而合法性遮罩用的是本體真武器
（如 hill_giant 巨棒 reach 3.0m）＝遮罩判合法、引擎判拒絕的接縫：嫁接身分在
1.5m<d≤3.0m 帶每回合把行動砸在 ERROR 上（dead_action 空轉、25 次/場）。

正解＝builder 給 ``"weapon": ""``（引擎既有的「施放者預設武器」語義，全下游
``get_weapon(action.get("weapon",""))`` 一致）。對本職使用者位元級等價
（六個本職職業 weapons[0] 全是長劍），對嫁接身體改用真武器＝接縫閉合。

通用反硬寫測試＝設計警覺原則的登錄表迴圈版：任何未來新增的 ATTACK builder
只要硬寫了施放者沒有的武器名，立即翻紅。
"""
from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.vec2 import Vec2
from trpg.engine.world_state import WorldState
from trpg.engine.combat import execute_action
from trpg.engine.skill import available_skills
from trpg.engine.abilities import ABILITY_REGISTRY
from trpg.scenarios.monsters import register_monsters
from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES

register_monsters()


def _giant_with(skill_ids, level=8):
    """hill_giant（巨棒 reach 3.0m、無長劍）＋嫁接技能＝GUI 組合空間的代表。"""
    g = ARCHETYPE_FACTORIES["hill_giant"](level=level)
    g.known_abilities = list(getattr(g, "known_abilities", []) or []) + list(skill_ids)
    g.position = Vec2(0.0, 0.0)
    return g


def _world(caster, enemy):
    ws = WorldState(characters={"c": caster, "e0": enemy},
                    scene="", dungeon_map=None)
    ws.party_ids = ["c"]
    ws.combat = CombatState(initiative_order=["c", "e0"])
    return ws


def test_grafted_maneuver_executes_within_own_weapon_reach():
    """mine|0|0195 最小重現：巨人（reach 3.0m）嫁接 distracting_strike、敵在
    2.0m——遮罩層（真武器 reach）判合法，引擎執行也必須合法（不得回 ERROR）。"""
    g = _giant_with(["distracting_strike"])
    enemy = Character(name="E", race="", class_="", level=1, stats=Stats(),
                      hp=60, max_hp=60, ac=10, is_npc=True, attitude=0)
    enemy.position = Vec2(2.0, 0.0)          # 幻影長劍 1.5m 之外、巨棒 3.0m 之內
    ws = _world(g, enemy)
    sk = next(s for s in available_skills(g, ws)
              if s.skill_id == "distracting_strike")
    act = sk.build_action("c", "e0", (2.0, 0.0))
    res = execute_action(act, ws)
    assert isinstance(res, dict) and res.get("type") != "ERROR", (
        f"嫁接身分的戰技在自身武器 reach 內被引擎拒絕＝幻影武器接縫: {res}")


def test_attack_builders_mask_reach_never_exceeds_engine_reach():
    """登錄表全掃＝遮罩/引擎距離同源不變量（對任意施放者身體）：

    遮罩側 rng = features.range_m（>0 時）否則 本體真武器 reach（model.py 距離
    門的規則）；引擎側 reach = get_weapon(builder 輸出的 weapon 名)。遮罩比引擎
    寬＝放行引擎會拒絕的距離＝dead_action 空轉接縫（本 bug）；窄只是保守。

    兩類合法寫法都自然通過：``"weapon": ""``（施放者預設武器＝兩側同一把）、
    吞噬家族式「天然武器＋顯式宣告 range_m 且與該武器 reach 一致」。"""
    g = _giant_with([])                       # 巨棒 reach 3.0＝非長劍身體代表
    own = {w.name for w in g.weapons}
    fallback = g.get_weapon("").range_normal
    offenders = []
    for sid, ab in sorted(ABILITY_REGISTRY.items()):
        if ab.builder is None:
            continue
        try:
            act = ab.builder("c", "e0", (1.0, 0.0), char=g)
        except Exception:
            continue                          # 需要特定前置的 builder 不在本測範圍
        if not isinstance(act, dict) or act.get("type") != "ATTACK":
            continue
        wname = act.get("weapon", "")
        engine_reach = g.get_weapon(wname).range_normal or 1.5
        rng = float(getattr(ab.features, "range_m", 0.0) or 0.0)
        mask_rng = rng if rng > 0.0 else fallback
        if mask_rng > engine_reach + 1e-6:
            offenders.append(f"{sid}(weapon={wname or '預設'}: "
                             f"mask={mask_rng} > engine={engine_reach})")
    assert not offenders, (
        "遮罩距離比引擎執行 reach 寬＝dead_action 接縫: " + ", ".join(offenders))
