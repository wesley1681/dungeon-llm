"""bug_miner 生成覆蓋守門（2026-07-02 用戶硬性要求）：

  「只要 GUI 能湊出來的組合，bug_miner 就要能湊出來。」

GUI（scripts/play_gui.py）可把 ABILITY_REGISTRY 裡任一能力嫁接到任一身分上；過去
bug_miner 只抽整個預建身分、沒有嫁接步驟，且 swallow 家族專屬的怪物被
onev1_viable_monsters(8.0) 擋在池外 → 「職業＋吞噬」這種組合生成機率是 0，使用者手動
兩三下就撞到、miner 卻永遠抓不到。這組測試盯死這個生成缺口不許回歸。

修正後兩者共用 scripts/gui_identity.build_identity（唯一真源），故覆蓋面相等。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from trpg.scenarios.monsters import register_monsters  # noqa: E402
register_monsters()
import bug_miner as BM  # noqa: E402
from trpg.scenarios.archetypes import CLASS_DEFS  # noqa: E402
from trpg.engine.abilities import ABILITY_REGISTRY  # noqa: E402


def _kit(ident):
    cd = CLASS_DEFS.get(ident)
    return [g.skill_id for g in cd.skills] if cd else []


def _target_status_abilities():
    return [sid for sid, ab in ABILITY_REGISTRY.items()
            if getattr(ab, "requires_target_status", None)
            or getattr(ab, "blocked_by_target_status", None)]


def test_graft_space_is_the_whole_gui_ability_list():
    # miner 的加掛技能池 == GUI 技能多選清單（都是整個 ABILITY_REGISTRY）。
    assert set(BM.GRAFT_SKILLS) == set(ABILITY_REGISTRY.keys())


def test_ident_pool_is_gui_catalog_superset():
    # 身分池涵蓋 GUI 下拉選單的每一個 id（職業＋全部怪物），含過去被排除的
    # 強怪（swallow 家族的宿主 behir/kraken/tarrasque）。
    gui_ids = {i for (_, i, _, _) in BM.identity_catalog()}
    assert gui_ids, "identity_catalog 不該為空"
    assert gui_ids <= set(BM.IDENTS)
    for host in ("tarrasque", "kraken", "behir"):
        assert host in BM.IDENTS, f"{host} 應在生成池（GUI 選得到就要抽得到）"


def test_target_status_abilities_are_reachable_by_generation():
    # 三個 target-status 前置能力過去生成機率為 0；現在必須能在隨機組合中被造出來。
    ts = set(_target_status_abilities())
    assert ts, "測試前提：登錄表要有 target-status 能力"
    seen = set()
    for seed in range(8):
        for sc in BM.sample_scenarios(120, seed):
            seen |= ts.intersection(_kit(sc.agents[0]))
    assert seen == ts, (
        f"生成器沒能造出全部 target-status 前置能力的組合；缺 {sorted(ts - seen)}"
        "——生成缺口沒補上（用戶手動撞得到、miner 卻抽不到）")


def test_sampling_is_deterministic_in_kit_content():
    # 同 (n, seed) 兩次抽樣，agent 身分的 kit 內容必須一致（臨時 id 字串可不同，
    # 但技能組合一致＝行為可重現、--replay 有效）。
    a = [tuple(_kit(sc.agents[0])) for sc in BM.sample_scenarios(60, 123)]
    b = [tuple(_kit(sc.agents[0])) for sc in BM.sample_scenarios(60, 123)]
    assert a == b
