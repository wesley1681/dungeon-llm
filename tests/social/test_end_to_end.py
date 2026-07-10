"""端到端劇本：對話→唬騙檢定→承諾→時鐘→到期→謊言拆穿。
這是整個非戰鬥管線的活文件——讀這個檔＝理解系統怎麼動。"""
from unittest.mock import patch
from tests.social.helpers import make_world
from trpg.engine.adjudicator import rule_check
from trpg.engine.consequences import commit_bundle
from trpg.engine.social_time import advance_time


def test_finn_deception_scenario():
    ws = make_world()

    # 1) 玩家：「我謊稱自己是灰鴉派來的先遣人」→ 裁決（引擎擲骰，DC=芬恩洞察14）
    with patch("trpg.engine.adjudicator.roll_d20", return_value=18):
        ruling = rule_check(ws, dc_from=("npc_insight", "finn"), reason="唬騙：假扮灰鴉先遣人")
    assert ruling.success

    # 2) 成功束：信念+態度+揭露（表一第1行的預設束）
    ok, facts = commit_bundle(ws, [
        {"template": "BELIEF_FLAG", "params": {
            "holder": "finn", "content": "Aria 是灰鴉的人",
            "flaw": {"kind": "flag", "flag_id": "real_raven_arrived"}}},
        {"template": "DISPOSITION", "params": {"observer": "finn", "delta": 2}},
        {"template": "REVEAL", "params": {"fact_id": "relic_at_finn",
                                          "source": "card", "npc_id": "finn"}},
    ], actor="party", ruling=ruling.as_dict())
    assert ok
    assert ws.social.npc_cards["finn"].disposition["party"] == 1
    assert "relic_at_finn" in ws.social.party_knowledge

    # 3) 世界自己往前走：交易時鐘 3 天，到期＝事跡敗露、芬恩被通緝
    ok, _ = commit_bundle(ws, [{"template": "CLOCK_SPAWN", "params": {
        "clock_id": "deal", "name": "聖物交易", "days": 3.0,
        "on_expire": [{"template": "CONDITION", "params": {
            "entity": "finn", "status": "通緝", "op": "+"}}]}}], actor="engine")
    assert ok

    # 4) 回溯防禦（規則6）：玩家宣稱「我早就在芬恩的酒裡下毒」→ 引擎查帳本
    assert not ws.social.ledger.has_prior(
        "OBJ_STATE", where=lambda c: "酒" in str(c.params.get("object_id", "")))

    # 5) 三天過去，時鐘到期
    advance_time(ws, 3 * 1440)
    assert ws.social.clocks["deal"].expired
    assert any(c["status"] == "通緝" for c in ws.social.conditions["finn"])

    # 6) 帳本審計：每筆 check commit 都帶得出 DC 與理由
    checks = ws.social.ledger.query(where=lambda c: c.ruling.get("kind") == "check")
    assert checks and all(c.ruling["dc"] == 14 for c in checks)
