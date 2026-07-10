from tests.social.helpers import make_world
from trpg.engine.consequences import commit_bundle
from trpg.engine.social_state import Faction


def _one(ws, template, params, actor="party", ruling=None):
    return commit_bundle(ws, [{"template": template, "params": params}],
                         actor=actor, ruling=ruling)


def test_disposition_applies_and_clamps():
    ws = make_world()
    ok, _ = _one(ws, "DISPOSITION", {"observer": "finn", "delta": 2})
    assert ok and ws.social.npc_cards["finn"].disposition["party"] == 1   # -1+2
    assert not _one(ws, "DISPOSITION", {"observer": "no_such", "delta": 1})[0]
    assert not _one(ws, "DISPOSITION", {"observer": "finn", "delta": 5})[0]  # N∈1~3


def test_disposition_axis_decay():
    ws = make_world()
    for _ in range(3):
        _one(ws, "DISPOSITION", {"observer": "finn", "delta": 3})
    # 第1次 +3、第2次遞減為 +2、第3次 +1 → -1+6=5（同時撞 +5 上限）
    assert ws.social.npc_cards["finn"].disposition["party"] == 5
    ok, msgs = _one(ws, "DISPOSITION", {"observer": "finn", "delta": 3})
    assert ok and ws.social.npc_cards["finn"].disposition["party"] == 5  # 遞減到 0 → no-op


def test_disposition_third_party_edge():
    ws = make_world()
    from trpg.engine.social_state import NPCCard
    ws.social.npc_cards["tomi"] = NPCCard(npc_id="tomi", name="托米")
    ok, _ = _one(ws, "DISPOSITION", {"observer": "finn", "target": "tomi", "delta": -2})
    assert ok and ws.social.npc_cards["finn"].disposition["tomi"] == -2  # 挑撥離間記帳


def test_belief_flag_requires_valid_flaw_predicate():
    ws = make_world()
    bad = {"holder": "finn", "content": "Aria=灰鴉的人",
           "flaw": {"kind": "兩個月亮同升"}}
    ok, msgs = _one(ws, "BELIEF_FLAG", bad)
    assert not ok  # 漏洞2：不可評估的破綻條件拒絕 commit
    good = {"holder": "finn", "content": "Aria=灰鴉的人",
            "flaw": {"kind": "flag", "flag_id": "met_real_raven"}}
    ok, _ = _one(ws, "BELIEF_FLAG", good)
    assert ok and ws.social.npc_cards["finn"].beliefs[0].content == "Aria=灰鴉的人"


def test_reveal_from_card_moves_into_party_ledger():
    ws = make_world()
    ok, _ = _one(ws, "REVEAL", {"fact_id": "relic_at_finn", "source": "card", "npc_id": "finn"})
    assert ok
    assert "relic_at_finn" in ws.social.party_knowledge
    assert ws.social.npc_cards["finn"].knowledge["relic_at_finn"].revealed
    assert not _one(ws, "REVEAL", {"fact_id": "ghost", "source": "card", "npc_id": "finn"})[0]


def test_reveal_oracle_commits_canon_and_blocks_reroll():
    ws = make_world()
    ok, _ = _one(ws, "REVEAL", {"fact_id": "smith_gone", "source": "oracle",
                                "text": "鐵匠上週被抓走了"})
    assert ok and ws.social.party_knowledge["smith_gone"].text == "鐵匠上週被抓走了"
    ok, msgs = _one(ws, "REVEAL", {"fact_id": "smith_gone", "source": "oracle",
                                   "text": "鐵匠其實還在"})
    assert not ok  # canon 不可重擲


def test_promise_and_faction_rep():
    ws = make_world()
    ok, _ = _one(ws, "PROMISE_FLAG", {"promise_id": "pay_captain", "a": "party",
                                      "b": "finn", "content": "登陸付雙倍"})
    assert ok and ws.social.promises["pay_captain"].status == "open"

    assert not _one(ws, "REP_FACTION", {"faction_id": "dock_gang", "delta": -2})[0]
    ws.social.factions["dock_gang"] = Faction("dock_gang", "碼頭幫")
    ok, _ = _one(ws, "REP_FACTION", {"faction_id": "dock_gang", "delta": -2})
    assert ok and ws.social.factions["dock_gang"].reputation == -2
