from tests.social.helpers import make_world
from trpg.engine.consequences import commit_bundle
from trpg.engine.dungeon_map import DungeonMap, Room

CHECK_OK = {"kind": "check", "dc": 12, "roll": 15, "success": True, "reason": "潛行割喉"}


def _one(ws, template, params, actor="party", ruling=None):
    return commit_bundle(ws, [{"template": template, "params": params}],
                         actor=actor, ruling=ruling)


def test_spawn_npc_and_item():
    ws = make_world()
    ok, _ = _one(ws, "SPAWN", {"kind": "npc", "entity_id": "sailor", "name": "老水手",
                               "source": "oracle"})
    assert ok and ws.social.npc_cards["sailor"].name == "老水手"
    assert not _one(ws, "SPAWN", {"kind": "npc", "entity_id": "sailor", "name": "x",
                                  "source": "oracle"})[0]   # 已存在
    assert not _one(ws, "SPAWN", {"kind": "npc", "entity_id": "y", "name": "y",
                                  "source": "wish"})[0]     # 非法來源
    ok, _ = _one(ws, "SPAWN", {"kind": "item", "entity_id": "解毒藥湯", "source": "craft",
                               "dst": "aria"})
    assert ok and "解毒藥湯" in ws.characters["aria"].gear


def test_entity_remove_needs_ruling_and_orphans_promises():
    ws = make_world()
    _one(ws, "PROMISE_FLAG", {"promise_id": "debt", "a": "party", "b": "finn",
                              "content": "還芬恩 50 金"})
    assert not _one(ws, "ENTITY_REMOVE", {"entity_id": "finn", "way": "死亡"})[0]  # 無裁決
    ok, _ = _one(ws, "ENTITY_REMOVE", {"entity_id": "finn", "way": "死亡",
                                       "leave_corpse": True}, ruling=CHECK_OK)
    assert ok
    assert ws.social.npc_cards["finn"].removed == "死亡"
    assert ws.social.promises["debt"].status == "orphaned"   # 壓測漏洞7：謀殺不是橡皮擦
    assert ws.social.object_states.get("屍體:芬恩") == "可見"


def test_recruit_wires_wage_standing_rule():
    ws = make_world()
    ok, _ = _one(ws, "RECRUIT", {"entity_id": "finn", "role": "眼線", "loyalty": 6,
                                 "wage": {"amount": 10, "interval_days": 1.0}})
    assert ok and ws.social.npc_cards["finn"].recruited["role"] == "眼線"
    assert "wage:finn" in ws.social.standing_rules


def test_travel_requires_connectivity():
    ws = make_world()
    ws.dungeon_map = DungeonMap(rooms={
        "dock": Room(id="dock", name="碼頭", description="", exits={"north": "market"}),
        "market": Room(id="market", name="市場", description="", exits={"south": "dock"}),
        "keep": Room(id="keep", name="城堡", description="", exits={}),
    }, current_room_id="dock")
    assert not _one(ws, "TRAVEL", {"who": "party", "dest": "keep"})[0]   # 不連通
    ok, _ = _one(ws, "TRAVEL", {"who": "party", "dest": "market", "hours": 1})
    assert ok and ws.dungeon_map.current_room_id == "market"
    assert ws.social.time_minutes == 60   # 耗時自動 TIME_ADVANCE


def test_aggregate_state_clamps_and_needs_catalog():
    ws = make_world()
    assert not _one(ws, "AGGREGATE_STATE", {"var": "貓密度", "region": "碼頭區", "delta": 1})[0]
    ok, _ = _one(ws, "AGGREGATE_STATE", {"var": "物價", "region": "碼頭區", "delta": -3})
    assert ok and ws.social.aggregates["物價:碼頭區"] == 2   # 預設 5，clamp 0..10
