from tests.social.helpers import make_world
from trpg.engine.consequences import commit_bundle


def _one(ws, template, params, actor="party"):
    return commit_bundle(ws, [{"template": template, "params": params}], actor=actor)


def test_obj_state_legal_transitions_only():
    ws = make_world()
    ws.social.object_states["石棺"] = "鎖上"
    assert _one(ws, "OBJ_STATE", {"object_id": "石棺", "new_state": "開啟"})[0]
    assert ws.social.object_states["石棺"] == "開啟"
    assert not _one(ws, "OBJ_STATE", {"object_id": "石棺", "new_state": "隱藏"})[0]  # 非法轉移
    assert not _one(ws, "OBJ_STATE", {"object_id": "幽靈門", "new_state": "開啟"})[0]  # 未登記


def test_transact_money_requires_funds():
    ws = make_world()   # party=100, finn=40
    ok, _ = _one(ws, "TRANSACT", {"kind": "money", "amount": 30, "src": "party", "dst": "finn"})
    assert ok and ws.social.accounts == {"party": 70, "finn": 70}
    assert not _one(ws, "TRANSACT", {"kind": "money", "amount": 999,
                                     "src": "party", "dst": "finn"})[0]


def test_transact_item_moves_gear():
    ws = make_world()
    ws.characters["aria"].gear.append("燈油")
    ok, _ = _one(ws, "TRANSACT", {"kind": "item", "item": "燈油", "src": "aria", "dst": "finn"})
    assert ok
    assert "燈油" not in ws.characters["aria"].gear
    assert "燈油" in ws.characters["finn"].gear
    assert not _one(ws, "TRANSACT", {"kind": "item", "item": "神劍",
                                     "src": "aria", "dst": "finn"})[0]  # 未持有


def test_resource_hp_clamps_to_max():
    ws = make_world()
    ws.characters["aria"].hp = 5
    ok, _ = _one(ws, "RESOURCE", {"entity": "aria", "resource": "hp", "delta": 100})
    assert ok and ws.characters["aria"].hp == 20   # max_hp
    ok, _ = _one(ws, "RESOURCE", {"entity": "aria", "resource": "補給", "delta": 3})
    assert ok and ws.social.resources[("aria", "補給")] == 3
    assert not _one(ws, "RESOURCE", {"entity": "aria", "resource": "魔力值", "delta": 1})[0]


def test_condition_add_and_remove():
    ws = make_world()
    ok, _ = _one(ws, "CONDITION", {"entity": "finn", "status": "通緝", "op": "+"})
    assert ok and any(c["status"] == "通緝" for c in ws.social.conditions["finn"])
    assert not _one(ws, "CONDITION", {"entity": "finn", "status": "中毒", "op": "-"})[0]  # 不在身
    ok, _ = _one(ws, "CONDITION", {"entity": "finn", "status": "通緝", "op": "-"})
    assert ok and not any(c["status"] == "通緝" for c in ws.social.conditions["finn"])
    assert not _one(ws, "CONDITION", {"entity": "finn", "status": "戀愛腦", "op": "+"})[0]  # 不在目錄
