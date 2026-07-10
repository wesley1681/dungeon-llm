from tests.social.helpers import make_world
from trpg.engine.consequences import commit_bundle
from trpg.engine.social_state import Clock
from trpg.engine.social_time import advance_time


def _one(ws, template, params, actor="party", ruling=None):
    return commit_bundle(ws, [{"template": template, "params": params}],
                         actor=actor, ruling=ruling)


def test_advance_time_ticks_and_fires_expiry():
    ws = make_world()
    ws.social.clocks["deal"] = Clock(
        clock_id="deal", name="聖物交易", remaining_days=1.0,
        on_expire=[{"template": "CONDITION",
                    "params": {"entity": "finn", "status": "通緝", "op": "+"}}])
    facts = advance_time(ws, 720)   # 半天
    assert ws.social.time_minutes == 720
    assert not ws.social.clocks["deal"].expired
    facts = advance_time(ws, 720)   # 到期
    assert ws.social.clocks["deal"].expired
    assert any(c["status"] == "通緝" for c in ws.social.conditions["finn"])
    assert any("通緝" in f for f in facts)


def test_expiry_bundle_lands_in_ledger_as_engine():
    ws = make_world()
    ws.social.clocks["c"] = Clock("c", "c", 0.5,
        on_expire=[{"template": "CONDITION",
                    "params": {"entity": "finn", "status": "受詛咒", "op": "+"}}])
    advance_time(ws, 1440)
    hits = ws.social.ledger.query(template_id="CONDITION", actor="engine")
    assert len(hits) == 1


def test_time_advance_template_calls_advance():
    ws = make_world()
    ok, _ = _one(ws, "TIME_ADVANCE", {"minutes": 60})
    assert ok and ws.social.time_minutes == 60
    assert not _one(ws, "TIME_ADVANCE", {"minutes": -5})[0]


def test_clock_modify_bounds_and_self_delay_cost():
    ws = make_world()
    ws.social.clocks["doom"] = Clock("doom", "末日", 2.0, owner="party")
    assert not _one(ws, "CLOCK_MODIFY", {"clock_id": "doom", "delta_days": 3})[0]  # |±|≤2
    # 壓測漏洞4：延後自己擁有的時鐘，無裁決 → 拒
    assert not _one(ws, "CLOCK_MODIFY", {"clock_id": "doom", "delta_days": 2})[0]
    ok, _ = _one(ws, "CLOCK_MODIFY", {"clock_id": "doom", "delta_days": 2},
                 ruling={"kind": "check", "dc": 15, "roll": 18, "success": True,
                         "reason": "付出代價拖延"})
    assert ok and ws.social.clocks["doom"].remaining_days == 4.0
    # 加速任何時鐘不需裁決
    ok, _ = _one(ws, "CLOCK_MODIFY", {"clock_id": "doom", "delta_days": -1})
    assert ok and ws.social.clocks["doom"].remaining_days == 3.0


def test_clock_spawn_rejects_self_reference():
    ws = make_world()
    bad = {"clock_id": "z", "name": "z", "days": 1.0,
           "on_expire": [{"template": "CLOCK_SPAWN",
                          "params": {"clock_id": "z", "name": "z", "days": 1.0,
                                     "on_expire": []}}]}
    assert not _one(ws, "CLOCK_SPAWN", bad)[0]   # 壓測漏洞4：不朽時鐘
    ok, _ = _one(ws, "CLOCK_SPAWN", {"clock_id": "verify", "name": "求證", "days": 1.0,
                                     "on_expire": []})
    assert ok and "verify" in ws.social.clocks


def test_cascade_depth_capped():
    ws = make_world()
    # c1 到期生 c2（0 天）……到期束再生時鐘的鏈不得無限遞迴
    ws.social.clocks["c1"] = Clock(
        "c1", "c1", 0.5,
        on_expire=[{"template": "CLOCK_SPAWN",
                    "params": {"clock_id": "c2", "name": "c2", "days": 0.0,
                               "on_expire": []}}])
    advance_time(ws, 1440)   # 不應 RecursionError
    assert "c2" in ws.social.clocks
