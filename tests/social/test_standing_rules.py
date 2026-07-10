from tests.social.helpers import make_world
from trpg.engine.consequences import commit_bundle
from trpg.engine.social_time import advance_time


def _mk_rule(ws, rule_id="morning", interval=1.0, effects=None, trigger=None, cancel=None):
    params = {"rule_id": rule_id,
              "trigger": trigger or {"kind": "periodic", "interval_days": interval},
              "effects": effects or [{"template": "RESOURCE",
                                      "params": {"entity": "aria", "resource": "補給",
                                                 "delta": 1}}]}
    if cancel:
        params["cancel"] = cancel
    return commit_bundle(ws, [{"template": "STANDING_RULE", "params": params}],
                         actor="party")


def test_periodic_rule_fires_on_interval():
    ws = make_world()
    ok, _ = _mk_rule(ws)
    assert ok
    advance_time(ws, 1440)   # 第 1 天
    assert ws.social.resources[("aria", "補給")] == 1
    advance_time(ws, 2880)   # 再過 2 天 → 補償觸發 2 次
    assert ws.social.resources[("aria", "補給")] == 3


def test_fires_per_tick_capped():
    ws = make_world()
    _mk_rule(ws)
    advance_time(ws, 1440 * 10)   # 一口氣 10 天：只補 MAX_FIRES_PER_RULE=3 次
    assert ws.social.resources[("aria", "補給")] == 3


def test_conditional_rule_and_cancel():
    ws = make_world()
    _mk_rule(ws, rule_id="alarm",
             trigger={"kind": "conditional",
                      "predicate": {"kind": "flag", "flag_id": "intruder"}},
             cancel={"kind": "flag", "flag_id": "camp_broken"})
    advance_time(ws, 60)
    assert ("aria", "補給") not in ws.social.resources   # 條件未真
    ws.social.flags.add("intruder")
    advance_time(ws, 60)
    assert ws.social.resources[("aria", "補給")] == 1     # 觸發一次
    ws.social.flags.add("camp_broken")
    advance_time(ws, 60)
    assert ws.social.resources[("aria", "補給")] == 1     # 已取消
    assert ws.social.standing_rules["alarm"].active is False


def test_rule_effects_may_not_nest_rule_makers():
    ws = make_world()
    ok, msgs = _mk_rule(ws, rule_id="loop", effects=[
        {"template": "STANDING_RULE",
         "params": {"rule_id": "x",
                    "trigger": {"kind": "periodic", "interval_days": 1},
                    "effects": []}}])
    assert not ok   # 壓測漏洞4：規則生規則


def test_rule_fire_lands_in_ledger():
    ws = make_world()
    _mk_rule(ws, rule_id="morning")
    advance_time(ws, 1440)
    hits = ws.social.ledger.query(actor="engine:rule:morning")
    assert len(hits) == 1
