from tests.social.helpers import make_world
from trpg.engine.predicates import validate_predicate, evaluate_predicate
from trpg.engine.social_state import Clock


def test_unknown_kind_rejected():
    ws = make_world()
    errs = validate_predicate({"kind": "兩個月亮同升"}, ws)
    assert errs  # 表二通用規則：不可評估的條件拒絕 commit


def test_missing_param_rejected():
    ws = make_world()
    assert validate_predicate({"kind": "clock_expired"}, ws)


def test_clock_expired_requires_existing_clock_and_evaluates():
    ws = make_world()
    assert validate_predicate({"kind": "clock_expired", "clock_id": "no_such"}, ws)
    ws.social.clocks["deal"] = Clock(clock_id="deal", name="交易", remaining_days=1.0)
    pred = {"kind": "clock_expired", "clock_id": "deal"}
    assert validate_predicate(pred, ws) == []
    assert evaluate_predicate(pred, ws) is False
    ws.social.clocks["deal"].expired = True
    assert evaluate_predicate(pred, ws) is True


def test_time_after_and_flag_and_disposition():
    ws = make_world()
    assert evaluate_predicate({"kind": "time_after", "minutes": 100}, ws) is False
    ws.social.time_minutes = 101
    assert evaluate_predicate({"kind": "time_after", "minutes": 100}, ws) is True

    assert evaluate_predicate({"kind": "flag", "flag_id": "seen"}, ws) is False
    ws.social.flags.add("seen")
    assert evaluate_predicate({"kind": "flag", "flag_id": "seen"}, ws) is True

    pred = {"kind": "disposition_at_least", "npc_id": "finn", "target": "party", "value": 0}
    assert evaluate_predicate(pred, ws) is False   # helpers 卡上 -1
    ws.social.npc_cards["finn"].disposition["party"] = 1
    assert evaluate_predicate(pred, ws) is True


def test_entity_removed():
    ws = make_world()
    pred = {"kind": "entity_removed", "entity_id": "finn"}
    assert validate_predicate(pred, ws) == []
    assert evaluate_predicate(pred, ws) is False
    ws.social.npc_cards["finn"].removed = "死亡"
    assert evaluate_predicate(pred, ws) is True
