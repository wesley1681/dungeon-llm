from unittest.mock import patch
from tests.social.helpers import make_world
from trpg.engine.adjudicator import (
    Ruling, rule_auto, rule_impossible, rule_check, rule_oracle, lookup_dc)


def test_lookup_dc_sources():
    ws = make_world()
    assert lookup_dc(ws, ("npc_insight", "finn")) == 14
    assert lookup_dc(ws, ("npc_will", "finn")) == 12
    ws.social.object_dcs["鐵門"] = 17
    assert lookup_dc(ws, ("object", "鐵門")) == 17
    assert lookup_dc(ws, ("object", "無此物")) is None
    assert lookup_dc(ws, ("info", "finn", "relic_at_finn")) == 15


def test_rule_check_rolls_engine_dice_and_records():
    ws = make_world()
    with patch("trpg.engine.adjudicator.roll_d20", return_value=17):
        r = rule_check(ws, dc_from=("npc_insight", "finn"), modifier=2, reason="唬騙")
    assert r.kind == "check" and r.dc == 14 and r.roll == 19 and r.success
    with patch("trpg.engine.adjudicator.roll_d20", return_value=3):
        r = rule_check(ws, dc=15, reason="撬鎖")
    assert not r.success and r.as_dict()["reason"] == "撬鎖"


def test_rule_check_requires_resolvable_dc():
    ws = make_world()
    r = rule_check(ws, dc_from=("object", "無此物"), reason="x")
    assert r.kind == "impossible"   # DC 無來源＝不裁決，不許 LLM 憑空編 DC


def test_rule_oracle_thresholds():
    ws = make_world()
    with patch("trpg.engine.adjudicator.roll_d20", return_value=7):
        assert rule_oracle(ws, likelihood="很可能", reason="有淺灘?").success
        assert not rule_oracle(ws, likelihood="不太可能", reason="有神劍?").success
    r = rule_oracle(ws, likelihood="包準有", reason="x")
    assert r.kind == "impossible"


def test_auto_and_impossible():
    a = rule_auto("無阻力")
    assert a.kind == "auto" and a.success
    i = rule_impossible("違反世界規則")
    assert i.kind == "impossible" and not i.success
