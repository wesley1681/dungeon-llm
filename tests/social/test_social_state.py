from tests.social.helpers import make_world
from trpg.engine.social_state import SocialState, Clock
from trpg.engine.world_state import WorldState


def test_worldstate_gets_social_by_default():
    ws = WorldState(characters={}, scene="")
    assert isinstance(ws.social, SocialState)
    assert ws.social.time_minutes == 0
    assert len(ws.social.ledger) == 0


def test_two_worlds_do_not_share_social():
    a = WorldState(characters={}, scene="")
    b = WorldState(characters={}, scene="")
    a.social.flags.add("x")
    assert "x" not in b.social.flags


def test_helper_builds_npc_card():
    ws = make_world()
    card = ws.social.npc_cards["finn"]
    assert card.disposition["party"] == -1
    assert card.knowledge["relic_at_finn"].min_disposition == 3


def test_social_roundtrip_serialization():
    ws = make_world()
    ws.social.ledger.append("REVEAL", {"fact_id": "x"}, actor="party", time_minutes=3)
    ws.social.clocks["deal"] = Clock(
        clock_id="deal", name="聖物交易", remaining_days=3.0,
        on_expire=[{"template": "REVEAL", "params": {"fact_id": "deal_done"}}])
    ws.social.resources[("aria", "補給")] = 2
    data = ws.social.to_dict()
    s2 = SocialState.from_dict(data)
    assert s2.npc_cards["finn"].insight == 14
    assert s2.npc_cards["finn"].knowledge["relic_at_finn"].dc == 15
    assert s2.clocks["deal"].remaining_days == 3.0
    assert s2.resources[("aria", "補給")] == 2
    assert len(s2.ledger) == 1
