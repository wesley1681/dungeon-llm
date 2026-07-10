from trpg.engine.ledger import Ledger, Commit


def test_append_assigns_monotonic_seq_and_freezes():
    led = Ledger()
    c0 = led.append("DISPOSITION", {"observer": "finn", "target": "party", "delta": 2},
                    actor="party", time_minutes=100)
    c1 = led.append("REVEAL", {"fact_id": "relic_at_finn"}, actor="party", time_minutes=110)
    assert (c0.seq, c1.seq) == (0, 1)
    assert len(led) == 2
    assert c0.ruling == {"kind": "none"}


def test_params_are_defensively_copied():
    led = Ledger()
    p = {"observer": "finn", "delta": 1}
    c = led.append("DISPOSITION", p, actor="party", time_minutes=0)
    p["delta"] = 99
    assert c.params["delta"] == 1


def test_query_filters_by_template_actor_and_where():
    led = Ledger()
    led.append("DISPOSITION", {"observer": "finn", "delta": 1}, actor="party", time_minutes=0)
    led.append("DISPOSITION", {"observer": "tomi", "delta": 1}, actor="party", time_minutes=5)
    led.append("TRANSACT", {"item": "金幣", "amount": 5}, actor="npc:finn", time_minutes=9)
    assert len(led.query(template_id="DISPOSITION")) == 2
    assert len(led.query(actor="npc:finn")) == 1
    hits = led.query(template_id="DISPOSITION", where=lambda c: c.params["observer"] == "tomi")
    assert len(hits) == 1 and hits[0].params["observer"] == "tomi"


def test_has_prior_supports_retroactive_defense():
    led = Ledger()
    assert not led.has_prior("OBJ_STATE", where=lambda c: c.params.get("object_id") == "酒杯")
    led.append("OBJ_STATE", {"object_id": "酒杯", "new_state": "下毒"}, actor="party", time_minutes=50)
    assert led.has_prior("OBJ_STATE", where=lambda c: c.params.get("object_id") == "酒杯")


def test_count_axis_counts_same_actor_axis_within_window():
    led = Ledger()
    for t in (0, 2000, 3000):
        led.append("DISPOSITION", {"observer": "finn", "target": "party", "delta": 3},
                   actor="party", time_minutes=t)
    led.append("DISPOSITION", {"observer": "tomi", "target": "party", "delta": 3},
               actor="party", time_minutes=2000)  # 不同軸（observer 不同）
    n = led.count_axis("DISPOSITION", "party", {"observer": "finn", "target": "party"},
                       since_minutes=1440, now_minutes=3010)
    assert n == 2  # t=2000 與 t=3000 在窗內（距 now <1440）；t=0 出窗；tomi 不同軸


def test_roundtrip_serialization():
    led = Ledger()
    led.append("REVEAL", {"fact_id": "x"}, actor="party", time_minutes=1,
               ruling={"kind": "check", "dc": 15, "roll": 17, "success": True, "reason": "唬騙"})
    led2 = Ledger.from_dicts(led.to_dicts())
    assert len(led2) == 1 and led2.query()[0].ruling["dc"] == 15
