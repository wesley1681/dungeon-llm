"""社會層測試共用建構器（比照 test_tag_parser 的 _make_world 慣例）。"""
from trpg.engine.character import Character, Stats
from trpg.engine.world_state import WorldState
from trpg.engine.social_state import NPCCard, KnowledgeItem


def make_char(cid: str, *, is_npc: bool = False, hp: int = 20) -> Character:
    c = Character(name=cid, race="人類", class_="平民", level=1, stats=Stats(),
                  hp=hp, max_hp=20, ac=10, is_npc=is_npc)
    return c


def make_world() -> WorldState:
    ws = WorldState(characters={"aria": make_char("aria"),
                                "finn": make_char("finn", is_npc=True)},
                    scene="碼頭區")
    ws.social.npc_cards["finn"] = NPCCard(
        npc_id="finn", name="芬恩",
        goals=["三天內把聖物賣給灰鴉換 500g"],
        disposition={"party": -1},
        knowledge={
            "relic_at_finn": KnowledgeItem(
                fact_id="relic_at_finn", text="聖物在芬恩手上",
                dc=15, min_disposition=3),
        },
        red_lines=["不出賣灰鴉的藏身處"],
        insight=14, will=12)
    ws.social.accounts["party"] = 100
    ws.social.accounts["finn"] = 40
    return ws
