from unittest.mock import patch
from trpg.engine.character import Character, Stats, CombatState
from trpg.engine.world_state import WorldState
from trpg.llm.tag_parser import parse_and_resolve


def _make_world():
    aria = Character(
        name="艾里亞", race="人類", class_="盜賊", level=3,
        stats=Stats(STR=10, DEX=16, CON=12, INT=13, WIS=11, CHA=14),
        hp=22, max_hp=22, ac=14, is_npc=False,
        proficiencies=["DEX", "INT"],
    )
    goblin = Character(
        name="地精甲", race="地精", class_="—", level=1,
        stats=Stats(STR=8, DEX=14, CON=10, INT=8, WIS=8, CHA=8),
        hp=7, max_hp=7, ac=13, is_npc=True,
    )
    return WorldState(characters={"aria": aria, "goblin_1": goblin}, scene="")


def test_no_tags_returns_unchanged():
    world = _make_world()
    text = "你們進入了地下城，空氣潮濕。"
    cleaned, results = parse_and_resolve(text, world)
    assert cleaned == text
    assert results == []


def test_damage_tag_reduces_hp():
    world = _make_world()
    with patch("trpg.engine.combat.roll", return_value=4):
        cleaned, results = parse_and_resolve(
            "地精攻擊！[DAMAGE: 1d6+2 to aria]", world
        )
    assert world.characters["aria"].hp == 22 - 4   # mock returns 4; modifier ignored by mock
    assert len(results) == 1
    assert "艾里亞" in results[0]
    assert "[DAMAGE:" not in cleaned


def test_heal_tag_restores_hp():
    world = _make_world()
    world.characters["aria"].hp = 10
    with patch("trpg.engine.combat.roll", return_value=5):
        cleaned, results = parse_and_resolve("[HEAL: aria 1d6]", world)
    assert world.characters["aria"].hp == 15
    assert "艾里亞" in results[0]


def test_status_tag_adds_effect():
    # status_effects 裝 StatusEffect 物件（非字串）——字串 in 永遠 False；
    # 用引擎原生 has_status 查名字。
    world = _make_world()
    cleaned, results = parse_and_resolve("[STATUS: aria +poisoned]", world)
    assert world.characters["aria"].has_status("poisoned")
    assert "艾里亞" in results[0]


def test_status_tag_removes_effect():
    # 舊版直接 append 字串再驗字串 not-in 物件列表＝永遠 True＝瞎測試。
    # 改用引擎路徑（+ 標籤）建立狀態，再驗 - 標籤真的移除。
    world = _make_world()
    parse_and_resolve("[STATUS: aria +poisoned]", world)
    assert world.characters["aria"].has_status("poisoned")
    cleaned, results = parse_and_resolve("[STATUS: aria -poisoned]", world)
    assert not world.characters["aria"].has_status("poisoned")


def test_initiative_tag_creates_combat_state():
    world = _make_world()
    with patch("trpg.engine.combat.roll", return_value=10):
        cleaned, results = parse_and_resolve("[INITIATIVE: start]", world)
    assert world.combat is not None
    assert len(world.combat.initiative_order) == 2
    assert len(results) == 1


def test_unknown_character_returns_error_string():
    world = _make_world()
    cleaned, results = parse_and_resolve("[DAMAGE: 1d6 to unknown_char]", world)
    assert "找不到" in results[0]
