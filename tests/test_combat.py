from trpg.engine.character import Stats, Character, CombatState


def test_stats_modifier_average():
    stats = Stats(STR=10, DEX=10, CON=10, INT=10, WIS=10, CHA=10)
    assert stats.modifier("STR") == 0


def test_stats_modifier_high():
    stats = Stats(STR=17, DEX=10, CON=10, INT=10, WIS=10, CHA=10)
    assert stats.modifier("STR") == 3  # (17-10)//2


def test_stats_modifier_low():
    stats = Stats(STR=8, DEX=10, CON=10, INT=10, WIS=10, CHA=10)
    assert stats.modifier("STR") == -1  # (8-10)//2


def test_character_proficiency_bonus_level1():
    char = _make_char(level=1)
    assert char.proficiency_bonus == 2


def test_character_proficiency_bonus_level5():
    char = _make_char(level=5)
    assert char.proficiency_bonus == 3


def test_character_is_alive():
    char = _make_char(hp=1)
    assert char.is_alive() is True


def test_character_is_dead_at_zero():
    char = _make_char(hp=0)
    assert char.is_alive() is False


def _make_char(name="Test", hp=20, max_hp=20, ac=12, level=1,
               str_=10, dex=10, is_npc=False):
    from trpg.engine.character import Character, Stats
    return Character(
        name=name, race="人類", class_="戰士", level=level,
        stats=Stats(STR=str_, DEX=dex, CON=10, INT=10, WIS=10, CHA=10),
        hp=hp, max_hp=max_hp, ac=ac, is_npc=is_npc,
    )


from unittest.mock import patch
from trpg.engine.combat import resolve_attack, apply_damage, make_saving_throw, apply_heal, roll_initiative
from trpg.engine.world_state import WorldState


def test_attack_hits_when_roll_exceeds_ac():
    attacker = _make_char("A", level=1)   # prof +2, STR mod 0 → total = roll+2
    target = _make_char("B", ac=5)
    with patch("trpg.engine.combat.roll", return_value=10):
        hit, total = resolve_attack(attacker, target)
    assert hit is True
    assert total == 12  # 10 + 0 (STR mod) + 2 (prof bonus)


def test_attack_misses_when_roll_below_ac():
    attacker = _make_char("A")
    target = _make_char("B", ac=20)
    with patch("trpg.engine.combat.roll", return_value=1):
        hit, total = resolve_attack(attacker, target)
    assert hit is False


def test_apply_damage_reduces_hp():
    target = _make_char("B", hp=20)
    with patch("trpg.engine.combat.roll", return_value=5):
        damage = apply_damage(target, "1d6")
    assert target.hp == 15
    assert damage == 5


def test_apply_damage_floors_at_zero():
    target = _make_char("B", hp=3)
    with patch("trpg.engine.combat.roll", return_value=10):
        apply_damage(target, "1d12")
    assert target.hp == 0


def test_apply_heal_restores_hp():
    target = _make_char("B", hp=10, max_hp=20)
    with patch("trpg.engine.combat.roll", return_value=6):
        healed = apply_heal(target, "1d8")
    assert target.hp == 16
    assert healed == 6


def test_apply_heal_caps_at_max_hp():
    target = _make_char("B", hp=18, max_hp=20)
    with patch("trpg.engine.combat.roll", return_value=10):
        apply_heal(target, "1d10")
    assert target.hp == 20


def test_saving_throw_success():
    char = _make_char("A", dex=16)  # DEX mod = +3
    with patch("trpg.engine.combat.roll", return_value=12):
        success, total = make_saving_throw(char, "DEX", dc=14)
    assert success is True
    assert total == 15  # 12 + 3


def test_saving_throw_failure():
    char = _make_char("A", dex=10)  # DEX mod = 0
    with patch("trpg.engine.combat.roll", return_value=5):
        success, total = make_saving_throw(char, "DEX", dc=14)
    assert success is False
    assert total == 5


def test_saving_throw_adds_proficiency_if_proficient():
    from trpg.engine.character import Character, Stats
    char = Character(
        name="A", race="人類", class_="戰士", level=1,
        stats=Stats(DEX=10), hp=10, max_hp=10, ac=12,
        proficiencies=["DEX"],
    )
    with patch("trpg.engine.combat.roll", return_value=10):
        success, total = make_saving_throw(char, "DEX", dc=12)
    assert total == 12  # 10 + 0 (mod) + 2 (prof)
    assert success is True


def test_roll_initiative_returns_sorted_order():
    from trpg.engine.character import Character, Stats
    world = WorldState(characters={
        "slow": _make_char("Slow", dex=8),   # DEX mod -1
        "fast": _make_char("Fast", dex=18),  # DEX mod +4
    }, scene="")
    side_effects = [5, 15]  # slow rolls 5, fast rolls 15
    with patch("trpg.engine.combat.roll", side_effect=side_effects):
        state = roll_initiative(world)
    assert state.initiative_order[0] == "fast"   # 15 + 4 = 19
    assert state.initiative_order[1] == "slow"   # 5 + (-1) = 4
