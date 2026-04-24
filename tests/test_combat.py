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
