import pytest
from trpg.engine.dice import roll


def test_d20_in_range():
    for _ in range(50):
        assert 1 <= roll("1d20") <= 20


def test_modifier_applied():
    # With d1 (always 1), result must equal 1 + modifier
    assert roll("1d1+3") == 4
    assert roll("1d1-1") == 0


def test_multiple_dice():
    for _ in range(50):
        assert 2 <= roll("2d6") <= 12


def test_no_count_prefix():
    for _ in range(50):
        assert 1 <= roll("d8") <= 8


def test_invalid_notation_raises():
    with pytest.raises(ValueError):
        roll("invalid")


def test_invalid_notation_raises_on_plain_number():
    with pytest.raises(ValueError):
        roll("5")
