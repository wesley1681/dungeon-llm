import pytest
from trpg.sandbox.frontend import Frontend, ScriptedFrontend


def test_protocol_can_be_imported():
    # Protocol existence and method names — a static check that we accept
    # ScriptedFrontend as a Frontend.
    assert hasattr(Frontend, "render")
    assert hasattr(Frontend, "prompt_action")
    assert hasattr(Frontend, "announce")


def test_scripted_returns_canned_actions_in_order():
    actions = [
        {"type": "MOVE", "target_position": [5.0, 5.0], "skill_id": "move"},
        None,  # end turn
    ]
    fe = ScriptedFrontend(actions)
    assert fe.prompt_action(None, None, {}) == actions[0]
    assert fe.prompt_action(None, None, {}) is None


def test_scripted_raises_when_exhausted():
    fe = ScriptedFrontend([None])
    fe.prompt_action(None, None, {})
    with pytest.raises(StopIteration):
        fe.prompt_action(None, None, {})


def test_scripted_records_render_and_announce_calls():
    fe = ScriptedFrontend([None])
    fe.render("ws_placeholder", "agent", "opponent")
    fe.announce("hello")
    fe.announce("world")
    assert fe.render_calls == 1
    assert fe.announcements == ["hello", "world"]


from io import StringIO
from rich.console import Console
from trpg.engine.vec2 import Vec2
from trpg.sandbox.setup import build_world_state
from trpg.sandbox.frontend import ConsoleFrontend


def _render_to_string(fe: ConsoleFrontend, ws) -> str:
    """Capture ConsoleFrontend output to a string for assertions."""
    buf = StringIO()
    fe._console = Console(file=buf, width=120, force_terminal=False)
    fe.render(ws, "agent", "opponent")
    return buf.getvalue()


def test_console_render_smoke():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker", level=5,
        agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0), terrain="empty",
    )
    fe = ConsoleFrontend()
    text = _render_to_string(fe, ws)
    # Should contain both marks
    assert "A" in text
    assert "O" in text


def test_console_render_lava_glyph():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker", level=5,
        agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0), terrain="lava_strip",
    )
    fe = ConsoleFrontend()
    text = _render_to_string(fe, ws)
    assert "░" in text   # lava glyph


def test_console_render_arena_walls():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker", level=5,
        agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0), terrain="arena",
    )
    fe = ConsoleFrontend()
    text = _render_to_string(fe, ws)
    assert "█" in text   # wall glyph
