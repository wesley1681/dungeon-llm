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


def test_console_prompt_action_returns_end_on_zero(monkeypatch):
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="berserker", level=3,
        agent_pos=Vec2(15.0, 15.5), opp_pos=Vec2(15.0, 16.5),  # adjacent
        terrain="empty",
    )
    fe = ConsoleFrontend()
    fe._console = Console(file=StringIO(), width=120, force_terminal=False)
    inputs = iter(["0"])  # 0 = end turn
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    action = fe.prompt_action(ws, ws.characters["agent"],
                               {"action": 1, "bonus_action": 1, "movement": 9.0})
    assert action is None


def test_console_prompt_action_picks_weapon_attack(monkeypatch):
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="berserker", level=3,
        agent_pos=Vec2(15.0, 15.5), opp_pos=Vec2(15.0, 16.5),  # adjacent
        terrain="empty",
    )
    fe = ConsoleFrontend()
    fe._console = Console(file=StringIO(), width=120, force_terminal=False)
    # Find the index of the first weapon attack in the available_skills list.
    from trpg.engine.skill import available_skills
    skills = available_skills(ws.characters["agent"], ws)
    weapon_idx = next(i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:"))
    # The menu is 1-indexed (1..N), with 0 reserved for "end".
    inputs = iter([str(weapon_idx + 1), "1"])  # pick weapon, then target entity #1 (opponent)
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    action = fe.prompt_action(ws, ws.characters["agent"],
                               {"action": 1, "bonus_action": 1, "movement": 9.0})
    assert action is not None
    assert action["type"] == "ATTACK"
    assert action["target"] == "opponent"


def test_console_prompt_action_picks_move_to_coord(monkeypatch):
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="berserker", level=3,
        agent_pos=Vec2(10.0, 10.0), opp_pos=Vec2(20.0, 20.0),
        terrain="empty",
    )
    fe = ConsoleFrontend()
    fe._console = Console(file=StringIO(), width=120, force_terminal=False)
    from trpg.engine.skill import available_skills
    skills = available_skills(ws.characters["agent"], ws)
    move_idx = next(i for i, s in enumerate(skills) if s.skill_id == "move")
    inputs = iter([str(move_idx + 1), "12 12"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    action = fe.prompt_action(ws, ws.characters["agent"],
                               {"action": 1, "bonus_action": 1, "movement": 9.0})
    assert action is not None
    assert action["type"] == "MOVE"


def test_console_prompt_action_reprompts_on_bad_index(monkeypatch):
    ws = build_world_state(
        agent_arch="berserker", opponent_arch="berserker", level=3,
        agent_pos=Vec2(15.0, 15.5), opp_pos=Vec2(15.0, 16.5),
        terrain="empty",
    )
    fe = ConsoleFrontend()
    fe._console = Console(file=StringIO(), width=120, force_terminal=False)
    # First two inputs are garbage; third is "end"
    inputs = iter(["abc", "999", "0"])
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))
    action = fe.prompt_action(ws, ws.characters["agent"],
                               {"action": 1, "bonus_action": 1, "movement": 9.0})
    assert action is None


def test_render_includes_character_card():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker", level=5,
        agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0), terrain="empty",
    )
    fe = ConsoleFrontend()
    text = _render_to_string(fe, ws)
    # Agent name should appear somewhere (in the card)
    assert ws.characters["agent"].name in text
