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
