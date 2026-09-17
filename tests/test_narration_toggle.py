"""The 旁白 (GM narration) off-switch — a testing aid.

When narration is off, the GM's exploration prose and combat flavour are skipped
(no LLM call), but mechanics run unchanged and the turn passes to the next actor.
The toggle is a meta-command intercepted in submit_player_input (the single input
point every front-end routes through), so it works in cli / desktop / web alike.
"""
from trpg.scenarios.dungeon import build_world_state, build_npc_agents
from trpg.game import GameSession, ExplorationPrompt, StatusMessage, _STOP
from trpg.llm.controllers import ActorController, HumanController


class _StubTagAgent:
    def generate_tags(self, actions, on_chunk=None):
        return "無"


class _RecordingGM:
    """Records whether narration was actually invoked."""
    def __init__(self):
        self.generate_calls = 0
        self.combat_calls = 0

    def generate(self, tag_results=None, on_chunk=None):
        self.generate_calls += 1
        return "（GM 旁白）"

    def combat_narrate(self, result_text, on_chunk=None):
        self.combat_calls += 1
        return "（戰鬥旁白）"


def _session(gm=None):
    ws = build_world_state()
    npc_agents = build_npc_agents(ws, model="stub", base_url="stub", backend="stub")
    session = GameSession(
        world_state=ws, gm=gm or _RecordingGM(), tag_agent=_StubTagAgent(),
        thor_agent=object(), npc_agents=npc_agents,
    )
    session.controllers["thor"] = ActorController()   # silent — no LLM
    assert isinstance(session.controllers["kaine"], HumanController)
    return session, ws


def _drain_status(session) -> str:
    """Return the text of the first StatusMessage currently queued, else ''."""
    while True:
        ev = session.next_event(block=False)
        if ev is None:
            return ""
        if isinstance(ev, StatusMessage):
            return ev.text


# ── The toggle command ────────────────────────────────────────────────────────

def test_toggle_command_flips_flag_without_queuing_an_action():
    session, _ = _session()
    assert session.narrate is True                 # default: narration on

    session.submit_player_input("/旁白")            # bare → toggle off
    assert session.narrate is False
    assert session._player_in.empty()              # consumed, not a game action
    assert "關閉" in _drain_status(session)

    session.submit_player_input("/旁白 on")
    assert session.narrate is True
    session.submit_player_input("/旁白 off")
    assert session.narrate is False

    session.submit_player_input("旁白")             # alias, toggles → on
    assert session.narrate is True
    session.submit_player_input("/narration off")  # english alias
    assert session.narrate is False


def test_real_action_is_not_intercepted():
    session, _ = _session()
    session.submit_player_input("我往北走")
    assert session._player_in.get_nowait() == "我往北走"
    assert session.narrate is True                 # untouched by a normal action


def test_toggle_repeats_the_pending_prompt():
    """A consumed meta-command must not leave every front-end waiting forever."""
    session, ws = _session()
    prompt = ExplorationPrompt(kaine=ws.characters["kaine"], gm_text="")
    session._emit(prompt)
    assert session.next_event(block=False) is prompt

    session.submit_player_input("/旁白 off")

    assert isinstance(session.next_event(block=False), StatusMessage)
    assert session.next_event(block=False) is prompt
    assert session._player_in.empty()


def test_rest_repeats_the_exploration_prompt():
    """Rest consumes one input, reports recovery, then asks for the real action."""
    session, ws = _session()
    submitted = iter(["short rest", "往北走"])
    emitted = []
    controller = HumanController(
        "kaine", lambda: next(submitted), emitted.append,
    )

    result = controller.take_exploration_turn(ws.characters["kaine"])

    assert result.text == "往北走"
    assert sum(isinstance(event, ExplorationPrompt) for event in emitted) == 2
    assert sum(isinstance(event, StatusMessage) for event in emitted) == 1


# ── Exploration gate ──────────────────────────────────────────────────────────

def test_narration_off_skips_gm_generate():
    gm = _RecordingGM()
    session, ws = _session(gm=gm)
    session.narrate = False
    session._player_in.put(_STOP)                  # kaine quits → turn ends
    done, _ = session._exploration_turn(skip_gm=False)
    assert done is True
    assert gm.generate_calls == 0                  # 旁白 off → no GM prose / LLM call
    assert all(e["speaker"] != "gm" for e in ws.narrative_log)  # nothing logged


def test_narration_on_calls_gm_generate():
    gm = _RecordingGM()
    session, ws = _session(gm=gm)
    assert session.narrate is True
    session._player_in.put(_STOP)
    session._exploration_turn(skip_gm=False)
    assert gm.generate_calls == 1                  # 旁白 on → GM narrates as usual


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
