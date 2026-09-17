"""Regression: leaving an NPC conversation must not re-trigger it next turn.

Bug: the player input that opens a conversation (e.g. "走向老柯說話") is stored
in GameSession._tag_actions and classified into [TALK] to enter the conversation.
_run_conversation only cleared that queue on its normal loop exit; the player's
"離開" took an early-return path that skipped the clear, so the stale input was
re-classified into [TALK] on the very next exploration turn — you could never
leave. Fix clears the queue on conversation entry (all exit paths covered).
"""
from trpg.scenarios.dungeon import build_world_state
from trpg.game import GameSession
from trpg.llm.controllers import ActorController, HumanController


class _StubTagAgent:
    def generate_tags(self, actions, on_chunk=None):
        return "無"

    def generate_conversation_tags(self, combined_input, npc_id=""):
        return "無"


class _StubNpcAgent:
    """Only the attributes _run_conversation reads before we leave.
    attitude/char_id are read by _mark_quests_offered, which now runs right after
    the NPC opening (before the loop) on every conversation."""
    attitude_label = "友善"
    attitude = 3            # 友善 — matches attitude_label
    char_id = "civilian"
    recruit_decision = ""
    pending_action = ""


def _build_session():
    """A session with every LLM-calling seam stubbed. Only the human controller
    (kaine) is real — it reads from the player-input queue."""
    ws = build_world_state()
    session = GameSession(
        world_state=ws,
        gm=object(),                        # not touched on the leave path
        tag_agent=_StubTagAgent(),
        thor_agent=object(),                # wrapped in a controller we overwrite
        npc_agents={"civilian": _StubNpcAgent()},  # ditto
    )
    # Silence the two LLM-driven slots: base ActorController is silent by default
    # (take_conversation_turn -> silent, take_npc_opening/response -> "").
    session.controllers["thor"] = ActorController()
    session.controllers["civilian"] = ActorController()
    assert isinstance(session.controllers["kaine"], HumanController)
    return session, ws


def test_leaving_conversation_clears_tag_actions():
    session, ws = _build_session()

    # Simulate the input that opened this conversation still sitting in the queue.
    session._tag_actions = ["凱恩：我走向老柯說話"]
    ws.pending_conversation = "civilian"

    # The player leaves.
    session.submit_player_input("離開")
    outcome = session._run_conversation("civilian")

    # The triggering input must be gone, or the next turn re-classifies it into
    # [TALK] and drops us straight back into the conversation.
    assert session._tag_actions == [], (
        "leaving a conversation left stale input queued -> would re-trigger [TALK]"
    )
    assert not outcome.game_over
    assert not outcome.triggered_combat


def test_quitting_conversation_clears_tag_actions():
    """The game-quit early-return path (get_input -> None) had the same leak."""
    from trpg.game import _STOP
    session, ws = _build_session()
    session._tag_actions = ["凱恩：我走向老柯說話"]

    # Feed the sentinel so the human controller returns quit=True mid-conversation.
    session._player_in.put(_STOP)
    outcome = session._run_conversation("civilian")

    assert session._tag_actions == []
    assert outcome.game_over


if __name__ == "__main__":
    test_leaving_conversation_clears_tag_actions()
    test_quitting_conversation_clears_tag_actions()
    print("ok")
