"""Headless tests for the Jev-backed conversation resolution pipeline."""
from types import SimpleNamespace

from trpg.scenarios.dungeon import build_world_state, build_npc_agents
from trpg.game import GameSession
from trpg.llm.controllers import ActorController, HumanController
from trpg.llm.dialogue_flow import DialogueDirector, CheckPlan
from trpg.engine.items import Consumable


class _StubTagAgent:
    # Only the connection attrs DialogueDirector.from_agent reads (all absent →
    # None, which is fine — the director is inert unless a stage is called).
    def generate_tags(self, actions, on_chunk=None):
        return "無"


def _session():
    ws = build_world_state()
    npc_agents = build_npc_agents(ws, model="stub", base_url="stub", backend="stub")
    session = GameSession(
        world_state=ws, gm=object(), tag_agent=_StubTagAgent(),
        player_agents={}, npc_agents=npc_agents,
    )
    session.controllers["thor"] = ActorController()
    for cid in npc_agents:
        session.controllers[cid] = ActorController()
    assert isinstance(session.controllers["kaine"], HumanController)
    return session, ws, npc_agents


# ── NpcAgent structural: attitude self-marking removed, new channels added ─────

def test_npc_agent_lost_attitude_self_marking():
    import trpg.llm.npc_agent as na
    assert not hasattr(na, "_MARKER_RE"), "[+/-] attitude marker must be gone"
    assert not hasattr(na, "_FORCE_DETAILS")
    _, _, npc_agents = _session()
    civ = npc_agents["civilian"]
    assert hasattr(civ, "revealed") and civ.revealed == []
    assert hasattr(civ, "pending_directives") and civ.pending_directives == []
    assert not hasattr(civ, "force_reveal")
    assert not hasattr(civ, "_skip_marker")


# ── Quest applier ─────────────────────────────────────────────────────────────

def test_quest_accept_complete_turnin_lifecycle():
    session, ws, npc = _session()
    q = ws.quests["moonlight_grass"]
    assert q.status == "inactive"

    session._apply_quest_events([{"event": "quest_accept", "id": "moonlight_grass"}])
    assert q.status == "active"

    session._apply_quest_events([{"event": "quest_complete", "id": "moonlight_grass"}])
    assert q.status == "completed", "narrative completion path active→completed"

    session._apply_quest_events([{"event": "quest_turnin", "id": "moonlight_grass"}])
    assert q.status == "turned_in"


def test_quest_event_bad_id_ignored():
    session, ws, npc = _session()
    session._apply_quest_events([{"event": "quest_accept", "id": "does_not_exist"}])
    # nothing crashes; real quest untouched
    assert ws.quests["moonlight_grass"].status == "inactive"


# ── DialogueDirector Jev + parse/filter seams ─────────────────────────────────

class _FakeSocialJevClient:
    def __init__(self, *, required=0.95, check_type="persuade", score=3.0):
        self.required = required
        self.check_type = check_type
        self.score = score
        self.call_count = 0
        self.last_state = None
        self.last_questions = None

    def system_one(self, *, state, questions, model):
        self.call_count += 1
        self.last_state = state
        self.last_questions = questions
        answers = {}
        for question_id, question in questions.items():
            if question.type == "noul":
                answers[question_id] = SimpleNamespace(noul=self.required)
            elif question.type == "choice":
                answers[question_id] = SimpleNamespace(
                    choice=self.check_type,
                    confidence=0.96,
                    probabilities={key: 0.0 for key in question.criteria},
                )
            else:
                answers[question_id] = SimpleNamespace(
                    score=self.score,
                    confidence=0.94,
                    probabilities={},
                )
        return SimpleNamespace(
            model="jev-test",
            answers=answers,
            nouls={key: value for key, value in answers.items()
                   if hasattr(value, "noul")},
            choices={key: value for key, value in answers.items()
                     if hasattr(value, "choice")},
            scores={key: value for key, value in answers.items()
                    if hasattr(value, "score")},
        )


class _FakeNpcActionJevClient:
    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities or {}
        self.error = error
        self.call_count = 0
        self.last_state = None
        self.last_questions = None

    def system_one(self, *, state, questions, model):
        self.call_count += 1
        self.last_state = state
        self.last_questions = questions
        if self.error:
            raise self.error
        answers = {
            question_id: SimpleNamespace(
                noul=self.probabilities.get(question_id, 0.0))
            for question_id in questions
        }
        return SimpleNamespace(model="jev-test", answers=answers, nouls=answers)


class _FakeChoiceJevClient:
    def __init__(self, selections):
        self.selections = selections
        self.call_count = 0
        self.last_state = None
        self.last_questions = None

    def system_one(self, *, state, questions, model):
        self.call_count += 1
        self.last_state = state
        self.last_questions = questions
        answers = {}
        for question_id, question in questions.items():
            if question.type == "noul":
                answers[question_id] = SimpleNamespace(
                    noul=float(self.selections.get(question_id, 0.0)))
            else:
                answers[question_id] = SimpleNamespace(
                    choice=self.selections.get(question_id, "none"),
                    confidence=0.99,
                    probabilities={},
                )
        return SimpleNamespace(
            model="jev-test",
            answers=answers,
            nouls={key: value for key, value in answers.items()
                   if hasattr(value, "noul")},
            choices={key: value for key, value in answers.items()
                     if hasattr(value, "choice")},
        )


def test_decide_check_uses_noul_choice_and_score_in_one_jev_call():
    _, ws, npc = _session()
    client = _FakeSocialJevClient(
        required=0.95, check_type="persuade", score=4.0)
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)
    plan = d.decide_check(
        [{"actor_id": "kaine", "text": "我求你告訴我"}],
        npc["civilian"], ws,
    )

    assert plan == CheckPlan(check="說服", dc=12, actor="kaine")
    assert client.call_count == 1
    assert {question.type for question in client.last_questions.values()} == {
        "noul", "choice", "score",
    }
    required = client.last_questions["social_0_required"]
    assert required.criteria["true"] and required.criteria["false"]
    assert client.last_state["player_actions"][0]["actor_id"] == "kaine"
    assert set(client.last_state) == {"player_actions", "npc"}
    score_prompt = client.last_questions["social_0_effectiveness"].instructions
    assert "假設" not in score_prompt
    assert "CHA" not in score_prompt
    assert "conversation_history" not in score_prompt


def test_decide_check_noul_is_a_yes_no_gate_not_an_intensity():
    _, ws, npc = _session()
    client = _FakeSocialJevClient(
        required=0.50, check_type="intimidate", score=6.0)
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    plan = d.decide_check(
        [{"actor_id": "kaine", "text": "你好啊"}], npc["civilian"], ws)

    assert plan.check is None


def test_npc_actions_include_actor_target_and_four_nouls():
    _, ws, npc = _session()
    client = _FakeNpcActionJevClient({
        "give": 0.96,
        "offer_quest": 0.88,
    })
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    actions = d.judge_npc_actions(
        "這瓶藥給你，也請幫我找回女兒。", npc["civilian"], ws)

    assert actions == [
        {"event": "give", "actor_id": "civilian", "target_id": "kaine"},
        {"event": "offer_quest", "actor_id": "civilian", "target_id": "kaine"},
    ]
    assert client.call_count == 1
    utterance = client.last_state["utterance"]
    assert utterance["text"] == "這瓶藥給你，也請幫我找回女兒。"
    assert utterance["actor"] == {"id": "civilian", "name": "老柯"}
    assert utterance["target"] == {"id": "kaine", "name": "凱恩"}
    assert "side" not in utterance["actor"]
    assert set(client.last_questions) == {
        "attack", "join", "give", "offer_quest",
    }
    assert {q.type for q in client.last_questions.values()} == {"noul"}


def test_judge_npc_actions_derives_none_and_falls_back_safely():
    _, ws, npc = _session()
    quiet = _FakeNpcActionJevClient()
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=quiet)
    assert d.judge_npc_actions("今天天氣不錯。", npc["civilian"], ws) == []

    broken = _FakeNpcActionJevClient(error=RuntimeError("offline"))
    d2 = DialogueDirector("m", "u", "b", ws, typesafe_client=broken)
    assert d2.judge_npc_actions("請跟我來。", npc["civilian"], ws) == []


def test_resolve_dialogue_actions_only_sends_valid_quest_candidates():
    _, ws, npc = _session()
    client = _FakeChoiceJevClient({"offered_quest": "quest_0"})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    events = d.resolve_dialogue_actions(
        "能請你們幫我採集月光草嗎？", [{
            "event": "offer_quest",
            "actor_id": "civilian",
            "target_id": "kaine",
        }], ws)

    assert events == [{
        "event": "offer_quest",
        "actor_id": "civilian",
        "target_id": "kaine",
        "id": "moonlight_grass",
    }]
    assert client.call_count == 1
    assert client.last_state["candidates"]["items"] == []
    assert client.last_state["candidates"]["quests"][0]["id"] == "moonlight_grass"


def test_resolve_give_uses_actor_inventory_and_target():
    _, ws, npc = _session()
    ws.characters["civilian"].consumables.append(
        Consumable("治療藥水", 1, "heal", "2d4+2"))
    client = _FakeChoiceJevClient({"give_item": "item_0"})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    events = d.resolve_dialogue_actions(
        "這瓶治療藥水給你。", [{
            "event": "give",
            "actor_id": "civilian",
            "target_id": "kaine",
        }], ws)

    assert events == [{
        "event": "give",
        "actor_id": "civilian",
        "target_id": "kaine",
        "item": "治療藥水",
    }]


def test_npc_join_offer_requires_player_acceptance_before_party_change():
    session, ws, npc = _session()
    civ = npc["civilian"]
    civ_char = ws.characters["civilian"]
    session.dialogue = SimpleNamespace(
        judge_npc_actions=lambda text, agent, state, target_id=None: [{
            "event": "join",
            "actor_id": "civilian",
            "target_id": "kaine",
        }],
        resolve_dialogue_actions=lambda text, events, state: events,
    )

    assert session._apply_npc_dialogue_actions(
        "我想跟你們一起冒險。", civ, civ_char) is False
    assert civ.pending_join_offer == {
        "proposer_id": "civilian",
        "responder_id": "kaine",
        "candidate_id": "civilian",
    }
    assert "civilian" not in ws.party_ids

    assert session._apply_player_dialogue_actions(
        [{"event": "join_accept", "candidate_id": "civilian"}],
        civ, civ_char) is False
    assert civ.pending_join_offer is None
    assert "civilian" in ws.party_ids


def test_player_attack_action_starts_combat_immediately():
    session, ws, npc = _session()
    civ = npc["civilian"]
    civ_char = ws.characters["civilian"]

    assert session._apply_player_dialogue_actions([{
        "event": "attack",
        "actor_id": "kaine",
        "target_id": "civilian",
    }], civ, civ_char) is True
    assert ws.combat is not None and ws.combat.active


def test_judge_player_attack_uses_noul():
    _, ws, npc = _session()
    client = _FakeChoiceJevClient({"attack": 0.95})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    events = d.judge_player_actions("我拔劍立刻砍向你。", npc["civilian"], ws)

    assert events == [{
        "event": "attack",
        "actor_id": "kaine",
        "target_id": "civilian",
    }]
    assert set(client.last_questions) == {"attack", "join", "give"}
    assert "offer_quest" not in client.last_questions
    assert client.last_questions["attack"].type == "noul"


def test_player_confirmation_uses_only_pending_offers():
    _, ws, npc = _session()
    civ = npc["civilian"]
    civ.pending_join_offer = {
        "proposer_id": "civilian",
        "responder_id": "kaine",
        "candidate_id": "civilian",
    }
    ws.quests["moonlight_grass"].offered = True
    client = _FakeChoiceJevClient({
        "join_response": "accept",
        "quest_0_response": "reject",
    })
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    events = d.judge_player_actions(
        "你可以加入，但我不接這個任務。", civ, ws)

    assert events == [
        {
            "event": "join_accept",
            "actor_id": "kaine",
            "target_id": "civilian",
            "candidate_id": "civilian",
        },
        {
            "event": "quest_reject",
            "id": "moonlight_grass",
            "actor_id": "kaine",
            "target_id": "civilian",
        },
    ]
    assert set(client.last_questions) == {
        "attack", "join", "give", "join_response", "quest_0_response",
    }


def test_player_join_offer_is_accepted_by_npc_without_second_player_consent():
    session, ws, npc = _session()
    civ = npc["civilian"]
    civ_char = ws.characters["civilian"]

    session._apply_player_dialogue_actions([{
        "event": "join",
        "actor_id": "kaine",
        "target_id": "civilian",
    }], civ, civ_char)
    assert civ.pending_join_offer == {
        "proposer_id": "kaine",
        "responder_id": "civilian",
        "candidate_id": "civilian",
    }

    client = _FakeChoiceJevClient({"join_response": "accept"})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)
    events = d.judge_npc_actions(
        "好，我加入你們。", civ, ws, target_id="kaine")

    assert events == [{
        "event": "join_accept",
        "actor_id": "civilian",
        "target_id": "kaine",
        "candidate_id": "civilian",
    }]
    session._apply_dialogue_actions(events, civ, civ_char)
    assert "civilian" in ws.party_ids
    assert civ.pending_join_offer is None


def test_player_give_uses_same_catalog_resolution_and_dispatch():
    session, ws, npc = _session()
    giver = ws.characters["kaine"]
    recipient = ws.characters["civilian"]
    giver.weapons = []
    giver.gear = []
    giver.consumables = [Consumable("治療藥水", 1, "heal", "2d4+2")]
    recipient.consumables = []
    client = _FakeChoiceJevClient({"give": 0.96, "give_item": "item_0"})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)

    events = d.judge_player_actions({
        "actor_id": "kaine",
        "text": "這瓶治療藥水給你。",
    }, npc["civilian"], ws)
    events = d.resolve_dialogue_actions("這瓶治療藥水給你。", events, ws)

    assert events == [{
        "event": "give",
        "actor_id": "kaine",
        "target_id": "civilian",
        "item": "治療藥水",
    }]
    session._apply_player_dialogue_actions(
        events, npc["civilian"], recipient)
    assert giver.get_consumable("治療藥水") is None
    assert recipient.get_consumable("治療藥水").quantity == 1


def test_judge_completion_detects_and_validates_id():
    _, ws, npc = _session()
    ws.quests["moonlight_grass"].status = "active"
    client = _FakeNpcActionJevClient({"complete_0": 0.91})
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)
    evs = d.judge_completion("你答應停手嗎", "好啦我不收保護費了", ws)
    assert evs == [{"event": "quest_complete", "id": "moonlight_grass"}]
    assert set(client.last_questions) == {"complete_0"}


def test_judge_completion_skips_when_no_active_quests():
    _, ws, npc = _session()
    # moonlight_grass is inactive; no active quests → empty scope → skip LLM call.
    client = _FakeNpcActionJevClient()
    d = DialogueDirector("m", "u", "b", ws, typesafe_client=client)
    assert d.judge_completion("嗨", "哼", ws) == []
    assert client.call_count == 0


# ── Integration: _run_conversation wires the 3 stages together ────────────────

class _StubNpcCtrl(ActorController):
    def take_npc_opening(self, char):
        return "你是誰？"
    def take_npc_response(self, char):
        return "好…我告訴你守衛室的事，也接下你的請求。"


class _StubDirector:
    """Canned outputs so the full pipeline runs deterministically."""
    def __init__(self):
        self.npc_action_calls = []
        self.player_confirmation_calls = []
    def decide_check(self, player_text, npc, ws):
        return CheckPlan(check="說服", dc=0, actor="kaine")
    def judge_npc_actions(self, npc_text, npc, ws, target_id=None):
        self.npc_action_calls.append(npc_text)
        return []
    def resolve_dialogue_actions(self, text, events, ws):
        return events
    def judge_player_actions(self, player_action, npc, ws):
        self.player_confirmation_calls.append(player_action)
        return []
    def judge_completion(self, player_text, npc_response, ws):
        return []


def test_run_conversation_chains_all_stages():
    session, ws, npc = _session()
    session.controllers["civilian"] = _StubNpcCtrl()
    session.dialogue = _StubDirector()
    civ = npc["civilian"]

    # 凱恩 speaks once (triggers the pipeline), then leaves.
    session.submit_player_input("我求求你，把守衛室的情況告訴我")
    session.submit_player_input("離開")
    outcome = session._run_conversation("civilian")

    assert not outcome.game_over
    assert session.dialogue.npc_action_calls == [
        "你是誰？",
        "好…我告訴你守衛室的事，也接下你的請求。",
    ]
    assert session.dialogue.player_confirmation_calls == [
        {
            "actor_id": "kaine",
            "actor_name": "凱恩",
            "text": "我求求你，把守衛室的情況告訴我",
        },
    ]
    # Leaving still clears the trigger queue (no regression of the leave fix).
    assert session._tag_actions == []


class _OfferingNpcCtrl(ActorController):
    """The NPC voices the quest offer in its OPENING line (as 老柯 does at 戒備+)."""
    def take_npc_opening(self, char):
        return "求求你們，能幫我採三朵月光草嗎？只要拿回來，我的藥草都分你們！"
    def take_npc_response(self, char):
        return "太好了，謝謝你們！"


class _GateFaithfulDirector:
    """Accepts a quest only after its spoken offer was recorded."""
    def decide_check(self, player_text, npc, ws):
        return CheckPlan(check=None, dc=0, actor="kaine")
    def judge_npc_actions(self, npc_text, npc, ws, target_id=None):
        if "月光草" not in npc_text:
            return []
        return [{
            "event": "offer_quest",
            "actor_id": "civilian",
            "target_id": target_id or "kaine",
        }]
    def resolve_dialogue_actions(self, text, events, ws):
        return [
            {**event, "id": "moonlight_grass"}
            if event.get("event") == "offer_quest" else event
            for event in events
        ]
    def judge_player_actions(self, player_action, npc, ws):
        if (ws.quests["moonlight_grass"].offered
                and "接下" in player_action["text"]):
            return [{
                "event": "quest_accept",
                "id": "moonlight_grass",
                "actor_id": player_action["actor_id"],
                "target_id": "civilian",
            }]
        return []
    def judge_completion(self, player_text, npc_response, ws):
        return []


def test_opening_offer_unlocks_first_turn_accept():
    # Regression: an offer voiced in the opening must be recorded before the
    # player's first reply is classified.
    session, ws, npc = _session()
    assert npc["civilian"].attitude >= 1          # 老柯 starts 戒備 → voices the offer
    q = ws.quests["moonlight_grass"]
    assert q.giver_id == "civilian" and q.status == "inactive" and not q.offered

    session.controllers["civilian"] = _OfferingNpcCtrl()
    session.dialogue = _GateFaithfulDirector()
    session.submit_player_input("好，我們接下你的委託")
    session.submit_player_input("離開")
    session._run_conversation("civilian")

    assert q.status == "active", (
        "a quest offered in the NPC opening must be acceptable on the player's "
        "first reply")


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))
