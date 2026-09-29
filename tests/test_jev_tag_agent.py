import json
from types import SimpleNamespace

import trpg.llm.tag_agent as tag_agent_module
from trpg.engine.character import Character, Stats
from trpg.engine.world_state import WorldState
from trpg.llm.tag_agent import ActionIntent, TagAgent
from trpg.llm.tag_parser import execute_all_tags


class _FakeNoul:
    def __init__(self, *, instructions):
        self.instructions = instructions


class _FakeTypeSafeClient:
    def __init__(self, probabilities=None, error=None):
        self.probabilities = probabilities or {}
        self.error = error
        self.last_state = None
        self.last_questions = None
        self.last_model = None
        self.call_count = 0

    def system_one(self, *, state, questions, model):
        self.call_count += 1
        if self.error:
            raise self.error
        self.last_state = state
        self.last_questions = questions
        self.last_model = model
        answers = {
            question_id: SimpleNamespace(
                noul=self.probabilities.get(question_id.rsplit("_", 1)[-1], 0.0)
            )
            for question_id in questions
        }
        return SimpleNamespace(model="jev-test", nouls=answers)


def _world():
    kaine = Character(
        name="艾里亞",
        race="人類",
        class_="測試",
        level=1,
        stats=Stats(),
        hp=10,
        max_hp=10,
        ac=10,
    )
    return WorldState(
        characters={"kaine": kaine},
        scene="測試場景",
        pc_ids=["kaine"],
        party_ids=["kaine"],
    )


def _agent(monkeypatch, probabilities=None, error=None):
    monkeypatch.setattr(tag_agent_module, "Noul", _FakeNoul)
    client = _FakeTypeSafeClient(probabilities, error)
    world = _world()
    agent = TagAgent(
        model="legacy-dialogue-model",
        world_state=world,
        base_url="http://legacy.invalid",
        backend="openai",
        typesafe_client=client,
    )
    return agent, client, world


def test_jev_receives_only_player_actions_as_state(monkeypatch):
    agent, client, _ = _agent(monkeypatch, {"pickup": 0.97})
    actions = ["艾里亞：我撿起火把"]

    judgment = agent.judge_intents(actions)

    assert client.last_state == {"player_actions": actions}
    assert client.last_model == "jev-latest"
    assert len(client.last_questions) == len(ActionIntent)
    assert judgment["model"] == "jev-test"


def test_judgment_contains_all_probabilities_and_detected_intents(monkeypatch):
    agent, _, _ = _agent(monkeypatch, {"pickup": 0.97, "use": 0.91})

    judgment = agent.judge_intents(["艾里亞：我撿起火把並使用它"])
    action = judgment["actions"][0]

    assert set(action["probabilities"]) == {intent.value for intent in ActionIntent}
    assert action["probabilities"]["pickup"] == 0.97
    assert action["probabilities"]["use"] == 0.91
    assert [item["intent"] for item in action["detected"]] == ["pickup", "use"]


def test_no_entity_resolution_or_world_mutation_occurs(monkeypatch):
    agent, _, world = _agent(monkeypatch, {"talk": 0.99, "travel": 0.98})
    before_scene = world.scene

    output = agent.generate_tags(
        ["艾里亞：我要跟他談談，然後往東走"],
        on_chunk=lambda *_args, **_kwargs: None,
    )
    parsed = json.loads(output)
    ok, errors = execute_all_tags(output, world)

    assert [item["intent"] for item in parsed["actions"][0]["detected"]] == [
        "talk",
        "travel",
    ]
    assert "[TALK:" not in output
    assert "[TRAVEL:" not in output
    assert ok == []
    assert errors == []
    assert world.scene == before_scene
    assert world.pending_conversation == ""


def test_common_rule_is_part_of_every_question(monkeypatch):
    agent, client, _ = _agent(monkeypatch)

    agent.judge_intents(["艾里亞：我不攻擊任何人"])

    for question in client.last_questions.values():
        assert "Negated actions" in question.instructions
        assert "do not judge whether the action is legal" in question.instructions


def test_empty_actions_skip_api_call(monkeypatch):
    agent, client, _ = _agent(monkeypatch)

    judgment = agent.judge_intents([])

    assert judgment["actions"] == []
    assert client.call_count == 0


def test_jev_failure_returns_structured_error_without_mutation(monkeypatch):
    agent, _, world = _agent(monkeypatch, error=TimeoutError("network timeout"))

    output = agent.generate_tags(
        ["艾里亞：往東走"],
        on_chunk=lambda *_args, **_kwargs: None,
    )
    parsed = json.loads(output)

    assert parsed["actions"] == []
    assert parsed["error"]["type"] == "TimeoutError"
    assert world.scene == "測試場景"
