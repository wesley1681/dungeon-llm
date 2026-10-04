import json
from types import SimpleNamespace

import trpg.llm.tag_agent as tag_agent_module
from trpg.engine.character import Character, Stats
from trpg.engine.dungeon_map import DungeonMap, Room
from trpg.engine.items import Chest, Consumable, Weapon
from trpg.engine.world_state import WorldState
from trpg.llm.tag_agent import ActionIntent, TagAgent


class _FakeNoul:
    type = "noul"

    def __init__(self, *, instructions):
        self.instructions = instructions


class _FakeChoice:
    type = "choice"

    def __init__(self, *, instructions, criteria):
        self.instructions = instructions
        self.criteria = criteria


class _FakeTypeSafeClient:
    def __init__(self, *, choices=None, nouls=None, error=None):
        self.selected_choices = choices or {}
        self.noul_values = nouls or {}
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
        answers = {}
        for question_id, question in questions.items():
            if question.type == "noul":
                answers[question_id] = SimpleNamespace(
                    noul=self.noul_values.get(question_id, 0.0),
                )
                continue
            choice = self.selected_choices.get(question_id, "none")
            probabilities = {
                option: (0.97 if option == choice else 0.01)
                for option in question.criteria
            }
            answers[question_id] = SimpleNamespace(
                choice=choice,
                confidence=0.96,
                probabilities=probabilities,
            )
        return SimpleNamespace(
            model="jev-test",
            answers=answers,
            nouls={qid: answer for qid, answer in answers.items() if hasattr(answer, "noul")},
            choices={qid: answer for qid, answer in answers.items() if hasattr(answer, "choice")},
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
        )


def _character(name, *, npc=False):
    return Character(
        name=name,
        race="人類",
        class_="冒險者",
        level=1,
        stats=Stats(),
        hp=10,
        max_hp=10,
        ac=10,
        is_npc=npc,
    )


def _world():
    kaine = _character("凱恩")
    kaine.weapons.append(Weapon("短劍", "1d6", "穿刺", "近戰"))
    kaine.consumables.append(Consumable("治療藥水", 2, "heal", "1d4"))
    thor = _character("索爾")
    guard = _character("守衛", npc=True)
    follower = _character("傭兵", npc=True)
    chest = Chest("鐵箱", ["寶石"], lock_dc=12)
    room = Room(
        id="hall",
        name="大廳",
        description="",
        exits={"east": "yard"},
        npc_ids=["guard", "follower"],
        loot=["火把", chest],
    )
    yard = Room(id="yard", name="庭院", description="", exits={"west": "hall"})
    dungeon_map = DungeonMap(rooms={"hall": room, "yard": yard}, current_room_id="hall")
    return WorldState(
        characters={
            "kaine": kaine,
            "thor": thor,
            "guard": guard,
            "follower": follower,
        },
        scene="大廳",
        pc_ids=["kaine", "thor"],
        party_ids=["kaine", "thor"],
        dungeon_map=dungeon_map,
    )


def _agent(monkeypatch, *, choices=None, nouls=None, error=None):
    monkeypatch.setattr(tag_agent_module, "Noul", _FakeNoul)
    monkeypatch.setattr(tag_agent_module, "Choice", _FakeChoice)
    client = _FakeTypeSafeClient(choices=choices, nouls=nouls, error=error)
    world = _world()
    agent = TagAgent(
        model="legacy-dialogue-model",
        world_state=world,
        base_url="http://legacy.invalid",
        backend="openai",
        typesafe_client=client,
    )
    return agent, client, world


def test_jev_receives_structured_actions_and_all_questions_in_one_request(monkeypatch):
    agent, client, _ = _agent(monkeypatch)
    actions = [{"actor_id": "kaine", "text": "我和守衛談話後往東走"}]

    judgment = agent.judge_actions(actions)

    assert client.call_count == 1
    assert client.last_state == {
        "player_actions": [
            {"actor_id": "kaine", "actor_name": "凱恩", "text": actions[0]["text"]}
        ]
    }
    assert client.last_model == "jev-latest"
    assert any(question.type == "choice" for question in client.last_questions.values())
    assert any(question.type == "noul" for question in client.last_questions.values())
    assert all("confiscate" not in qid for qid in client.last_questions)
    assert all("detain" not in qid and "drag" not in qid for qid in client.last_questions)
    assert judgment["model"] == "jev-test"
    assert judgment["usage"] == {"input_tokens": 123, "output_tokens": 45}


def test_choice_answers_build_typed_commands_without_tag_strings(monkeypatch):
    choices = {
        "action_0_talk_target": "option_0",       # 守衛
        "action_0_travel_direction": "option_2",  # east
    }
    agent, client, _ = _agent(monkeypatch, choices=choices)

    judgment = agent.judge_actions([
        {"actor_id": "kaine", "text": "我先和守衛談話，然後往東走"}
    ])

    assert judgment["commands"] == [
        {"action": "talk", "actor_id": "kaine", "target_id": "guard"},
        {"action": "travel", "actor_id": "kaine", "direction": "east"},
    ]
    assert client.call_count == 1
    assert "[TALK:" not in json.dumps(judgment, ensure_ascii=False)


def test_give_requires_both_recipient_and_item_choices(monkeypatch):
    agent, _, _ = _agent(
        monkeypatch,
        choices={
            "action_0_give_recipient": "option_0",
            "action_0_give_item": "none",
        },
    )

    judgment = agent.judge_actions([
        {"actor_id": "kaine", "text": "我給守衛一些東西"}
    ])

    assert all(command["action"] != "give" for command in judgment["commands"])


def test_unlock_requires_explicit_strength_or_dexterity(monkeypatch):
    agent, _, _ = _agent(
        monkeypatch,
        choices={
            "action_0_unlock_target": "option_0",
            "action_0_unlock_method": "none",
        },
    )

    judgment = agent.judge_actions([
        {"actor_id": "kaine", "text": "我打開鐵箱的鎖"}
    ])

    assert all(command["action"] != "unlock" for command in judgment["commands"])


def test_search_uses_noul_threshold(monkeypatch):
    agent, _, _ = _agent(
        monkeypatch,
        nouls={"action_0_search_requested": 0.91},
    )

    judgment = agent.judge_actions([
        {"actor_id": "kaine", "text": "我仔細觀察四周"}
    ])

    assert {"action": "search", "actor_id": "kaine"} in judgment["commands"]


def test_npc_can_choose_follow_or_join_party(monkeypatch):
    agent, _, _ = _agent(
        monkeypatch,
        choices={
            "action_0_follow_target": "option_0",
            "action_1_join_party_target": "option_1",
        },
    )

    judgment = agent.judge_actions([
        {"actor_id": "follower", "text": "我跟著凱恩，但不打架"},
        {"actor_id": "guard", "text": "我加入索爾的隊伍一起戰鬥"},
    ])

    assert {"action": "follow", "actor_id": "follower", "target_id": "kaine"} in judgment["commands"]
    assert {"action": "join_party", "actor_id": "guard", "target_id": "thor"} in judgment["commands"]


def test_empty_actions_skip_api_call(monkeypatch):
    agent, client, _ = _agent(monkeypatch)

    judgment = agent.judge_actions([])

    assert judgment["actions"] == []
    assert judgment["commands"] == []
    assert client.call_count == 0


def test_jev_failure_returns_no_commands_and_does_not_mutate_world(monkeypatch):
    agent, _, world = _agent(monkeypatch, error=TimeoutError("network timeout"))
    before_room = world.dungeon_map.current_room_id

    commands = agent.generate_commands(
        [{"actor_id": "kaine", "text": "我往東走"}],
        on_chunk=lambda *_args, **_kwargs: None,
    )

    assert commands == []
    assert agent.last_judgment["error"]["type"] == "TimeoutError"
    assert world.dungeon_map.current_room_id == before_room


def test_only_confirmed_action_types_are_exposed():
    assert {intent.value for intent in ActionIntent} == {
        "talk", "give", "follow", "join_party", "travel",
        "pickup", "use", "unlock", "search", "attack",
    }
