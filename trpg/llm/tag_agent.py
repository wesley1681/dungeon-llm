from __future__ import annotations

import json
import math
import pathlib
from enum import Enum
from typing import Any, Mapping

from ..engine.world_state import WorldState
from .action_executor import build_action_candidates

try:  # Tests can inject fake question classes/client before the SDK is installed.
    from typesafe_sdk import Choice, Noul, TypeSafeClient
except ImportError:  # pragma: no cover - only before dependencies are installed
    Choice = None
    Noul = None
    TypeSafeClient = None


_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEFAULT_THRESHOLD = 0.80
_NONE = "none"


class ActionIntent(str, Enum):
    TALK = "talk"
    GIVE = "give"
    FOLLOW = "follow"
    JOIN_PARTY = "join_party"
    TRAVEL = "travel"
    PICKUP = "pickup"
    USE = "use"
    UNLOCK = "unlock"
    SEARCH = "search"
    ATTACK = "attack"


def _question_id(action_index: int, intent: ActionIntent, field: str) -> str:
    return f"action_{action_index}_{intent.value}_{field}"


class TagAgent:
    """Use one Jev request to turn narrated actions into typed game commands."""

    def __init__(
        self,
        model: str,
        world_state: WorldState,
        base_url: str,
        backend: str,
        options: dict | None = None,
        api_key: str | None = None,
        *,
        typesafe_api_key: str | None = None,
        typesafe_model: str = "jev-latest",
        intent_threshold: float = _DEFAULT_THRESHOLD,
        typesafe_client: Any | None = None,
    ):
        # Retained for DialogueDirector.from_agent; not sent to TypeSafe.
        self.model = model
        self.world_state = world_state
        self.base_url = base_url
        self.backend = backend
        self.options = options or {}
        self.api_key = api_key

        self.typesafe_api_key = typesafe_api_key
        self.typesafe_model = typesafe_model
        self.intent_threshold = float(intent_threshold)
        if not 0.0 <= self.intent_threshold <= 1.0:
            raise ValueError("intent_threshold must be between 0 and 1")
        self._typesafe_client = typesafe_client
        self.last_judgment: dict[str, Any] | None = None

    def _normalize_actions(self, player_actions: list[Any]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for raw in player_actions:
            if isinstance(raw, Mapping):
                actor_id = str(raw.get("actor_id", ""))
                text = str(raw.get("text", ""))
            else:
                value = str(raw)
                prefix, separator, remainder = value.partition("：")
                if not separator:
                    prefix, separator, remainder = value.partition(":")
                actor_id = ""
                if separator:
                    actor_id = next(
                        (
                            cid for cid, character in self.world_state.characters.items()
                            if character.name == prefix.strip()
                        ),
                        "",
                    )
                    text = remainder.strip()
                else:
                    text = value
                if not actor_id and len(self.world_state.pc_ids) == 1:
                    actor_id = self.world_state.pc_ids[0]

            character = self.world_state.characters.get(actor_id)
            normalized.append({
                "actor_id": actor_id,
                "actor_name": character.name if character else actor_id,
                "text": text,
            })
        return normalized

    @staticmethod
    def _choice(
        instructions: str,
        candidates: list[dict[str, Any]],
        *,
        none_meaning: str,
    ) -> tuple[Any, dict[str, Any]]:
        if Choice is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")
        criteria: dict[str, Any] = {_NONE: none_meaning}
        values: dict[str, Any] = {_NONE: None}
        for index, candidate in enumerate(candidates):
            option_id = f"option_{index}"
            criteria[option_id] = candidate
            values[option_id] = candidate
        return Choice(instructions=instructions, criteria=criteria), values

    def _build_questions(
        self, actions: list[dict[str, str]]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if Noul is None or Choice is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")

        questions: dict[str, Any] = {}
        bindings: dict[str, dict[str, Any]] = {}
        common_none = "敘述沒有表達這項行動，或沒有明確指向任何候選。"

        def add_choice(
            index: int,
            intent: ActionIntent,
            field: str,
            instructions: str,
            candidates: list[dict[str, Any]],
            *,
            required: bool = True,
        ) -> None:
            question_id = _question_id(index, intent, field)
            question, values = self._choice(
                instructions, candidates, none_meaning=common_none,
            )
            questions[question_id] = question
            bindings[question_id] = {
                "index": index,
                "intent": intent.value,
                "field": field,
                "required": required,
                "values": values,
            }

        for index, action in enumerate(actions):
            actor_id = action["actor_id"]
            actor = self.world_state.characters.get(actor_id)
            if actor is None or not actor.is_alive() or not action["text"].strip():
                continue
            path = f"`player_actions[{index}].text`"
            candidates = build_action_candidates(self.world_state, actor_id)

            if candidates["room_npcs"]:
                add_choice(
                    index, ActionIntent.TALK, "target",
                    f"{path} 明確表示現在要與哪一名 NPC 交談？",
                    candidates["room_npcs"],
                )

            if candidates["room_characters"] and candidates["inventory"]:
                add_choice(
                    index, ActionIntent.GIVE, "recipient",
                    f"{path} 明確表示現在要把物品交給哪一名角色？",
                    candidates["room_characters"],
                )
                add_choice(
                    index, ActionIntent.GIVE, "item",
                    f"{path} 明確表示現在要完整給出哪一件自身持有的物品？",
                    candidates["inventory"],
                )

            if actor.is_npc and candidates["room_player_characters"]:
                add_choice(
                    index, ActionIntent.FOLLOW, "target",
                    f"{path} 明確表示行動者要跟隨哪一名玩家角色移動，但不加入戰鬥？",
                    candidates["room_player_characters"],
                )
                add_choice(
                    index, ActionIntent.JOIN_PARTY, "target",
                    f"{path} 明確表示行動者自己決定加入哪一名玩家角色的隊伍，並參與戰鬥？",
                    candidates["room_player_characters"],
                )

            add_choice(
                index, ActionIntent.TRAVEL, "direction",
                f"{path} 明確表示現在要往哪個方向移動？只判斷立即移動。",
                [
                    {"id": "north", "name": "北"},
                    {"id": "south", "name": "南"},
                    {"id": "east", "name": "東"},
                    {"id": "west", "name": "西"},
                ],
            )

            if candidates["pickup_items"]:
                add_choice(
                    index, ActionIntent.PICKUP, "item",
                    f"{path} 明確表示現在要拾起哪一件周遭物品？",
                    candidates["pickup_items"],
                )

            if candidates["inventory"]:
                add_choice(
                    index, ActionIntent.USE, "item",
                    f"{path} 明確表示現在要使用或消耗哪一件自身持有的物品？",
                    candidates["inventory"],
                )
                use_targets = [
                    {"id": actor_id, "name": actor.name},
                    *candidates["room_characters"],
                ]
                add_choice(
                    index, ActionIntent.USE, "target",
                    f"{path} 明確表示使用物品要作用在哪一名角色？沒有指定或不需要目標時選 none。",
                    use_targets,
                    required=False,
                )

            if candidates["locked_objects"]:
                add_choice(
                    index, ActionIntent.UNLOCK, "target",
                    f"{path} 明確表示現在要解鎖哪一個上鎖物件？",
                    candidates["locked_objects"],
                )
                add_choice(
                    index, ActionIntent.UNLOCK, "method",
                    f"{path} 明確表示要使用力量還是敏捷開鎖？未指定時選 none。",
                    [
                        {"id": "strength", "name": "力量"},
                        {"id": "dexterity", "name": "敏捷"},
                    ],
                )

            search_id = _question_id(index, ActionIntent.SEARCH, "requested")
            questions[search_id] = Noul(instructions=(
                f"{path} 是否明確表示現在要主動觀察、檢查、調查或搜索周遭環境？"
                "單純描述看見的事物、否定、假設、未來計畫或提及先前搜索都回答否。"
            ))
            bindings[search_id] = {
                "index": index,
                "intent": ActionIntent.SEARCH.value,
                "field": "requested",
                "required": True,
            }

            if candidates["room_npcs"]:
                add_choice(
                    index, ActionIntent.ATTACK, "target",
                    f"{path} 明確表示現在要向哪一名 NPC 發起攻擊？不判斷武器或攻擊方式。",
                    candidates["room_npcs"],
                )

        return questions, bindings

    def _call_jev(self, state: dict[str, Any], questions: dict[str, Any]):
        if self._typesafe_client is not None:
            return self._typesafe_client.system_one(
                state=state,
                questions=questions,
                model=self.typesafe_model,
            )
        if TypeSafeClient is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")
        with TypeSafeClient(api_key=self.typesafe_api_key) as client:
            return client.system_one(
                state=state,
                questions=questions,
                model=self.typesafe_model,
            )

    @staticmethod
    def _noul_value(response: Any, question_id: str) -> float:
        answers = getattr(response, "nouls", None)
        if answers is None:
            answers = getattr(response, "answers", None)
        if answers is None or question_id not in answers:
            raise ValueError(f"Jev 回應缺少 Noul：{question_id}")
        value = float(answers[question_id].noul)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Jev 回傳無效 Noul：{question_id}={value!r}")
        return value

    @staticmethod
    def _choice_value(response: Any, question_id: str) -> tuple[str, float, dict[str, float]]:
        answers = getattr(response, "choices", None)
        if answers is None:
            answers = getattr(response, "answers", None)
        if answers is None or question_id not in answers:
            raise ValueError(f"Jev 回應缺少 Choice：{question_id}")
        answer = answers[question_id]
        choice = str(answer.choice)
        confidence = float(answer.confidence)
        probabilities = {
            str(key): float(value)
            for key, value in dict(answer.probabilities).items()
        }
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Jev 回傳無效 Choice confidence：{question_id}")
        return choice, confidence, probabilities

    @staticmethod
    def _command(
        intent: str,
        action: dict[str, str],
        fields: dict[str, Any],
    ) -> dict[str, Any] | None:
        base = {"action": intent, "actor_id": action["actor_id"]}
        if intent == "talk":
            return {**base, "target_id": fields["target"]["id"]}
        if intent == "give":
            return {
                **base,
                "target_id": fields["recipient"]["id"],
                "item": fields["item"],
            }
        if intent in {"follow", "join_party"}:
            return {**base, "target_id": fields["target"]["id"]}
        if intent == "travel":
            return {**base, "direction": fields["direction"]["id"]}
        if intent == "pickup":
            return {**base, "item": fields["item"]}
        if intent == "use":
            command = {**base, "item": fields["item"]}
            if fields.get("target"):
                command["target_id"] = fields["target"]["id"]
            return command
        if intent == "unlock":
            return {
                **base,
                "target": fields["target"],
                "method": fields["method"]["id"],
            }
        if intent == "search":
            return base
        if intent == "attack":
            return {**base, "target_id": fields["target"]["id"]}
        return None

    def _write_debug(self, name: str, payload: Any) -> None:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        path = _DEBUG_DIR / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8", errors="replace")
        else:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
                errors="replace",
            )

    def judge_actions(self, player_actions: list[Any]) -> dict[str, Any]:
        """Return Jev answers plus Python-built structured commands."""
        actions = self._normalize_actions(player_actions)
        state = {"player_actions": actions}
        if not actions:
            judgment = {
                "model": self.typesafe_model,
                "threshold": self.intent_threshold,
                "actions": [],
                "answers": {},
                "commands": [],
            }
            self.last_judgment = judgment
            return judgment
        questions, bindings = self._build_questions(actions)

        if not questions:
            judgment = {
                "model": self.typesafe_model,
                "threshold": self.intent_threshold,
                "actions": actions,
                "answers": {},
                "commands": [],
            }
            self.last_judgment = judgment
            return judgment

        self._write_debug(
            "tag_agent_context.json",
            {
                "state": state,
                "model": self.typesafe_model,
                "questions": {
                    question_id: {
                        "type": question.type,
                        "instructions": question.instructions,
                        **(
                            {"criteria": question.criteria}
                            if hasattr(question, "criteria") else {}
                        ),
                    }
                    for question_id, question in questions.items()
                },
            },
        )
        response = self._call_jev(state, questions)

        answer_records: dict[str, Any] = {}
        selected: dict[int, dict[str, dict[str, Any]]] = {}
        required: dict[int, dict[str, set[str]]] = {}
        blocked: set[tuple[int, str]] = set()

        for question_id, binding in bindings.items():
            index = binding["index"]
            intent = binding["intent"]
            field = binding["field"]
            required.setdefault(index, {}).setdefault(intent, set())
            if binding.get("required", True):
                required[index][intent].add(field)

            question = questions[question_id]
            if question.type == "noul":
                probability = self._noul_value(response, question_id)
                accepted = probability >= self.intent_threshold
                answer_records[question_id] = {
                    "type": "noul",
                    "probability": probability,
                    "accepted": accepted,
                }
                if accepted:
                    selected.setdefault(index, {}).setdefault(intent, {})[field] = True
                else:
                    blocked.add((index, intent))
                continue

            choice, confidence, probabilities = self._choice_value(response, question_id)
            values = binding["values"]
            if choice not in values:
                raise ValueError(f"Jev 選擇了未知選項：{question_id}={choice}")
            value = values[choice]
            answer_records[question_id] = {
                "type": "choice",
                "choice": choice,
                "value": value,
                "confidence": confidence,
                "probabilities": probabilities,
            }
            if value is not None:
                selected.setdefault(index, {}).setdefault(intent, {})[field] = value
            elif binding.get("required", True):
                blocked.add((index, intent))

        commands: list[dict[str, Any]] = []
        per_action_commands: dict[int, list[dict[str, Any]]] = {}
        for index, action in enumerate(actions):
            for intent in ActionIntent:
                name = intent.value
                fields = selected.get(index, {}).get(name, {})
                needed = required.get(index, {}).get(name)
                if needed is None or (index, name) in blocked or not needed.issubset(fields):
                    continue
                command = self._command(name, action, fields)
                if command is not None:
                    commands.append(command)
                    per_action_commands.setdefault(index, []).append(command)

        action_results = [
            {**action, "commands": per_action_commands.get(index, [])}
            for index, action in enumerate(actions)
        ]
        usage = getattr(response, "usage", None)
        judgment = {
            "model": getattr(response, "model", self.typesafe_model),
            "threshold": self.intent_threshold,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            },
            "actions": action_results,
            "answers": answer_records,
            "commands": commands,
        }
        self.last_judgment = judgment
        self._write_debug("tag_agent_decision.json", judgment)
        return judgment

    def judge_intents(self, player_actions: list[Any]) -> dict[str, Any]:
        """Backward-compatible name for callers of the previous API."""
        return self.judge_actions(player_actions)

    def _safe_judgment(self, player_actions: list[Any]) -> dict[str, Any]:
        try:
            return self.judge_actions(player_actions)
        except Exception as exc:
            judgment = {
                "model": self.typesafe_model,
                "threshold": self.intent_threshold,
                "actions": [],
                "answers": {},
                "commands": [],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            self.last_judgment = judgment
            self._write_debug("tag_agent_decision.json", judgment)
            return judgment

    def generate_commands(self, player_actions: list[Any], on_chunk=None) -> list[dict[str, Any]]:
        """Judge a batch once, emit its JSON trace, and return executable commands."""
        judgment = self._safe_judgment(player_actions)
        result = json.dumps(judgment, ensure_ascii=False, indent=2)
        self._write_debug("tag_agent_output.txt", result)
        if on_chunk:
            on_chunk(result, thinking=False)
        else:
            print(result)
        return judgment["commands"]

    def generate_tags(self, player_actions: list[Any], on_chunk=None) -> str:
        """Compatibility wrapper returning JSON; it never generates tag strings."""
        judgment = self._safe_judgment(player_actions)
        result = json.dumps(judgment, ensure_ascii=False, indent=2)
        self._write_debug("tag_agent_output.txt", result)
        if on_chunk:
            on_chunk(result, thinking=False)
        else:
            print(result)
        return result
