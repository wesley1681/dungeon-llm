from __future__ import annotations

import json
import math
import pathlib
from enum import Enum
from typing import Any

from ..engine.world_state import WorldState

try:  # Tests can inject a fake client before the optional SDK is installed.
    from typesafe_sdk import Noul, TypeSafeClient
except ImportError:  # pragma: no cover - only before dependencies are installed
    Noul = None
    TypeSafeClient = None


_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"
_DEFAULT_THRESHOLD = 0.80


class ActionIntent(str, Enum):
    TALK = "talk"
    GIVE = "give"
    CONFISCATE = "confiscate"
    DETAIN = "detain"
    DRAG = "drag"
    FOLLOW = "follow"
    RECRUIT = "recruit"
    TRAVEL = "travel"
    PICKUP = "pickup"
    USE = "use"
    UNLOCK = "unlock"
    SEARCH = "search"
    ATTACK = "attack"


_INTENT_LABELS = {
    ActionIntent.TALK: "對話",
    ActionIntent.GIVE: "給予",
    ActionIntent.CONFISCATE: "沒收",
    ActionIntent.DETAIN: "拘留",
    ActionIntent.DRAG: "拖走",
    ActionIntent.FOLLOW: "跟隨",
    ActionIntent.RECRUIT: "加入隊伍",
    ActionIntent.TRAVEL: "移動",
    ActionIntent.PICKUP: "拾取",
    ActionIntent.USE: "使用",
    ActionIntent.UNLOCK: "開鎖",
    ActionIntent.SEARCH: "觀察",
    ActionIntent.ATTACK: "發起攻擊",
}

_INTENT_MEANINGS = {
    ActionIntent.TALK: "initiate a conversation with a specific person",
    ActionIntent.GIVE: "give a specific item to another person",
    ActionIntent.CONFISCATE: "confiscate or forcibly take a specific item from another person",
    ActionIntent.DETAIN: "detain, arrest, restrain, or hold a person in custody",
    ActionIntent.DRAG: "forcibly drag, carry, or take a person away",
    ActionIntent.FOLLOW: "follow a specific person's movement",
    ActionIntent.RECRUIT: "ask a specific person to join the player's party",
    ActionIntent.TRAVEL: "move now in a cardinal direction: north, south, east, or west",
    ActionIntent.PICKUP: "pick up or take a specific object from the surroundings",
    ActionIntent.USE: "use or consume a specific item",
    ActionIntent.UNLOCK: "unlock or force open a specific locked object",
    ActionIntent.SEARCH: "actively observe, inspect, investigate, or search the surroundings",
    ActionIntent.ATTACK: "initiate an attack against a specific person",
}


def _question_id(action_index: int, intent: ActionIntent) -> str:
    return f"action_{action_index}_{intent.value}"


class TagAgent:
    """Use Jev only to judge action intents; do not resolve or execute them.

    ``world_state`` and the legacy LLM connection attributes remain on this
    object solely for compatibility with ``DialogueDirector.from_agent``. They
    are never included in the TypeSafe request.
    """

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

    def _build_questions(self, player_actions: list[str]) -> dict[str, Any]:
        if Noul is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")

        common_rule = (
            "Answer yes only when the referenced action explicitly states an immediate "
            "intent to perform this action now. Negated actions, hypothetical examples, "
            "future plans, observations about what someone else did, and mere mentions "
            "must be answered no. Judge intent only; do not judge whether the action is legal."
        )
        questions: dict[str, Any] = {}
        for index in range(len(player_actions)):
            for intent in ActionIntent:
                instructions = (
                    f"Does `player_actions[{index}]` clearly express an intent to "
                    f"{_INTENT_MEANINGS[intent]}? {common_rule}"
                )
                questions[_question_id(index, intent)] = Noul(instructions=instructions)
        return questions

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
            raise ValueError(f"Jev 回應缺少問題：{question_id}")
        value = float(answers[question_id].noul)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Jev 回傳無效機率：{question_id}={value!r}")
        return value

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

    def judge_intents(self, player_actions: list[str]) -> dict[str, Any]:
        """Return Jev probabilities and thresholded intent labels only.

        This method performs no entity extraction, legality validation, tag
        construction, or world-state mutation.
        """
        state = {"player_actions": list(player_actions)}
        if not player_actions:
            judgment = {
                "model": self.typesafe_model,
                "threshold": self.intent_threshold,
                "actions": [],
            }
            self.last_judgment = judgment
            return judgment

        questions = self._build_questions(player_actions)
        self._write_debug(
            "tag_agent_context.json",
            {
                "state": state,
                "model": self.typesafe_model,
                "questions": {
                    question_id: {
                        "type": "noul",
                        "instructions": question.instructions,
                    }
                    for question_id, question in questions.items()
                },
            },
        )
        response = self._call_jev(state, questions)

        action_results = []
        for index, player_action in enumerate(player_actions):
            probabilities: dict[str, float] = {}
            detected = []
            for intent in ActionIntent:
                probability = self._noul_value(response, _question_id(index, intent))
                probabilities[intent.value] = probability
                if probability >= self.intent_threshold:
                    detected.append(
                        {
                            "intent": intent.value,
                            "label": _INTENT_LABELS[intent],
                            "probability": probability,
                        }
                    )
            action_results.append(
                {
                    "index": index,
                    "player_action": player_action,
                    "probabilities": probabilities,
                    "detected": detected,
                }
            )

        judgment = {
            "model": getattr(response, "model", self.typesafe_model),
            "threshold": self.intent_threshold,
            "actions": action_results,
        }
        self.last_judgment = judgment
        self._write_debug("tag_agent_decision.json", judgment)
        return judgment

    def generate_tags(self, player_actions: list[str], on_chunk=None) -> str:
        """Compatibility wrapper returning the unprocessed judgment as JSON.

        The game currently passes this string to ``execute_all_tags``. Because
        this output contains no ``[TAG: ...]`` syntax, no game mechanic fires.
        Future handling can consume ``judge_intents`` or ``last_judgment``.
        """
        try:
            judgment = self.judge_intents(player_actions)
        except Exception as exc:
            judgment = {
                "model": self.typesafe_model,
                "threshold": self.intent_threshold,
                "actions": [],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            self.last_judgment = judgment
            self._write_debug("tag_agent_decision.json", judgment)

        result = json.dumps(judgment, ensure_ascii=False, indent=2)
        self._write_debug("tag_agent_output.txt", result)
        if on_chunk:
            on_chunk(result, thinking=False)
        else:
            print(result)
        return result
