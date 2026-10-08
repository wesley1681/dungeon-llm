"""Jev-backed conversation decisions.

NPC and player utterances use a directional action catalog with explicit actor
and target identities. Callers expose only the actions available to that
speaker: NPCs may request_to_join, while players may invite_to_party. Only give
and offer_quest need a second Jev call to select a valid world-state candidate.
Pending replies and narrative quest completion are judged alongside or after
those actions as appropriate.

This module only decides. World-state mutation lives in ``game.py``.
"""
from __future__ import annotations
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from .action_executor import build_action_candidates

try:  # Tests can inject a fake client; runtime uses the installed SDK.
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
except ImportError:  # pragma: no cover - only before dependencies are installed
    Choice = None
    Noul = None
    Score = None
    TypeSafeClient = None

_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"

_ATTITUDE_LABELS = ["敵意", "戒備", "中立", "友好", "信任"]

# ── Catalogs ──────────────────────────────────────────────────────────────────
# The check catalog (all CHA-based today; extend here to add new skills/stats).
CHECK_TYPES = ("威嚇", "欺瞞", "說服")
_CHECK_FROM_CHOICE = {
    "intimidate": "威嚇",
    "deceive": "欺瞞",
    "persuade": "說服",
}
_SOCIAL_CHECK_THRESHOLD = 0.80

# Score levels run from least to most likely to work. Jev returns their
# probability-weighted position, which code rounds to the nearest rubric level.
_SOCIAL_EFFECTIVENESS_LEVELS = [
    "幾乎不可能奏效：要求完全不可接受，謊言明顯荒謬，或威脅毫無可信度。",
    "極難奏效：嚴重違反 NPC 的利益、信念或安全，且沒有足夠理由讓步。",
    "很難奏效：NPC 必須承擔明顯損失或風險，說法也有重大弱點。",
    "有明顯阻力但可能奏效：要求有代價，但理由、交換或手段具有一定份量。",
    "合理且有機會奏效：要求大致合理，說法可信，NPC 的損失或風險有限。",
    "很容易奏效：明顯符合 NPC 利益、說法高度可信，或威脅具體且可信。",
    "幾乎必然奏效：NPC 幾乎沒有拒絕理由，接受是自然且低風險的選擇。",
]
_DC_BY_EFFECTIVENESS_LEVEL = (25, 20, 16, 14, 12, 10, 8)

_COMPLETE_EVENT = "quest_complete"

# Shared action catalog. Callers select a subset, so unavailable actions never
# become Jev questions. "none" is derived when no Noul clears its threshold.
DIALOGUE_ACTIONS = {
    "attack": {
        "threshold": 0.90,
        "instructions": (
            "`utterance.text` 是否表示 `utterance.actor` 已經在這一刻對 "
            "`utterance.target` 開始攻擊？"
        ),
        "true": "發言者明確動手、施法攻擊，或宣告立即對對方發動攻擊。",
        "false": "只是生氣、辱罵、警告、拒絕，或威脅將來可能攻擊。",
    },
    "invite_to_party": {
        "threshold": 0.80,
        "instructions": (
            "`utterance.text` 是否表示 `utterance.actor` 主動邀請 "
            "`utterance.target` 加入或長期跟隨 `utterance.actor` 所屬的隊伍？"
        ),
        "true": (
            "`utterance.actor` 明確邀請 `utterance.target` 加入、同行，"
            "或長期成為 `utterance.actor` 所屬隊伍的成員。"
        ),
        "false": (
            "`utterance.actor` 是請求加入 `utterance.target` 所屬的隊伍、"
            "只是暫時幫忙或同行一小段、談論隊伍，或回覆既有的加入提議。"
        ),
    },
    "request_to_join": {
        "threshold": 0.80,
        "instructions": (
            "`utterance.text` 是否表示 `utterance.actor` 主動請求加入或長期跟隨 "
            "`utterance.target` 所屬的隊伍？"
        ),
        "true": (
            "`utterance.actor` 明確請求加入、同行，或長期成為 "
            "`utterance.target` 所屬隊伍的成員。"
        ),
        "false": (
            "`utterance.actor` 是邀請 `utterance.target` 加入自己的隊伍、"
            "只是暫時幫忙或同行一小段、談論隊伍，或回覆既有的加入提議。"
        ),
    },
    "give": {
        "threshold": 0.90,
        "instructions": (
            "`utterance.text` 是否表示 `utterance.actor` 現在就要把自己的物品交給 "
            "`utterance.target`？"
        ),
        "true": "發言者明確完成贈與或交付，接收者現在取得某項物品。",
        "false": "只是答應以後給、提出交易、描述物品，或把物品當作未來報酬。",
    },
    "offer_quest": {
        "threshold": 0.80,
        "instructions": (
            "`utterance.text` 是否由 `utterance.actor` 向 `utterance.target` 提出一項"
            "新的任務或委託，需要對方決定是否接受？"
        ),
        "true": "發言者明確請對方完成某個目標或行動。",
        "false": "只是提及問題、提供背景、談論已有委託，或交付任務報酬。",
    },
}

NPC_DIALOGUE_ACTIONS = ("attack", "request_to_join", "give", "offer_quest")
PLAYER_DIALOGUE_ACTIONS = ("attack", "invite_to_party", "give")


@dataclass
class CheckPlan:
    check: str | None = None      # one of CHECK_TYPES, or None (no check)
    dc: int = 0
    actor: str = ""               # char_id of the PC making the attempt


@dataclass
class CheckResult:
    check: str
    success: bool
    total: int
    dc: int
    actor: str


class DialogueDirector:
    """Owns the conversation-resolution model stages. Built from the same
    connection info as the other agents (see DialogueDirector.from_agent)."""

    def __init__(self, model: str, base_url: str, backend: str, world_state,
                 options: dict | None = None, api_key: str | None = None,
                 *, typesafe_api_key: str | None = None,
                 typesafe_model: str = "jev-latest",
                 social_check_threshold: float = _SOCIAL_CHECK_THRESHOLD,
                 typesafe_client: Any | None = None):
        self.model      = model
        self.base_url   = base_url
        self.backend    = backend
        self.world_state = world_state
        self.options    = options or {}
        self.api_key    = api_key
        self.typesafe_api_key = typesafe_api_key
        self.typesafe_model = typesafe_model
        self.social_check_threshold = float(social_check_threshold)
        if not 0.0 <= self.social_check_threshold <= 1.0:
            raise ValueError("social_check_threshold must be between 0 and 1")
        self._typesafe_client = typesafe_client

    @classmethod
    def from_agent(cls, agent, world_state=None) -> "DialogueDirector":
        """Build from any agent exposing the connection attrs (e.g. TagAgent).
        Tolerates stubs missing the attrs (fields default to None) — the object
        is inert until a stage method actually calls the backend."""
        return cls(
            model=getattr(agent, "model", None),
            base_url=getattr(agent, "base_url", None),
            backend=getattr(agent, "backend", None),
            world_state=world_state if world_state is not None
                        else getattr(agent, "world_state", None),
            options=getattr(agent, "options", None),
            api_key=getattr(agent, "api_key", None),
            typesafe_api_key=getattr(agent, "typesafe_api_key", None),
            typesafe_model=getattr(agent, "typesafe_model", "jev-latest"),
            social_check_threshold=getattr(
                agent, "intent_threshold", _SOCIAL_CHECK_THRESHOLD),
            typesafe_client=getattr(agent, "_typesafe_client", None),
        )

    # ── Stage 1 — Jev 社交檢定判定 ────────────────────────────────────────────

    @staticmethod
    def _normalise_social_actions(player_actions, ws) -> list[dict[str, str]]:
        """Accept the new structured input and retain string-call compatibility."""
        if isinstance(player_actions, str):
            actor_id = _default_actor(ws)
            actor = ws.characters.get(actor_id)
            return [{
                "actor_id": actor_id,
                "actor_name": actor.name if actor else actor_id,
                "text": player_actions,
            }]

        actions: list[dict[str, str]] = []
        for raw in player_actions or []:
            if not isinstance(raw, Mapping):
                continue
            actor_id = str(raw.get("actor_id", ""))
            actor = ws.characters.get(actor_id)
            text = str(raw.get("text", "")).strip()
            if actor is None or actor.is_npc or not actor.is_alive() or not text:
                continue
            actions.append({
                "actor_id": actor_id,
                "actor_name": actor.name,
                "text": text,
            })
        return actions

    @staticmethod
    def _build_social_questions(actions: list[dict[str, str]]) -> dict[str, Any]:
        if Noul is None or Choice is None or Score is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")

        questions: dict[str, Any] = {}
        for index, _action in enumerate(actions):
            path = f"`player_actions[{index}].text`"
            questions[f"social_{index}_required"] = Noul(
                instructions=(
                    f"{path} 是否直接對 `npc` 提出一項具有明確目的、且結果存在不確定性的"
                    "社交嘗試？"
                ),
                criteria={
                    "true": (
                        "玩家正試圖用說服、欺瞞或威嚇，讓 NPC 採取行動、交出物品、"
                        "加入隊伍，或改變決定。"
                    ),
                    "false": (
                        "只是寒暄、閒聊、普通提問、陳述、回答、隊友間交談、沒有要求 NPC"
                        " 做事，或已直接動手攻擊。僅語氣兇狠而沒有明確威脅也屬於 false。"
                    ),
                },
            )
            # These answers run in the same request and are used only when the
            # Noul above clears the configured threshold.
            questions[f"social_{index}_type"] = Choice(
                instructions=(
                    f"判斷 {path} 主要使用的社交手段。"
                ),
                criteria={
                    "persuade": "以真誠請求、協商、交換利益或講道理使 NPC 願意配合。",
                    "deceive": "以虛假身分、捏造事實、隱瞞關鍵資訊或誤導使 NPC 相信。",
                    "intimidate": "以明確威脅、可信的暴力後果或強迫命令使 NPC 因恐懼服從。",
                },
            )
            questions[f"social_{index}_effectiveness"] = Score(
                instructions=(
                    f"評估 {path} 對目前 `npc` 有多容易奏效。依手段考慮要求的利益與"
                    "代價、謊言可信度或威脅可信度，以及 NPC 的個性與態度。"
                ),
                criteria=_SOCIAL_EFFECTIVENESS_LEVELS,
            )
        return questions

    def _call_jev(self, state: dict[str, Any], questions: dict[str, Any]):
        if self._typesafe_client is not None:
            return self._typesafe_client.system_one(
                state=state, questions=questions, model=self.typesafe_model)
        if TypeSafeClient is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")
        with TypeSafeClient(api_key=self.typesafe_api_key) as client:
            return client.system_one(
                state=state, questions=questions, model=self.typesafe_model)

    @staticmethod
    def _typed_answer(response, group: str, question_id: str):
        answers = getattr(response, group, None)
        if answers is not None and question_id in answers:
            return answers[question_id]
        answers = getattr(response, "answers", None)
        if answers is None or question_id not in answers:
            raise ValueError(f"Jev 回應缺少答案：{question_id}")
        return answers[question_id]

    @staticmethod
    def _effectiveness_to_dc(score: float) -> int:
        if not math.isfinite(score):
            raise ValueError(f"Jev 回傳無效 Score：{score!r}")
        level = max(0, min(
            len(_DC_BY_EFFECTIVENESS_LEVEL) - 1,
            int(math.floor(score + 0.5)),
        ))
        return _DC_BY_EFFECTIVENESS_LEVEL[level]

    def decide_check(self, player_actions, npc, ws) -> CheckPlan:
        """Use one Jev call to decide presence, type, actor, and DC rubric."""
        actions = self._normalise_social_actions(player_actions, ws)
        if not actions:
            return CheckPlan()

        state = {
            "player_actions": actions,
            "npc": {
                "id": getattr(npc, "char_id", ""),
                "name": getattr(getattr(npc, "char", None), "name", "NPC"),
                "personality": getattr(npc, "_personality", ""),
                "attitude": npc.attitude,
                "attitude_label": _ATTITUDE_LABELS[npc.attitude],
            },
        }
        questions = self._build_social_questions(actions)
        (_DEBUG_DIR / "dialogue_check_context.json").write_text(
            json.dumps({
                "state": state,
                "model": self.typesafe_model,
                "questions": {
                    question_id: {
                        "type": question.type,
                        "instructions": question.instructions,
                        "criteria": question.criteria,
                    }
                    for question_id, question in questions.items()
                },
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )

        try:
            response = self._call_jev(state, questions)
            candidates = []
            debug_answers = {}
            for index, action in enumerate(actions):
                required_id = f"social_{index}_required"
                type_id = f"social_{index}_type"
                score_id = f"social_{index}_effectiveness"
                required_answer = self._typed_answer(response, "nouls", required_id)
                type_answer = self._typed_answer(response, "choices", type_id)
                score_answer = self._typed_answer(response, "scores", score_id)
                probability = float(required_answer.noul)
                score = float(score_answer.score)
                check = _CHECK_FROM_CHOICE.get(str(type_answer.choice))
                debug_answers[str(index)] = {
                    "required_probability": probability,
                    "type": str(type_answer.choice),
                    "type_confidence": float(type_answer.confidence),
                    "effectiveness_score": score,
                    "score_confidence": float(score_answer.confidence),
                }
                if probability >= self.social_check_threshold and check:
                    candidates.append((probability, index, check, score, action))
        except Exception as exc:
            (_DEBUG_DIR / "dialogue_check_output.json").write_text(
                json.dumps({
                    "error": {"type": type(exc).__name__, "message": str(exc)}
                }, ensure_ascii=False, indent=2),
                encoding="utf-8", errors="replace",
            )
            return CheckPlan()

        (_DEBUG_DIR / "dialogue_check_output.json").write_text(
            json.dumps({
                "model": getattr(response, "model", self.typesafe_model),
                "threshold": self.social_check_threshold,
                "answers": debug_answers,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        if not candidates:
            return CheckPlan()
        _, _, check, score, action = max(candidates, key=lambda item: item[0])
        return CheckPlan(
            check=check,
            dc=self._effectiveness_to_dc(score),
            actor=action["actor_id"],
        )

    # ── Shared dialogue-action classifier ────────────────────────────────────
    @staticmethod
    def _action_questions(allowed_actions) -> tuple[tuple[str, ...], dict[str, Any]]:
        if Noul is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")
        requested = tuple(allowed_actions or ())
        unknown = set(requested) - set(DIALOGUE_ACTIONS)
        if unknown:
            raise ValueError(f"未知的對話行動：{sorted(unknown)}")
        questions = {
            action: Noul(
                instructions=DIALOGUE_ACTIONS[action]["instructions"],
                criteria={
                    "true": DIALOGUE_ACTIONS[action]["true"],
                    "false": DIALOGUE_ACTIONS[action]["false"],
                },
            )
            for action in requested
        }
        return requested, questions

    @staticmethod
    def _character_ref(ws, char_id: str) -> dict[str, str]:
        char = ws.characters.get(char_id)
        return {"id": char_id, "name": char.name if char else char_id}

    def judge_dialogue_actions(
            self, text: str, actor_id: str, target_id: str, allowed_actions,
            ws, *, pending_join: dict | None = None,
            pending_quests=None, turnins=None,
            debug_name: str = "dialogue_action") -> list[dict]:
        """Judge one utterance using the shared catalog and optional replies."""
        utterance = str(text or "").strip()
        if not utterance or Choice is None or Noul is None:
            return []

        actions, questions = self._action_questions(allowed_actions)
        pending_quests = list(pending_quests or [])
        turnins = list(turnins or [])
        join_for_actor = (
            pending_join if isinstance(pending_join, dict)
            and pending_join.get("responder_id") == actor_id else None
        )
        response_criteria = {
            "accept": "發言者明確同意或答應這項提議。",
            "reject": "發言者明確拒絕、不願意或要求取消這項提議。",
            "none": "發言者沒有回答這項提議、語意不明，或正在談其他事情。",
        }
        if join_for_actor:
            questions["join_response"] = Choice(
                instructions=(
                    "`utterance.text` 對 `pending.join` 的加入隊伍提議，"
                    "是接受、拒絕，還是沒有明確回答？"
                ),
                criteria=response_criteria,
            )
        for index, _quest in enumerate(pending_quests):
            questions[f"quest_{index}_response"] = Choice(
                instructions=(
                    f"`utterance.text` 對 `pending.quests[{index}]` 的任務邀請，"
                    "是接受、拒絕，還是沒有明確回答？"
                ),
                criteria=response_criteria,
            )
        for index, _quest in enumerate(turnins):
            questions[f"turnin_{index}"] = Noul(
                instructions=(
                    f"`utterance.text` 是否向對方明確回報已完成 "
                    f"`pending.turnins[{index}]`，並準備提交或領取報酬？"
                ),
                criteria={
                    "true": "發言者明確回報完成任務、交付成果或要求結算報酬。",
                    "false": "只是詢問進度、談論任務，或沒有回報完成。",
                },
            )

        state = {
            "utterance": {
                "text": utterance,
                "actor": self._character_ref(ws, actor_id),
                "target": self._character_ref(ws, target_id),
            },
            "pending": {
                "join": join_for_actor,
                "quests": [{
                    "id": q.id,
                    "title": q.title,
                    "description": q.description,
                } for q in pending_quests],
                "turnins": [{
                    "id": q.id,
                    "title": q.title,
                    "description": q.description,
                } for q in turnins],
            },
        }
        context_path = _DEBUG_DIR / f"{debug_name}_context.json"
        output_path = _DEBUG_DIR / f"{debug_name}_output.json"
        context_path.write_text(
            json.dumps({
                "state": state,
                "model": self.typesafe_model,
                "allowed_actions": list(actions),
                "questions": {
                    question_id: {
                        "type": question.type,
                        "instructions": question.instructions,
                        "criteria": question.criteria,
                    }
                    for question_id, question in questions.items()
                },
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )

        try:
            response = self._call_jev(state, questions)
            events: list[dict] = []
            probabilities = {}
            for action in actions:
                probability = float(self._typed_answer(
                    response, "nouls", action).noul)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Jev 回傳無效 Noul：{action}={probability!r}")
                probabilities[action] = probability
                if probability >= DIALOGUE_ACTIONS[action]["threshold"]:
                    events.append({
                        "event": action,
                        "actor_id": actor_id,
                        "target_id": target_id,
                    })

            if join_for_actor:
                answer = str(self._typed_answer(
                    response, "choices", "join_response").choice)
                if answer in ("accept", "reject"):
                    # A response to an existing proposal is not a new proposal.
                    events = [
                        e for e in events
                        if e["event"] not in ("invite_to_party", "request_to_join")
                    ]
                    events.append({
                        "event": f"join_{answer}",
                        "actor_id": actor_id,
                        "target_id": join_for_actor.get("proposer_id", target_id),
                        "candidate_id": join_for_actor.get("candidate_id", ""),
                    })
            for index, quest in enumerate(pending_quests):
                answer = str(self._typed_answer(
                    response, "choices", f"quest_{index}_response").choice)
                if answer in ("accept", "reject"):
                    events.append({
                        "event": f"quest_{answer}",
                        "id": quest.id,
                        "actor_id": actor_id,
                        "target_id": target_id,
                    })
            for index, quest in enumerate(turnins):
                probability = float(self._typed_answer(
                    response, "nouls", f"turnin_{index}").noul)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Jev 回傳無效 Noul：turnin_{index}={probability!r}")
                if probability >= 0.80:
                    events.append({
                        "event": "quest_turnin",
                        "id": quest.id,
                        "actor_id": actor_id,
                        "target_id": target_id,
                    })
        except Exception as exc:
            output_path.write_text(
                json.dumps({
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "events": [],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8", errors="replace",
            )
            return []

        output_path.write_text(
            json.dumps({
                "model": getattr(response, "model", self.typesafe_model),
                "probabilities": probabilities,
                "events": events,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        return events

    def judge_npc_actions(self, npc_text: str, npc, ws,
                          target_id: str | None = None) -> list[dict]:
        npc_id = getattr(npc, "char_id", "")
        return self.judge_dialogue_actions(
            npc_text, npc_id, target_id or _default_actor(ws),
            NPC_DIALOGUE_ACTIONS, ws,
            pending_join=getattr(npc, "pending_join_offer", None),
            debug_name="dialogue_npc_action",
        )

    def judge_player_actions(self, player_action, npc, ws) -> list[dict]:
        """Judge one player's utterance plus replies to pending NPC offers."""
        npc_id = getattr(npc, "char_id", "")
        if isinstance(player_action, Mapping):
            actor_id = str(player_action.get("actor_id", ""))
            text = str(player_action.get("text", ""))
        else:
            actor_id = _default_actor(ws)
            text = str(player_action or "")
        quests = [q for q in ws.quests.values()
                  if (q.giver_id == npc_id and q.status == "inactive"
                      and q.offered)]
        turnins = [q for q in ws.quests.values()
                   if q.giver_id == npc_id and q.status == "completed"]
        return self.judge_dialogue_actions(
            text, actor_id, npc_id, PLAYER_DIALOGUE_ACTIONS, ws,
            pending_join=getattr(npc, "pending_join_offer", None),
            pending_quests=quests,
            turnins=turnins,
            debug_name="dialogue_player_action",
        )

    def resolve_dialogue_actions(self, text: str, events: list[dict], ws) -> list[dict]:
        """Resolve inventory/quest candidates for shared coarse action events."""
        resolved = [dict(event) for event in events
                    if event.get("event") not in ("give", "offer_quest")]
        give_event = next(
            (event for event in events if event.get("event") == "give"), None)
        quest_event = next(
            (event for event in events if event.get("event") == "offer_quest"), None)

        items: list[dict[str, Any]] = []
        giver = ws.characters.get(give_event.get("actor_id")) if give_event else None
        if giver is not None:
            items = build_action_candidates(
                ws, give_event.get("actor_id", ""))["inventory"]

        quest_giver_id = quest_event.get("actor_id") if quest_event else ""
        quests = [q for q in ws.quests.values()
                  if (q.giver_id == quest_giver_id
                      and q.status == "inactive" and not q.offered)]
        if Choice is None:
            return resolved

        questions: dict[str, Any] = {}
        item_keys = {f"item_{i}": item for i, item in enumerate(items)}
        quest_keys = {f"quest_{i}": q.id for i, q in enumerate(quests)}
        if item_keys:
            questions["give_item"] = Choice(
                instructions=(
                    "假設 `utterance.text` 確實表示交付物品，選出 "
                    "`candidates.items` 中由發言者交付的物品。"
                ),
                criteria={
                    **{key: f"發言者持有的「{item['name']}」。"
                       for key, item in item_keys.items()},
                    "none": "發言沒有明確對應任何候選物品。",
                },
            )
        if quest_keys:
            questions["offered_quest"] = Choice(
                instructions=(
                    "假設 `utterance.text` 確實提出新任務，選出 "
                    "`candidates.quests` 中被提出的任務。"
                ),
                criteria={
                    **{key: f"任務「{q.title}」：{q.description}"
                       for key, q in zip(quest_keys, quests)},
                    "none": "發言沒有明確對應任何候選任務。",
                },
            )
        if not questions:
            return resolved

        source_event = give_event or quest_event or {}
        actor_id = source_event.get("actor_id", "")
        target_id = source_event.get("target_id", "")
        state = {
            "utterance": {
                "text": str(text or "").strip(),
                "actor": self._character_ref(ws, actor_id),
                "target": self._character_ref(ws, target_id),
            },
            "candidates": {
                "items": [{"key": key, **item}
                          for key, item in item_keys.items()],
                "quests": [{
                    "key": key,
                    "id": q.id,
                    "title": q.title,
                    "description": q.description,
                } for key, q in zip(quest_keys, quests)],
            },
        }
        try:
            response = self._call_jev(state, questions)
            if "give_item" in questions and give_event:
                key = str(self._typed_answer(
                    response, "choices", "give_item").choice)
                item = item_keys.get(key)
                if item:
                    resolved.append({**give_event, "item": item})
            if "offered_quest" in questions and quest_event:
                key = str(self._typed_answer(
                    response, "choices", "offered_quest").choice)
                qid = quest_keys.get(key)
                if qid:
                    resolved.append({**quest_event, "id": qid})
        except Exception as exc:
            (_DEBUG_DIR / "dialogue_action_detail_output.json").write_text(
                json.dumps({
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "events": resolved,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8", errors="replace",
            )
            return resolved

        (_DEBUG_DIR / "dialogue_action_detail_output.json").write_text(
            json.dumps({"state": state, "events": resolved},
                       ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        return resolved

    # ── Completion judge — minimal active-quest context, post-response ────────
    def judge_completion(self, player_text: str, npc_response: str, ws) -> list[dict]:
        """Use Jev to judge narrative completion of active quests."""
        active = [q for q in ws.quests.values() if q.status == "active"]
        if not active:
            return []
        if Noul is None:
            return []

        state = {
            "player_reply": {"text": str(player_text or "").strip()},
            "npc_reply": {"text": str(npc_response or "").strip()},
            "active_quests": [{
                "id": q.id,
                "title": q.title,
                "description": q.description,
            } for q in active],
        }
        questions = {
            f"complete_{index}": Noul(
                instructions=(
                    f"這輪 `player_reply.text` 與 `npc_reply.text` 是否已經真正達成 "
                    f"`active_quests[{index}]` 的任務目標？"
                ),
                criteria={
                    "true": "NPC 的實際回應或這輪結果已明確完成任務目標。",
                    "false": "只是敷衍、討價還價、答應考慮，或尚未真正達成目標。",
                },
            )
            for index, _quest in enumerate(active)
        }
        try:
            response = self._call_jev(state, questions)
            events = []
            probabilities = {}
            for index, quest in enumerate(active):
                question_id = f"complete_{index}"
                probability = float(self._typed_answer(
                    response, "nouls", question_id).noul)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Jev 回傳無效 Noul：{question_id}={probability!r}")
                probabilities[question_id] = probability
                if probability >= 0.85:
                    events.append({"event": _COMPLETE_EVENT, "id": quest.id})
        except Exception as exc:
            (_DEBUG_DIR / "dialogue_complete_output.json").write_text(
                json.dumps({
                    "error": {"type": type(exc).__name__, "message": str(exc)}
                }, ensure_ascii=False, indent=2),
                encoding="utf-8", errors="replace",
            )
            return []

        (_DEBUG_DIR / "dialogue_complete_output.json").write_text(
            json.dumps({
                "state": state,
                "probabilities": probabilities,
                "events": events,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        return events

def _default_actor(ws) -> str:
    """The human PC if present, else the first living PC."""
    if "kaine" in ws.characters and ws.characters["kaine"].is_alive():
        return "kaine"
    for cid in ws.pc_ids:
        c = ws.characters.get(cid)
        if c and not c.is_npc and c.is_alive():
            return cid
    return "kaine"
