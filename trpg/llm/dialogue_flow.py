"""Conversation resolution pipeline — focused model stages per round.

Replaces the old single `generate_conversation_tags` classifier. Each model call
stays focused on ONE job so tasks don't bleed together:

  1. decide_check     — 玩家輸入 → 要不要檢定？哪個(威嚇/欺瞞/說服)？誰？DC 多少？
  2. arrange_events   — 玩家輸入 + 檢定結果 → 事件目錄挑觸發（reveal/give/suggest_join/
                        attitude/attack）＋ **條件加入** 玩家驅動的 quest_accept（任務已被
                        提出）/ quest_turnin（任務已完成）
  3. judge_npc_actions — NPC 發言 → Jev 初步行動分類
  4. judge_completion — 已承接(active)任務名稱+內容 + 玩家輸入 + NPC 回應 → 判定 quest_complete

Checks and events are INDEPENDENT catalogs — NOT a 1:1 mapping. A single passed
說服 can arrange 揭露情報 OR 物品轉移 OR 建議入夥; the arranger picks.

Quest handling is split by nature (a quest event is not one thing):
  · accept / turn-in are PLAYER-INPUT-driven → they ride arrange_events (stage 2,
    pre-response), and are injected into its context ONLY when in scope: accept
    once the quest has been OFFERED (Quest.offered, set after the NPC voices it),
    turn-in once it is COMPLETED. The `offered` gate structurally prevents the
    same-turn false-accept (offer and accept can't be the same turn).
  · complete is NPC-REPLY-driven and needs flexible judgement → its OWN agent
    (judge_completion), post-response, whose context is ONLY the accepted quests.

This module only DECIDES. Execution (mutating world state) lives in game.py's
event appliers, which reuse the existing tag_parser dispatchers.
"""
from __future__ import annotations
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from .backend import stream_chat

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

# Stage-2 (pre-response) event catalog, split by nature:
#   non-quest — reveal/give/suggest_join are CONCESSIONS (only on a passed check);
#               attitude/attack are unconditional.
#   quest     — quest_accept/quest_turnin are PLAYER-INPUT-driven and gated by
#               quest state: accept only once the quest has been OFFERED, turn-in
#               only once it is COMPLETED. They ride the same arrange_events call
#               (per design) but are conditionally injected into its context.
_NONQUEST_EVENTS   = ("reveal", "give", "suggest_join", "attitude", "attack")
_STAGE2_QUEST_EVENTS = ("quest_accept", "quest_turnin")
STAGE2_EVENTS = _NONQUEST_EVENTS + _STAGE2_QUEST_EVENTS
# quest_complete lives in its OWN dedicated agent (judge_completion), post-response,
# whose context is ONLY the accepted (active) quests — flexible persuade-type goals.
_COMPLETE_EVENT = "quest_complete"

# NPC reply classification is deliberately coarse. It answers only what kind
# of action the NPC explicitly expressed; entity resolution and world-state
# mutation belong to later stages. Actions are independent, so one reply may
# trigger more than one Noul. "none" is derived by Python.
NPC_ACTIONS = ("attack", "join", "give", "offer_quest")
_NPC_ACTION_THRESHOLDS = {
    "attack": 0.90,
    "join": 0.80,
    "give": 0.90,
    "offer_quest": 0.80,
}


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


def _first_json(text: str, opener: str):
    """Extract and parse the first JSON value (array or object) in `text`.
    Robust to models that wrap it in prose / code fences. Returns None on fail."""
    close = "]" if opener == "[" else "}"
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == opener:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (ValueError, json.JSONDecodeError):
                    return None
    return None


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

    def _ask(self, system: str, user: str, num_predict: int, temperature: float,
             debug_name: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        (_DEBUG_DIR / debug_name).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        opts = {**self.options, "temperature": temperature, "num_predict": num_predict}
        try:
            result = stream_chat(
                self.base_url, self.model, messages, opts,
                think=False, on_chunk=None, backend=self.backend, timeout=30,
                api_key=self.api_key,
            )
        except Exception:
            # A transient backend failure must not crash the conversation — the
            # caller's parse falls back to "no check / no events".
            return ""
        result = result.strip()
        # Persist the raw decision too (not just the context) so each stage's
        # judgement is directly inspectable, e.g. dialogue_event_output.txt.
        (_DEBUG_DIR / debug_name.replace("_context.json", "_output.txt")).write_text(
            result, encoding="utf-8", errors="replace")
        return result

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
                        "提供非一般性情報，或改變決定。"
                    ),
                    "false": (
                        "只是寒暄、閒聊、普通提問、陳述、回答、隊友間交談、沒有要求 NPC"
                        " 做事，或已直接動手攻擊。僅語氣兇狠而沒有明確威脅也屬於 false。"
                    ),
                },
            )
            # These two speculative questions run in the same request. Code uses
            # them only when the Noul above clears the configured threshold.
            questions[f"social_{index}_type"] = Choice(
                instructions=(
                    f"假設 {path} 構成社交檢定，選出玩家主要使用的手段。"
                ),
                criteria={
                    "persuade": "以真誠請求、協商、交換利益或講道理使 NPC 願意配合。",
                    "deceive": "以虛假身分、捏造事實、隱瞞關鍵資訊或誤導使 NPC 相信。",
                    "intimidate": "以明確威脅、可信的暴力後果或強迫命令使 NPC 因恐懼服從。",
                },
            )
            questions[f"social_{index}_effectiveness"] = Score(
                instructions=(
                    f"假設 {path} 構成社交檢定，評估它對目前 `npc` 有多容易奏效。"
                    "不要考慮玩家的 CHA、技能加值、骰點或運氣。依手段考慮要求的利益與"
                    "代價、謊言可信度或威脅可信度，並考慮 NPC 的個性、態度及"
                    "`conversation_history` 中是否重複使用相同手段。"
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
            "conversation_history": _prior_conversation_script(
                ws, getattr(npc, "char_id", ""), actions),
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

    # ── Stage 2 — 事件安排 ────────────────────────────────────────────────────
    @staticmethod
    def _build_npc_action_questions() -> dict[str, Any]:
        if Noul is None:
            raise RuntimeError("缺少 typesafe-sdk；請先安裝 requirements.txt 中的依賴")
        return {
            "attack": Noul(
                instructions="`npc_reply.text` 是否表示 NPC 已經在這一刻開始攻擊？",
                criteria={
                    "true": "NPC 明確動手、發動攻擊，或宣告立即進入戰鬥。",
                    "false": "只是生氣、辱罵、警告、拒絕，或威脅將來可能攻擊。",
                },
            ),
            "join": Noul(
                instructions="`npc_reply.text` 是否明確表示 NPC 想加入或跟隨玩家隊伍？",
                criteria={
                    "true": "NPC 明確提議加入、同行，或長期成為玩家隊伍的成員。",
                    "false": "只是暫時幫忙、提供建議、交付任務，或沒有明確表示要加入隊伍。",
                },
            ),
            "give": Noul(
                instructions="`npc_reply.text` 是否表示 NPC 現在就要把物品交給玩家？",
                criteria={
                    "true": "NPC 明確完成贈與或交付，玩家現在取得某項物品。",
                    "false": "只是答應以後給、提出交易、描述物品，或把物品當作未來報酬。",
                },
            ),
            "offer_quest": Noul(
                instructions=(
                    "`npc_reply.text` 是否向玩家提出一項新的任務或委託，"
                    "需要玩家決定是否接受？"
                ),
                criteria={
                    "true": "NPC 明確請玩家完成某個目標或行動。",
                    "false": "只是提及問題、提供背景、談論已有委託，或交付任務報酬。",
                },
            ),
        }

    def judge_npc_actions(self, npc_text: str) -> list[str]:
        """Coarsely classify one NPC utterance with one Jev request."""
        text = str(npc_text or "").strip()
        if not text:
            return ["none"]

        state = {"npc_reply": {"text": text}}
        questions = self._build_npc_action_questions()
        (_DEBUG_DIR / "dialogue_npc_action_context.json").write_text(
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
            probabilities = {}
            actions = []
            for action in NPC_ACTIONS:
                answer = self._typed_answer(response, "nouls", action)
                probability = float(answer.noul)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"Jev 回傳無效 Noul：{action}={probability!r}")
                probabilities[action] = probability
                if probability >= _NPC_ACTION_THRESHOLDS[action]:
                    actions.append(action)
        except Exception as exc:
            (_DEBUG_DIR / "dialogue_npc_action_output.json").write_text(
                json.dumps({
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "actions": ["none"],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8", errors="replace",
            )
            return ["none"]

        result = actions or ["none"]
        (_DEBUG_DIR / "dialogue_npc_action_output.json").write_text(
            json.dumps({
                "model": getattr(response, "model", self.typesafe_model),
                "thresholds": _NPC_ACTION_THRESHOLDS,
                "probabilities": probabilities,
                "actions": result,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8", errors="replace",
        )
        return result

    # ── Stage 2 — 事件安排 ───────────────────────────────────────────
    _EVENT_SYSTEM = (
        "你是 D&D 5e 對話結果編排者，只做一件事：看玩家發言與檢定結果，決定這輪要觸發"
        "哪些「事件」。可以是 0 個、1 個或多個（排列組合）。\n\n"
        "## 對象 NPC\n{npc_brief}\n"
        "## NPC 身上的物品（可被說服交出）\n{npc_items}\n"
        "## NPC 知道、但尚未告訴玩家的情報（編號）\n{secrets}\n\n"
        "## 檢定結果\n{check_line}\n\n"
        "## 事件目錄\n"
        "reveal      揭露情報：把上面某幾條情報告訴玩家。**僅在檢定成功時可用**。\n"
        "give        物品轉移：NPC 把自己的一件物品交給玩家。**僅在檢定成功時可用**。\n"
        "suggest_join 建議入夥：讓 NPC 認真考慮加入隊伍（最終仍由它自己決定）。**僅在檢定成功時可用**。\n"
        "attitude    好感度增減：改變 NPC 對玩家的態度，delta 為整數（可負）。任何時候可用。\n"
        "attack      翻臉開戰：NPC 或玩家動手，進入戰鬥。任何時候可用（例如玩家動手、或威脅過火觸怒 NPC）。\n"
        "{quest_section}"
        "\n## 判斷原則\n"
        "- reveal/give/suggest_join 是「被贏得的讓步」，檢定失敗或根本沒檢定就**不可**輸出。\n"
        "- 只輸出玩家實際要求、且情境合理的事件；玩家沒要的東西不要硬給。\n"
        "- 威嚇成功常伴隨 attitude 下降（怨恨）；欺瞞失敗常伴隨 attitude 下降（識破）。\n"
        "- reveal 的 indices 只能取自上面的情報編號；give 的 item 必須是上面列出的物品名。\n\n"
        "## 輸出（只輸出一行 JSON 陣列，無其他文字）\n"
        '例：[{{"event":"reveal","indices":[0,2]}}, {{"event":"attitude","delta":-1}}]\n'
        '例：[{{"event":"give","item":"手斧","to":"kaine"}}]\n'
        "沒有任何事件時輸出 []。"
    )

    @staticmethod
    def _build_quest_section(ws, accept_ids: set, turnin_ids: set) -> str:
        """The conditionally-injected quest block: accept only appears once the
        quest is OFFERED, turn-in only once COMPLETED. Empty when neither."""
        def _lines(ids):
            return "\n".join(f"    [{q.id}] {q.title}｜{q.description}"
                             for q in ws.quests.values() if q.id in ids)
        parts = []
        if accept_ids:
            parts.append(
                "quest_accept  **玩家這一輪的發言**明確答應接下「可接受的任務」中的某項"
                "（例：『好，我幫你』『交給我們』）。\n"
                "              ⚠ NPC 提出／拜託任務**不算**接受；玩家只是打聽、寒暄、還沒回應"
                "就**不要**觸發。id 取自下列。\n"
                "  可接受的任務：\n" + _lines(accept_ids) + "\n"
                '  觸發時這樣輸出（鍵名務必用 id）：{"event":"quest_accept","id":"上面的任務id"}\n')
        if turnin_ids:
            parts.append(
                "quest_turnin  玩家向委託人回報並領取「可提交的任務」的獎勵。id 取自下列。\n"
                "  可提交的任務：\n" + _lines(turnin_ids) + "\n"
                '  觸發時這樣輸出（鍵名務必用 id）：{"event":"quest_turnin","id":"上面的任務id"}\n')
        return "".join(parts)

    def arrange_events(self, player_text: str, result: CheckResult | None,
                       npc, ws) -> list[dict]:
        """Stage 2: pick events given the input + check outcome. Non-quest events
        (reveal/give/suggest_join/attitude/attack) plus, when in scope, the
        player-input-driven quest events accept (offered) / turnin (completed)."""
        npc_id = getattr(npc, "char_id", "")
        secrets = getattr(npc, "_secrets", []) or []
        # Only secrets not yet revealed are offerable.
        revealed = set(getattr(npc, "revealed", []) or [])
        secret_lines = "\n".join(
            f"  [{i}] {s}" for i, s in enumerate(secrets) if s not in revealed
        ) or "  （無）"
        if result is None:
            check_line = "本輪未進行檢定（reveal/give/suggest_join 不可用）。"
        else:
            check_line = (
                f"{result.check} 檢定 {'成功' if result.success else '失敗'}"
                f"（擲出 {result.total} vs DC {result.dc}）。"
            )
        # Conditionally-available quest events (design): accept only after the
        # quest has been OFFERED to the player; turn-in only once COMPLETED.
        accept_ids = {q.id for q in ws.quests.values()
                      if q.giver_id == npc_id and q.status == "inactive" and q.offered}
        turnin_ids = {q.id for q in ws.quests.values()
                      if q.giver_id == npc_id and q.status == "completed"}
        valid_qids = accept_ids | turnin_ids

        system = self._EVENT_SYSTEM.format(
            npc_brief=_npc_brief(npc), npc_items=_npc_items(npc),
            secrets=secret_lines, check_line=check_line,
            quest_section=self._build_quest_section(ws, accept_ids, turnin_ids),
        )
        raw = self._ask(system, f"玩家發言：\n{player_text}",
                        num_predict=200, temperature=0.2,
                        debug_name="dialogue_event_context.json")
        arr = _first_json(raw, "[")
        if not isinstance(arr, list):
            return []
        out = []
        for e in arr:
            if not isinstance(e, dict):
                continue
            ev = e.get("event")
            if ev in _NONQUEST_EVENTS:
                out.append(e)
            elif ev in _STAGE2_QUEST_EVENTS:
                qid = e.get("id") or e.get("quest_id")   # models vary on the key name
                if qid in valid_qids:
                    out.append({"event": ev, "id": qid})  # normalise to "id"
        return out

    # ── Completion judge — its own agent, minimal context, post-response ──────
    _COMPLETE_SYSTEM = (
        "你是任務完成裁判，只做一件事：看玩家發言與 NPC 回應，判斷這一輪對話有沒有讓某個"
        "「進行中的任務」達成目標。這類任務通常需要彈性判斷（例：任務要你說服某人停手，而"
        "NPC 在回應中真的答應停手了）。\n\n"
        "## 進行中的任務（玩家已承接）\n{active}\n\n"
        "## 原則\n"
        "- 只有 NPC 的回應**真正體現**任務目標被達成才觸發；只是嘴上敷衍、還在討價還價、"
        "或只答應「考慮看看」，都**不算**。\n"
        "- id 只能取自上面清單。沒有達成任何任務時輸出 []。\n\n"
        "## 輸出（只輸出一行 JSON 陣列，無其他文字）\n"
        '例：[{{"event":"quest_complete","id":"stop_boss"}}]\n'
        "沒有任何達成時輸出 []。"
    )

    def judge_completion(self, player_text: str, npc_response: str, ws) -> list[dict]:
        """Dedicated post-response agent: did this exchange fulfil any ACTIVE quest's
        goal? Context is ONLY the accepted quests (name+content) — NPC-agnostic, since
        the current NPC may be the target of a quest given by someone else."""
        active = [f"  [{q.id}] {q.title}｜{q.description}"
                  for q in ws.quests.values() if q.status == "active"]
        if not active:
            return []   # nothing accepted → skip the call entirely
        system = self._COMPLETE_SYSTEM.format(active="\n".join(active))
        user = f"玩家發言：\n{player_text}\n\nNPC 回應：\n{npc_response}"
        raw = self._ask(system, user, num_predict=60, temperature=0.1,
                        debug_name="dialogue_complete_context.json")
        arr = _first_json(raw, "[")
        if not isinstance(arr, list):
            return []
        valid = {q.id for q in ws.quests.values() if q.status == "active"}
        out = []
        for e in arr:
            if isinstance(e, dict) and e.get("event") == _COMPLETE_EVENT:
                qid = e.get("id") or e.get("quest_id")   # tolerate the key name
                if qid in valid:
                    out.append({"event": _COMPLETE_EVENT, "id": qid})
        return out


# ── context helpers ───────────────────────────────────────────────────────────

def _npc_brief(npc) -> str:
    char = getattr(npc, "char", None)
    name = getattr(char, "name", getattr(npc, "char_id", "NPC"))
    att = npc.attitude
    return f"{name}（態度：{_ATTITUDE_LABELS[att]} {att}/4）\n{getattr(npc, '_personality', '')}"


def _npc_items(npc) -> str:
    char = getattr(npc, "char", None)
    if char is None:
        return "  （無）"
    names = [w.name for w in getattr(char, "weapons", [])]
    names += [f"{c.name}×{c.quantity}" for c in getattr(char, "consumables", [])
              if c.quantity > 0]
    return "\n".join(f"  {n}" for n in names) or "  （無）"


def _prior_conversation_script(ws, npc_id: str,
                               actions: list[dict[str, str]]) -> str:
    """Render history without duplicating the current, already-logged PC lines."""
    entries = list(ws.log_for(npc_id))
    count = len(actions)
    if count and len(entries) >= count:
        tail = entries[-count:]
        if all(
            entry.get("speaker") == action["actor_id"]
            and entry.get("text") == action["text"]
            for entry, action in zip(tail, actions)
        ):
            entries = entries[:-count]
    return "\n".join(
        f"{ws.speaker_label(entry['speaker'])}：{entry['text']}"
        for entry in entries
    )


def _default_actor(ws) -> str:
    """The human PC if present, else the first living PC."""
    if "kaine" in ws.characters and ws.characters["kaine"].is_alive():
        return "kaine"
    for cid in ws.pc_ids:
        c = ws.characters.get(cid)
        if c and not c.is_npc and c.is_alive():
            return cid
    return "kaine"
