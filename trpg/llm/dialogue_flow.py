"""Conversation resolution pipeline — focused LLM stages per round.

Replaces the old single `generate_conversation_tags` classifier. Each LLM call
stays focused on ONE job so tasks don't bleed together:

  1. decide_check     — 玩家輸入 → 要不要檢定？哪個(威嚇/欺瞞/說服)？誰？DC 多少？
  2. arrange_events   — 玩家輸入 + 檢定結果 → 事件目錄挑觸發（reveal/give/suggest_join/
                        attitude/attack）＋ **條件加入** 玩家驅動的 quest_accept（任務已被
                        提出）/ quest_turnin（任務已完成）
  3. judge_completion — 已承接(active)任務名稱+內容 + 玩家輸入 + NPC 回應 → 判定 quest_complete

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
import pathlib
from dataclasses import dataclass

from .backend import stream_chat
from .social_dc_agent import SocialDcAgent
from .log_render import render_script

_DEBUG_DIR = pathlib.Path(__file__).parent.parent / "debug"

_ATTITUDE_LABELS = ["敵意", "戒備", "中立", "友好", "信任"]

# ── Catalogs ──────────────────────────────────────────────────────────────────
# The check catalog (all CHA-based today; extend here to add new skills/stats).
CHECK_TYPES = ("威嚇", "欺瞞", "說服")
_CHECK_EN = {"威嚇": "intimidate", "欺瞞": "deceive", "說服": "persuade"}

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
    """Owns the three conversation-resolution LLM calls. Built from the same
    connection info as the other agents (see DialogueDirector.from_agent)."""

    def __init__(self, model: str, base_url: str, backend: str, world_state,
                 options: dict | None = None, api_key: str | None = None):
        self.model      = model
        self.base_url   = base_url
        self.backend    = backend
        self.world_state = world_state
        self.options    = options or {}
        self.api_key    = api_key
        # DC calibration stays single-source in the tuned SocialDcAgent.
        self._dc_agent  = SocialDcAgent(model, base_url, backend, api_key=api_key)

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

    # ── Stage 1 — 檢定判定 ────────────────────────────────────────────────────
    _CHECK_SYSTEM = (
        "你是 D&D 5e 對話裁判，只做一件事：判斷玩家這輪對 NPC 的發言，是否構成一個"
        "需要擲骰的社交嘗試；若是，指出是哪一種、由哪位玩家發起。\n\n"
        "## 對象 NPC\n{npc_brief}\n\n"
        "## 三種社交檢定\n"
        "威嚇 = 明確威脅、恐嚇、拔武器示威、帶暴力後果的命令\n"
        "欺瞞 = 謊稱身份、捏造事實、欺騙\n"
        "說服 = 懇求、勸說、要求對方做某件具體的事（含要求交出物品、加入隊伍、停止某行為）\n\n"
        "## 玩家角色ID\n{pc_ids}\n\n"
        "## 規則\n"
        "- 只有玩家的話「直接針對該 NPC」且有成敗風險時才算檢定。\n"
        "- 普通提問、寒暄、閒聊、回答、隊員之間交談 → 不是檢定。\n"
        "- 僅口氣兇、催促、皺眉不算威嚇；威嚇要有明確威脅內容。\n"
        "- 若玩家只是動手攻擊（已揮劍/出拳），那不是社交檢定 → 輸出 none。\n\n"
        "## 輸出（只輸出一行 JSON，無其他文字）\n"
        '{{"check": "威嚇"|"欺瞞"|"說服"|"none", "actor": "<玩家ID>"}}\n'
        "不需檢定時 check 填 none、actor 留空字串。"
    )

    def decide_check(self, player_text: str, npc, ws) -> CheckPlan:
        """Stage 1: is a check needed? which? who? + DC (from SocialDcAgent)."""
        pc_ids = "、".join(
            f"{cid}＝{c.name}" for cid, c in ws.characters.items()
            if not c.is_npc and c.is_alive()
        )
        system = self._CHECK_SYSTEM.format(
            npc_brief=_npc_brief(npc), pc_ids=pc_ids or "kaine＝凱恩",
        )
        raw = self._ask(system, f"玩家發言：\n{player_text}",
                        num_predict=40, temperature=0.1,
                        debug_name="dialogue_check_context.json")
        obj = _first_json(raw, "{") or {}
        check = obj.get("check")
        if check not in CHECK_TYPES:
            return CheckPlan()   # no check
        actor = obj.get("actor") or ""
        if actor not in ws.characters or ws.characters[actor].is_npc:
            actor = _default_actor(ws)
        # DC via the tuned, repeat-penalty-aware SocialDcAgent (single source).
        try:
            dc = self._dc_agent.estimate(
                social_type=_CHECK_EN[check], attempt_text=player_text,
                personality=getattr(npc, "_personality", ""),
                attitude=npc.attitude, attitude_label=_ATTITUDE_LABELS[npc.attitude],
                conv_log_text=render_script(ws, getattr(npc, "char_id", "")),
            )
        except Exception:
            dc = 12   # safe default DC on estimator failure
        return CheckPlan(check=check, dc=dc, actor=actor)

    # ── Stage 2 — 事件安排 ────────────────────────────────────────────────────
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


def _default_actor(ws) -> str:
    """The human PC if present, else the first living PC."""
    if "kaine" in ws.characters and ws.characters["kaine"].is_alive():
        return "kaine"
    for cid in ws.pc_ids:
        c = ws.characters.get(cid)
        if c and not c.is_npc and c.is_alive():
            return cid
    return "kaine"
