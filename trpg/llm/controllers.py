"""ActorController — non-combat slot dispatch for each character.

Each char_id in a GameSession is paired with one controller. Combat sub-action
decisions go through a CombatPolicy (engine.combat_policy) — controllers are
only responsible for exploration narration and conversation turns.

Adding a new playable character type = write a new ActorController subclass
and register it; combat doesn't care.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ExplorationOutput:
    text: str = ""        # actor's narration / action — caller logs and uses in tag_actions
    quit: bool = False    # quit the game


@dataclass
class ConversationOutput:
    text: str = ""        # what the actor said (empty if silent)
    silent: bool = False  # actor explicitly chose silence — caller skips this slot
    leave: bool = False   # end conversation
    quit: bool = False    # quit the game


class ActorController:
    """Base class for non-combat slots. Defaults are no-ops so a follower-style
    controller can omit them and stay silent automatically."""

    def take_exploration_turn(self, char, *, gm_text: str = "",
                              prior_remarks: dict[str, str] | None = None
                              ) -> ExplorationOutput:
        return ExplorationOutput()

    def take_conversation_turn(self, char, npc_char, attitude_label: str
                               ) -> ConversationOutput:
        return ConversationOutput(silent=True)

    # ── NPC-side conversation (only LLMNpcController implements) ────────────
    def take_npc_opening(self, char) -> str:
        return ""

    def take_npc_response(self, char) -> str:
        return ""


# ── Human ────────────────────────────────────────────────────────────────────

class HumanController(ActorController):
    """Reads from a stdin/queue callback for exploration / conversation slots.

    Combat is auto-resolved by the CombatPolicy assigned to this character —
    HumanController is no longer involved in combat sub-action decisions.
    """
    def __init__(self, char_id: str, get_input, emit_event):
        self.char_id = char_id
        self.get_input = get_input
        self.emit_event = emit_event

    def take_exploration_turn(self, char, *, gm_text: str = "",
                              prior_remarks: dict[str, str] | None = None
                              ) -> ExplorationOutput:
        from ..game import ExplorationPrompt
        while True:
            self.emit_event(ExplorationPrompt(
                kaine=char, gm_text=gm_text,
                prior_remarks=dict(prior_remarks or {}),
            ))
            raw = self.get_input()
            if raw is None:
                return ExplorationOutput(quit=True)
            text = raw.strip()
            if text.lower() == "quit":
                return ExplorationOutput(quit=True)
            # REST commands: intercept before TagAgent so they don't confuse
            # the LLM. Apply immediately and prompt again for next action.
            rest_lower = text.lower()
            if rest_lower in ("短休", "short rest", "短休息", "rest short"):
                self._do_rest(char, "short")
                continue
            if rest_lower in ("長休", "long rest", "長休息", "rest long"):
                self._do_rest(char, "long")
                continue
            if text:
                return ExplorationOutput(text=text)

    def _do_rest(self, char, rest_type: str) -> None:
        from ..engine.character import rest_character
        from ..game import StatusMessage
        result = rest_character(char, rest_type)
        label = "短休" if rest_type == "short" else "長休"
        recovered = "、".join(result.get("recovered", [])) or "（無可回復資源）"
        self.emit_event(StatusMessage(f"{label}完成。恢復：{recovered}"))

    def take_conversation_turn(self, char, npc_char, attitude_label: str
                               ) -> ConversationOutput:
        from ..game import ConversationPrompt
        self.emit_event(ConversationPrompt(
            npc_name=npc_char.name, kaine=char,
            attitude_label=attitude_label,
        ))
        while True:
            raw = self.get_input()
            if raw is None:
                return ConversationOutput(quit=True)
            text = raw.strip()
            if not text:
                continue
            if text.lower() in ("離開", "結束", "quit"):
                return ConversationOutput(leave=True)
            return ConversationOutput(text=text)


# ── LLM-driven controllers ───────────────────────────────────────────────────

class LLMPlayerController(ActorController):
    """Wraps a PlayerAgent for exploration narration + conversation lines.
    Combat is handled by an attached CombatPolicy, not by this controller."""

    def __init__(self, agent, emit_event):
        self.agent = agent
        self.emit_event = emit_event

    def take_exploration_turn(self, char, *, gm_text: str = "",
                              prior_remarks: dict[str, str] | None = None
                              ) -> ExplorationOutput:
        from ..game import StreamChunk
        nudge = (f"## 現在請\n以 {char.name} 的身份描述你的下一步——"
                 "做什麼動作、看什麼、或對隊友說什麼。")
        text = self.agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk(self.agent.char_id, c, thinking)
            ),
        )
        return ExplorationOutput(text=text)

    def take_conversation_turn(self, char, npc_char, attitude_label: str
                               ) -> ConversationOutput:
        from ..game import StreamChunk
        nudge = (f"現在輪到你（{char.name}），對方是 {npc_char.name}（態度：{attitude_label}）。"
                 "如果有話要說直接說出來；如果選擇保持沉默，輸出 [SILENT]。")
        text = self.agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk(self.agent.char_id, c, thinking)
            ),
        )
        if "[SILENT]" in text.upper() or not text.strip():
            return ConversationOutput(silent=True)
        return ConversationOutput(text=text.strip())


class LLMNpcController(ActorController):
    """Wraps an NpcAgent for conversation-side dialogue. Combat is handled by
    the actor's CombatPolicy, not this controller."""

    def __init__(self, agent, emit_event):
        self.agent = agent
        self.emit_event = emit_event

    def take_npc_opening(self, char) -> str:
        from ..game import StreamChunk
        nudge = (f"## 現在請\n以 {char.name} 的身份，根據以上歷史和當前態度，"
                 "用第一人稱繁體中文簡短回應走近的冒險者（開場第一句）。")
        return self.agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk("npc_talk", c, actor=char.name)
            ),
        )

    def take_npc_response(self, char) -> str:
        from ..game import StreamChunk
        nudge = (f"## 現在請\n以 {char.name} 的身份，根據以上對話歷史和當前態度，"
                 "用第一人稱繁體中文簡短回應對方剛才說的話。")
        return self.agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk("npc_talk", c, actor=char.name)
            ),
        )
