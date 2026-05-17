"""ActorController — unified interface for everything that takes a combat turn.

Each char_id in a GameSession is paired with one controller. The combat loop
calls `controller.take_sub_action(char, ctx)` and acts on the returned
ActorDecision; it does not know or care whether the controller wraps a human,
an LLM player agent, or an LLM NPC agent.

Adding a new playable character type = write a new ActorController subclass
and register it; the combat loop changes nothing.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

from ..engine.combat import CombatContext


_END_RE = re.compile(r'<\s*END\s*>', re.IGNORECASE)
_FLEE_RE = re.compile(r'<\s*FLEE\s*>', re.IGNORECASE)


def _strip_marker(text: str, pat: re.Pattern) -> tuple[str, bool]:
    matched = bool(pat.search(text))
    cleaned = pat.sub("", text).strip()
    return cleaned, matched


@dataclass
class ActorDecision:
    description: str = ""   # natural language for arbiter; "" = end-only, no action
    ended: bool = False     # turn ends after this sub-action
    fled: bool = False      # actor flees combat entirely (NPC only)
    quit: bool = False      # quit the game (Human only)


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
    """Base class. All controllers implement take_sub_action; the non-combat
    methods default to "this actor doesn't participate in that slot" so a
    follower-style controller can omit them and stay silent automatically."""

    # ── Combat (always implemented by participating actors) ─────────────────
    def take_sub_action(self, char, ctx: CombatContext) -> ActorDecision:
        raise NotImplementedError

    def on_invalid_action(self, reason: str, suggestion: str) -> ActorDecision | None:
        """Called when arbiter rejects the controller's description.
        Return a fresh ActorDecision to retry, or None to drop this sub-action."""
        return None

    # ── Exploration / conversation (overridden by PC-style controllers) ─────
    def take_exploration_turn(self, char, *, gm_text: str = "",
                              prior_remarks: dict[str, str] | None = None
                              ) -> ExplorationOutput:
        """Returns an ExplorationOutput. Default is a no-op (silent) — used by
        non-PC controllers (followers, enemy NPCs)."""
        return ExplorationOutput()

    def take_conversation_turn(self, char, npc_char, attitude_label: str
                               ) -> ConversationOutput:
        """Returns a ConversationOutput. Default is silent — used by non-PC
        controllers."""
        return ConversationOutput(silent=True)

    # ── NPC-side conversation methods (only LLMNpcController implements) ────
    def take_npc_opening(self, char) -> str:
        """Generate the NPC's opening line when a conversation starts. Default
        empty string — only LLMNpcController implements this."""
        return ""

    def take_npc_response(self, char) -> str:
        """Generate the NPC's reply after the PCs spoke. Default empty."""
        return ""


# ── Human ────────────────────────────────────────────────────────────────────

class HumanController(ActorController):
    """Reads from a stdin/queue callback and emits CombatPrompt events.

    Construction:
      char_id     — which character this controls (for emit context)
      get_input   — () -> str | None; blocking input read. None = quit signal.
      emit_event  — (event) -> None; pushes UI events (CombatPrompt etc.).
      end_inputs  — set of strings that mean "end my turn" (e.g. {"結束", "end"})
    """
    def __init__(self, char_id: str, get_input, emit_event, end_inputs: set[str]):
        self.char_id = char_id
        self.get_input = get_input
        self.emit_event = emit_event
        self.end_inputs = end_inputs

    def take_sub_action(self, char, ctx: CombatContext) -> ActorDecision:
        # Import inside method to avoid circular game.py import
        from ..game import CombatPrompt, format_aria_combat_info

        # Emit prompt (caller assembles UI)
        prompt = CombatPrompt(
            aria=char, enemies=ctx.enemies,
            info_text=format_aria_combat_info(char, ctx),
        )
        self.emit_event(prompt)

        raw = self.get_input()
        if raw is None:
            return ActorDecision(quit=True)
        text = raw.strip()
        if not text:
            return ActorDecision()    # empty: caller re-prompts on next iteration
        if text.lower() in self.end_inputs:
            return ActorDecision(ended=True)
        if text.lower() == "quit":
            return ActorDecision(quit=True)
        return ActorDecision(description=text)

    def on_invalid_action(self, reason: str, suggestion: str) -> ActorDecision | None:
        raw = self.get_input()
        if raw is None:
            return ActorDecision(quit=True)
        text = raw.strip()
        if not text:
            return None
        if text.lower() in self.end_inputs:
            return ActorDecision(ended=True)
        if text.lower() == "quit":
            return ActorDecision(quit=True)
        return ActorDecision(description=text)

    # ── Non-combat (UI-driven) ──────────────────────────────────────────────
    def take_exploration_turn(self, char, *, gm_text: str = "",
                              prior_remarks: dict[str, str] | None = None
                              ) -> ExplorationOutput:
        from ..game import ExplorationPrompt
        self.emit_event(ExplorationPrompt(
            aria=char, gm_text=gm_text,
            prior_remarks=dict(prior_remarks or {}),
        ))
        while True:
            raw = self.get_input()
            if raw is None:
                return ExplorationOutput(quit=True)
            text = raw.strip()
            if text.lower() == "quit":
                return ExplorationOutput(quit=True)
            if text:
                return ExplorationOutput(text=text)
            # empty input — re-prompt by looping (caller's UI typically
            # handles the re-prompt itself, but block here defensively)

    def take_conversation_turn(self, char, npc_char, attitude_label: str
                               ) -> ConversationOutput:
        from ..game import ConversationPrompt
        self.emit_event(ConversationPrompt(
            npc_name=npc_char.name, aria=char,
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

# LLM controllers cap retries so a misbehaving model can't loop forever on
# the same rejected sub-action. Caller (game.py _execute_sub_action) hands the
# rejection reason to on_invalid_action; the controller regenerates with that
# reason injected into the next prompt so the model can pick something else.
_LLM_RETRIES_PER_SUB_ACTION = 1


class LLMPlayerController(ActorController):
    """Wraps a PlayerAgent. Builds a Thor-style combat nudge from CombatContext
    and lets the agent generate a sub-action description; strips <END> marker."""

    def __init__(self, agent, emit_event):
        self.agent = agent
        self.emit_event = emit_event
        self._last_char = None
        self._last_ctx: CombatContext | None = None
        self._retries_left = 0

    def _build_nudge(self, char, ctx: CombatContext) -> str:
        combat_tactics = (self.agent.combat_tactics.rstrip() + "\n\n"
                          if self.agent.combat_tactics else "")
        return (
            f"{combat_tactics}"
            f"【戰鬥回合 {ctx.round_num}】\n"
            f"HP：{char.hp}/{char.max_hp}\n"
            f"武器：{ctx.weapons_str}\n"
            f"盟友：{ctx.allies_str}\n"
            f"敵人：{ctx.enemies_str}\n"
            f"從下列**選一個** sub-action 輸出（不要組合）：\n"
            f"- 攻擊：「我用 [武器] 攻擊 [敵人]」\n"
            f"- 移動：「我衝上去」「我後退」（單次最多 9m）\n"
            f"- 閃避：「我閃避」「我專注防禦」\n"
            f"- 躲藏：「我躲到 X 後面」\n"
            f"- 結束本回合：單獨輸出 <END>\n"
            f"完成本動作想直接結束回合，訊息結尾加 <END>（例：「我用長劍砍他。<END>」）。"
        )

    def _generate(self, char, ctx: CombatContext, error_feedback: str) -> ActorDecision:
        from ..game import StreamChunk
        nudge = self._build_nudge(char, ctx)
        if error_feedback:
            nudge += f"\n\n## 系統訊息\n上次行動被拒：{error_feedback}\n請改選不同的 sub-action。"
        actor = char.name
        desc = self.agent.generate(
            nudge=nudge,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk("pc_combat", c, actor=actor)
            ),
        )
        desc, ended = _strip_marker(desc, _END_RE)
        return ActorDecision(description=desc, ended=ended)

    def take_sub_action(self, char, ctx: CombatContext) -> ActorDecision:
        self._last_char = char
        self._last_ctx = ctx
        self._retries_left = _LLM_RETRIES_PER_SUB_ACTION
        return self._generate(char, ctx, error_feedback="")

    def on_invalid_action(self, reason: str, suggestion: str) -> ActorDecision | None:
        if self._retries_left <= 0 or self._last_char is None or self._last_ctx is None:
            return None
        self._retries_left -= 1
        feedback = reason + (f"（{suggestion}）" if suggestion else "")
        return self._generate(self._last_char, self._last_ctx, error_feedback=feedback)

    # ── Non-combat ──────────────────────────────────────────────────────────
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
    """Wraps an NpcAgent in combat mode. Builds the combat nudge here (tactics
    + situation + action menu) and calls agent.generate(combat=True, nudge=...).
    Parses <FLEE>/<END> markers off the response. Mirrors LLMPlayerController."""

    def __init__(self, agent, emit_event):
        self.agent = agent
        self.emit_event = emit_event
        self._last_char = None
        self._last_ctx: CombatContext | None = None
        self._retries_left = 0

    def _build_nudge(self, char, ctx: CombatContext) -> str:
        combat_tactics = (self.agent.combat_tactics.rstrip() + "\n\n"
                          if self.agent.combat_tactics else "")
        return (
            f"{combat_tactics}"
            f"【戰鬥回合 {ctx.round_num}】\n"
            f"HP：{char.hp}/{char.max_hp}\n"
            f"武器：{ctx.weapons_str}\n"
            f"盟友：{ctx.allies_str}\n"
            f"敵人：{ctx.enemies_str}\n"
            f"從下列**選一個** sub-action 輸出（不要組合）：\n"
            f"- 攻擊：「我用 [武器] 攻擊 [敵人]」\n"
            f"- 移動：「我衝上去」「我後退」（單次最多 9m）\n"
            f"- 閃避：「我閃避」「我專注防禦」\n"
            f"- 躲藏：「我躲到 X 後面」\n"
            f"- 結束本回合：單獨輸出 <END>\n"
            f"完成本動作想結束回合，訊息結尾加 <END>。\n"
            f"逃跑：訊息結尾加 <FLEE>，立刻離開戰場。"
        )

    def _generate(self, char, ctx: CombatContext, error_feedback: str) -> ActorDecision:
        from ..game import StreamChunk
        nudge = self._build_nudge(char, ctx)
        if error_feedback:
            nudge += f"\n\n## 系統訊息\n上次行動被拒：{error_feedback}\n請改選不同的 sub-action。"
        actor = char.name
        desc = self.agent.generate(
            nudge=nudge, combat=True,
            on_chunk=lambda c, thinking=False: self.emit_event(
                StreamChunk("npc", c, actor=actor)
            ),
        )
        # <FLEE> overrides everything (leave combat); otherwise check <END>.
        desc, fled = _strip_marker(desc, _FLEE_RE)
        if fled:
            return ActorDecision(fled=True)
        desc, ended = _strip_marker(desc, _END_RE)
        return ActorDecision(description=desc, ended=ended)

    def take_sub_action(self, char, ctx: CombatContext) -> ActorDecision:
        self._last_char = char
        self._last_ctx = ctx
        self._retries_left = _LLM_RETRIES_PER_SUB_ACTION
        return self._generate(char, ctx, error_feedback="")

    def on_invalid_action(self, reason: str, suggestion: str) -> ActorDecision | None:
        if self._retries_left <= 0 or self._last_char is None or self._last_ctx is None:
            return None
        self._retries_left -= 1
        feedback = reason + (f"（{suggestion}）" if suggestion else "")
        return self._generate(self._last_char, self._last_ctx, error_feedback=feedback)

    # ── Conversation (NPC side) ─────────────────────────────────────────────
    def take_npc_opening(self, char) -> str:
        from ..game import StreamChunk
        nudge = (f"## 現在請\n以 {char.name} 的身份，根據以上歷史和當前態度，"
                 "用第一人稱繁體中文簡短回應走近的冒險者（開場第一句）。")
        # Opening = first interaction; no prior signal to map to [+/-].
        self.agent._skip_marker = True
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
