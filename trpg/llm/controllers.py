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


class ActorController:
    """Base class. All controllers implement take_sub_action; on_invalid_action
    is optional and defaults to None (give up the sub-action)."""

    def take_sub_action(self, char, ctx: CombatContext) -> ActorDecision:
        raise NotImplementedError

    def on_invalid_action(self, reason: str, suggestion: str) -> ActorDecision | None:
        """Called when arbiter rejects the controller's description.
        Return a fresh ActorDecision to retry, or None to drop this sub-action."""
        return None


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
        return (
            f"【戰鬥回合 {ctx.round_num}】\n"
            f"HP：{char.hp}/{char.max_hp}\n"
            f"武器：{ctx.weapons_str}\n"
            f"盟友：{ctx.allies_str}\n"
            f"敵人：{ctx.enemies_str}\n"
            f"做一個 sub-action（攻擊 / 移動 / 閃避）；想結束本回合就在訊息結尾加 <END>。\n"
            f"例：「我衝向哥布林。」不加 <END>→系統會問你下一步；"
            f"「我用長劍砍他。<END>」→ 砍完直接結束。"
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


class LLMNpcController(ActorController):
    """Wraps an NpcAgent. Delegates to NpcAgent.combat_action() which already
    returns (desc, fled, ended); maps to ActorDecision."""

    def __init__(self, agent, emit_event):
        self.agent = agent
        self.emit_event = emit_event
        self._last_char = None
        self._last_ctx: CombatContext | None = None
        self._retries_left = 0

    def _generate(self, char, ctx: CombatContext, error_feedback: str) -> ActorDecision:
        from ..game import StreamChunk
        actor = char.name
        desc, fled, ended = self.agent.combat_action(
            ctx.weapons_str, ctx.allies_str, ctx.enemies_str,
            resources=ctx.resources,
            error_feedback=error_feedback,
            on_chunk=lambda c, thinking=False: self.emit_event(StreamChunk("npc", c, actor=actor)),
        )
        if fled:
            return ActorDecision(fled=True)
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
