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
