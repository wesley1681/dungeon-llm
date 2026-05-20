"""Frontend abstraction for the sparring sandbox.

A Frontend is anything that can:
  - render the world (for human observers)
  - prompt for an action_dict on the user's turn
  - announce out-of-band events (opponent moves, round transitions, deaths)

The combat driver depends only on this interface. Concrete impls:
  ScriptedFrontend  — test double, returns a canned list of actions
  ConsoleFrontend   — rich-powered terminal UI (Task 7)
"""
from __future__ import annotations
from typing import Protocol


class Frontend(Protocol):
    def render(self, ws, agent_id: str, opp_id: str) -> None: ...
    def prompt_action(self, ws, actor, resources: dict) -> dict | None: ...
    def announce(self, msg: str) -> None: ...


class ScriptedFrontend:
    """Test double. Returns a pre-recorded sequence of actions from prompt_action.

    Action dicts (or None for "end turn") are pulled in order. Exhausting the
    list raises StopIteration so tests catch unintended extra prompts.
    """

    def __init__(self, actions: list[dict | None]):
        self._actions = iter(actions)
        self.render_calls = 0
        self.announcements: list[str] = []

    def render(self, ws, agent_id: str, opp_id: str) -> None:
        self.render_calls += 1

    def prompt_action(self, ws, actor, resources: dict) -> dict | None:
        return next(self._actions)

    def announce(self, msg: str) -> None:
        self.announcements.append(msg)
