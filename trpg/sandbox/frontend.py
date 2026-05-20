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


from rich.console import Console
from rich.text import Text

from ..engine.vec2 import TerrainType


_GLYPHS = {
    TerrainType.NORMAL:    ("·", "dim"),
    TerrainType.BLOCKED:   ("█", "white"),
    TerrainType.DANGEROUS: ("░", "yellow"),
    TerrainType.DIFFICULT: ("▒", "cyan"),
}


class ConsoleFrontend:
    """Rich-powered terminal frontend.

    Rendering samples the battlefield at integer-metre coords, so each rendered
    cell == one metre. The 30×30 m default field maps to a 30×30 char grid.
    Characters are drawn on top of terrain glyphs at their rounded positions.
    """

    def __init__(self) -> None:
        self._console = Console()

    def announce(self, msg: str) -> None:
        self._console.print(msg)

    def render(self, ws, agent_id: str, opp_id: str) -> None:
        bf = ws.combat.battlefield
        agent = ws.characters[agent_id]
        opp = ws.characters[opp_id]
        from ..engine.vec2 import Vec2

        text = Text()
        n_cols = int(bf.width)
        n_rows = int(bf.height)
        ax, ay = int(agent.position.x), int(agent.position.y)
        ox, oy = int(opp.position.x), int(opp.position.y)
        # Top axis (every-other column gets its low digit)
        text.append("    " + "".join(
            str(x % 10) if x % 2 == 0 else " " for x in range(n_cols)
        ) + "\n")
        for y in range(n_rows):
            text.append(f"{y:3d} ")
            for x in range(n_cols):
                if (x, y) == (ax, ay):
                    text.append("A", style="bold blue")
                elif (x, y) == (ox, oy):
                    text.append("O", style="bold red")
                else:
                    tt = bf.terrain_at(Vec2(x + 0.5, y + 0.5))
                    glyph, style = _GLYPHS.get(tt, ("?", "magenta"))
                    text.append(glyph, style=style)
            text.append("\n")
        self._console.print(text)

    def prompt_action(self, ws, actor, resources: dict) -> dict | None:
        # Implemented in Task 8. The stub raises so a missing impl is loud.
        raise NotImplementedError("ConsoleFrontend.prompt_action — implemented in Task 8")
