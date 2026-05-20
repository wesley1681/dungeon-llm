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
        # Top axis: two rows — tens digit (only at every multiple of 10),
        # then ones digit (every-other column). Distinguishes x=0 / 10 / 20.
        text.append("    " + "".join(
            str(x // 10) if x % 10 == 0 and x > 0 else " " for x in range(n_cols)
        ) + "\n")
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

        from rich.table import Table
        card = Table.grid(padding=(0, 2))
        card.add_column()
        card.add_column()
        card.add_row(_char_card(agent, "blue"), _char_card(opp, "red"))
        self._console.print(card)

    def prompt_action(self, ws, actor, resources: dict) -> dict | None:
        from ..engine.skill import available_skills, TargetType
        from ..rl.obs import partition_entities

        ENTITY_TT = (TargetType.SINGLE_ENEMY, TargetType.SINGLE_ALLY,
                     TargetType.MULTI_ENEMY, TargetType.MULTI_ALLY)
        GRID_TT = (TargetType.POINT, TargetType.LINE, TargetType.CONE)

        # Resolve actor's char_id via reverse lookup
        actor_id = next(
            (cid for cid, c in ws.characters.items() if c is actor), None
        )

        # available_skills() prepends an "end" sentinel at index 0; drop it
        # from the menu since we offer a dedicated "0) end" option below.
        skills = [s for s in available_skills(actor, ws) if s.skill_id != "end"]
        # Show the menu (0 = end; 1..N = skills)
        self._console.print(f"\n[bold]{actor.name} 的回合 — 資源: "
                            f"action={resources.get('action',0)} "
                            f"bonus={resources.get('bonus_action',0)} "
                            f"move={resources.get('movement',0):.1f}m[/]")
        self._console.print("[dim]  0) end (結束回合)[/]")
        for i, sk in enumerate(skills, start=1):
            self._console.print(f"  {i:2d}) {sk.skill_id}   "
                                f"({sk.features.target_type.name})")

        # Skill selection
        while True:
            raw = input("選擇動作 # > ").strip()
            try:
                idx = int(raw)
            except ValueError:
                self._console.print(f"[red]請輸入數字，不是 {raw!r}[/]")
                continue
            if idx == 0:
                return None
            if 1 <= idx <= len(skills):
                break
            self._console.print(f"[red]超出範圍 (0..{len(skills)})[/]")

        sk = skills[idx - 1]
        tt = sk.features.target_type

        # Target
        if tt == TargetType.SELF:
            return sk.build_action(actor_id, actor_id, None)

        if tt in ENTITY_TT:
            allies, enemies = partition_entities(ws, actor_id)
            # Entity slot list: 0=self, 1..=allies, then enemies
            slots = {0: actor_id}
            for k, aid in enumerate(allies[:2], start=1):
                slots[k] = aid
            for k, eid in enumerate(enemies[:3], start=len(allies[:2]) + 1):
                slots[k] = eid
            self._console.print("[dim]目標:[/]")
            for k, cid in slots.items():
                c = ws.characters[cid]
                self._console.print(
                    f"  {k}) {c.name} (HP {c.hp}/{c.max_hp})"
                )
            while True:
                raw = input("目標 # > ").strip()
                try:
                    tidx = int(raw)
                except ValueError:
                    self._console.print(f"[red]請輸入數字[/]")
                    continue
                if tidx in slots:
                    return sk.build_action(actor_id, slots[tidx], None)
                self._console.print(f"[red]目標不存在[/]")

        if tt in GRID_TT:
            while True:
                raw = input("目標座標 (x y) > ").strip().split()
                if len(raw) != 2:
                    self._console.print(f"[red]需要兩個數字 x y[/]")
                    continue
                try:
                    x, y = float(raw[0]), float(raw[1])
                except ValueError:
                    self._console.print(f"[red]無效座標[/]")
                    continue
                from ..engine.vec2 import Vec2
                return sk.build_action(actor_id, None, Vec2(x, y))

        # Fallback (should not reach)
        return sk.build_action(actor_id, None, None)


def _char_card(char, color: str) -> "Text":
    from rich.text import Text
    t = Text()
    t.append(f"{char.name}  (HP {char.hp}/{char.max_hp}  AC {char.ac})\n",
             style=f"bold {color}")
    # HP bar
    bar_w = 20
    filled = int(bar_w * max(0, char.hp) / max(1, char.max_hp))
    t.append("HP " + "█" * filled + "░" * (bar_w - filled) + "\n",
             style=color)
    # Spell slots
    if char.spell_slots:
        slot_repr = " ".join(
            f"L{lvl}:" + "●" * n
            for lvl, n in sorted(char.spell_slots.items())
        )
        t.append(f"法位 {slot_repr}\n", style="cyan")
    if getattr(char, "concentrating_on", None):
        t.append(f"集中: {char.concentrating_on}\n", style="magenta")
    return t
