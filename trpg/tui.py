import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Input
from textual.containers import Horizontal
from textual import work

from .scenarios.dungeon import build_world_state, OPENING_SCENE, THOR_PERSONALITY
from .llm.gm_agent import GMAgent
from .llm.player_agent import PlayerAgent
from .llm.tag_parser import parse_and_resolve
from .main import (
    check_ollama, MODEL,
    GM_THINK, GM_SHOW_THINKING, GM_OPTIONS,
    THOR_THINK, THOR_SHOW_THINKING, THOR_OPTIONS,
)


class LineBuffer:
    """Buffer streaming chunks and flush complete lines to a RichLog via call_from_thread."""

    def __init__(self, log: RichLog, app: App, style: str = ""):
        self.log = log
        self.app = app
        self.style = style
        self._buf = ""

    def push(self, chunk: str) -> None:
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                text = f"[{self.style}]{line}[/{self.style}]" if self.style else line
                self.app.call_from_thread(self.log.write, text)

    def flush(self) -> None:
        if self._buf.strip():
            text = f"[{self.style}]{self._buf}[/{self.style}]" if self.style else self._buf
            self.app.call_from_thread(self.log.write, text)
        self._buf = ""


class TRPGApp(App):
    CSS = """
    Screen { layout: vertical; }

    #gm-panel {
        height: 45%;
        border: tall cyan;
    }

    #bottom-row {
        height: 45%;
    }

    #thor-panel {
        width: 1fr;
        border: tall yellow;
    }

    #aria-panel {
        width: 1fr;
        border: tall green;
    }

    Input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [("ctrl+q", "quit", "退出")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="gm-panel", markup=True, wrap=True, highlight=False)
        with Horizontal(id="bottom-row"):
            yield RichLog(id="thor-panel", markup=True, wrap=True, highlight=False)
            yield RichLog(id="aria-panel", markup=True, wrap=True, highlight=False)
        yield Input(
            placeholder="輸入你（艾里亞）的行動，按 Enter 確認… (quit 退出 / status 查看狀態)",
            id="player-input",
            disabled=True,
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "⚔  地下城探索：失竊的護符"

        self.gm_log = self.query_one("#gm-panel", RichLog)
        self.thor_log = self.query_one("#thor-panel", RichLog)
        self.aria_log = self.query_one("#aria-panel", RichLog)

        self.gm_log.border_title = "GM"
        self.thor_log.border_title = "索爾（AI 玩家）"
        self.aria_log.border_title = "你（艾里亞）"

        self.world_state = build_world_state()
        self.gm = GMAgent(
            model=MODEL, world_state=self.world_state,
            think=GM_THINK, show_thinking=GM_SHOW_THINKING,
            options=GM_OPTIONS,
        )
        self.thor_agent = PlayerAgent(
            model=MODEL,
            character=self.world_state.characters["thor"],
            personality=THOR_PERSONALITY,
            think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
            options=THOR_OPTIONS,
        )

        self._input_event = threading.Event()
        self._input_value = ""
        self.player_actions = []

        self.aria_log.write(f"[bold]{OPENING_SCENE}[/bold]")
        self.run_game_loop()

    # ── Input handling ────────────────────────────────────────────────────

    def _enable_input(self) -> None:
        inp = self.query_one("#player-input", Input)
        inp.disabled = False
        inp.focus()

    def _disable_input(self) -> None:
        self.query_one("#player-input", Input).disabled = True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        if not val:
            return
        self.query_one("#player-input").value = ""
        self._disable_input()
        self._input_value = val
        self._input_event.set()

    def _wait_for_input(self) -> str:
        self._input_event.clear()
        self.call_from_thread(self._enable_input)
        self._input_event.wait()
        return self._input_value

    # ── Game loop (runs in background thread) ─────────────────────────────

    @work(thread=True)
    def run_game_loop(self) -> None:
        while True:
            # 1. GM generates ─────────────────────────────────────────────
            self.call_from_thread(self.gm_log.write, "\n[bold cyan]【GM】[/bold cyan]")

            gm_buf = LineBuffer(self.gm_log, self)
            think_buf = LineBuffer(self.gm_log, self, style="dim italic")
            in_thinking = False

            def gm_on_chunk(chunk: str, thinking: bool = False) -> None:
                nonlocal in_thinking
                if thinking:
                    if not in_thinking:
                        in_thinking = True
                        self.call_from_thread(self.gm_log.write, "[dim]── 思考中 ──[/dim]")
                    think_buf.push(chunk)
                else:
                    if in_thinking:
                        in_thinking = False
                        think_buf.flush()
                        self.call_from_thread(self.gm_log.write, "[dim]── 回答 ──[/dim]")
                    gm_buf.push(chunk)

            gm_raw = self.gm.generate(self.player_actions, on_chunk=gm_on_chunk)
            gm_buf.flush()
            think_buf.flush()

            # 2. Tag resolution ───────────────────────────────────────────
            gm_text, tag_results = parse_and_resolve(gm_raw, self.world_state)
            if tag_results:
                result_str = "　".join(tag_results)
                self.call_from_thread(
                    self.gm_log.write,
                    f"[dim]（規則結算：{result_str}）[/dim]",
                )
                self.gm.notify_results(tag_results)
            self.world_state.event_log.append(f"GM: {gm_text}")

            # 3. Death check ──────────────────────────────────────────────
            for cid, char in self.world_state.characters.items():
                if not char.is_npc and not char.is_alive():
                    self.call_from_thread(
                        self.gm_log.write,
                        f"[bold red]💀 {char.name} 倒下了！遊戲結束。[/bold red]",
                    )
                    return

            # 4. Thor sees GM narration ───────────────────────────────────
            self.call_from_thread(self.thor_log.write, f"\n[cyan]【GM】[/cyan] {gm_text}")
            self.call_from_thread(self.thor_log.write, "[bold yellow]【索爾】[/bold yellow]")

            thor_buf = LineBuffer(self.thor_log, self)

            def thor_on_chunk(chunk: str, thinking: bool = False) -> None:
                if not thinking:
                    thor_buf.push(chunk)

            thor_response = self.thor_agent.generate(gm_text, on_chunk=thor_on_chunk)
            thor_buf.flush()
            self.world_state.event_log.append(f"索爾：{thor_response}")

            # 5. Aria sees GM narration + Thor response ───────────────────
            self.call_from_thread(self.aria_log.write, f"\n[cyan]【GM】[/cyan] {gm_text}")
            self.call_from_thread(
                self.aria_log.write,
                f"[yellow]【索爾】[/yellow] {thor_response}",
            )

            aria = self.world_state.characters["aria"]
            status_str = "、".join(aria.status_effects) if aria.status_effects else "無"
            self.call_from_thread(
                self.aria_log.write,
                f"\n[bold green]【艾里亞｜HP {aria.hp}/{aria.max_hp} AC {aria.ac}｜狀態：{status_str}】[/bold green]",
            )

            # 6. Wait for player input ────────────────────────────────────
            human_input = self._wait_for_input()

            if human_input.lower() == "quit":
                self.call_from_thread(self.exit)
                return

            if human_input.lower() == "status":
                inv = "、".join(aria.inventory) if aria.inventory else "無"
                self.call_from_thread(
                    self.aria_log.write,
                    f"[bold]HP {aria.hp}/{aria.max_hp}  AC {aria.ac}[/bold]\n道具：{inv}\n狀態：{status_str}",
                )
                self.player_actions = []
                continue

            self.call_from_thread(
                self.aria_log.write,
                f"[bold green]{human_input}[/bold green]",
            )
            self.world_state.event_log.append(f"艾里亞：{human_input}")

            self.player_actions = [
                f"索爾：{thor_response}",
                f"艾里亞：{human_input}",
            ]


def main() -> None:
    check_ollama(MODEL)
    TRPGApp().run()


if __name__ == "__main__":
    main()
