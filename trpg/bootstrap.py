"""Assemble and launch the graphical TRPG application."""
import os
import sys

from .scenarios.dungeon import (
    build_world_state, build_npc_agents, build_player_agents,
)
from .llm.gm_agent import GMAgent
from .llm.tag_agent import TagAgent
from .llm import config as llm_config
from .rl.combat_model import load_combat_policy
from .game import GameSession

os.environ.setdefault("GGML_FLASH_ATTENTION", "0")

GM_THINK = False
GM_SHOW_THINKING = False
GM_OPTIONS = {"temperature": 0.8, "num_predict": 2048, "repeat_penalty": 1.15}

TAG_OPTIONS = {"temperature": 0.2, "num_predict": 200}

def init_game() -> dict:
    """Resolve the backend, build + start a GameSession.

    Returns {"session": GameSession, "world_state": WorldState}.
    Raises llm_config.BackendConfigError when no backend is configured.
    """
    cfg = llm_config.resolve()
    url, model, bk, api_key = cfg["base_url"], cfg["model"], cfg["backend"], cfg["api_key"]
    world_state = build_world_state()
    session = GameSession(
        world_state=world_state,
        gm=GMAgent(model=model, world_state=world_state,
                   think=GM_THINK, show_thinking=GM_SHOW_THINKING,
                   options=GM_OPTIONS, base_url=url, backend=bk, api_key=api_key),
        tag_agent=TagAgent(model=model, world_state=world_state,
                           base_url=url, backend=bk,
                           options=TAG_OPTIONS, api_key=api_key),
        player_agents=build_player_agents(
            world_state, model, url, bk, api_key=api_key,
        ),
        npc_agents=build_npc_agents(world_state, model, url, bk, api_key=api_key),
        default_combat_policy=load_combat_policy(),   # allies+monsters ← general model
    )
    session.start()
    return {"session": session, "world_state": world_state}


def _fatal(message: str) -> None:
    """Show a startup failure before the main application window exists."""
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("無法啟動", message)
    root.destroy()


def _run_game() -> None:
    """Assemble the game and run its Tkinter event loop."""
    from .gui import GameGUI

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        state = init_game()
    except llm_config.BackendConfigError as exc:
        _fatal(str(exc))
        return
    except SystemExit:
        _fatal("後端連線失敗。請先執行 start_llama_server.ps1 選擇模型／啟動伺服器。")
        return
    GameGUI(state).mainloop()


def main() -> None:
    """Start the LLM backend, run the game, then stop the backend."""
    from start_llama_server import ServerStartError, start_server, stop_server

    try:
        server_process = start_server()
    except ServerStartError as exc:
        _fatal(str(exc))
        return

    try:
        _run_game()
    finally:
        stop_server(server_process)


if __name__ == "__main__":
    main()
