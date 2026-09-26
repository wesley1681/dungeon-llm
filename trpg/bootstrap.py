"""Build the ready-to-run GameSession used by the desktop application."""
import os
import sys

import requests

from .scenarios.dungeon import (
    build_world_state, build_npc_agents,
    THOR_PERSONALITY, THOR_TACTICS_GENERAL, THOR_TACTICS_COMBAT,
)
from .llm.gm_agent import GMAgent
from .llm.tag_agent import TagAgent
from .llm.player_agent import PlayerAgent
from .llm import config as llm_config
from .rl.combat_model import load_combat_policy
from .game import GameSession

os.environ.setdefault("GGML_FLASH_ATTENTION", "0")

GM_THINK = False
GM_SHOW_THINKING = False
GM_OPTIONS = {"temperature": 0.8, "num_predict": 2048, "repeat_penalty": 1.15}

TAG_OPTIONS = {"temperature": 0.2, "num_predict": 200}

THOR_THINK = False
THOR_SHOW_THINKING = False
THOR_OPTIONS = {"temperature": 0.9, "num_predict": 150}


def check_ollama(model: str, base_url: str) -> None:
    """Stop startup with a useful message if Ollama is unavailable."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        installed = [item["name"] for item in response.json().get("models", [])]
        if model not in installed and f"{model}:latest" not in installed:
            print(f"錯誤：找不到模型 '{model}'\n目前已安裝：{installed}")
            sys.exit(1)
    except requests.ConnectionError:
        print("錯誤：無法連線 Ollama，請確認服務正在運行。")
        sys.exit(1)


def init_game() -> dict:
    """Resolve the backend, build + start a GameSession.

    Returns {"session": GameSession, "world_state": WorldState}.
    Raises llm_config.BackendConfigError when no backend is configured, and may
    sys.exit via check_ollama when an Ollama model is missing/unreachable.
    """
    cfg = llm_config.resolve()
    url, model, bk, api_key = cfg["base_url"], cfg["model"], cfg["backend"], cfg["api_key"]
    if bk == "ollama":
        check_ollama(model, url)

    world_state = build_world_state()
    session = GameSession(
        world_state=world_state,
        gm=GMAgent(model=model, world_state=world_state,
                   think=GM_THINK, show_thinking=GM_SHOW_THINKING,
                   options=GM_OPTIONS, base_url=url, backend=bk, api_key=api_key),
        tag_agent=TagAgent(model=model, world_state=world_state,
                           base_url=url, backend=bk,
                           options=TAG_OPTIONS, api_key=api_key),
        thor_agent=PlayerAgent(model=model, char_id="thor",
                               character=world_state.characters["thor"],
                               personality=THOR_PERSONALITY,
                               tactics=THOR_TACTICS_GENERAL,
                               combat_tactics=THOR_TACTICS_COMBAT,
                               world_state=world_state,
                               think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
                               options=THOR_OPTIONS, base_url=url, backend=bk, api_key=api_key),
        npc_agents=build_npc_agents(world_state, model, url, bk, api_key=api_key),
        default_combat_policy=load_combat_policy(),   # allies+monsters ← general model
    )
    session.start()
    return {"session": session, "world_state": world_state}
