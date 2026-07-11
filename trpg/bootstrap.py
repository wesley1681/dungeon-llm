"""Single construction point for a ready-to-run GameSession.

Every front-end (web.py, desktop.py) assembles the game identically: resolve the
LLM backend (llm_config.resolve — i.e. whatever start_llama_server.ps1 wrote),
wire the GM / tag / player / NPC agents, attach the trained general combat model
as the ally+monster policy, and start the session thread. Keeping it here means a
change to HOW the game is built (a new PC, a different combat policy, another
agent) lands in ONE place instead of once per front-end.

The reasoning-heavy constants (GM_THINK, *_OPTIONS, …) still live in cli.py — the
one config home for those — and are imported from there.
"""
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
from .cli import (
    check_ollama,
    GM_THINK, GM_SHOW_THINKING, GM_OPTIONS,
    TAG_OPTIONS,
    THOR_THINK, THOR_SHOW_THINKING, THOR_OPTIONS,
)


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
