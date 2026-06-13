"""Terminal entry point — event consumer only, no game logic."""
import os
import sys
import time
import pathlib

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("GGML_FLASH_ATTENTION", "0")

import requests
from .scenarios.dungeon import (
    build_world_state, build_npc_agents, OPENING_SCENE,
    THOR_PERSONALITY, THOR_TACTICS_GENERAL, THOR_TACTICS_COMBAT,
)
from .llm.gm_agent import GMAgent
from .llm.tag_agent import TagAgent
from .llm.player_agent import PlayerAgent
from .game import (
    GameSession,
    TagResult, StreamChunk, ActionResult,
    RoundStart, CombatStart, CombatEnd,
    ExplorationPrompt, CombatPrompt,
    ConversationPrompt, StatusMessage, GameOver,
)

OLLAMA_URL   = "http://localhost:11434"
LLAMACPP_URL = "http://localhost:11435"
BACKEND      = "ollama"
MODEL        = "qwen3.6-prism"
LLAMACPP_MODEL = "Qwen3.6-27B-TQ3_4S"

def _backend_url():  return LLAMACPP_URL if BACKEND == "llamacpp" else OLLAMA_URL
def _active_model(): return LLAMACPP_MODEL if BACKEND == "llamacpp" else MODEL

GM_THINK         = False
GM_SHOW_THINKING = False
GM_OPTIONS = {"temperature": 0.8, "num_predict": 2048, "repeat_penalty": 1.15}

TAG_OPTIONS = {"temperature": 0.2, "num_predict": 200}

THOR_THINK         = False
THOR_SHOW_THINKING = False
THOR_OPTIONS = {"temperature": 0.9, "num_predict": 150}

# When True, sub-action JSON gets echoed under each ActionResult line.
DEBUG_COMBAT_ACTION = True

CLAUDE_TEST     = False
CLAUDE_RESPONSE = pathlib.Path(__file__).parent.parent / "claude_response.txt"


def _claude_input() -> str:
    print("<<<CLAUDE_TURN>>>", flush=True)
    while not CLAUDE_RESPONSE.exists():
        time.sleep(0.3)
    text = CLAUDE_RESPONSE.read_text(encoding="utf-8").strip()
    CLAUDE_RESPONSE.unlink()
    return text


def _read_player_input() -> str:
    return _claude_input() if CLAUDE_TEST else input().strip()


def check_ollama(model: str) -> None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m["name"] for m in r.json().get("models", [])]
        if model not in installed:
            print(f"錯誤：找不到模型 '{model}'\n已安裝：{installed}")
            sys.exit(1)
    except requests.ConnectionError:
        print("錯誤：無法連線 Ollama，請確認服務正在運行。")
        sys.exit(1)


def print_status(world_state) -> None:
    print("\n" + "─" * 50)
    for _, char in world_state.characters.items():
        if not char.is_npc:
            bar = "█" * (char.hp * 10 // char.max_hp) + "░" * (10 - char.hp * 10 // char.max_hp)
            print(f"  {char.name}  HP [{bar}] {char.hp}/{char.max_hp}  AC {char.ac}")
    print("─" * 50)


def run_game() -> None:
    url   = _backend_url()
    model = _active_model()
    bk    = BACKEND

    if bk == "ollama":
        check_ollama(model)

    world_state = build_world_state()

    session = GameSession(
        world_state = world_state,
        gm          = GMAgent(model=model, world_state=world_state,
                              think=GM_THINK, show_thinking=GM_SHOW_THINKING,
                              options=GM_OPTIONS, base_url=url, backend=bk),
        tag_agent   = TagAgent(model=model, world_state=world_state,
                               base_url=url, backend=bk, options=TAG_OPTIONS),
        thor_agent  = PlayerAgent(model=model,
                                  char_id="thor",
                                  character=world_state.characters["thor"],
                                  personality=THOR_PERSONALITY,
                                  tactics=THOR_TACTICS_GENERAL,
                                  combat_tactics=THOR_TACTICS_COMBAT,
                                  world_state=world_state,
                                  think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
                                  options=THOR_OPTIONS, base_url=url, backend=bk),
        npc_agents  = build_npc_agents(world_state, model, url, bk),
    )

    print("\n" + "═" * 60)
    print(f"  ⚔  {world_state.scenario_name}")
    print("═" * 60)
    print(OPENING_SCENE)
    print('\n輸入你（凱恩）的行動。輸入 "quit" 退出，"status" 查看狀態。')
    print("═" * 60 + "\n")

    session.start()

    # ── Event consumer loop ───────────────────────────────────────────────────
    _seen: dict[str, bool] = {}   # tracks "first chunk" per source per turn

    while True:
        event = session.next_event()
        if event is None:
            continue

        if isinstance(event, TagResult):
            _seen.clear()
            print("\n【標籤】", end="")
            if event.ok:
                print("  （" + "　".join(event.ok) + "）")
            if event.errors:
                print("  ⚠ 標籤錯誤（已忽略）：" + "　".join(event.errors))
            if not event.ok and not event.errors:
                print("  （無）")

        elif isinstance(event, StreamChunk):
            src = event.source
            if src == "gm":
                if not _seen.get("gm"):
                    print("\n【GM】", end="")
                    _seen["gm"] = True
                if not event.thinking or GM_SHOW_THINKING:
                    print(event.text, end="", flush=True)
            elif src == "thor":
                if not _seen.get("thor"):
                    thor_name = world_state.characters["thor"].name
                    print(f"\n【{thor_name}】", end="")
                    _seen["thor"] = True
                if not event.thinking:
                    print(event.text, end="", flush=True)
            elif src == "npc":
                if not _seen.get(f"npc_{event.actor}"):
                    print(f"\n  【{event.actor} 的回合】\n  ", end="")
                    _seen[f"npc_{event.actor}"] = True
                print(event.text, end="", flush=True)
            elif src == "pc_combat":
                slot_key = f"pc_combat_{event.actor}"
                if not _seen.get(slot_key):
                    print(f"\n  【{event.actor} 的回合】\n  ", end="")
                    _seen[slot_key] = True
                print(event.text, end="", flush=True)
            elif src == "narrate":
                if not _seen.get("narrate_current"):
                    print("\n  ", end="")
                    _seen["narrate_current"] = True
                print(event.text, end="", flush=True)
            elif src == "npc_talk":
                if not _seen.get("npc_talk"):
                    print(f"\n【{event.actor}】", end="")
                    _seen["npc_talk"] = True
                print(event.text, end="", flush=True)
            # "tag" source: not streamed to terminal (fast + low-info)

        elif isinstance(event, ActionResult):
            _seen.pop("narrate_current", None)  # reset narrate slot for next action
            print(f"\n  {event.summary}")
            if DEBUG_COMBAT_ACTION and event.debug:
                print(f"  [動作] {event.debug}")

        elif isinstance(event, CombatStart):
            print(f"\n  先攻順序：{'→'.join(event.order)}")

        elif isinstance(event, RoundStart):
            print(f"\n{'─'*50}\n  第 {event.number} 回合")
            _seen.pop("narrate_current", None)

        elif isinstance(event, CombatEnd):
            print("\n✨ 所有敵人已倒下！戰鬥結束。")
            if event.loot:
                print(f"  💰 可拾取：{'、'.join(event.loot)}")

        elif isinstance(event, ExplorationPrompt):
            print()  # newline after GM stream
            aria       = event.aria
            inv        = "、".join(aria.inventory)      if aria.inventory      else "無"
            status_str = "、".join(fx.name for fx in aria.status_effects) if aria.status_effects else "無"
            print(f"\n【凱恩｜HP {aria.hp}/{aria.max_hp} AC {aria.ac}"
                  f"｜道具：{inv}｜狀態：{status_str}】", end="")
            while True:
                human_input = _read_player_input()
                if not human_input:
                    continue
                if human_input.lower() == "status":
                    print_status(world_state)
                    print(f"\n【凱恩｜HP {aria.hp}/{aria.max_hp}】", end="")
                    continue
                break
            session.submit_player_input(human_input)
            _seen.clear()

        elif isinstance(event, CombatPrompt):
            aria       = event.aria
            inv        = "、".join(aria.inventory)      if aria.inventory      else "無"
            status_str = "、".join(fx.name for fx in aria.status_effects) if aria.status_effects else "無"
            if event.info_text:
                print(f"\n{event.info_text}")
            else:
                enemy_str = "、".join(f"{n}（{c}）" for c, n in event.enemies.items())
                print(f"\n  敵人：{enemy_str}")
            print(f"  【凱恩｜HP {aria.hp}/{aria.max_hp} AC {aria.ac}"
                  f"｜{inv}｜狀態：{status_str}】", end="")
            human_input = _read_player_input()
            session.submit_player_input(human_input)

        elif isinstance(event, ConversationPrompt):
            aria   = event.aria
            prompt = (f"\n【凱恩｜HP {aria.hp}/{aria.max_hp}】"
                      f"【{event.npc_name} 態度：{event.attitude_label}】"
                      f"（輸入「離開」結束對話）")
            print(prompt, end="")
            while True:
                human_input = _read_player_input()
                if not human_input:
                    continue
                if human_input.lower() == "status":
                    print_status(world_state)
                    print(prompt, end="")
                    continue
                break
            session.submit_player_input(human_input)
            _seen.clear()

        elif isinstance(event, StatusMessage):
            print(f"\n  {event.text}")

        elif isinstance(event, GameOver):
            print(f"\n{event.reason}")
            session.stop()
            break


if __name__ == "__main__":
    run_game()
