import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import requests
from .scenarios.dungeon import build_world_state, OPENING_SCENE, THOR_PERSONALITY
from .llm.gm_agent import GMAgent
from .llm.player_agent import PlayerAgent
from .llm.tag_parser import parse_and_resolve

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4-abliterix:latest"


def check_ollama(model: str) -> None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m["name"] for m in r.json().get("models", [])]
        if model not in installed:
            print(f"錯誤：找不到模型 '{model}'")
            print(f"已安裝：{installed}")
            sys.exit(1)
    except requests.ConnectionError:
        print("錯誤：無法連線 Ollama，請確認服務正在運行。")
        sys.exit(1)


def print_status(world_state) -> None:
    print("\n" + "─" * 50)
    for cid, char in world_state.characters.items():
        if not char.is_npc:
            bar = "█" * (char.hp * 10 // char.max_hp) + "░" * (10 - char.hp * 10 // char.max_hp)
            print(f"  {char.name}  HP [{bar}] {char.hp}/{char.max_hp}  AC {char.ac}")
    print("─" * 50)


def run_game(model: str = MODEL) -> None:
    check_ollama(model)
    world_state = build_world_state()

    gm = GMAgent(model=model, world_state=world_state)
    thor_agent = PlayerAgent(
        model=model,
        character=world_state.characters["thor"],
        personality=THOR_PERSONALITY,
    )

    print("\n" + "═" * 60)
    print(f"  ⚔  {world_state.scenario_name}")
    print("═" * 60)
    print(OPENING_SCENE)
    print('\n輸入你（艾里亞）的行動。輸入 "quit" 退出，"status" 查看狀態。')
    print("═" * 60 + "\n")

    player_actions: list[str] = []  # empty on opening turn

    while True:
        # 1. GM generates narration
        print("\n【GM】", end="")
        gm_raw = gm.generate(player_actions)

        # 2. Parse tags → update WorldState
        gm_text, tag_results = parse_and_resolve(gm_raw, world_state)
        if tag_results:
            print("  （規則結算：" + "　".join(tag_results) + "）")

        # 3. Log
        world_state.event_log.append(f"GM: {gm_text}")

        # 4. Check if any PC is dead
        for cid, char in world_state.characters.items():
            if not char.is_npc and not char.is_alive():
                print(f"\n💀 {char.name} 倒下了！遊戲結束。")
                return

        # 5. AI player (Thor) acts
        thor_response = thor_agent.generate(gm_text)
        print(f"\n【{world_state.characters['thor'].name}】{thor_response}")
        world_state.event_log.append(f"索爾：{thor_response}")

        # 6. Human player acts
        print(f"\n【你（艾里亞）】", end="")
        human_input = input().strip()

        if not human_input:
            continue
        if human_input.lower() == "quit":
            print("\n冒險結束。再見！")
            break
        if human_input.lower() == "status":
            print_status(world_state)
            player_actions = []
            continue

        world_state.event_log.append(f"艾里亞：{human_input}")

        # 7. Collect actions for next GM turn
        player_actions = [
            f"索爾：{thor_response}",
            f"艾里亞：{human_input}",
        ]


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    run_game(model)
