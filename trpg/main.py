import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import requests
from .scenarios.dungeon import build_world_state, OPENING_SCENE, THOR_PERSONALITY
from .llm.gm_agent import GMAgent
from .llm.player_agent import PlayerAgent
from .llm.arbiter import ArbiterAgent
from .llm.tag_parser import parse_and_resolve
from .engine.combat import execute_action, format_result

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4-abliterix:latest"

# ── GM 設定 ──────────────────────────────────────────
GM_THINK         = False
GM_SHOW_THINKING = False
GM_OPTIONS = {
    "temperature": 0.8,
    "num_predict": 4096,
}

# ── 除錯 ─────────────────────────────────────────────
DEBUG_ARBITER = True   # 顯示判定器 JSON 輸出

# ── AI 玩家（索爾）設定 ──────────────────────────────
THOR_THINK         = False
THOR_SHOW_THINKING = False
THOR_OPTIONS = {
    "temperature": 0.9,
    "num_predict": 150,
}


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


def _alive_enemies(world_state) -> dict[str, str]:
    return {
        cid: char.name
        for cid, char in world_state.characters.items()
        if char.is_npc and char.is_alive()
    }


def _alive_pcs(world_state) -> dict[str, str]:
    return {
        cid: char.name
        for cid, char in world_state.characters.items()
        if not char.is_npc and char.is_alive()
    }


# ── 戰鬥循環 ─────────────────────────────────────────

def run_combat(world_state, gm: GMAgent, thor_agent: PlayerAgent,
               arbiter: ArbiterAgent, model: str) -> None:
    """按先攻順序逐回合執行戰鬥，直到一方全滅。"""

    combat = world_state.combat
    print(f"\n  先攻順序：{'→'.join(world_state.characters[c].name for c in combat.initiative_order if c in world_state.characters)}")

    while True:
        # 檢查戰鬥是否結束
        if not _alive_enemies(world_state):
            print("\n✨ 所有敵人已倒下！戰鬥結束。")
            world_state.combat.active = False
            return
        if not _alive_pcs(world_state):
            print("\n💀 全員倒下！遊戲結束。")
            sys.exit(0)

        combat.round_number += 1
        print(f"\n{'─'*50}\n  第 {combat.round_number} 回合")

        for cid in combat.initiative_order:
            char = world_state.characters.get(cid)
            if not char or not char.is_alive():
                continue

            print(f"\n  【{char.name} 的回合】", end="")

            resources = {"action": True, "bonus_action": True, "movement": 9}
            result_summary = ""

            if char.is_npc:
                # NPC 回合：GM 選擇行動
                targets = _alive_pcs(world_state)
                if not targets:
                    break

                target_list = "、".join(f"{n}（{c}）" for c, n in targets.items())
                weapons_str = "、".join(w.name for w in char.weapons) or "無武器"
                npc_prompt = [
                    f"現在是 {char.name}（{cid}）的回合，HP {char.hp}/{char.max_hp}。",
                    f"{char.name} 的武器：{weapons_str}",
                    f"可攻擊目標：{target_list}",
                    "請選擇一個行動，用自然語言一句話描述，必須使用實際武器名稱，例如「地精甲揮彎刀攻擊艾里亞」。",
                ]
                print()
                npc_raw = gm.generate(npc_prompt, on_chunk=lambda c, t=False: print(c, end="", flush=True) if not t else None)
                print()

                # 用判定器解析 NPC 行動
                npc_action = arbiter.parse(
                    player_action=npc_raw,
                    actor_id=cid,
                    actor_name=char.name,
                    available_targets=targets,
                    resources=resources,
                    actor_char=char,
                )

                if DEBUG_ARBITER:
                    print(f"  [判定器] {json.dumps(npc_action, ensure_ascii=False)}")
                if npc_action.get("valid"):
                    result = execute_action(npc_action, world_state)
                    result_summary = format_result(npc_raw, result, char.name)
                    print(f"  {result_summary}")
                else:
                    print(f"  （{char.name} 無法行動：{npc_action.get('reason', '未知')}）")

            elif cid == "thor":
                # 索爾回合：AI 玩家描述行動
                context = (
                    f"現在是你（索爾）的回合。\n"
                    f"存活的敵人：{'、'.join(_alive_enemies(world_state).values())}\n"
                    f"你的 HP：{char.hp}/{char.max_hp}"
                )
                print()
                thor_desc = thor_agent.generate(context)
                print(thor_desc)

                thor_action = arbiter.parse(
                    player_action=thor_desc,
                    actor_id="thor",
                    actor_name=char.name,
                    available_targets=_alive_enemies(world_state),
                    resources=resources,
                    actor_char=char,
                )

                if DEBUG_ARBITER:
                    print(f"  [判定器] {json.dumps(thor_action, ensure_ascii=False)}")
                if thor_action.get("valid"):
                    result = execute_action(thor_action, world_state)
                    result_summary = format_result(thor_desc, result, char.name)
                    print(f"  {result_summary}")
                else:
                    print(f"  （索爾無法行動：{thor_action.get('reason')}）")

            elif cid == "aria":
                # 艾里亞回合：真人輸入
                aria = world_state.characters["aria"]
                inv = "、".join(aria.inventory) if aria.inventory else "無"
                status_str = "、".join(aria.status_effects) if aria.status_effects else "無"
                enemies = _alive_enemies(world_state)
                enemy_str = "、".join(f"{n}（{c}）" for c, n in enemies.items())

                print(f"\n  敵人：{enemy_str}")
                print(f"  【艾里亞｜HP {aria.hp}/{aria.max_hp} AC {aria.ac}｜{inv}｜狀態：{status_str}】", end="")
                human_input = input().strip()

                if not human_input:
                    print("  （跳過回合）")
                    continue
                if human_input.lower() == "quit":
                    print("\n冒險結束。再見！")
                    sys.exit(0)

                aria_action = arbiter.parse(
                    player_action=human_input,
                    actor_id="aria",
                    actor_name=aria.name,
                    available_targets=enemies,
                    resources=resources,
                    actor_char=aria,
                )

                if DEBUG_ARBITER:
                    print(f"  [判定器] {json.dumps(aria_action, ensure_ascii=False)}")
                if not aria_action.get("valid"):
                    print(f"  ❌ {aria_action.get('reason')}")
                    print(f"  💡 {aria_action.get('suggestion')}")
                    print(f"  【重新輸入】", end="")
                    human_input = input().strip()
                    if human_input:
                        aria_action = arbiter.parse(human_input, "aria", aria.name,
                                                    enemies, resources)
                        if DEBUG_ARBITER:
                            print(f"  [判定器] {json.dumps(aria_action, ensure_ascii=False)}")

                if aria_action.get("valid"):
                    result = execute_action(aria_action, world_state)
                    result_summary = format_result(human_input, result, aria.name)
                    print(f"  {result_summary}")

            # GM 簡短描述這個回合的結果
            if result_summary:
                gm_brief_prompt = [
                    f"用一句生動的繁體中文描述以下戰鬥結果（不超過50字，不要加分析或機制行）：",
                    result_summary,
                ]
                print("\n  ", end="")
                gm.generate(gm_brief_prompt,
                            on_chunk=lambda c, t=False: print(c, end="", flush=True) if not t else None)
                print()

            # 檢查戰鬥是否提前結束
            if not _alive_enemies(world_state) or not _alive_pcs(world_state):
                break


# ── 主探索循環 ────────────────────────────────────────

def run_game(model: str = MODEL) -> None:
    check_ollama(model)
    world_state = build_world_state()

    gm      = GMAgent(model=model, world_state=world_state,
                      think=GM_THINK, show_thinking=GM_SHOW_THINKING,
                      options=GM_OPTIONS)
    thor_agent = PlayerAgent(model=model,
                             character=world_state.characters["thor"],
                             personality=THOR_PERSONALITY,
                             think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
                             options=THOR_OPTIONS)
    arbiter = ArbiterAgent(model=model)

    print("\n" + "═" * 60)
    print(f"  ⚔  {world_state.scenario_name}")
    print("═" * 60)
    print(OPENING_SCENE)
    print('\n輸入你（艾里亞）的行動。輸入 "quit" 退出，"status" 查看狀態。')
    print("═" * 60 + "\n")

    player_actions: list[str] = []

    while True:
        # 1. GM 敘事
        print("\n【GM】", end="")
        gm_raw = gm.generate(player_actions)

        # 2. 只解析 --- 後的敘事段落
        narrative = gm_raw.split("---", 1)[-1] if "---" in gm_raw else gm_raw
        gm_text, tag_results = parse_and_resolve(narrative, world_state)

        # 3. 偵測到 INITIATIVE → 進入戰鬥循環
        if world_state.combat and world_state.combat.active:
            if tag_results:
                print("  （" + "　".join(tag_results) + "）")
            gm.notify_results(tag_results)
            world_state.event_log.append(f"GM: {gm_text}")
            run_combat(world_state, gm, thor_agent, arbiter, model)
            player_actions = ["（戰鬥結束）"]
            continue

        if tag_results:
            print("  （規則結算：" + "　".join(tag_results) + "）")

        world_state.event_log.append(f"GM: {gm_text}")
        if tag_results:
            gm.notify_results(tag_results)

        # 4. 死亡檢查
        for cid, char in world_state.characters.items():
            if not char.is_npc and not char.is_alive():
                print(f"\n💀 {char.name} 倒下了！遊戲結束。")
                return

        # 5. 索爾行動
        print(f"\n【{world_state.characters['thor'].name}】", end="")
        thor_response = thor_agent.generate(gm_text)
        world_state.event_log.append(f"索爾：{thor_response}")

        # 6. 玩家行動
        aria = world_state.characters["aria"]
        inv        = "、".join(aria.inventory)       if aria.inventory       else "無"
        status_str = "、".join(aria.status_effects)  if aria.status_effects  else "無"
        print(f"\n【你（艾里亞）｜HP {aria.hp}/{aria.max_hp} AC {aria.ac}｜道具：{inv}｜狀態：{status_str}】", end="")
        human_input = input().strip()

        if not human_input:
            print("（請輸入你的行動）", end="")
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
        player_actions = [f"索爾：{thor_response}", f"艾里亞：{human_input}"]


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    run_game(model)
