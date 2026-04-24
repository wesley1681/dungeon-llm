import sys
import queue
import threading

sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

from .scenarios.dungeon import build_world_state, OPENING_SCENE, THOR_PERSONALITY
from .llm.gm_agent import GMAgent
from .llm.player_agent import PlayerAgent
from .llm.arbiter import ArbiterAgent
from .llm.tag_parser import parse_and_resolve, parse_travel_only
from .engine.combat import execute_action, format_result
from .main import (
    check_ollama, MODEL,
    GM_THINK, GM_SHOW_THINKING, GM_OPTIONS,
    THOR_THINK, THOR_SHOW_THINKING, THOR_OPTIONS,
    DEBUG_ARBITER,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stream_agent(agent_fn, *args, **kwargs):
    q: queue.Queue = queue.Queue()

    def on_chunk(chunk: str, thinking: bool = False) -> None:
        q.put((chunk, thinking))

    def run() -> None:
        try:
            agent_fn(*args, on_chunk=on_chunk, **kwargs)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def _alive_enemies(world_state) -> dict[str, str]:
    if world_state.dungeon_map:
        room_ids = set(world_state.dungeon_map.current_room.enemy_ids)
        return {cid: c.name for cid, c in world_state.characters.items()
                if c.is_npc and c.is_alive() and cid in room_ids}
    return {cid: c.name for cid, c in world_state.characters.items()
            if c.is_npc and c.is_alive()}


def _alive_pcs(world_state) -> dict[str, str]:
    return {cid: c.name for cid, c in world_state.characters.items()
            if not c.is_npc and c.is_alive()}


def _fmt_gm(thinking: str, response: str) -> str:
    if thinking and GM_SHOW_THINKING:
        return f"*[思考]*\n{thinking}\n\n---\n\n{response}"
    return response


def _aria_status(world_state) -> str:
    aria = world_state.characters["aria"]
    status   = "、".join(aria.status_effects) if aria.status_effects else "無"
    weapons  = "、".join(w.name for w in aria.weapons) or "無"
    usable   = [
        f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
        for c in aria.consumables if c.effect_type != "ammo" and c.quantity > 0
    ]
    items = "、".join(usable) or "無"
    return (
        f"*HP {aria.hp}/{aria.max_hp}  AC {aria.ac}  狀態：{status}*\n"
        f"⚔ 武器：{weapons}\n"
        f"🎒 道具：{items}"
    )


_CUTOFF_TAGS = {"[INITIATIVE:"}


def _stream_gm(gm_msgs: list, state: dict, player_actions: list):
    """
    Stream GM output, yield (gm_msgs, state) after each chunk.
    Cuts off immediately when a combat-triggering tag is detected.
    Returns the final gm_response string.
    """
    gm = state["gm"]
    gm_response = gm_thinking = ""
    cut = False

    for chunk, thinking in _stream_agent(gm.generate, player_actions):
        if thinking:
            gm_thinking += chunk
        else:
            gm_response += chunk

            # Only cut off tags that appear AFTER the "---" separator
            # (the 機制: planning line before --- should be ignored)
            # Wait until the closing ] arrives so parse_and_resolve sees a full tag.
            if "---" in gm_response:
                narrative_part = gm_response.split("---", 1)[1]
                for tag in _CUTOFF_TAGS:
                    if tag in narrative_part:
                        idx = gm_response.index(tag, gm_response.index("---"))
                        end = gm_response.find("]", idx)
                        if end != -1:
                            gm_response = gm_response[: end + 1]
                            cut = True
                        break

        gm_msgs[-1]["content"] = _fmt_gm(gm_thinking, gm_response)
        yield gm_msgs, state, gm_response

        if cut:
            break

    return  # caller uses the last yielded gm_response


def _init_game() -> dict:
    world_state = build_world_state()
    return {
        "world_state": world_state,
        "gm": GMAgent(
            model=MODEL, world_state=world_state,
            think=GM_THINK, show_thinking=GM_SHOW_THINKING,
            options=GM_OPTIONS,
        ),
        "thor_agent": PlayerAgent(
            model=MODEL,
            character=world_state.characters["thor"],
            personality=THOR_PERSONALITY,
            think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
            options=THOR_OPTIONS,
        ),
        "arbiter": ArbiterAgent(model=MODEL),
        "thor_response": "",
    }


# ── Combat turn processor ─────────────────────────────────────────────────────

def _combat_turns(gm_msgs, thor_msgs, aria_msgs, state):
    """
    Auto-process NPC and Thor turns in order.
    Yields (gm_msgs, thor_msgs, aria_msgs, state) after each action.
    Returns (stops yielding) when it's Aria's turn or combat ends.
    """
    world_state = state["world_state"]
    gm          = state["gm"]
    thor_agent  = state["thor_agent"]
    arbiter     = state["arbiter"]
    combat      = world_state.combat
    order_len   = len(combat.initiative_order)

    while combat.active:
        enemies = _alive_enemies(world_state)
        pcs     = _alive_pcs(world_state)

        if not enemies:
            combat.active = False
            # Mark room cleared and show loot
            msg = "✨ **所有敵人已倒下！戰鬥結束。**"
            if world_state.dungeon_map:
                room = world_state.dungeon_map.current_room
                room.cleared = True
                loot = room.loot_names()
                if loot:
                    msg += f"\n\n💰 **可拾取物品：{'、'.join(loot)}**\n（告訴GM你想拿什麼）"
            gm_msgs = gm_msgs + [{"role": "assistant", "content": msg}]
            yield gm_msgs, thor_msgs, aria_msgs, state
            return

        if not pcs:
            gm_msgs = gm_msgs + [{"role": "assistant",
                                   "content": "💀 **全員倒下！遊戲結束。**"}]
            yield gm_msgs, thor_msgs, aria_msgs, state
            return

        cid  = combat.initiative_order[combat.current_turn_index % order_len]
        char = world_state.characters.get(cid)

        if not char or not char.is_alive():
            combat.current_turn_index += 1
            continue

        # ── Aria's turn: stop and wait for player input ───────────────────────
        if cid == "aria":
            aria_msgs = aria_msgs + [{
                "role": "assistant",
                "content": (
                    f"**⚔ 輪到你了！**\n\n"
                    f"敵人：{'、'.join(f'{n}（{c}）' for c, n in enemies.items())}\n\n"
                    f"{_aria_status(world_state)}"
                ),
            }]
            yield gm_msgs, thor_msgs, aria_msgs, state
            return  # pause — resume when player submits

        # ── NPC turn ──────────────────────────────────────────────────────────
        if char.is_npc:
            if not pcs:
                break
            pc_str      = "、".join(f"{n}（{c}）" for c, n in pcs.items())
            weapons_str = "、".join(w.name for w in char.weapons) or "無武器"
            npc_prompt  = (
                f"現在是 {char.name}（{cid}）的回合，HP {char.hp}/{char.max_hp}。\n"
                f"{char.name} 的武器：{weapons_str}\n"
                f"可攻擊目標：{pc_str}\n"
                f"用一句話描述 {char.name} 選擇的行動（30字以內），必須使用實際武器名稱，不要描述玩家角色的反應。"
            )

            gm_msgs = gm_msgs + [
                {"role": "user",      "content": f"（{char.name} 的回合）"},
                {"role": "assistant", "content": ""},
            ]
            npc_desc = ""
            for chunk, thinking in _stream_agent(gm.combat_narrate, npc_prompt):
                if not thinking:
                    npc_desc += chunk
                    gm_msgs[-1]["content"] = npc_desc
                    yield gm_msgs, thor_msgs, aria_msgs, state

            npc_action = arbiter.parse(
                player_action=npc_desc,
                actor_id=cid, actor_name=char.name,
                available_targets=pcs,
                resources={"action": True, "bonus_action": True, "movement": 9},
                actor_char=char,
            )
            debug_str = f"\n\n`[判定器] {npc_action}`" if DEBUG_ARBITER else ""

            if npc_action.get("valid"):
                result      = execute_action(npc_action, world_state)
                result_text = format_result(npc_desc, result, char.name)
                gm_msgs[-1]["content"] += f"{debug_str}\n\n`{result_text}`"
                yield gm_msgs, thor_msgs, aria_msgs, state
                brief = f"【戰鬥結算】用一句（30字以內）繁體中文敘述此結果，直接輸出敘事，不加格式欄位：\n{result_text}"
                gm_msgs = gm_msgs + [{"role": "assistant", "content": ""}]
                for chunk, thinking in _stream_agent(gm.combat_narrate, brief):
                    if not thinking:
                        gm_msgs[-1]["content"] += chunk
                        yield gm_msgs, thor_msgs, aria_msgs, state
            else:
                gm_msgs[-1]["content"] += f"{debug_str}\n\n*無效行動：{npc_action.get('reason')}*"
                yield gm_msgs, thor_msgs, aria_msgs, state

        # ── Thor's turn ───────────────────────────────────────────────────────
        elif cid == "thor":
            thor_char = world_state.characters["thor"]
            context   = (
                f"現在是你（索爾）的回合。\n"
                f"存活的敵人：{'、'.join(f'{n}（{c}）' for c, n in enemies.items())}\n"
                f"你的 HP：{thor_char.hp}/{thor_char.max_hp}"
            )

            thor_msgs = thor_msgs + [
                {"role": "user",      "content": context},
                {"role": "assistant", "content": ""},
            ]
            thor_desc = ""
            for chunk, thinking in _stream_agent(thor_agent.generate, context):
                if not thinking:
                    thor_desc += chunk
                    thor_msgs[-1]["content"] = thor_desc
                    yield gm_msgs, thor_msgs, aria_msgs, state

            thor_action = arbiter.parse(
                player_action=thor_desc,
                actor_id="thor", actor_name=thor_char.name,
                available_targets=enemies,
                resources={"action": True, "bonus_action": True, "movement": 9},
                actor_char=thor_char,
            )
            debug_str = f"\n\n`[判定器] {thor_action}`" if DEBUG_ARBITER else ""

            if thor_action.get("valid"):
                result      = execute_action(thor_action, world_state)
                result_text = format_result(thor_desc, result, thor_char.name)
                thor_msgs[-1]["content"] += f"{debug_str}\n\n`{result_text}`"
                aria_msgs = aria_msgs + [{
                    "role": "assistant",
                    "content": f"**索爾：** {thor_desc}\n\n`{result_text}`",
                }]
                yield gm_msgs, thor_msgs, aria_msgs, state
                brief = f"【戰鬥結算】用一句（30字以內）繁體中文敘述此結果，直接輸出敘事，不加格式欄位：\n{result_text}"
                gm_msgs = gm_msgs + [{"role": "assistant", "content": ""}]
                for chunk, thinking in _stream_agent(gm.combat_narrate, brief):
                    if not thinking:
                        gm_msgs[-1]["content"] += chunk
                        yield gm_msgs, thor_msgs, aria_msgs, state
            else:
                thor_msgs[-1]["content"] += f"{debug_str}\n\n*無效行動：{thor_action.get('reason')}*"
                yield gm_msgs, thor_msgs, aria_msgs, state

        combat.current_turn_index += 1

        # New round
        if combat.current_turn_index % order_len == 0:
            combat.round_number += 1
            gm_msgs = gm_msgs + [{
                "role": "assistant",
                "content": f"---\n**第 {combat.round_number} 回合**",
            }]
            yield gm_msgs, thor_msgs, aria_msgs, state


# ── Opening (page load) ───────────────────────────────────────────────────────

def on_load():
    state       = _init_game()
    world_state = state["world_state"]
    gm          = state["gm"]

    gm_msgs   = [{"role": "user", "content": "（開場）"},
                 {"role": "assistant", "content": ""}]
    thor_msgs : list = []
    aria_msgs = [{"role": "user", "content": OPENING_SCENE}]

    for gm_msgs, state, gm_response in _stream_gm(gm_msgs, state, []):
        yield gm_msgs, thor_msgs, aria_msgs, state

    _room_before = world_state.dungeon_map.current_room_id if world_state.dungeon_map else None
    if "---" in gm_response:
        _pre, _narrative = gm_response.split("---", 1)
        tag_results = parse_travel_only(_pre, world_state)
    else:
        _narrative, tag_results = gm_response, []
    _skip = {"TRAVEL"} if world_state.dungeon_map and world_state.dungeon_map.current_room_id != _room_before else None
    gm_text, _extra = parse_and_resolve(_narrative, world_state, skip_tags=_skip)
    tag_results = tag_results + _extra
    if tag_results:
        rules = "\n".join(f"- {r}" for r in tag_results)
        gm_msgs[-1]["content"] += f"\n\n**規則結算：**\n{rules}"
        gm.notify_results(tag_results)
    world_state.event_log.append(f"GM: {gm_text}")

    # Combat started during opening?
    if world_state.combat and world_state.combat.active:
        world_state.combat.current_turn_index = 0
        for gm_msgs, thor_msgs, aria_msgs, state in _combat_turns(
                gm_msgs, thor_msgs, aria_msgs, state):
            yield gm_msgs, thor_msgs, aria_msgs, state
        return

    # Normal exploration: Thor responds, Aria sees
    thor_msgs = [{"role": "user", "content": gm_text},
                 {"role": "assistant", "content": ""}]
    thor_response = ""
    for chunk, thinking in _stream_agent(state["thor_agent"].generate, gm_text):
        if not thinking:
            thor_response += chunk
            thor_msgs[-1]["content"] = thor_response
            yield gm_msgs, thor_msgs, aria_msgs, state

    world_state.event_log.append(f"索爾：{thor_response}")
    state["thor_response"] = thor_response

    aria_msgs.append({
        "role": "assistant",
        "content": (f"**GM：** {gm_text}\n\n"
                    f"**索爾：** {thor_response}\n\n"
                    f"{_aria_status(world_state)}"),
    })
    yield gm_msgs, thor_msgs, aria_msgs, state


# ── Submit (player action) ────────────────────────────────────────────────────

def on_submit(human_input: str,
              gm_msgs: list, thor_msgs: list, aria_msgs: list,
              state: dict):
    if not human_input.strip() or state is None:
        yield gm_msgs, thor_msgs, aria_msgs, state, ""
        return

    world_state = state["world_state"]
    gm          = state["gm"]
    thor_agent  = state["thor_agent"]
    arbiter     = state["arbiter"]

    # ── COMBAT MODE ───────────────────────────────────────────────────────────
    if world_state.combat and world_state.combat.active:
        aria    = world_state.characters["aria"]
        enemies = _alive_enemies(world_state)

        aria_msgs = aria_msgs + [{"role": "user", "content": human_input}]

        aria_action = arbiter.parse(
            player_action=human_input,
            actor_id="aria", actor_name=aria.name,
            available_targets=enemies,
            resources={"action": True, "bonus_action": True, "movement": 9},
            actor_char=aria,
        )
        debug_str = f"\n\n`[判定器] {aria_action}`" if DEBUG_ARBITER else ""

        if not aria_action.get("valid"):
            aria_msgs = aria_msgs + [{
                "role": "assistant",
                "content": (f"❌ **{aria_action.get('reason')}**\n\n"
                            f"💡 {aria_action.get('suggestion')}{debug_str}"),
            }]
            yield gm_msgs, thor_msgs, aria_msgs, state, ""
            return

        result      = execute_action(aria_action, world_state)
        result_text = format_result(human_input, result, aria.name)
        aria_msgs = aria_msgs + [{
            "role": "assistant",
            "content": f"`{result_text}`{debug_str}",
        }]
        world_state.event_log.append(f"艾里亞：{human_input}")
        yield gm_msgs, thor_msgs, aria_msgs, state, ""

        brief = f"【戰鬥結算】用一句（30字以內）繁體中文敘述此結果，直接輸出敘事，不加格式欄位：\n{result_text}"
        gm_msgs = gm_msgs + [{"role": "assistant", "content": ""}]
        for chunk, thinking in _stream_agent(gm.combat_narrate, brief):
            if not thinking:
                gm_msgs[-1]["content"] += chunk
                yield gm_msgs, thor_msgs, aria_msgs, state, ""

        world_state.combat.current_turn_index += 1

        # Auto-process until next Aria turn or combat end
        for gm_msgs, thor_msgs, aria_msgs, state in _combat_turns(
                gm_msgs, thor_msgs, aria_msgs, state):
            yield gm_msgs, thor_msgs, aria_msgs, state, ""

        # Still in combat (paused for Aria's next turn) — wait for next submit
        if world_state.combat and world_state.combat.active:
            return

        # ── Combat ended — transition back to exploration ──────────────────────
        world_state.combat = None
        player_actions = ["（戰鬥結束，所有敵人已被擊敗）"]
        gm_msgs = gm_msgs + [{"role": "user",      "content": "（戰鬥結束）"},
                              {"role": "assistant", "content": ""}]
        yield gm_msgs, thor_msgs, aria_msgs, state, ""

        for gm_msgs, state, gm_response in _stream_gm(gm_msgs, state, player_actions):
            yield gm_msgs, thor_msgs, aria_msgs, state, ""

        if "---" in gm_response:
            _pre, _narrative = gm_response.split("---", 1)
            tag_results = parse_travel_only(_pre, world_state)
        else:
            _narrative, tag_results = gm_response, []
        gm_text, _extra = parse_and_resolve(_narrative, world_state)
        tag_results = tag_results + _extra
        if tag_results:
            gm_msgs[-1]["content"] += "\n\n**規則結算：**\n" + "\n".join(f"- {r}" for r in tag_results)
            gm.notify_results(tag_results)
        world_state.event_log.append(f"GM: {gm_text}")
        yield gm_msgs, thor_msgs, aria_msgs, state, ""

        # New combat triggered in aftermath narration
        if world_state.combat and world_state.combat.active:
            world_state.combat.current_turn_index = 0
            for gm_msgs, thor_msgs, aria_msgs, state in _combat_turns(
                    gm_msgs, thor_msgs, aria_msgs, state):
                yield gm_msgs, thor_msgs, aria_msgs, state, ""
            return

        thor_msgs = thor_msgs + [{"role": "user",      "content": gm_text},
                                  {"role": "assistant", "content": ""}]
        thor_response = ""
        for chunk, thinking in _stream_agent(thor_agent.generate, gm_text):
            if not thinking:
                thor_response += chunk
                thor_msgs[-1]["content"] = thor_response
                yield gm_msgs, thor_msgs, aria_msgs, state, ""

        world_state.event_log.append(f"索爾：{thor_response}")
        state["thor_response"] = thor_response
        aria_msgs = aria_msgs + [{
            "role": "assistant",
            "content": (f"**GM：** {gm_text}\n\n"
                        f"**索爾：** {thor_response}\n\n"
                        f"{_aria_status(world_state)}"),
        }]
        yield gm_msgs, thor_msgs, aria_msgs, state, ""
        return

    # ── EXPLORATION MODE ──────────────────────────────────────────────────────
    thor_response_prev = state.get("thor_response", "")
    aria_msgs = aria_msgs + [{"role": "user", "content": human_input}]
    world_state.event_log.append(f"艾里亞：{human_input}")

    player_actions  = [f"索爾：{thor_response_prev}", f"艾里亞：{human_input}"]
    actions_display = "\n".join(f"• {a}" for a in player_actions)
    gm_msgs = gm_msgs + [{"role": "user",      "content": actions_display},
                          {"role": "assistant", "content": ""}]
    yield gm_msgs, thor_msgs, aria_msgs, state, ""

    for gm_msgs, state, gm_response in _stream_gm(gm_msgs, state, player_actions):
        yield gm_msgs, thor_msgs, aria_msgs, state, ""

    _room_before = world_state.dungeon_map.current_room_id if world_state.dungeon_map else None
    if "---" in gm_response:
        _pre, _narrative = gm_response.split("---", 1)
        tag_results = parse_travel_only(_pre, world_state)
    else:
        _narrative, tag_results = gm_response, []
    _skip = {"TRAVEL"} if world_state.dungeon_map and world_state.dungeon_map.current_room_id != _room_before else None
    gm_text, _extra = parse_and_resolve(_narrative, world_state, skip_tags=_skip)
    tag_results = tag_results + _extra
    if tag_results:
        rules = "\n".join(f"- {r}" for r in tag_results)
        gm_msgs[-1]["content"] += f"\n\n**規則結算：**\n{rules}"
        gm.notify_results(tag_results)
    world_state.event_log.append(f"GM: {gm_text}")
    yield gm_msgs, thor_msgs, aria_msgs, state, ""

    # Death check
    for cid, char in world_state.characters.items():
        if not char.is_npc and not char.is_alive():
            gm_msgs[-1]["content"] += f"\n\n💀 **{char.name} 倒下了！遊戲結束。**"
            yield gm_msgs, thor_msgs, aria_msgs, state, ""
            return

    # Combat started?
    if world_state.combat and world_state.combat.active:
        world_state.combat.current_turn_index = 0
        for gm_msgs, thor_msgs, aria_msgs, state in _combat_turns(
                gm_msgs, thor_msgs, aria_msgs, state):
            yield gm_msgs, thor_msgs, aria_msgs, state, ""
        return

    # Normal exploration: Thor responds
    thor_msgs = thor_msgs + [{"role": "user",      "content": gm_text},
                              {"role": "assistant", "content": ""}]
    thor_response = ""
    for chunk, thinking in _stream_agent(thor_agent.generate, gm_text):
        if not thinking:
            thor_response += chunk
            thor_msgs[-1]["content"] = thor_response
            yield gm_msgs, thor_msgs, aria_msgs, state, ""

    world_state.event_log.append(f"索爾：{thor_response}")
    state["thor_response"] = thor_response

    aria_msgs = aria_msgs + [{
        "role": "assistant",
        "content": (f"**GM：** {gm_text}\n\n"
                    f"**索爾：** {thor_response}\n\n"
                    f"{_aria_status(world_state)}"),
    }]
    yield gm_msgs, thor_msgs, aria_msgs, state, ""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="⚔ 地下城探索") as demo:
        gr.Markdown("# ⚔ 地下城探索：失竊的護符")

        state = gr.State()

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### 🎲 GM")
                gm_chat = gr.Chatbot(height=550, show_label=False)
            with gr.Column():
                gr.Markdown("### 🛡️ 索爾（AI 玩家）")
                thor_chat = gr.Chatbot(height=550, show_label=False)
            with gr.Column():
                gr.Markdown("### 🗡️ 你（艾里亞）")
                aria_chat = gr.Chatbot(height=550, show_label=False)

        with gr.Row():
            input_box = gr.Textbox(
                placeholder="輸入你（艾里亞）的行動，按 Enter 確認…",
                show_label=False, scale=5,
            )
            submit_btn = gr.Button("確認", scale=1, variant="primary")

        demo.load(fn=on_load, outputs=[gm_chat, thor_chat, aria_chat, state])

        inputs  = [input_box, gm_chat, thor_chat, aria_chat, state]
        outputs = [gm_chat, thor_chat, aria_chat, state, input_box]
        submit_btn.click(on_submit, inputs=inputs, outputs=outputs)
        input_box.submit(on_submit, inputs=inputs, outputs=outputs)

    return demo


def main() -> None:
    check_ollama(MODEL)
    build_ui().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
