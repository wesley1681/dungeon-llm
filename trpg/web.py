"""Gradio web entry point — event consumer only, no game logic."""
import sys
import queue as _queue
import threading

sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

from .scenarios.dungeon import OPENING_SCENE
from .game import (
    TagResult, StreamChunk, ActionResult,
    RoundStart, CombatStart, CombatEnd,
    ExplorationPrompt, CombatPrompt,
    ConversationPrompt, StatusMessage, GameOver, QuestComplete,
)
from .engine.quests import objective_progress_str
from .llm import config as llm_config
from . import web_combat
from .cli import check_ollama, GM_SHOW_THINKING, DEBUG_COMBAT_ACTION


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kaine_status(world_state) -> str:
    kaine     = world_state.characters["kaine"]
    status   = "、".join(fx.name for fx in kaine.status_effects) if kaine.status_effects else "無"
    weapons  = "、".join(w.name for w in kaine.weapons) or "無"
    usable   = [
        f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
        for c in kaine.consumables if c.effect_type != "ammo" and c.quantity > 0
    ]
    items = "、".join(usable) or "無"
    quest_parts = []
    for q in world_state.quests.values():
        if q.status == "active":
            prog = objective_progress_str(q, world_state)
            quest_parts.append(f"📋 {q.title}（{prog}）" if prog else f"📋 {q.title}")
        elif q.status == "completed":
            giver = world_state.characters.get(q.giver_id)
            gname = giver.name if giver else q.giver_id
            quest_parts.append(f"✅ {q.title}（回去找 {gname}）")
    quest_line = ("\n" + "　".join(quest_parts)) if quest_parts else ""
    return (
        f"*HP {kaine.hp}/{kaine.max_hp}  AC {kaine.ac}  狀態：{status}*\n"
        f"⚔ 武器：{weapons}\n"
        f"🎒 道具：{items}"
        f"{quest_line}"
    )


def _init_game() -> dict:
    # Single construction point shared with the desktop front-end (bootstrap.py).
    from .bootstrap import init_game
    try:
        return init_game()
    except llm_config.BackendConfigError as e:
        raise RuntimeError(str(e)) from e


# ── Event → Gradio renderer ───────────────────────────────────────────────────

def _consume_until_prompt(state, gm_msgs, thor_msgs, kaine_msgs):
    """Consume GameSession events, yield Gradio updates, stop at player prompt.

    Yields (gm_msgs, thor_msgs, kaine_msgs, state, "").
    """
    session     = state["session"]
    world_state = state["world_state"]

    # Per-turn tracking
    tag_lines:   list[str] = []
    gm_text      = ""
    thor_text    = ""
    last_source  = None  # for opening new message slots

    def _ensure_gm_slot(src):
        nonlocal last_source
        if last_source != src:
            gm_msgs.append({"role": "assistant", "content": ""})
            last_source = src

    def _ensure_thor_slot(src):
        nonlocal last_source
        if last_source != src:
            thor_msgs.append({"role": "assistant", "content": ""})
            last_source = src

    while True:
        event = session.next_event(timeout=120)
        if event is None:
            break

        # ── TagResult ─────────────────────────────────────────────────────────
        if isinstance(event, TagResult):
            tag_lines = [f"- {r}" for r in event.ok] + [f"- ⚠ {e}" for e in event.errors]
            if tag_lines:
                gm_msgs.append({"role": "assistant",
                                "content": "**【機制結算】**\n" + "\n".join(tag_lines)})
                last_source = "tag"
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── StreamChunk ───────────────────────────────────────────────────────
        elif isinstance(event, StreamChunk):
            src = event.source

            if src == "gm":
                if not event.thinking or GM_SHOW_THINKING:
                    if last_source not in ("gm", "tag_then_gm"):
                        if last_source == "tag":
                            # Tag block already rendered — append separator into same slot
                            gm_msgs[-1]["content"] += "\n\n---\n\n"
                            last_source = "tag_then_gm"
                        elif tag_lines:
                            tag_block = "**【機制結算】**\n" + "\n".join(tag_lines)
                            gm_msgs.append({"role": "assistant",
                                            "content": tag_block + "\n\n---\n\n"})
                            last_source = "tag_then_gm"
                        else:
                            gm_msgs.append({"role": "assistant", "content": ""})
                            last_source = "gm"
                    gm_msgs[-1]["content"] += event.text
                    gm_text += event.text
                    yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

            elif src == "thor":
                if not event.thinking:
                    _ensure_thor_slot("thor")
                    thor_msgs[-1]["content"] += event.text
                    thor_text += event.text
                    yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

            elif src == "npc":
                actor_key = f"npc_{event.actor}"
                if last_source != actor_key:
                    gm_msgs.append({"role": "user",      "content": f"（{event.actor} 的回合）"})
                    gm_msgs.append({"role": "assistant", "content": ""})
                    last_source = actor_key
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

            elif src == "pc_combat":
                # All LLM-PC combat streams route to the Thor panel for now.
                # When a 2nd LLM-controlled PC is added, this needs per-actor panels.
                if not event.thinking:
                    slot_key = f"pc_combat_{event.actor}"
                    _ensure_thor_slot(slot_key)
                    thor_msgs[-1]["content"] += event.text
                    yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

            elif src == "narrate":
                _ensure_gm_slot("narrate")
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

            elif src == "npc_talk":
                if last_source != "npc_talk":
                    gm_msgs.append({"role": "user",      "content": f"（{event.actor}）"})
                    gm_msgs.append({"role": "assistant", "content": ""})
                    last_source = "npc_talk"
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── ActionResult ──────────────────────────────────────────────────────
        elif isinstance(event, ActionResult):
            debug_str = f"\n\n`[動作] {event.debug}`" if DEBUG_COMBAT_ACTION else ""
            result_line = f"\n\n`{event.summary}`{debug_str}"
            # Append to whatever the last open slot is (npc / pc_combat message)
            if gm_msgs and last_source and last_source.startswith("npc"):
                gm_msgs[-1]["content"] += result_line
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            elif last_source and last_source.startswith("pc_combat_"):
                if thor_msgs:
                    thor_msgs[-1]["content"] += result_line
                # Also show in kaine panel (uses event.actor — not hardcoded)
                last_line = thor_msgs[-1]['content'].split('`')[0].strip() if thor_msgs else ''
                kaine_msgs.append({"role": "assistant",
                                   "content": f"**{event.actor}：**{last_line}\n\n`{event.summary}`"})
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            else:
                # 凱恩 action result
                kaine_msgs.append({"role": "assistant", "content": f"`{event.summary}`{debug_str}"})
                yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            last_source = "result"

        # ── CombatStart ───────────────────────────────────────────────────────
        elif isinstance(event, CombatStart):
            order_str = "→".join(event.order)
            gm_msgs.append({"role": "assistant",
                             "content": f"⚔ **先攻順序：{order_str}**"})
            last_source = "combat_start"
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── RoundStart ────────────────────────────────────────────────────────
        elif isinstance(event, RoundStart):
            gm_msgs.append({"role": "assistant",
                             "content": f"---\n**第 {event.number} 回合**"})
            last_source = "round"
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── CombatEnd ─────────────────────────────────────────────────────────
        elif isinstance(event, CombatEnd):
            state["combat_view"] = None      # hide the battlefield/controls
            msg = "✨ **所有敵人已倒下！戰鬥結束。**"
            if event.loot:
                msg += f"\n\n💰 **可拾取：{'、'.join(event.loot)}**\n（告訴GM你想拿什麼）"
            gm_msgs.append({"role": "assistant", "content": msg})
            last_source = "combat_end"
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── ExplorationPrompt ─────────────────────────────────────────────────
        elif isinstance(event, ExplorationPrompt):
            # Thor panel: ensure final content is set
            if thor_msgs and thor_text:
                thor_msgs[-1]["content"] = thor_text

            # 凱恩 panel: combined summary — GM + each prior PC remark + status
            remark_lines = "\n\n".join(f"**{name}：** {text}"
                                       for name, text in event.prior_remarks.items())
            summary = f"**GM：** {event.gm_text}"
            if remark_lines:
                summary += "\n\n" + remark_lines
            summary += f"\n\n{_kaine_status(world_state)}"
            kaine_msgs.append({"role": "assistant", "content": summary})
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            return  # stop — wait for player submit

        # ── CombatPrompt ──────────────────────────────────────────────────────
        elif isinstance(event, CombatPrompt):
            # Structured state for the GUI battlefield/skill/target controls
            # (the wrapper reads state["combat_view"] to show them). ctx carries
            # everything; fall back to text-only if it's absent (older emitters).
            if event.ctx is not None:
                state["combat_view"] = web_combat.combat_view_state(
                    event.kaine, event.ctx, world_state)
            if event.info_text:
                info_block = event.info_text
            else:
                enemies_str = "、".join(f"{n}（{c}）" for c, n in event.enemies.items())
                info_block = f"敵人：{enemies_str}"
            kaine_msgs.append({"role": "assistant",
                               "content": (f"**⚔ 輪到你了！**（可用下方戰場面板操作，或直接打字）\n\n"
                                           f"```\n{info_block}\n```\n\n"
                                           f"{_kaine_status(world_state)}")})
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            return  # stop — wait for player submit

        # ── ConversationPrompt ────────────────────────────────────────────────
        elif isinstance(event, ConversationPrompt):
            kaine_msgs.append({"role": "assistant",
                               "content": (f"**【與 {event.npc_name} 對話中｜態度：{event.attitude_label}】**\n\n"
                                           f"{_kaine_status(world_state)}\n\n"
                                           f"*輸入「離開」結束對話*")})
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            return  # wait for player input

        # ── StatusMessage ─────────────────────────────────────────────────────
        elif isinstance(event, StatusMessage):
            kaine_msgs.append({"role": "assistant", "content": event.text})
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── QuestComplete ─────────────────────────────────────────────────────
        elif isinstance(event, QuestComplete):
            msg = f"✅ **任務達成：{event.title}**（可回去找 {event.giver_name} 回報）"
            gm_msgs.append({"role": "assistant", "content": msg})
            kaine_msgs.append({"role": "assistant", "content": msg})
            last_source = "quest"
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""

        # ── GameOver ──────────────────────────────────────────────────────────
        elif isinstance(event, GameOver):
            gm_msgs.append({"role": "assistant",
                             "content": f"💀 **{event.reason}**"})
            yield gm_msgs[:], thor_msgs[:], kaine_msgs[:], state, ""
            return


# ── Combat GUI: battlefield image + skill/target controls ──────────────────────
# The whole thing rides the existing text-command input path: a GUI selection is
# synthesised into the same command string a player could type (web_combat.
# command_for), so no new game seam is needed. The wrapper below turns each
# streamed frame into updates for [battlefield, controls, skill_radio, target].

_BATTLE_PX = 600


def _combat_updates(state):
    """(battlefield_img, combat_controls, skill_radio, target_dropdown) updates
    for the current frame — read purely from state (world_state + combat_view)."""
    if not state:
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(), gr.update())
    ws = state["world_state"]
    active = bool(getattr(ws, "combat", None) and ws.combat.active)
    cv = state.get("combat_view")
    if not active or not ws.combat.initiative_order:
        return (gr.update(value=None, visible=False), gr.update(visible=False),
                gr.update(), gr.update())
    actor_id = cv["actor_id"] if cv else ws.combat.initiative_order[0]
    img, transform = web_combat.render_battlefield(ws, actor_id, px=_BATTLE_PX)
    state["view_transform"] = transform      # (ox, oy, scale) — for click→metre mapping
    if cv:      # the human's turn — show the action controls
        skills = [s["display_name"] for s in cv["skills"]]
        targets = list({**cv["enemies"], **cv["allies"]}.values())
        return (gr.update(value=img, visible=True),
                gr.update(visible=True),
                gr.update(choices=skills, value=(skills[0] if skills else None)),
                gr.update(choices=targets, value=(targets[0] if targets else None)))
    # an NPC's turn — battlefield only, controls hidden
    return (gr.update(value=img, visible=True), gr.update(visible=False),
            gr.update(), gr.update())


# ── Opening (page load) ───────────────────────────────────────────────────────

def on_load():
    state = _init_game()

    gm_msgs   = [{"role": "user", "content": "（開場）"}]
    thor_msgs : list = []
    kaine_msgs = [{"role": "user", "content": OPENING_SCENE}]

    for gm_msgs, thor_msgs, kaine_msgs, state, _ in _consume_until_prompt(
            state, gm_msgs, thor_msgs, kaine_msgs):
        yield (gm_msgs, thor_msgs, kaine_msgs, state) + _combat_updates(state)


# ── Submit (player action — typed OR synthesised from the combat GUI) ──────────

def on_submit(human_input: str,
              gm_msgs: list, thor_msgs: list, kaine_msgs: list,
              state: dict):
    if not human_input.strip() or state is None:
        yield (gm_msgs, thor_msgs, kaine_msgs, state, "") + _combat_updates(state)
        return

    state["combat_view"] = None      # player is acting → hide controls until next prompt
    session = state["session"]
    session.submit_player_input(human_input)
    kaine_msgs = kaine_msgs + [{"role": "user", "content": human_input}]
    yield (gm_msgs, thor_msgs, kaine_msgs, state, "") + _combat_updates(state)

    for gm_msgs, thor_msgs, kaine_msgs, state, _ in _consume_until_prompt(
            state, gm_msgs[:], thor_msgs[:], kaine_msgs[:]):
        yield (gm_msgs, thor_msgs, kaine_msgs, state, "") + _combat_updates(state)


def on_combat_act(skill_label, target_label,
                  gm_msgs, thor_msgs, kaine_msgs, state):
    """Battlefield 'act' button → synthesise a command string → normal submit."""
    cv = state.get("combat_view") if state else None
    skill = next((s for s in cv["skills"] if s["display_name"] == skill_label),
                 None) if cv else None
    if skill is None:
        yield (gm_msgs, thor_msgs, kaine_msgs, state, "") + _combat_updates(state)
        return
    name2id = {n: i for i, n in {**cv["enemies"], **cv["allies"]}.items()}
    tid = name2id.get(target_label)
    cmd = web_combat.command_for(skill, target_id=tid, cell=state.get("move_cell"))
    state["move_cell"] = None
    yield from on_submit(cmd, gm_msgs, thor_msgs, kaine_msgs, state)


def on_dodge(gm_msgs, thor_msgs, kaine_msgs, state):
    yield from on_submit("閃避", gm_msgs, thor_msgs, kaine_msgs, state)


def on_end_turn(gm_msgs, thor_msgs, kaine_msgs, state):
    yield from on_submit("結束", gm_msgs, thor_msgs, kaine_msgs, state)


def on_grid_click(state, evt: gr.SelectData):
    """Click a battlefield cell → remember it as the move/AoE destination.
    Uses the last render's view transform (auto-fit zoom) to map pixel→metre."""
    if not state:
        return state, ""
    ws = state["world_state"]
    if not (getattr(ws, "combat", None) and ws.combat.active):
        return state, ""
    ox, oy, scale = state.get("view_transform", (0.0, 0.0, _BATTLE_PX / 30.0))
    try:
        px, py = evt.index
    except Exception:
        return state, ""
    mx, my = round(px / scale + ox, 1), round(py / scale + oy, 1)
    state["move_cell"] = (mx, my)
    return state, f"已選格：({mx}, {my})m —— 選「移動」或範圍技能後按〔行動〕"


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
                gr.Markdown("### 🗡️ 你（凱恩）")
                kaine_chat = gr.Chatbot(height=550, show_label=False)

        # Combat panel — hidden until a fight starts, then shows the battlefield
        # (updates every combat action) plus your turn's skill/target controls.
        with gr.Row():
            battlefield_img = gr.Image(
                label="⚔ 戰場（藍=你、綠=隊友、紅=敵人；點格子選移動/範圍目標）",
                visible=False, interactive=False, height=560)
            with gr.Column(visible=False, scale=1) as combat_controls:
                gr.Markdown("**戰鬥操作**")
                combat_hint = gr.Markdown("")
                skill_radio = gr.Radio(label="技能", choices=[])
                target_dropdown = gr.Dropdown(label="目標（單體技能用）", choices=[])
                with gr.Row():
                    act_btn = gr.Button("行動", variant="primary")
                    dodge_btn = gr.Button("閃避")
                    end_btn = gr.Button("結束回合")

        with gr.Row():
            input_box = gr.Textbox(
                placeholder="輸入你（凱恩）的行動，按 Enter 確認…（戰鬥中也可打字）",
                show_label=False, scale=5,
            )
            submit_btn = gr.Button("確認", scale=1, variant="primary")

        combat_out = [battlefield_img, combat_controls, skill_radio, target_dropdown]
        demo.load(fn=on_load, outputs=[gm_chat, thor_chat, kaine_chat, state] + combat_out)

        full_out = [gm_chat, thor_chat, kaine_chat, state, input_box] + combat_out
        sub_in   = [input_box, gm_chat, thor_chat, kaine_chat, state]
        submit_btn.click(on_submit, inputs=sub_in, outputs=full_out)
        input_box.submit(on_submit, inputs=sub_in, outputs=full_out)

        quick_in = [gm_chat, thor_chat, kaine_chat, state]
        act_btn.click(on_combat_act,
                      inputs=[skill_radio, target_dropdown] + quick_in, outputs=full_out)
        dodge_btn.click(on_dodge, inputs=quick_in, outputs=full_out)
        end_btn.click(on_end_turn, inputs=quick_in, outputs=full_out)
        battlefield_img.select(on_grid_click, inputs=[state], outputs=[state, combat_hint])

    return demo


def main() -> None:
    try:
        cfg = llm_config.resolve()
    except llm_config.BackendConfigError as e:
        print(f"錯誤：{e}")
        return
    if cfg["backend"] == "ollama":
        check_ollama(cfg["model"], cfg["base_url"])
    build_ui().launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
