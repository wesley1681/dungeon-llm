"""Gradio web entry point — event consumer only, no game logic."""
import sys
import queue as _queue
import threading

sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

from .scenarios.dungeon import (
    build_world_state, build_npc_agents, OPENING_SCENE,
    THOR_PERSONALITY, THOR_TACTICS_GENERAL, THOR_TACTICS_COMBAT,
)
from .llm.gm_agent import GMAgent
from .llm.tag_agent import TagAgent
from .llm.player_agent import PlayerAgent
from .llm.arbiter import ArbiterAgent
from .game import (
    GameSession,
    TagResult, StreamChunk, ActionResult,
    RoundStart, CombatStart, CombatEnd,
    ExplorationPrompt, CombatPrompt,
    ConversationPrompt, StatusMessage, GameOver, QuestComplete,
)
from .engine.quests import objective_progress_str
from .cli import (
    check_ollama, MODEL,
    GM_THINK, GM_SHOW_THINKING, GM_OPTIONS,
    TAG_OPTIONS,
    THOR_THINK, THOR_SHOW_THINKING, THOR_OPTIONS,
    DEBUG_ARBITER,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _aria_status(world_state) -> str:
    aria     = world_state.characters["aria"]
    status   = "、".join(fx.name for fx in aria.status_effects) if aria.status_effects else "無"
    weapons  = "、".join(w.name for w in aria.weapons) or "無"
    usable   = [
        f"{c.name}×{c.quantity}" if c.quantity > 1 else c.name
        for c in aria.consumables if c.effect_type != "ammo" and c.quantity > 0
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
        f"*HP {aria.hp}/{aria.max_hp}  AC {aria.ac}  狀態：{status}*\n"
        f"⚔ 武器：{weapons}\n"
        f"🎒 道具：{items}"
        f"{quest_line}"
    )


def _init_game() -> dict:
    world_state = build_world_state()
    session = GameSession(
        world_state = world_state,
        gm          = GMAgent(model=MODEL, world_state=world_state,
                              think=GM_THINK, show_thinking=GM_SHOW_THINKING,
                              options=GM_OPTIONS),
        tag_agent   = TagAgent(model=MODEL, world_state=world_state,
                               base_url="http://localhost:11434", backend="ollama",
                               options=TAG_OPTIONS),
        thor_agent  = PlayerAgent(model=MODEL,
                                  char_id="thor",
                                  character=world_state.characters["thor"],
                                  personality=THOR_PERSONALITY,
                                  tactics=THOR_TACTICS_GENERAL,
                                  combat_tactics=THOR_TACTICS_COMBAT,
                                  world_state=world_state,
                                  think=THOR_THINK, show_thinking=THOR_SHOW_THINKING,
                                  options=THOR_OPTIONS),
        arbiter     = ArbiterAgent(model=MODEL),
        npc_agents  = build_npc_agents(world_state, MODEL,
                                       "http://localhost:11434", "ollama"),
    )
    session.start()
    return {"session": session, "world_state": world_state}


# ── Event → Gradio renderer ───────────────────────────────────────────────────

def _consume_until_prompt(state, gm_msgs, thor_msgs, aria_msgs):
    """Consume GameSession events, yield Gradio updates, stop at player prompt.

    Yields (gm_msgs, thor_msgs, aria_msgs, state, "").
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
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

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
                    yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

            elif src == "thor":
                if not event.thinking:
                    _ensure_thor_slot("thor")
                    thor_msgs[-1]["content"] += event.text
                    thor_text += event.text
                    yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

            elif src == "npc":
                actor_key = f"npc_{event.actor}"
                if last_source != actor_key:
                    gm_msgs.append({"role": "user",      "content": f"（{event.actor} 的回合）"})
                    gm_msgs.append({"role": "assistant", "content": ""})
                    last_source = actor_key
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

            elif src == "pc_combat":
                # All LLM-PC combat streams route to the Thor panel for now.
                # When a 2nd LLM-controlled PC is added, this needs per-actor panels.
                if not event.thinking:
                    slot_key = f"pc_combat_{event.actor}"
                    _ensure_thor_slot(slot_key)
                    thor_msgs[-1]["content"] += event.text
                    yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

            elif src == "narrate":
                _ensure_gm_slot("narrate")
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

            elif src == "npc_talk":
                if last_source != "npc_talk":
                    gm_msgs.append({"role": "user",      "content": f"（{event.actor}）"})
                    gm_msgs.append({"role": "assistant", "content": ""})
                    last_source = "npc_talk"
                gm_msgs[-1]["content"] += event.text
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── ActionResult ──────────────────────────────────────────────────────
        elif isinstance(event, ActionResult):
            debug_str = f"\n\n`[判定器] {event.debug}`" if DEBUG_ARBITER else ""
            result_line = f"\n\n`{event.summary}`{debug_str}"
            # Append to whatever the last open slot is (npc / pc_combat message)
            if gm_msgs and last_source and last_source.startswith("npc"):
                gm_msgs[-1]["content"] += result_line
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            elif last_source and last_source.startswith("pc_combat_"):
                if thor_msgs:
                    thor_msgs[-1]["content"] += result_line
                # Also show in aria panel (uses event.actor — not hardcoded)
                last_line = thor_msgs[-1]['content'].split('`')[0].strip() if thor_msgs else ''
                aria_msgs.append({"role": "assistant",
                                   "content": f"**{event.actor}：**{last_line}\n\n`{event.summary}`"})
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            else:
                # Aria action result
                aria_msgs.append({"role": "assistant", "content": f"`{event.summary}`{debug_str}"})
                yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            last_source = "result"

        # ── CombatStart ───────────────────────────────────────────────────────
        elif isinstance(event, CombatStart):
            order_str = "→".join(event.order)
            gm_msgs.append({"role": "assistant",
                             "content": f"⚔ **先攻順序：{order_str}**"})
            last_source = "combat_start"
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── RoundStart ────────────────────────────────────────────────────────
        elif isinstance(event, RoundStart):
            gm_msgs.append({"role": "assistant",
                             "content": f"---\n**第 {event.number} 回合**"})
            last_source = "round"
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── CombatEnd ─────────────────────────────────────────────────────────
        elif isinstance(event, CombatEnd):
            msg = "✨ **所有敵人已倒下！戰鬥結束。**"
            if event.loot:
                msg += f"\n\n💰 **可拾取：{'、'.join(event.loot)}**\n（告訴GM你想拿什麼）"
            gm_msgs.append({"role": "assistant", "content": msg})
            last_source = "combat_end"
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── ExplorationPrompt ─────────────────────────────────────────────────
        elif isinstance(event, ExplorationPrompt):
            # Thor panel: ensure final content is set
            if thor_msgs and thor_text:
                thor_msgs[-1]["content"] = thor_text

            # Aria panel: combined summary — GM + each prior PC remark + status
            remark_lines = "\n\n".join(f"**{name}：** {text}"
                                       for name, text in event.prior_remarks.items())
            summary = f"**GM：** {event.gm_text}"
            if remark_lines:
                summary += "\n\n" + remark_lines
            summary += f"\n\n{_aria_status(world_state)}"
            aria_msgs.append({"role": "assistant", "content": summary})
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            return  # stop — wait for player submit

        # ── CombatPrompt ──────────────────────────────────────────────────────
        elif isinstance(event, CombatPrompt):
            if event.info_text:
                info_block = event.info_text
            else:
                enemies_str = "、".join(f"{n}（{c}）" for c, n in event.enemies.items())
                info_block = f"敵人：{enemies_str}"
            aria_msgs.append({"role": "assistant",
                               "content": (f"**⚔ 輪到你了！**\n\n"
                                           f"```\n{info_block}\n```\n\n"
                                           f"{_aria_status(world_state)}")})
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            return  # stop — wait for player submit

        # ── ConversationPrompt ────────────────────────────────────────────────
        elif isinstance(event, ConversationPrompt):
            aria_msgs.append({"role": "assistant",
                               "content": (f"**【與 {event.npc_name} 對話中｜態度：{event.attitude_label}】**\n\n"
                                           f"{_aria_status(world_state)}\n\n"
                                           f"*輸入「離開」結束對話*")})
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            return  # wait for player input

        # ── StatusMessage ─────────────────────────────────────────────────────
        elif isinstance(event, StatusMessage):
            aria_msgs.append({"role": "assistant", "content": event.text})
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── QuestComplete ─────────────────────────────────────────────────────
        elif isinstance(event, QuestComplete):
            msg = f"✅ **任務達成：{event.title}**（可回去找 {event.giver_name} 回報）"
            gm_msgs.append({"role": "assistant", "content": msg})
            aria_msgs.append({"role": "assistant", "content": msg})
            last_source = "quest"
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""

        # ── GameOver ──────────────────────────────────────────────────────────
        elif isinstance(event, GameOver):
            gm_msgs.append({"role": "assistant",
                             "content": f"💀 **{event.reason}**"})
            yield gm_msgs[:], thor_msgs[:], aria_msgs[:], state, ""
            return


# ── Opening (page load) ───────────────────────────────────────────────────────

def on_load():
    state = _init_game()

    gm_msgs   = [{"role": "user", "content": "（開場）"}]
    thor_msgs : list = []
    aria_msgs = [{"role": "user", "content": OPENING_SCENE}]

    for gm_msgs, thor_msgs, aria_msgs, state, _ in _consume_until_prompt(
            state, gm_msgs, thor_msgs, aria_msgs):
        yield gm_msgs, thor_msgs, aria_msgs, state


# ── Submit (player action) ────────────────────────────────────────────────────

def on_submit(human_input: str,
              gm_msgs: list, thor_msgs: list, aria_msgs: list,
              state: dict):
    if not human_input.strip() or state is None:
        yield gm_msgs, thor_msgs, aria_msgs, state, ""
        return

    session = state["session"]
    session.submit_player_input(human_input)
    aria_msgs = aria_msgs + [{"role": "user", "content": human_input}]
    yield gm_msgs, thor_msgs, aria_msgs, state, ""

    for gm_msgs, thor_msgs, aria_msgs, state, _ in _consume_until_prompt(
            state, gm_msgs[:], thor_msgs[:], aria_msgs[:]):
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
                gr.Markdown("### 🗡️ 你（凱恩）")
                aria_chat = gr.Chatbot(height=550, show_label=False)

        with gr.Row():
            input_box = gr.Textbox(
                placeholder="輸入你（凱恩）的行動，按 Enter 確認…",
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
