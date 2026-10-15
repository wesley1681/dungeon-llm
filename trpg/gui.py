"""Tkinter graphical user interface for the narrative dungeon game.

The GameSession runs on its own thread while this app polls its event queue from
the Tk main loop. Narration appears in a scrolling transcript, and combat uses a
battlefield image with native skill buttons.
"""
from __future__ import annotations
import pathlib
import tkinter as tk
from tkinter import ttk, scrolledtext

from PIL import ImageTk

from . import gui_combat
from .game import (
    TagResult, StreamChunk, ActionResult, RoundStart, CombatStart, CombatEnd,
    ExplorationPrompt, CombatPrompt, ConversationPrompt, StatusMessage,
    QuestComplete, GameOver,
)

_PUMP_MS = 40          # event-poll cadence (ms) on the tk main loop
_BATTLE_PX = 680       # battlefield image side (px)
_MELEE_FONT = ("Microsoft JhengHei", 11)
_MONO_FONT = ("Consolas", 10)

_STAT_LABELS = {
    "STR": "力量",
    "DEX": "敏捷",
    "CON": "體質",
    "INT": "智力",
    "WIS": "感知",
    "CHA": "魅力",
}

_STATUS_LABELS = {
    "dodging": "閃避",
    "hidden": "躲藏",
    "reckless": "魯莽攻擊",
    "blessed": "祝福",
    "shielded": "法盾",
    "raging": "狂暴",
    "restrained": "束縛",
    "paralyzed": "麻痺",
    "frightened": "恐懼",
    "poisoned": "中毒",
    "stunned": "震懾",
    "petrified": "石化",
    "asleep": "沉睡",
    "swallowed": "被吞噬",
}

# Combat trace log (repo root, overwritten each launch). The GUI transcript
# is in-memory only; this file provides the details needed to diagnose a fight —
# it records the full roster (team / attitude / position / HP / driving policy),
# every round, every action, so a bug ("civilian dragged into the fight", "ally
# hides in a corner") can be diagnosed from data instead of guessed at.
_LOG_PATH = pathlib.Path(__file__).resolve().parent.parent / "desktop_game_log.txt"

# skills handled by dedicated buttons — filtered out of the generated skill list
# so we never render duplicate move/dodge/end buttons.
_DEDICATED = {"end", "move", "dodge", "hide", "disengage"}


def _object_name(value) -> str:
    """Return the display name of an equipment/status value."""
    return str(getattr(value, "name", value))


def character_panel_data(character, character_id: str | None = None) -> dict:
    """Build display-only character data without depending on Tk widgets."""
    from .engine.abilities import ABILITY_REGISTRY

    if character.is_dead():
        status = "死亡"
    elif character.is_dying():
        saves = character.death_saves
        status = f"瀕死（成功 {saves.get('successes', 0)}／失敗 {saves.get('failures', 0)}）"
    else:
        status_names = []
        for effect in list(character.status_effects):
            name = _object_name(effect)
            status_names.append(_STATUS_LABELS.get(name, name))
        if character.concentrating_on:
            concentration = ABILITY_REGISTRY.get(character.concentrating_on)
            concentration_name = (
                concentration.display_name
                if concentration is not None else character.concentrating_on
            )
            status_names.append(f"專注：{concentration_name}")
        status = "、".join(status_names) or "正常"

    abilities = []
    ability_ids = dict.fromkeys(
        list(character.known_abilities) + list(character.reactions))
    for skill_id in ability_ids:
        ability = ABILITY_REGISTRY.get(skill_id)
        name = ability.display_name if ability is not None else skill_id
        if skill_id in character.ability_uses:
            uses = character.ability_uses[skill_id]
            name += f"（剩 {uses} 次）" if uses is not None else "（不限次數）"
        abilities.append(name)

    spell_slots = [
        f"{level} 環 × {remaining}"
        for level, remaining in sorted(character.spell_slots.items())
    ]
    if character.spellcasting_ability:
        stat_id = character.spellcasting_ability
        stat_label = _STAT_LABELS.get(stat_id, stat_id)
        spell_slots.insert(0, f"施法屬性：{stat_label} {stat_id}")

    stats = []
    for stat_id, label in _STAT_LABELS.items():
        value = getattr(character.stats, stat_id)
        stats.append({
            "id": stat_id,
            "label": label,
            "value": value,
            "modifier": character.stats.modifier(stat_id),
        })

    return {
        "id": character_id or character.id,
        "name": character.name,
        "race": character.race,
        "class": character.class_,
        "level": character.level,
        "hp": character.hp,
        "max_hp": character.max_hp,
        "ac": character.ac,
        "status": status,
        "stats": stats,
        "skills": list(character.skill_proficiencies),
        "saving_throws": list(character.saving_throw_proficiencies),
        "abilities": abilities,
        "spell_slots": spell_slots,
        "weapons": [_object_name(item) for item in list(character.weapons)],
        "consumables": [
            f"{item.name} × {item.quantity}" for item in list(character.consumables)
        ],
        "equipment": [_object_name(item) for item in list(character.equipment)],
        "gear": [str(item) for item in list(character.gear)],
    }


def party_panel_data(world_state) -> list[dict]:
    """Return party members in party order, skipping stale character ids."""
    return [
        character_panel_data(world_state.characters[character_id], character_id)
        for character_id in list(world_state.party_ids)
        if character_id in world_state.characters
    ]


class GameGUI(tk.Tk):
    def __init__(self, state: dict):
        super().__init__()
        self.title("⚔ 地下城探索：失竊的護符")
        self.geometry("1280x860")
        self.minsize(1024, 720)
        self.session = state["session"]
        self.world_state = state["world_state"]
        self._human_id = next(
            (
                cid for cid, character in self.world_state.characters.items()
                if character.controller_type == "human"
            ),
            "kaine",
        )

        self._last_speaker = None          # transcript: whose line is currently open
        self._awaiting = None              # None | 'explore' | 'combat' | 'conversation'
        self._combat_view = None           # combat_view_state payload on the human's turn
        self._transform = (0.0, 0.0, _BATTLE_PX / 30.0)   # (ox, oy, scale) px<->metre
        self._pending_skill = None         # skill dict awaiting a target/cell click
        self._aim = None                   # None | 'point' | 'enemy' | 'ally'
        self._bf_photo = None              # keep a ref so tk doesn't GC the image
        self._party_data = None            # last rendered display snapshot
        self._selected_party_id = None
        try:
            self._logf = open(_LOG_PATH, "w", encoding="utf-8")
        except Exception:
            self._logf = None

        self._build_ui()
        self._refresh_party_panel(force=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._append(
            f"你（{self._name(self._human_id)}）", self.world_state.scene, "you")
        self.after(_PUMP_MS, self._pump)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # input row stays available in both exploration and combat modes.
        bottom = ttk.Frame(self, padding=6)
        bottom.pack(side="bottom", fill="x")
        self.entry = ttk.Entry(bottom, font=_MELEE_FONT)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self._on_enter())
        self.send_btn = ttk.Button(bottom, text="確認", command=self._on_enter)
        self.send_btn.pack(side="left", padx=6)
        self._set_input_enabled(False)

        # Exploration and combat are sibling views. Combat replaces the whole
        # content area; ending combat restores the transcript and party panel.
        self.content = ttk.Frame(self)
        self.content.pack(side="top", fill="both", expand=True)

        self.exploration = ttk.Frame(self.content)
        self.exploration.pack(fill="both", expand=True)
        panes = ttk.PanedWindow(self.exploration, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=6, pady=6)

        transcript = ttk.Frame(panes)
        panes.add(transcript, weight=3)
        self.log = scrolledtext.ScrolledText(
            transcript, wrap="word", font=_MELEE_FONT, state="disabled",
            bg="#0d1117", fg="#d0d6de", insertbackground="#d0d6de", padx=8, pady=6)
        self.log.pack(fill="both", expand=True)
        for tag, color in (("gm", "#e0c86a"), ("llm_player", "#6ab7ff"),
                           ("npc", "#9ad07a"), ("you", "#ff9a76"),
                           ("sys", "#8b93a1"), ("result", "#c0a0ff")):
            self.log.tag_config(tag, foreground=color)
        self.log.tag_config("speaker", font=("Microsoft JhengHei", 11, "bold"))

        sidebar = ttk.Frame(panes, width=340)
        panes.add(sidebar, weight=1)
        ttk.Label(sidebar, text="隊伍", font=("Microsoft JhengHei", 14, "bold")).pack(
            anchor="w", padx=8, pady=(4, 2))

        party_scroll = ttk.Frame(sidebar)
        party_scroll.pack(fill="both", expand=True)
        self.party_canvas = tk.Canvas(
            party_scroll, width=330, highlightthickness=0, borderwidth=0)
        party_bar = ttk.Scrollbar(
            party_scroll, orient="vertical", command=self.party_canvas.yview)
        self.party_canvas.configure(yscrollcommand=party_bar.set)
        party_bar.pack(side="right", fill="y")
        self.party_canvas.pack(side="left", fill="both", expand=True)
        self.party_body = ttk.Frame(self.party_canvas, padding=(8, 4))
        self._party_window = self.party_canvas.create_window(
            (0, 0), window=self.party_body, anchor="nw")
        self.party_body.bind(
            "<Configure>",
            lambda _event: self.party_canvas.configure(
                scrollregion=self.party_canvas.bbox("all")))
        self.party_canvas.bind(
            "<Configure>",
            lambda event: self.party_canvas.itemconfigure(
                self._party_window, width=event.width))

        # Full content-area combat view — packed only during a fight.
        self.combat = ttk.Frame(self.content, padding=10)
        self.bf = tk.Label(self.combat, bg="#181c24", cursor="crosshair")
        self.bf.pack(side="left", anchor="center", padx=(0, 10))
        self.bf.bind("<Button-1>", self._on_bf_click)

        # The battlefield has a fixed useful size, but the controls can be
        # taller than a small window. Keep them in their own scrolling pane so
        # every skill and dedicated action remains reachable.
        combat_side = ttk.Frame(self.combat)
        combat_side.pack(side="left", fill="both", expand=True)
        self.combat_canvas = tk.Canvas(
            combat_side, highlightthickness=0, borderwidth=0)
        combat_bar = ttk.Scrollbar(
            combat_side, orient="vertical", command=self.combat_canvas.yview)
        self.combat_canvas.configure(yscrollcommand=combat_bar.set)
        combat_bar.pack(side="right", fill="y")
        self.combat_canvas.pack(side="left", fill="both", expand=True)
        ctrl = ttk.Frame(self.combat_canvas, padding=(10, 0))
        self._combat_ctrl_window = self.combat_canvas.create_window(
            (0, 0), window=ctrl, anchor="nw")
        ctrl.bind(
            "<Configure>",
            lambda _event: self.combat_canvas.configure(
                scrollregion=self.combat_canvas.bbox("all")))
        self.combat_canvas.bind(
            "<Configure>",
            lambda event: self.combat_canvas.itemconfigure(
                self._combat_ctrl_window, width=event.width))

        self.combat_status = ttk.Label(ctrl, text="", font=_MONO_FONT, justify="left")
        self.combat_status.pack(anchor="w")
        ttk.Label(ctrl, text="你的招式：").pack(anchor="w", pady=(6, 0))
        self.skill_frame = ttk.Frame(ctrl)
        self.skill_frame.pack(anchor="w", fill="x")
        btns = ttk.Frame(ctrl)
        btns.pack(anchor="w", pady=6)
        ttk.Button(btns, text="移動", command=self._pick_move).pack(side="left")
        ttk.Button(btns, text="閃避", command=lambda: self._submit("閃避")).pack(side="left", padx=4)
        ttk.Button(btns, text="脫離", command=lambda: self._submit("脫離")).pack(side="left")
        ttk.Button(btns, text="結束回合", command=lambda: self._submit("結束")).pack(side="left", padx=4)
        ttk.Button(btns, text="取消瞄準", command=self._cancel_aim).pack(side="left")
        self.aim_hint = ttk.Label(ctrl, text="", foreground="#e0c86a")
        self.aim_hint.pack(anchor="w")
        self.bind_all("<MouseWheel>", self._route_mousewheel, add="+")

    @staticmethod
    def _is_within(widget, ancestor) -> bool:
        while widget is not None:
            if widget == ancestor:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _route_mousewheel(self, event):
        """Scroll the sidebar under the pointer, including all child widgets."""
        units = int(-event.delta / 120)
        if self._is_within(event.widget, self.party_canvas):
            self.party_canvas.yview_scroll(units, "units")
            return "break"
        if self._is_within(event.widget, self.combat_canvas):
            self.combat_canvas.yview_scroll(units, "units")
            return "break"
        return None

    def _select_party_member(self, character_id: str):
        self._selected_party_id = character_id
        self._refresh_party_panel(force=True)

    def _detail_section(self, parent, title: str, items: list[str]):
        box = ttk.LabelFrame(parent, text=title, padding=(7, 4))
        box.pack(fill="x", pady=(0, 6))
        ttk.Label(
            box, text="、".join(items) if items else "無",
            justify="left", wraplength=285,
        ).pack(anchor="w", fill="x")

    def _refresh_party_panel(self, force: bool = False):
        data = party_panel_data(self.world_state)
        if not force and data == self._party_data:
            return
        self._party_data = data

        ids = [member["id"] for member in data]
        if self._selected_party_id not in ids:
            self._selected_party_id = ids[0] if ids else None

        for widget in self.party_body.winfo_children():
            widget.destroy()

        if not data:
            ttk.Label(self.party_body, text="目前沒有隊伍成員").pack(anchor="w")
            return

        for member in data:
            selected = member["id"] == self._selected_party_id
            row = ttk.Frame(self.party_body, relief="solid", borderwidth=1, padding=6)
            row.pack(fill="x", pady=(0, 6))
            prefix = "▶ " if selected else ""
            ttk.Button(
                row,
                text=f"{prefix}{member['name']}　Lv.{member['level']} {member['class']}",
                command=lambda character_id=member["id"]:
                    self._select_party_member(character_id),
            ).pack(fill="x")
            hp_row = ttk.Frame(row)
            hp_row.pack(fill="x", pady=(5, 1))
            hp_max = max(1, member["max_hp"])
            ttk.Progressbar(
                hp_row, maximum=hp_max,
                value=max(0, min(member["hp"], hp_max)),
            ).pack(side="left", fill="x", expand=True)
            ttk.Label(
                hp_row, text=f"  HP {member['hp']}/{member['max_hp']}"
            ).pack(side="left")
            ttk.Label(
                row, text=f"AC {member['ac']}　狀態：{member['status']}",
                foreground="#9aa4b2",
            ).pack(anchor="w")

        member = next(
            item for item in data if item["id"] == self._selected_party_id)
        ttk.Separator(self.party_body).pack(fill="x", pady=(4, 8))
        ttk.Label(
            self.party_body, text=member["name"],
            font=("Microsoft JhengHei", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            self.party_body,
            text=f"{member['race']}／{member['class']}　等級 {member['level']}　AC {member['ac']}",
        ).pack(anchor="w", pady=(0, 7))

        stats_box = ttk.LabelFrame(self.party_body, text="屬性", padding=(7, 4))
        stats_box.pack(fill="x", pady=(0, 6))
        for index, stat in enumerate(member["stats"]):
            text = (
                f"{stat['label']} {stat['id']}  {stat['value']} "
                f"({stat['modifier']:+d})"
            )
            ttk.Label(stats_box, text=text).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 12), pady=1)

        self._detail_section(self.party_body, "技能熟練", member["skills"])
        self._detail_section(self.party_body, "豁免熟練", member["saving_throws"])
        self._detail_section(self.party_body, "職業能力／法術", member["abilities"])
        self._detail_section(self.party_body, "法術位", member["spell_slots"])
        self._detail_section(self.party_body, "武器", member["weapons"])
        self._detail_section(self.party_body, "消耗品", member["consumables"])
        self._detail_section(self.party_body, "穿戴裝備", member["equipment"])
        self._detail_section(self.party_body, "一般道具", member["gear"])

    # ── event pump (tk main loop) ───────────────────────────────────────────────
    def _pump(self):
        try:
            drained = 0
            while drained < 400:                 # cap per tick so the UI stays responsive
                ev = self.session.next_event(block=False)
                if ev is None:
                    break
                self._handle(ev)
                drained += 1
            if drained and self._combat_active():
                self._redraw_battlefield()
        finally:
            self._refresh_party_panel()
            self.after(_PUMP_MS, self._pump)

    def _handle(self, ev):
        if isinstance(ev, StreamChunk):
            self._handle_stream(ev)
        elif isinstance(ev, TagResult):
            lines = [f"  • {r}" for r in ev.ok] + [f"  ⚠ {e}" for e in ev.errors]
            if lines:
                self._append("【機制結算】", "\n".join(lines), "sys")
                self._log("[TAG] " + " | ".join(ev.ok + [f"ERR:{e}" for e in ev.errors]))
        elif isinstance(ev, ActionResult):
            self._append_inline(f"  〔{ev.summary}〕\n", "result")
            self._log(f"[ACTION] {ev.actor}｜{ev.debug}｜{ev.summary}")
        elif isinstance(ev, CombatStart):
            self._append("戰鬥", "⚔ 先攻順序：" + "→".join(ev.order), "sys")
            self._log("\n========== COMBAT START ==========")
            self._log("先攻順序: " + " → ".join(ev.order))
            self._dump_roster()
            self._show_combat()
        elif isinstance(ev, RoundStart):
            self._append("回合", f"—— 第 {ev.number} 回合 ——", "sys")
            self._log(f"\n----- Round {ev.number} -----")
        elif isinstance(ev, CombatEnd):
            loot = f"　可拾取：{'、'.join(ev.loot)}" if ev.loot else ""
            self._append("戰鬥", "✨ 敵人已倒下，戰鬥結束。" + loot, "sys")
            self._log("---- COMBAT END ----")
            self._dump_roster()
            self._hide_combat()
        elif isinstance(ev, ExplorationPrompt):
            # narration already arrived as StreamChunks — just open the input.
            self._prompt("explore")
        elif isinstance(ev, CombatPrompt):
            if ev.ctx is not None:
                self._combat_view = gui_combat.combat_view_state(
                    ev.kaine, ev.ctx, self.world_state)
            self._show_combat()
            self._prompt("combat")
            self._log("[你的回合] 等待玩家輸入")
        elif isinstance(ev, ConversationPrompt):
            self._append("系統", f"【與 {ev.npc_name} 對話中｜態度：{ev.attitude_label}】"
                                  f"　輸入「離開」結束對話", "sys")
            self._prompt("conversation")
        elif isinstance(ev, StatusMessage):
            self._append("系統", ev.text, "sys")
            self._log(f"[STATUS] {ev.text}")
        elif isinstance(ev, QuestComplete):
            self._append("系統", f"✅ 任務達成：{ev.title}（回去找 {ev.giver_name} 回報）", "sys")
        elif isinstance(ev, GameOver):
            self._append("系統", f"💀 {ev.reason}", "sys")
            self._log(f"[GAME OVER] {ev.reason}")
            self._set_input_enabled(False)
            self._awaiting = None

    def _handle_stream(self, ev: StreamChunk):
        if ev.thinking:                    # GUI hides chain-of-thought
            return
        src = ev.source
        if src in ("gm", "narrate"):
            speaker, tag = "GM", "gm"
        elif src == "pc_combat" or (
            src in self.world_state.characters
            and self.world_state.characters[src].controller_type == "llm_player"
        ):
            speaker, tag = self._name(ev.actor or src), "llm_player"
        elif src in ("npc", "npc_talk"):
            speaker, tag = self._name(ev.actor) if ev.actor else "NPC", "npc"
        else:
            speaker, tag = "GM", "gm"
        self._stream(speaker, ev.text, tag)

    # ── transcript helpers ──────────────────────────────────────────────────────
    def _name(self, cid: str) -> str:
        c = self.world_state.characters.get(cid)
        return c.name if c else cid

    def _stream(self, speaker: str, text: str, tag: str):
        self.log.config(state="normal")
        if self._last_speaker != speaker:
            self.log.insert("end", f"\n{speaker}：", ("speaker", tag))
            self._last_speaker = speaker
        self.log.insert("end", text, (tag,))
        self.log.see("end")
        self.log.config(state="disabled")

    def _append(self, speaker: str, text: str, tag: str):
        self.log.config(state="normal")
        self.log.insert("end", f"\n{speaker}：", ("speaker", tag))
        self.log.insert("end", f"{text}\n", (tag,))
        self.log.see("end")
        self.log.config(state="disabled")
        self._last_speaker = None

    def _append_inline(self, text: str, tag: str):
        self.log.config(state="normal")
        self.log.insert("end", text, (tag,))
        self.log.see("end")
        self.log.config(state="disabled")

    # ── combat trace log (desktop_game_log.txt) ─────────────────────────────────
    def _log(self, msg: str):
        if self._logf is None:
            return
        try:
            self._logf.write(msg + "\n")
            self._logf.flush()
        except Exception:
            pass

    def _dump_roster(self):
        """Full roster — the data needed to see who is actually in the fight and
        why: team (by attitude), position, HP, and which policy drives each."""
        if self._logf is None:
            return
        ws = self.world_state
        combat = getattr(ws, "combat", None)
        order = list(combat.initiative_order) if combat else []
        room = ws.dungeon_map.current_room if getattr(ws, "dungeon_map", None) else None
        room_npcs = list(room.npc_ids) if room else []
        self._log(f"當前房間 npc_ids: {room_npcs}")
        self._log(f"party_ids: {list(ws.party_ids)}   pc_ids: {list(ws.pc_ids)}")
        self._log("roster (id｜名｜類｜attitude｜HP｜pos｜init｜party｜room｜policy):")
        for cid, c in ws.characters.items():
            if not c.is_alive():
                continue
            kind = "NPC" if c.is_npc else "PC"
            pos = getattr(c, "position", None)
            posf = f"({pos.x:.1f},{pos.y:.1f})" if pos is not None else "?"
            pol = type((getattr(self.session, "policies", {}) or {}).get(cid)).__name__
            self._log(
                f"  {cid:14} {c.name:8} {kind:3} att={getattr(c, 'attitude', '?')} "
                f"HP={c.hp}/{c.max_hp} {posf} init={cid in order} "
                f"party={cid in ws.party_ids} room={cid in room_npcs} pol={pol}")

    # ── input gating ────────────────────────────────────────────────────────────
    def _set_input_enabled(self, on: bool):
        st = "normal" if on else "disabled"
        self.entry.config(state=st)
        self.send_btn.config(state=st)

    def _prompt(self, kind: str):
        self._awaiting = kind
        self._set_input_enabled(True)
        self.entry.focus_set()
        if kind == "combat":
            self._show_combat_controls()

    def _on_enter(self):
        text = self.entry.get().strip()
        if not text or self._awaiting is None:
            return
        self.entry.delete(0, "end")
        self._submit(text)

    def _submit(self, text: str):
        """The single funnel: typed input AND synthesised combat commands both
        go through here (echo, gate off, then hand the command to the session)."""
        if self._awaiting is None:
            return
        self._append(f"你（{self._name(self._human_id)}）", text, "you")
        self._awaiting = None
        self._cancel_aim()
        self._set_input_enabled(False)
        self._clear_skill_buttons()        # hide controls until the next prompt
        self.session.submit_player_input(text)

    # ── combat panel ────────────────────────────────────────────────────────────
    def _combat_active(self) -> bool:
        c = getattr(self.world_state, "combat", None)
        return bool(c and c.active and c.initiative_order)

    def _show_combat(self):
        if self.exploration.winfo_ismapped():
            self.exploration.pack_forget()
        if not self.combat.winfo_ismapped():
            self.combat.pack(fill="both", expand=True)
        self._redraw_battlefield()

    def _hide_combat(self):
        self._combat_view = None
        self._cancel_aim()
        self._clear_skill_buttons()
        if self.combat.winfo_ismapped():
            self.combat.pack_forget()
        if not self.exploration.winfo_ismapped():
            self.exploration.pack(fill="both", expand=True)
        self._refresh_party_panel(force=True)

    def _redraw_battlefield(self):
        actor_id = self._combat_view["actor_id"] if self._combat_view else self._human_id
        img, transform = gui_combat.render_battlefield(
            self.world_state, actor_id, px=_BATTLE_PX)
        self._transform = transform
        self._bf_photo = ImageTk.PhotoImage(img)
        self.bf.config(image=self._bf_photo)

    def _clear_skill_buttons(self):
        for w in self.skill_frame.winfo_children():
            w.destroy()
        self.combat_status.config(text="（等待其他角色行動…）" if self._combat_active() else "")

    def _show_combat_controls(self):
        self._clear_skill_buttons()
        cv = self._combat_view
        if not cv:
            return
        r = cv["resources"]
        self.combat_status.config(text=(
            f"第 {cv['round']} 回合　你的回合\n"
            f"資源：動作 {r.get('action', 0)}／附贈 {r.get('bonus_action', 0)}"
            f"／移動 {r.get('movement', 0):.1f}m"))
        for s in cv["skills"]:
            if s["skill_id"] in _DEDICATED:
                continue
            ttk.Button(self.skill_frame, text=s["display_name"],
                       command=lambda sk=s: self._pick_skill(sk)).pack(
                anchor="w", fill="x", pady=1)

    # ── targeting ───────────────────────────────────────────────────────────────
    def _pick_move(self):
        if self._awaiting != "combat":
            return
        self._pending_skill = {"skill_id": "move", "display_name": "移動",
                               "target_kind": "point"}
        self._aim = "point"
        self.aim_hint.config(text="點戰場選定要移動到的位置")

    def _pick_skill(self, sk: dict):
        if self._awaiting != "combat":
            return
        kind = sk["target_kind"]
        cv = self._combat_view
        if kind == "none":
            self._submit(gui_combat.command_for(sk))
        elif kind in ("multi_enemy", "multi_ally"):
            pool = cv["enemies"] if "enemy" in kind else cv["allies"]
            self._submit(gui_combat.command_for(sk, target_id=",".join(pool.keys())))
        elif kind in ("enemy", "ally"):
            pool = cv["enemies"] if kind == "enemy" else cv["allies"]
            if len(pool) == 1:
                self._submit(gui_combat.command_for(sk, target_id=next(iter(pool))))
            else:
                self._pending_skill = sk
                self._aim = kind
                self.aim_hint.config(
                    text=f"點戰場上的{'敵人' if kind == 'enemy' else '隊友'}選目標")
        else:   # point / line / cone
            self._pending_skill = sk
            self._aim = "point"
            self.aim_hint.config(text="點戰場選定位置")

    def _on_bf_click(self, evt):
        if not (self._combat_active() and self._pending_skill and self._aim):
            return
        ox, oy, scale = self._transform
        mx, my = round(evt.x / scale + ox, 1), round(evt.y / scale + oy, 1)
        sk = self._pending_skill
        if self._aim == "point":
            self._cancel_aim()
            self._submit(gui_combat.command_for(sk, cell=(mx, my)))
        else:   # enemy / ally — snap to the nearest matching combatant
            pool = (self._combat_view["enemies"] if self._aim == "enemy"
                    else self._combat_view["allies"])
            tid = self._nearest(mx, my, pool)
            if tid:
                self._cancel_aim()
                self._submit(gui_combat.command_for(sk, target_id=tid))
            else:
                self.aim_hint.config(text="沒點到目標，再點一次靠近目標的位置")

    def _nearest(self, mx: float, my: float, pool: dict):
        best, best_d = None, 2.5      # metres — must click reasonably close
        for c in self._combat_view["combatants"]:
            if c["id"] in pool and c["alive"]:
                d = ((c["x"] - mx) ** 2 + (c["y"] - my) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = c["id"], d
        return best

    def _cancel_aim(self):
        self._pending_skill = None
        self._aim = None
        self.aim_hint.config(text="")

    # ── shutdown ────────────────────────────────────────────────────────────────
    def _on_close(self):
        try:
            self.session.stop()
        except Exception:
            pass
        if self._logf is not None:
            try:
                self._logf.close()
            except Exception:
                pass
        self.destroy()
