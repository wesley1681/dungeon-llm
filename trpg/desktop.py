"""Desktop (tkinter) front-end for the narrative dungeon game.

Same GameSession event stream as web.py — the session runs on its own thread,
this app polls next_event() on the tk main loop (an `after` loop, non-blocking)
and renders each event. Narration flows into one scrolling transcript. When a
fight starts a combat panel appears in place: a battlefield image (reusing
web_combat.render_battlefield — auto-fit zoom, CJK labels, the same pixel<->metre
transform) plus native skill buttons. Picking a skill (then a target/cell) is
synthesised into the very command string the text box accepts (web_combat.
command_for), so combat rides the existing input path — no new game seam.

Run:  python -m trpg.desktop
"""
from __future__ import annotations
import sys
import pathlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from PIL import ImageTk

from . import web_combat
from .llm import config as llm_config
from .scenarios.dungeon import OPENING_SCENE
from .game import (
    TagResult, StreamChunk, ActionResult, RoundStart, CombatStart, CombatEnd,
    ExplorationPrompt, CombatPrompt, ConversationPrompt, StatusMessage,
    QuestComplete, GameOver,
)

_PUMP_MS = 40          # event-poll cadence (ms) on the tk main loop
_BATTLE_PX = 560       # battlefield image side (px)
_MELEE_FONT = ("Microsoft JhengHei", 11)
_MONO_FONT = ("Consolas", 10)

# Combat trace log (repo root, overwritten each launch). The desktop transcript
# is in-memory only; this file is what to hand Claude when a fight misbehaves —
# it records the full roster (team / attitude / position / HP / driving policy),
# every round, every action, so a bug ("civilian dragged into the fight", "ally
# hides in a corner") can be diagnosed from data instead of guessed at.
_LOG_PATH = pathlib.Path(__file__).resolve().parent.parent / "desktop_game_log.txt"

# skills handled by dedicated buttons — filtered out of the generated skill list
# so we never render duplicate move/dodge/end buttons.
_DEDICATED = {"end", "move", "dodge", "hide", "disengage"}


class DesktopApp(tk.Tk):
    def __init__(self, state: dict):
        super().__init__()
        self.title("⚔ 地下城探索：失竊的護符")
        self.geometry("1120x840")
        self.session = state["session"]
        self.world_state = state["world_state"]

        self._last_speaker = None          # transcript: whose line is currently open
        self._awaiting = None              # None | 'explore' | 'combat' | 'conversation'
        self._combat_view = None           # combat_view_state payload on the human's turn
        self._transform = (0.0, 0.0, _BATTLE_PX / 30.0)   # (ox, oy, scale) px<->metre
        self._pending_skill = None         # skill dict awaiting a target/cell click
        self._aim = None                   # None | 'point' | 'enemy' | 'ally'
        self._bf_photo = None              # keep a ref so tk doesn't GC the image
        try:
            self._logf = open(_LOG_PATH, "w", encoding="utf-8")
        except Exception:
            self._logf = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._append("你（凱恩）", OPENING_SCENE, "you")
        self.after(_PUMP_MS, self._pump)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # transcript (fills the space above the combat panel + input row)
        top = ttk.Frame(self, padding=6)
        top.pack(side="top", fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(
            top, wrap="word", font=_MELEE_FONT, state="disabled",
            bg="#0d1117", fg="#d0d6de", insertbackground="#d0d6de", padx=8, pady=6)
        self.log.pack(fill="both", expand=True)
        for tag, color in (("gm", "#e0c86a"), ("thor", "#6ab7ff"),
                           ("npc", "#9ad07a"), ("you", "#ff9a76"),
                           ("sys", "#8b93a1"), ("result", "#c0a0ff")):
            self.log.tag_config(tag, foreground=color)
        self.log.tag_config("speaker", font=("Microsoft JhengHei", 11, "bold"))

        # combat panel — packed in only during a fight (see _show_combat)
        self.combat = ttk.Frame(self, padding=6)
        self.bf = tk.Label(self.combat, bg="#181c24", cursor="crosshair")
        self.bf.pack(side="left")
        self.bf.bind("<Button-1>", self._on_bf_click)
        ctrl = ttk.Frame(self.combat, padding=(10, 0))
        ctrl.pack(side="left", fill="both", expand=True)
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

        # input row (always at the very bottom)
        bottom = ttk.Frame(self, padding=6)
        bottom.pack(side="bottom", fill="x")
        self.entry = ttk.Entry(bottom, font=_MELEE_FONT)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self._on_enter())
        self.send_btn = ttk.Button(bottom, text="確認", command=self._on_enter)
        self.send_btn.pack(side="left", padx=6)
        self._set_input_enabled(False)

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
                self._combat_view = web_combat.combat_view_state(
                    ev.aria, ev.ctx, self.world_state)
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
        if ev.thinking:                    # desktop view hides chain-of-thought
            return
        src = ev.source
        if src in ("gm", "narrate"):
            speaker, tag = "GM", "gm"
        elif src in ("thor", "pc_combat"):
            speaker, tag = self._name(ev.actor) if ev.actor else "索爾", "thor"
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
        go through here, mirroring web.on_submit (echo, gate off, hand to session)."""
        if self._awaiting is None:
            return
        self._append("你（凱恩）", text, "you")
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
        if not self.combat.winfo_ismapped():
            self.combat.pack(side="bottom", fill="x")
        self._redraw_battlefield()

    def _hide_combat(self):
        self._combat_view = None
        self._cancel_aim()
        self._clear_skill_buttons()
        if self.combat.winfo_ismapped():
            self.combat.pack_forget()

    def _redraw_battlefield(self):
        actor_id = self._combat_view["actor_id"] if self._combat_view else "aria"
        img, transform = web_combat.render_battlefield(
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
            self._submit(web_combat.command_for(sk))
        elif kind in ("multi_enemy", "multi_ally"):
            pool = cv["enemies"] if "enemy" in kind else cv["allies"]
            self._submit(web_combat.command_for(sk, target_id=",".join(pool.keys())))
        elif kind in ("enemy", "ally"):
            pool = cv["enemies"] if kind == "enemy" else cv["allies"]
            if len(pool) == 1:
                self._submit(web_combat.command_for(sk, target_id=next(iter(pool))))
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
            self._submit(web_combat.command_for(sk, cell=(mx, my)))
        else:   # enemy / ally — snap to the nearest matching combatant
            pool = (self._combat_view["enemies"] if self._aim == "enemy"
                    else self._combat_view["allies"])
            tid = self._nearest(mx, my, pool)
            if tid:
                self._cancel_aim()
                self._submit(web_combat.command_for(sk, target_id=tid))
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


def _fatal(msg: str):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("無法啟動", msg)
    root.destroy()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from .bootstrap import init_game
    try:
        state = init_game()
    except llm_config.BackendConfigError as e:
        _fatal(str(e))
        return
    except SystemExit:
        _fatal("後端連線失敗。請先執行 start_llama_server.ps1 選擇模型／啟動伺服器。")
        return
    DesktopApp(state).mainloop()


if __name__ == "__main__":
    main()
