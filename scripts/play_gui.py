"""圖形化 1v1 對戰測試台：人類操作 vs 通用模型 (uni_v7)。

啟動： python scripts/play_gui.py [模型路徑]
  預設模型 = models/unified/uni_v7.pt

兩個畫面：
  1) 選身分：雙方各自從「標準職業 / 標準怪物」挑一個身分，可選等級、佈局；
     並可額外加上天賦（抗性/免疫/易傷、pack_tactics、再生、不死韌性、額外攻擊）
     與技能（從技能登錄表多選）。對手席固定由通用模型(盲化)駕駛。
  2) 戰鬥：30×30 格子地圖，點技能鈕選招、點地圖選移動/瞄點目標、按「結束回合」。
     單體攻擊在 1v1 自動鎖定唯一敵人；自療類自動指向自己。

只讀引擎/模型既有介面，不改任何訓練程式。
"""
from __future__ import annotations

import os
import sys
import stat
import tkinter as tk
from tkinter import ttk, messagebox

# 讓 `import trpg` 與同目錄 scripts 模組可用（從 repo root 或 scripts/ 啟動皆可）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from trpg.rl.env_v2 import CombatEnvV2  # noqa: E402
from trpg.rl.model import CombatPolicyNet  # noqa: E402
from trpg.engine.skill import available_skills  # noqa: E402
from trpg.engine.abilities import ABILITY_REGISTRY  # noqa: E402
from trpg.engine.items import WEAPON_DEFS  # noqa: E402
from trpg.engine.vec2 import Vec2, TerrainType  # noqa: E402
from gui_identity import (  # noqa: E402  身分組裝＝與 bug_miner 共用的唯一真源
    DMG_TYPES, RESIST_MODES, LAYOUTS, build_identity, identity_catalog,
)
from trpg.scenarios.monsters import register_monsters  # noqa: E402
import trpg.rl.neural_policy as _npmod  # noqa: E402
from trpg.rl.obs import ENEMY_SLOT_START, N_ENEMY_SLOTS  # noqa: E402
from trpg.rl.action import _entity_id_at_slot, decode_action  # noqa: E402
from trpg.rl.model import apply_resource_mask  # noqa: E402

register_monsters()  # 讓怪物 id 進入 CLASS_DEFS / ARCHETYPE_FACTORIES

N_GRID = 30
# DMG_TYPES / RESIST_MODES / LAYOUTS 及身分組裝（resolve_identity / identity_catalog
# / build_identity）已移到 scripts/gui_identity.py，與 bug_miner 共用同一份。

# 固定戰鬥紀錄檔（repo 根目錄），每場開戰覆蓋。有問題直接讓 Claude 看這個檔。
LOG_PATH = os.path.join(_ROOT, "combat_log.txt")


def _set_log_writable():
    if os.path.exists(LOG_PATH):
        try:
            os.chmod(LOG_PATH, stat.S_IWRITE | stat.S_IREAD)
        except Exception:
            pass


def _set_log_readonly():
    if os.path.exists(LOG_PATH):
        try:
            os.chmod(LOG_PATH, stat.S_IREAD)   # 唯讀 → 編輯器不會試圖存檔覆寫
        except Exception:
            pass


def describe_char(c) -> str:
    """把一個 Character 的完整身分攤平成多行字串，供詳細紀錄使用。"""
    weaps = ", ".join(f"{w.name}({w.damage_dice}/{w.damage_type}/{w.range_type})"
                      for w in getattr(c, "weapons", [])) or "—"
    abil = ", ".join(getattr(c, "known_abilities", [])) or "—"
    mults = getattr(c, "damage_multipliers", {}) or {}
    mult_s = ", ".join(f"{k}×{v:g}" for k, v in mults.items()) or "無"
    st = getattr(c, "stats", None)
    stat_s = ""
    if st is not None:
        stat_s = " ".join(f"{a}{getattr(st, a, '?')}"
                          for a in ("STR", "DEX", "CON", "INT", "WIS", "CHA"))
    slots = getattr(c, "spell_slots", {}) or {}
    extra = []
    for attr, lbl in (("attacks_per_action", "額外攻擊"),
                      ("pack_tactics", "集團戰術"),
                      ("undead_fortitude", "不死韌性"),
                      ("regeneration", "再生"),
                      ("sneak_attack_dice", "偷襲")):
        v = getattr(c, attr, None)
        if v:
            extra.append(f"{lbl}={v}")
    return (
        f"  名稱 {c.name}  職別 {getattr(c,'class_','?')}  Lv{c.level}\n"
        f"  HP {c.hp}/{c.max_hp}  AC {c.ac}  屬性 {stat_s}\n"
        f"  武器: {weaps}\n"
        f"  技能: {abil}\n"
        f"  施法位: {slots if slots else '無'}\n"
        f"  傷害倍率(抗性/免疫/易傷): {mult_s}\n"
        f"  其他天賦: {', '.join(extra) if extra else '無'}"
    )


# ───────────────────────── 對手政策：盲化 + 行動記錄 ─────────────────────────
# env.reset() 內部 `from .neural_policy import NeuralCombatPolicy` 在呼叫當下才
# 解析模組屬性，因此這裡先把模組屬性換成「強制 blind=True 並記錄決策」的子類，
# 之後 env.use_self_play_opponent(net) 建出來的對手政策就會是盲化版（符合 uni_v7
# 的訓練契約：自身 archetype one-hot 必須為零）。
_BaseNCP = _npmod.NeuralCombatPolicy


class GuiOppPolicy(_BaseNCP):
    sink = None  # callable(actor_id, actor, ws, decision)

    def __init__(self, net, device="cpu", blind=False):
        super().__init__(net, device=device, blind=True)

    def decide(self, actor_id, actor, world_state, resources, round_num):
        dec = super().decide(actor_id, actor, world_state, resources, round_num)
        if GuiOppPolicy.sink is not None:
            try:
                GuiOppPolicy.sink(actor_id, actor, world_state, dec)
            except Exception:
                pass
        return dec


_npmod.NeuralCombatPolicy = GuiOppPolicy


# 傳奇行動（boss 在別人回合結束時動作）發生在正常動作流之外，HP 會「無故」下降。
# 攔截 env_v2.run_legendary_actions，把它造成的 HP 變化回報給當前戰鬥畫面，
# 讓紀錄看得出「這是傳奇行動扣的血」。
import trpg.rl.env_v2 as _envmod  # noqa: E402
_orig_legendary = _envmod.run_legendary_actions
_LEG_SINK = {"fn": None}


def _traced_legendary(ws, cid, rnd):
    before = {k: v.hp for k, v in ws.characters.items()}
    r = _orig_legendary(ws, cid, rnd)
    fn = _LEG_SINK["fn"]
    if fn is not None:
        for k, v in ws.characters.items():
            if v.hp != before.get(k, v.hp):
                try:
                    fn(cid, k, before[k], v.hp)
                except Exception:
                    pass
    return r


_envmod.run_legendary_actions = _traced_legendary


def load_model(path: str):
    net = CombatPolicyNet(hidden=128)
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


# 身分組裝（resolve_identity / build_identity / identity_catalog）＝gui_identity.py


# ───────────────────────── 選身分面板 ─────────────────────────
class IdentityPanel(ttk.LabelFrame):
    def __init__(self, master, title, catalog):
        super().__init__(master, text=title, padding=8)
        self.catalog = catalog
        labels = [c[0] for c in catalog]

        ttk.Label(self, text="身分：").grid(row=0, column=0, sticky="w")
        self.id_var = tk.StringVar(value=labels[0])
        self.id_box = ttk.Combobox(self, textvariable=self.id_var,
                                   values=labels, width=34, state="readonly")
        self.id_box.grid(row=0, column=1, columnspan=3, sticky="we", pady=2)
        self.id_box.bind("<<ComboboxSelected>>", self._on_pick)

        ttk.Label(self, text="等級：").grid(row=1, column=0, sticky="w")
        self.level_var = tk.IntVar(value=5)
        ttk.Spinbox(self, from_=1, to=24, textvariable=self.level_var,
                    width=5).grid(row=1, column=1, sticky="w", pady=2)

        # 抗性/免疫/易傷（單一傷害型，作為 damage_table 天賦加掛）
        ttk.Label(self, text="抗性天賦：").grid(row=2, column=0, sticky="w")
        self.dmg_var = tk.StringVar(value=DMG_TYPES[0])
        ttk.Combobox(self, textvariable=self.dmg_var, values=DMG_TYPES,
                     width=6, state="readonly").grid(row=2, column=1, sticky="w")
        self.resist_var = tk.StringVar(value="無")
        ttk.Combobox(self, textvariable=self.resist_var,
                     values=list(RESIST_MODES.keys()), width=12,
                     state="readonly").grid(row=2, column=2, columnspan=2,
                                             sticky="w")

        # 常用天賦勾選
        self.t_pack = tk.BooleanVar()
        self.t_regen = tk.BooleanVar()
        self.t_undead = tk.BooleanVar()
        self.t_extra = tk.BooleanVar()
        trow = ttk.Frame(self)
        trow.grid(row=3, column=0, columnspan=4, sticky="w", pady=2)
        ttk.Checkbutton(trow, text="集團戰術", variable=self.t_pack).pack(side="left")
        ttk.Checkbutton(trow, text="再生(10)", variable=self.t_regen).pack(side="left")
        ttk.Checkbutton(trow, text="不死韌性", variable=self.t_undead).pack(side="left")
        ttk.Checkbutton(trow, text="額外攻擊", variable=self.t_extra).pack(side="left")

        # 額外技能多選
        ttk.Label(self, text="加掛技能(可多選)：").grid(row=4, column=0,
                                                  columnspan=4, sticky="w")
        sf = ttk.Frame(self)
        sf.grid(row=5, column=0, columnspan=4, sticky="we")
        self.skill_ids = sorted(ABILITY_REGISTRY.keys())
        self.skill_list = tk.Listbox(sf, selectmode="multiple", height=6,
                                     exportselection=False, width=40)
        for sid in self.skill_ids:
            disp = getattr(ABILITY_REGISTRY[sid], "display_name", "")
            self.skill_list.insert("end", f"{sid}  ({disp})")
        sb = ttk.Scrollbar(sf, orient="vertical", command=self.skill_list.yview)
        self.skill_list.config(yscrollcommand=sb.set)
        self.skill_list.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

    def _on_pick(self, _evt=None):
        idx = self.id_box.current()
        _, _id, is_mon, nat = self.catalog[idx]
        if is_mon:
            self.level_var.set(nat)  # 怪物預設帶到自然等級

    def selection(self):
        idx = self.id_box.current()
        _, base_id, _is_mon, _nat = self.catalog[idx]
        extra_skills = [self.skill_ids[i] for i in self.skill_list.curselection()]
        mult = RESIST_MODES[self.resist_var.get()]
        bool_traits = [k for k, v in (("pack_tactics", self.t_pack),
                                      ("regeneration", self.t_regen),
                                      ("undead_fortitude", self.t_undead),
                                      ("extra_attack", self.t_extra)) if v.get()]
        # 與 bug_miner 共用同一組裝器（gui_identity.build_identity）＝兩邊組合空間同源
        rid = build_identity(
            base_id, extra_skills,
            dmg_type=self.dmg_var.get() if mult is not None else None,
            resist_mult=mult, bool_traits=bool_traits)
        return rid, int(self.level_var.get())


# ───────────────────────── 戰鬥畫面 ─────────────────────────
class CombatScreen(ttk.Frame):
    CELL = 18

    def __init__(self, master, app, env, my_id, opp_id):
        super().__init__(master, padding=6)
        self.app = app
        self.env = env
        self.my_id = my_id
        self.opp_id = opp_id
        self.pending_skill = None   # (idx, target_type_name)
        self.aim_mode = False
        self._hp_snap = {}

        # 左：地圖
        left = ttk.Frame(self)
        left.pack(side="left", fill="y")
        sz = N_GRID * self.CELL
        self.canvas = tk.Canvas(left, width=sz, height=sz, bg="#101418",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.status = ttk.Label(left, text="", font=("Consolas", 10))
        self.status.pack(anchor="w", pady=3)

        # 右：資訊 + 技能 + 紀錄
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.info = ttk.Label(right, text="", font=("Consolas", 10),
                              justify="left")
        self.info.pack(anchor="w")
        ttk.Separator(right).pack(fill="x", pady=4)
        ttk.Label(right, text="你的招式（點選或按數字鍵）：").pack(anchor="w")
        self.skill_frame = ttk.Frame(right)
        self.skill_frame.pack(anchor="w", fill="x")
        btns = ttk.Frame(right)
        btns.pack(anchor="w", pady=4)
        ttk.Button(btns, text="結束回合 (E)", command=self.end_turn).pack(side="left")
        ttk.Button(btns, text="取消瞄準 (Esc)", command=self.cancel_aim).pack(side="left", padx=4)
        ttk.Button(btns, text="重新開始", command=self.app.show_setup).pack(side="left")
        ttk.Label(right, text="戰鬥紀錄：").pack(anchor="w", pady=(6, 0))
        logf = ttk.Frame(right)
        logf.pack(anchor="w", fill="both", expand=True)
        self.log = tk.Text(logf, width=46, height=16, font=("Consolas", 9),
                           state="disabled", bg="#0c0f12", fg="#cdd6df")
        lsb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        self.log.config(yscrollcommand=lsb.set)
        self.log.pack(side="left", fill="both", expand=True)
        lsb.pack(side="left", fill="y")

        # 鍵盤
        top = self.winfo_toplevel()
        top.bind("<Key>", self._on_key)

        # 固定詳細紀錄檔（覆蓋）。打仗時可寫、閒置時設唯讀，避免編輯器存檔衝突。
        self._logfile = None
        try:
            _set_log_writable()
            self._logfile = open(LOG_PATH, "w", encoding="utf-8")
            self._write_header()
        except Exception as e:
            print("無法開啟紀錄檔:", e)

        GuiOppPolicy.sink = self._log_opp
        _LEG_SINK["fn"] = self._log_legendary
        self._last_opp_turn = None   # (actor_id, round) 去重＝敵方回合分隔用
        self._snapshot_hp()
        self._log("=== 戰鬥開始 ===")
        self.refresh()

    def _write_header(self):
        ws = self.env.ws
        meta = getattr(self.app, "last_setup", {})
        me = ws.characters[self.my_id]
        opp = ws.characters[self.opp_id]
        order = ws.combat.initiative_order
        lines = [
            "=" * 60,
            "TRPG 1v1 戰鬥詳細紀錄 (combat_log.txt)",
            f"模型(對手, 盲化): {os.path.basename(self.app.model_path)}",
            f"地圖佈局: {meta.get('layout','?')}   "
            f"等級 你={meta.get('my_lvl','?')} 敵={meta.get('opp_lvl','?')}",
            f"身分 id  你={self.my_id}  敵={self.opp_id}",
            f"先攻順序: {order}",
            "-" * 60,
            "[你方]",
            describe_char(me),
            "[對手 — 由通用模型盲化駕駛]",
            describe_char(opp),
            "-" * 60,
            f"起始位置  你=({me.position.x:.1f},{me.position.y:.1f})  "
            f"敵=({opp.position.x:.1f},{opp.position.y:.1f})  "
            f"距離={me.position.distance_to(opp.position):.1f}m",
            "=" * 60,
            "",
        ]
        self._flog("\n".join(lines))

    def _flog(self, msg):
        """只寫檔（不上畫面）的詳細紀錄。"""
        if self._logfile is not None:
            try:
                self._logfile.write(msg + "\n")
                self._logfile.flush()
            except Exception:
                pass

    # ---- 紀錄 ----
    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")
        self._flog(msg)   # 同步寫入詳細紀錄檔

    def _cname(self, cid):
        # 注意：不可命名為 _name —— Tkinter widget 內部會佔用 self._name (字串)。
        c = self.env.ws.characters.get(cid)
        side = "你" if cid in self.env.agent_ids else "敵"
        return f"{side}:{c.name}" if c else cid

    def _state_line(self):
        ws = self.env.ws
        me = ws.characters[self.my_id]
        opp = ws.characters[self.opp_id]
        d = me.position.distance_to(opp.position)
        r = self.env.resources
        return (
            f"狀態 你 HP{max(0,me.hp)}/{me.max_hp}@({me.position.x:.1f},"
            f"{me.position.y:.1f})[{self._statuses(me)}] | "
            f"敵 HP{max(0,opp.hp)}/{opp.max_hp}@({opp.position.x:.1f},"
            f"{opp.position.y:.1f})[{self._statuses(opp)}] | 距離{d:.1f}m | "
            f"資源 動作{r.get('action',0)}/附贈{r.get('bonus_action',0)}/"
            f"移動{r.get('movement',0):.1f}"
        )

    def _log_opp(self, actor_id, actor, ws, dec):
        if dec is None:
            return
        # 一個 agent-step 內環境會連跑好幾個對手回合（你被石化/束縛時尤其明顯）。
        # decide 每個子動作呼叫一次，同一回合的 (actor_id, round) 相同 → 只在回合
        # 真正換人/換輪時插分隔線，避免多回合擠成一坨看不清（眼魔一回合 1 次
        # eye_rays＝3 道，連打幾輪才是好幾個 eye_rays）。
        rnd = ws.combat.round_number if ws.combat else -1
        key = (actor_id, rnd)
        if key != getattr(self, "_last_opp_turn", None):
            self._last_opp_turn = key
            hp = ws.characters[actor_id].hp if actor_id in ws.characters else "?"
            self._log(f"  --- 敵方回合 round={rnd}  {self._cname(actor_id)} "
                      f"HP{hp} ---")
        if getattr(dec, "action", None):
            ad = dec.action
            sk = ad.get("skill_id") or ad.get("type", "?")
            tgt = ad.get("target")
            tname = self._cname(tgt) if tgt else ""
            self._log(f"  [模型] {self._cname(actor_id)} → {sk} {tname}".rstrip())

    def _log_legendary(self, end_cid, victim_cid, old, new):
        verb = "HP" if new < old else "HP↑"
        self._log(f"  [傳奇行動] {self._cname(end_cid)}回合結束觸發 → "
                  f"{self._cname(victim_cid)} {verb} {old}→{new}")

    def _snapshot_hp(self):
        self._hp_snap = {cid: c.hp for cid, c in self.env.ws.characters.items()}

    def _log_hp_delta(self):
        for cid, c in self.env.ws.characters.items():
            old = self._hp_snap.get(cid, c.hp)
            if c.hp != old:
                self._log(f"     {self._cname(cid)} HP {old}→{c.hp}")
        self._snapshot_hp()

    # ---- 互動 ----
    def _enemy_slot(self):
        for s in range(ENEMY_SLOT_START, ENEMY_SLOT_START + N_ENEMY_SLOTS):
            eid = _entity_id_at_slot(self.env.ws, self.my_id, s)
            if eid and self.env.ws.characters[eid].is_alive():
                return s
        return ENEMY_SLOT_START

    def _do_step(self, action):
        if self.app.done:
            return
        ws = self.env.ws
        cid = self.env.current_agent_id
        rnd = ws.combat.round_number
        # 解碼（不改狀態）僅供詳細紀錄，看清這個三元組實際變成什麼動作
        try:
            decoded = decode_action(action, ws, cid) if cid in self.env.agent_ids else None
        except Exception as e:
            decoded = f"<decode error: {e}>"
        self._flog(f"\n--- step | round={rnd} 你方動作 action={tuple(action)} ---")
        self._flog(f"  decoded = {decoded if decoded is not None else 'END(結束回合)'}")
        self._snapshot_hp()
        obs, reward, term, trunc, info = self.env.step(action)
        # 玩家自身動作結果
        res = (info or {}).get("action_result")
        if res and isinstance(res, dict):
            self._flog(f"  result = {res}")
            rtype = res.get("type", "")
            if rtype == "ERROR":
                self._log(f"  [你] 動作無法執行：{res.get('reason', res.get('msg', ''))}")
            elif rtype:
                self._log(f"  [你] {rtype} {res.get('detail', '')}".rstrip())
        self._log_hp_delta()
        self._flog("  " + self._state_line())
        self._flog(f"  reward={reward:.3f} term={term} trunc={trunc} "
                   f"下一個行動者={self.env.current_agent_id}")
        self.pending_skill = None
        self.aim_mode = False
        if term or trunc:
            self._finish(term, trunc)
        self.refresh()

    def end_turn(self):
        # decode_action 把「超出範圍的 skill_idx」視為結束回合（它不接受 None）。
        if not self.app.done:
            self._log("-- 你結束回合 --")
            self._do_step((999, 0, 0))

    def cancel_aim(self):
        self.pending_skill = None
        self.aim_mode = False
        self.refresh()

    def _on_key(self, evt):
        if self.app.done:
            return
        k = evt.keysym.lower()
        if k in ("e", "return"):
            self.end_turn()
        elif k == "escape":
            self.cancel_aim()
        elif k.isdigit():
            n = int(k)
            if 1 <= n <= len(self._skill_buttons):
                self._skill_buttons[n - 1].invoke()

    def _on_canvas_click(self, evt):
        if self.app.done or not self.aim_mode or self.pending_skill is None:
            return
        ix = max(0, min(N_GRID - 1, int(evt.x / self.CELL)))
        iy = max(0, min(N_GRID - 1, int(evt.y / self.CELL)))
        cell = ix * N_GRID + iy   # 與 action._grid_cell_to_xy 對應
        idx, _tt = self.pending_skill
        self._do_step((idx, 0, cell))

    def _legal_skill_flags(self, skills):
        """重用模型用的 apply_resource_mask 算出每個可用招式本回合是否合法
        （資源成本/移動力/射程/視線）。人類點招也須守同一套規則，否則會像
        繞過遮罩、一回合內無限次攻擊。回傳與 skills 等長的 bool list。"""
        n = len(skills)
        try:
            zeros = torch.zeros(1, n)
            masked = apply_resource_mask(zeros, self.env.resources,
                                         self.env.ws, self.my_id)
            return [masked[0, i].item() > -1e8 for i in range(n)]
        except Exception:
            return [True] * n

    def _pick_skill(self, idx, ttn, label):
        # 資源/合法性檢查：付不起或非法的招直接擋下（與模型同規則）
        skills = available_skills(self.env.ws.characters[self.my_id], self.env.ws)
        flags = self._legal_skill_flags(skills)
        if idx < len(flags) and not flags[idx]:
            self._log(f"  「{label}」現在不可用（資源不足/超出射程/無視線）")
            return
        # 依目標型態決定後續流程
        if "SELF" in ttn or "ALLY" in ttn:        # 自身/友方 → 1v1 指向自己
            self._do_step((idx, 0, 0))
        elif any(t in ttn for t in ("POINT", "LINE", "CONE", "AREA")) or label == "move":
            self.aim_mode = True
            self.pending_skill = (idx, ttn)
            self._log(f"  選了「{label}」：請點地圖選定位置")
            self.refresh()
        else:                                     # 單體 → 鎖定唯一敵人
            self._do_step((idx, self._enemy_slot(), 0))

    # ---- 結束 ----
    def _finish(self, term, trunc):
        self.app.done = True
        me = self.env.ws.characters[self.my_id]
        opp = self.env.ws.characters[self.opp_id]
        if trunc:
            txt = "回合耗盡，平手 / 超時"
        elif me.is_alive() and not opp.is_alive():
            txt = "★ 你贏了！ ★"
        elif opp.is_alive() and not me.is_alive():
            txt = "✗ 你被通用模型擊敗 ✗"
        else:
            txt = "同歸於盡（雙方倒下）"
        self._log("=== " + txt + " ===")
        self._flog("\n" + "=" * 60)
        self._flog(f"終局: {txt}")
        self._flog(f"  你 {me.name} HP {max(0,me.hp)}/{me.max_hp} alive={me.is_alive()}")
        self._flog(f"  敵 {opp.name} HP {max(0,opp.hp)}/{opp.max_hp} alive={opp.is_alive()}")
        self._flog(f"  總步數 step_count={self.env._step_count}  "
                   f"round={self.env.ws.combat.round_number}")
        self._flog("=" * 60)
        if self._logfile is not None:
            try:
                self._logfile.close()
            except Exception:
                pass
            self._logfile = None
        _LEG_SINK["fn"] = None
        _set_log_readonly()
        messagebox.showinfo("戰鬥結束", txt)

    # ---- 繪製 ----
    def refresh(self):
        self._draw_map()
        ws = self.env.ws
        me = ws.characters[self.my_id]
        opp = ws.characters[self.opp_id]
        cur = self.env.current_agent_id
        turn = "你的回合" if cur in self.env.agent_ids else "模型回合"
        res = self.env.resources
        self.info.config(text=(
            f"回合數 round={ws.combat.round_number}   {turn}\n"
            f"資源  動作={res.get('action',0)} 附贈={res.get('bonus_action',0)} "
            f"移動={res.get('movement',0):.1f}m\n\n"
            f"你 {me.name}  HP {max(0,me.hp)}/{me.max_hp}  AC{me.ac}  "
            f"Lv{me.level}\n"
            f"   狀態: {self._statuses(me)}\n"
            f"敵 {opp.name}  HP {max(0,opp.hp)}/{opp.max_hp}  AC{opp.ac}  "
            f"Lv{opp.level}\n"
            f"   狀態: {self._statuses(opp)}"
        ))
        self.status.config(text=("瞄準中：點地圖格選位置（Esc 取消）"
                                 if self.aim_mode else ""))
        self._build_skill_buttons()

    def _statuses(self, c):
        try:
            ss = [getattr(s, "name", str(s)) for s in getattr(c, "status_effects", [])]
        except Exception:
            ss = []
        return ", ".join(ss) if ss else "—"

    def _build_skill_buttons(self):
        for w in self.skill_frame.winfo_children():
            w.destroy()
        self._skill_buttons = []
        ws = self.env.ws
        if self.app.done or self.env.current_agent_id not in self.env.agent_ids:
            ttk.Label(self.skill_frame, text="(等待模型行動…)").pack(anchor="w")
            return
        skills = available_skills(ws.characters[self.my_id], ws)
        flags = self._legal_skill_flags(skills)   # 本回合資源/射程允許的招
        n = 0
        for idx, s in enumerate(skills):
            sid = getattr(s, "skill_id", str(idx))
            if sid == "end":
                continue
            tt = getattr(getattr(s, "features", None), "target_type", None)
            ttn = getattr(tt, "name", "") or str(tt)
            disp = getattr(s, "display_name", "") or sid
            n += 1
            legal = flags[idx] if idx < len(flags) else True
            label = f"{n}. {disp}" + ("" if legal else "（不可用）")
            b = ttk.Button(self.skill_frame, text=label,
                           state=("normal" if legal else "disabled"),
                           command=lambda i=idx, t=ttn, l=sid: self._pick_skill(i, t, l))
            b.pack(anchor="w", fill="x", pady=1)
            self._skill_buttons.append(b)

    def _draw_map(self):
        cv = self.canvas
        cv.delete("all")
        bf = self.env.ws.combat.battlefield
        cs = self.CELL
        for ix in range(N_GRID):
            for iy in range(N_GRID):
                x0, y0 = ix * cs, iy * cs
                color = "#171c22"
                try:
                    t = bf.terrain_at(Vec2(ix + 0.5, iy + 0.5))
                    if t == TerrainType.BLOCKED:
                        color = "#5a5f66"
                    elif t == getattr(TerrainType, "DIFFICULT", None):
                        color = "#3a3326"
                    elif t == getattr(TerrainType, "DANGEROUS", None):
                        color = "#46201c"
                except Exception:
                    pass
                cv.create_rectangle(x0, y0, x0 + cs, y0 + cs,
                                    fill=color, outline="#262c33")
        # 角色
        for cid, c in self.env.ws.characters.items():
            if not c.is_alive():
                continue
            ix = int(c.position.x)
            iy = int(c.position.y)
            cx = (ix + 0.5) * cs
            cy = (iy + 0.5) * cs
            r = cs * 0.42
            mine = cid in self.env.agent_ids
            fill = "#3a86ff" if mine else "#ef476f"
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill,
                           outline="white", width=2)
            cv.create_text(cx, cy, text=("我" if mine else "敵"),
                           fill="white", font=("Consolas", 9, "bold"))
            # HP 條
            frac = max(0.0, c.hp) / max(1, c.max_hp)
            bw = cs * 1.4
            bx = cx - bw / 2
            by = cy - r - 6
            cv.create_rectangle(bx, by, bx + bw, by + 4, fill="#222", outline="")
            cv.create_rectangle(bx, by, bx + bw * frac, by + 4,
                                fill="#06d6a0" if mine else "#ffd166", outline="")


# ───────────────────────── App ─────────────────────────
class App(tk.Tk):
    def __init__(self, model_path):
        super().__init__()
        self.title("TRPG 1v1 對戰測試台 — 你 vs 通用模型")
        self.model_path = model_path
        self.net = load_model(model_path)
        self.catalog = identity_catalog()
        self.done = False
        self.frame = None
        self.show_setup()

    def _clear(self):
        if self.frame is not None:
            self.frame.destroy()
        self.frame = None

    def show_setup(self):
        self._clear()
        self.done = False
        f = ttk.Frame(self, padding=10)
        f.pack(fill="both", expand=True)
        self.frame = f
        ttk.Label(f, text=f"對手模型：{os.path.basename(self.model_path)}（盲化）",
                  font=("Consolas", 10)).grid(row=0, column=0, columnspan=2,
                                              sticky="w", pady=(0, 6))
        self.p_me = IdentityPanel(f, "你方（人類操作）", self.catalog)
        self.p_me.grid(row=1, column=0, sticky="nsew", padx=4)
        self.p_opp = IdentityPanel(f, "對手（通用模型駕駛）", self.catalog)
        self.p_opp.grid(row=1, column=1, sticky="nsew", padx=4)

        bottom = ttk.Frame(f)
        bottom.grid(row=2, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(bottom, text="地圖佈局：").pack(side="left")
        self.layout_var = tk.StringVar(value="open")
        ttk.Combobox(bottom, textvariable=self.layout_var, values=LAYOUTS,
                     width=10, state="readonly").pack(side="left")
        ttk.Button(bottom, text="開始戰鬥 ▶", command=self.start_combat).pack(
            side="left", padx=12)

    def start_combat(self):
        try:
            my_id, my_lvl = self.p_me.selection()
            opp_id, opp_lvl = self.p_opp.selection()
        except Exception as e:
            messagebox.showerror("身分設定錯誤", str(e))
            return
        env = CombatEnvV2(n_agents=1, n_opps=1)
        GuiOppPolicy.sink = None
        env.use_self_play_opponent(self.net)
        try:
            env.reset(level=my_lvl, opp_level=opp_lvl,
                      layout=self.layout_var.get(),
                      agent_archs=[my_id], opp_archs=[opp_id])
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("開戰失敗", str(e))
            return
        # 關掉前一場可能還開著的紀錄檔
        old = getattr(self, "frame", None)
        lf = getattr(old, "_logfile", None)
        if lf is not None:
            try:
                lf.close()
            except Exception:
                pass
        self.last_setup = {"layout": self.layout_var.get(),
                           "my_lvl": my_lvl, "opp_lvl": opp_lvl}
        self._clear()
        self.done = False
        # 對戰雙方在 initiative 洗牌後，agent 一定是 agent_0、opp 是 opp_0
        cs = CombatScreen(self, self, env, env.agent_ids[0], env.opp_ids[0])
        cs.pack(fill="both", expand=True)
        self.frame = cs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "models", "unified", "uni_v7.pt")
    if not os.path.exists(path):
        print(f"找不到模型：{path}")
        sys.exit(1)
    App(path).mainloop()


if __name__ == "__main__":
    main()
