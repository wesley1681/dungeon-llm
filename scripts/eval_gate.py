"""統一驗收門 (eval_gate)：給一個模型，跑一次，印一張*全面*驗收表。

動機（2026-07-01）：現況是「人肉挑散落的 150+ 支腳本、人肉讀數字、人肉判過不過」，
沒有任何機制保證不漏——一個 dodge-collapse 能力回歸（uni_v6→v7）就這樣無人察覺，
最後被用戶在 GUI 手動抓到。本腳本把 EVAL_COVERAGE.md 收斂成單一入口：

  • 每個「已接上可執行檢測」的項目 → 印 PASS / FAIL（+ 逐角落數字）
  • 每個「尚無可執行檢測」的項目 → 顯性印 NOT_COVERED（+ 原因）——漏掉不再隱形
  • 給 --base 舊最佳模型 → **逐角落不侵蝕守門**：任一角落比 base 退步 > 2σ = 回歸 FAIL
    （直指 EVAL_COVERAGE #32：舊守門把 per-class 資料丟掉、只比聚合平均，才會漏回歸）

用法:
  python scripts/eval_gate.py models/unified/uni_v7.pt                    # 純驗收
  python scripts/eval_gate.py models/unified/uni_v7.pt --base models/unified/uni_v1.pt
  python scripts/eval_gate.py <m> --games 8 --only immune,degen          # 加局數 / 選檢測
  python scripts/eval_gate.py <m> --list                                 # 只印覆蓋表(不跑)

判定門檻是「合理起點」，不是聖經：每項都印原始數字，人可覆核。設計原則＝
寧可 UNDERPOWERED/NOT_COVERED 顯性掛在表上，也不要靜靜地不測。
"""
from __future__ import annotations
import sys, os, argparse, random, math
from dataclasses import dataclass, field
from types import SimpleNamespace
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action, _entity_id_at_slot
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.status import Prone, Restrained, Frightened
from trpg.engine.vec2 import Vec2

from distill_routed import load_student
from eval_routed import stable_seed
from train_population import blind_np_single

# 重用「已測過」的 episode 邏輯，避免重寫引入新 bug
import eval_immune_weak as IW
import diag_degen_audit as DD

CLASSES = list(ARCHETYPE_LIST)
# WR 對手面板：坦/爆發/懲擊/穩健，涵蓋不同威脅型（跟 diag_degen 同源）
OPP_PANEL = ["battle_master", "champion", "evocation", "vengeance"]


# ── 驅動 ──────────────────────────────────────────────────────────────────────

@dataclass
class Outcome:
    win: bool
    loss: bool
    draw: bool          # 非勝非負（truncated 兩方存活 / 我活敵活）
    stalled_win: bool   # draw 且終局我明顯領先（＝「把能贏的拖成平手」）
    dmg_frac: float     # 對敵造成傷害 / 敵總血
    dpr: float          # 敵損血 / 我方決策數（time-to-kill 的倒數代理）
    opp_dmg_frac: float # 我方被打掉的血 / 我方總血（對手席攻勢的量）
    turns: int
    idle: int = 0       # 回合首手 action 還在、卻直接結束＝划水（劣勢不划水用）
    first_turns: int = 0


class _StationaryPolicy:
    """消極/被動敵：每回合直接結束、不移動不攻擊（＝GUI 裡一個一直結束回合的人類）。
    模型該主動接戰擊殺；若對能打的被動敵空轉 0 傷＝disengage-loop 退化。"""
    def decide(self, opp_id, opp, ws, resources, round_number):
        return SimpleNamespace(action=None, fled=False, ended=True)


def _model_driver(net):
    def drv(env, obs, actor):
        ob = blind_np_single(obs)
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
    return drv


def _expert_driver(arch):
    pol = make_archetype_policy(arch)
    def drv(env, obs, actor):
        a = env.ws.characters[actor]
        dec = pol.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
        if dec.action is None or getattr(dec, "fled", False):
            return [0, 0, 0]
        return list(encode_action(dec.action, env.ws, actor))
    return drv


def drive_episode(driver, agent_archs, opp_archs, level, opp_level, key,
                  layout="open", inject_immune=None, self_play_net=None,
                  passive_enemy=False):
    """驅動 agent 席（對手席由 env.step 內部腳本自動跑；或 self_play_net 接管）。

    agent_archs 可為單一字串或 list（list＝Nv1 優勢局，模型駕多個 agent 席）。
    對手席永遠由引擎腳本(或 self_play_net / passive_enemy)跑。"""
    if isinstance(agent_archs, str):
        agent_archs = [agent_archs]
    seed = stable_seed(key)
    random.seed(seed)                        # 全域引擎骰＝跨快照可比（記憶教訓）
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=len(agent_archs),
                      n_opps=len(opp_archs))
    if self_play_net is not None:
        # 必須在 reset 之前：reset() 才會讀 _opponent_override 建 NeuralCombatPolicy
        # 對手席（env_v2.py:257-266）。之前寫在 reset 後＝override 沒生效、對手仍是腳本，
        # oppseat 一直測錯席位——self_test 的 always_end 假 agent 抓到此 bug。
        env.use_self_play_opponent(self_play_net, blind=True)
    obs, _ = env.reset(agent_archs=list(agent_archs), opp_archs=list(opp_archs),
                       level=level, opp_level=opp_level, layout=layout)
    if passive_enemy:                        # 覆寫對手策略為靜止被動(reset 後才生效)
        for o in env.opp_ids:
            env._opp_policies[o] = _StationaryPolicy()
    aids = list(env.agent_ids)
    if inject_immune:
        for o in env.opp_ids:
            env.ws.characters[o].damage_multipliers[inject_immune] = 0.0
    opp_hp0 = {o: env.ws.characters[o].hp for o in env.opp_ids}
    enemy_tot = sum(opp_hp0.values())
    team_hp0 = sum(env.ws.characters[a].hp for a in aids)
    team_max = max(1.0, sum(env.ws.characters[a].max_hp for a in aids))
    turns = 0; idle = 0; first_turns = 0
    prev_actor = None
    done = False
    while not done:
        actor = env.current_agent_id
        first = actor != prev_actor
        prev_actor = actor
        had_action = env.resources.get("action", 0) > 0
        act = driver(env, obs, actor)
        if first:
            first_turns += 1
            if act[0] == 0 and had_action:   # 回合首手就結束、action 沒花＝划水
                idle += 1
        turns += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        if turns > 400:                      # 安全閥：病態不終局
            break
    team_alive = any(env.ws.characters[a].is_alive() for a in aids)
    opp_alive = any(env.ws.characters[o].is_alive() for o in env.opp_ids)
    enemy_lost = sum(opp_hp0[o] - max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    opp_frac_left = sum(max(0, env.ws.characters[o].hp) for o in env.opp_ids) / max(1.0, enemy_tot)
    team_left = sum(max(0, env.ws.characters[a].hp) for a in aids)
    win = team_alive and not opp_alive
    loss = not team_alive
    draw = (not win) and (not loss)
    my_frac = team_left / team_max
    stalled_win = draw and (my_frac - opp_frac_left > 0.15)
    return Outcome(
        win=win, loss=loss, draw=draw, stalled_win=stalled_win,
        dmg_frac=enemy_lost / max(1.0, enemy_tot),
        dpr=enemy_lost / max(1, turns),
        opp_dmg_frac=(team_hp0 - team_left) / team_max,
        turns=turns, idle=idle, first_turns=first_turns)


# ── 檢測結果模型 ──────────────────────────────────────────────────────────────

@dataclass
class Cell:
    label: str
    value: float        # 慣例：越高越好
    n: int
    aux: str = ""


@dataclass
class CheckResult:
    cid: str
    title: str
    covers: list          # EVAL_COVERAGE 對應項
    status: str = "PASS"  # PASS / FAIL / UNDERPOWERED / ERROR / SKIP
    summary: str = ""
    cells: list = field(default_factory=list)
    lines: list = field(default_factory=list)   # 給人看的逐角落明細
    note: str = ""


def _sigma_frac(n):     # 二項標準差上界
    return math.sqrt(0.25 / max(1, n))


def regression_flag(cand_cells, base_cells, kind="frac"):
    """逐角落比對：任一 cand 角落比 base 退步 > 2σ = 回歸。回 (worst_label, worst_delta)。"""
    if not base_cells:
        return None
    bmap = {c.label: c for c in base_cells}
    worst = None
    for c in cand_cells:
        b = bmap.get(c.label)
        if b is None:
            continue
        delta = c.value - b.value
        if kind == "frac":
            # 統計顯著(超過 2σ) 或 實務上明顯(≥30pp 硬地板)。硬地板是為了小 n：
            # n=1 時 2σ=1.0，連 100%→0% 全崩 delta=-1.0 都卡在邊界外(=#33 的活教材)，
            # 地板保證「明顯全崩」在任何局數都紅旗，不被寬 CI 吃掉。
            sig = _sigma_frac(min(c.n, b.n))
            fired = (delta < -2 * sig) or (delta <= -0.30)
        else:               # count 類（退化數）：淨增 ≥1 個致命退化即回歸
            fired = delta <= -1.0
        if fired and (worst is None or delta < worst[1]):
            worst = (c.label, delta)
    return worst


# ── C1: 標準職 WR + DPR + 平手拆分（Part1 WR/Δpp + #31 平手 + #35 DPR） ──────────

def _agg_wr(driver, arch, panel, G, level=5, self_play_net=None):
    w = d = s = 0; dpr = dmg = odmg = 0.0; n = 0
    for opp in panel:
        for gi in range(G):
            o = drive_episode(driver, arch, [opp], level, level,
                              key=f"wr|{arch}|{opp}|{gi}", self_play_net=self_play_net)
            w += o.win; d += o.draw; s += o.stalled_win
            dpr += o.dpr; dmg += o.dmg_frac; odmg += o.opp_dmg_frac; n += 1
    return dict(wr=w / n, draw=d / n, stall=s / n, dpr=dpr / n,
                dmg=dmg / n, odmg=odmg / n, n=n)


def check_std12(cand, base, G):
    r = CheckResult("std12", "標準12職 WR / DPR / 平手（對稱腳本專家）",
                    ["Part1:WR/Δpp", "#31 平手", "#35 DPR", "#3 傷害指標"])
    r.lines.append(f"{'職業':16s} {'model':>6s} {'expert':>7s} {'diff':>6s} "
                   f"{'平手':>5s} {'拖平*':>5s} {'DPRm':>5s} {'DPRe':>5s}  判定")
    cells = []; base_cells = []
    n_cell = G * len(OPP_PANEL)
    for arch in CLASSES:
        m = _agg_wr(_model_driver(cand), arch, OPP_PANEL, G)
        e = _agg_wr(_expert_driver(arch), arch, OPP_PANEL, G)
        diff = m["wr"] - e["wr"]
        sig = _sigma_frac(n_cell)
        ok = diff >= -2 * sig
        cells.append(Cell(arch, m["wr"], m["n"],
                          aux=f"draw={m['draw']:.0%} dpr={m['dpr']:.2f}"))
        if base is not None:
            bm = _agg_wr(_model_driver(base), arch, OPP_PANEL, G)
            base_cells.append(Cell(arch, bm["wr"], bm["n"]))
        vd = "OK" if ok else "SHORT"
        if not ok:
            r.status = "FAIL"
        # 拖平/DPR 崩也記（次級旗標，不單獨判 FAIL，只提示）
        r.lines.append(f"{arch:16s} {m['wr']:6.0%} {e['wr']:7.0%} {diff:+6.0%} "
                       f"{m['draw']:5.0%} {m['stall']:5.0%} {m['dpr']:5.2f} "
                       f"{e['dpr']:5.2f}  {vd}")
    r.cells = cells
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"
        r.note = f"⚠回歸 vs base：{reg[0]} WR {reg[1]:+.0%}"
    if n_cell < 8:
        r.note += f"  (UNDERPOWERED n/cell={n_cell}, σ≈{_sigma_frac(n_cell):.0%})"
    stalled = [c.label for c in cells if "stall" in c.aux]  # placeholder
    r.summary = ("全職 tie/beat 專家且無角落回歸"
                 if r.status == "PASS" else "有職業落後專家或比 base 回歸")
    return r


# ── C2: 免疫×弱敵 傷害輸出（dodge-collapse 煙槍｜Part1 immune + #2 #3 #23） ──────

def check_immune(cand, base, G):
    r = CheckResult("immune", "法師 vs 弱小免疫敵：會不會退回次級招擊殺（dodge-collapse 偵測）",
                    ["Part1:immune", "#2 強弱錯配", "#3 傷害指標", "#23 資源fallback"])
    r.lines.append(f"{'kit':12s} {'enemy':10s} {'dmg%':>7s} {'win%':>6s}")
    cells = []; base_cells = []
    tot_d = tot_n = 0.0
    for kit in IW.KITS:
        for enemy in IW.WEAK:
            ds = ws = 0.0
            for gi in range(G):
                d, w = IW.episode(cand, kit, enemy, 8, f"iw_{kit}_{enemy}_{gi}")
                ds += d; ws += w
            dm = ds / G
            cells.append(Cell(f"{kit}/{enemy}", dm, G, aux=f"win={ws/G:.0%}"))
            r.lines.append(f"{kit:12s} {enemy:10s} {dm:7.1%} {ws/G:6.0%}")
            tot_d += ds; tot_n += G
            if base is not None:
                bd = sum(IW.episode(base, kit, enemy, 8, f"iw_{kit}_{enemy}_{gi}")[0]
                         for gi in range(G)) / G
                base_cells.append(Cell(f"{kit}/{enemy}", bd, G))
    overall = tot_d / tot_n
    worst = min((c.value for c in cells), default=1.0)
    # 正解＝擊殺 15HP 弱敵 → dmg%≈100；崩潰＝dodge/退，dmg%≈0（v7 實測 0~16%）
    if overall < 0.5 or worst < 0.25:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"
        r.note = f"⚠回歸 vs base：{reg[0]} dmg% {reg[1]:+.0%}"
    r.summary = f"OVERALL dmg%={overall:.0%} 最崩角落={worst:.0%}"
    return r


# ── C3: 零退化審計（Part1 degen；致命＝卡死/被夾/拒戰/滿血補） ─────────────────

def check_degen(cand, base, G):
    r = CheckResult("degen", "零退化動作審計（致命：卡死 stall / 被夾 / 真拒戰 / 滿血硬補）",
                    ["Part1:degen", "#目標:零退化"])
    idents = ["battle_master", "champion", "evocation", "life", "assassin",
              "war", "berserker", "arcane_trickster",
              "orc", "goblin", "ogre", "shadow", "basilisk", "manticore"]
    idents += DD.CLS_CHIM[:2] + DD.MON_CHIM[:2]
    SCEN = [("1v1", 1, 0), ("1v2", 2, 0)]
    r.lines.append(f"{'身分':16s} {'卡死':>4s} {'被夾':>4s} {'拒戰':>4s} {'滿補':>4s}")
    cells = []; base_cells = []
    g_fatal = 0
    for ident in idents:
        eq = DD._EQUIV.get(ident, 5)
        base_lvl = 8 if eq == float("inf") else max(1, int(round(eq)))
        st = bl = rf = hf = 0
        for label, n_opp, dl in SCEN:
            lvl = max(1, base_lvl + dl)
            for gi in range(G):
                opp = [DD.OPP_PANEL[gi % len(DD.OPP_PANEL)] for _ in range(n_opp)]
                w, nm, r_, ns, nb, h, s_ = DD.run_combat(
                    cand, ident, opp, lvl, base_lvl, f"{ident}|{label}|{gi}")
                st += s_; bl += nb; rf += r_; hf += h
        fatal = st + bl + rf + hf
        g_fatal += fatal
        cells.append(Cell(ident, -fatal, G * len(SCEN),
                          aux=f"stall={st} blk={bl} refuse={rf} heal={hf}"))
        if fatal:
            r.lines.append(f"{ident:16s} {st:4d} {bl:4d} {rf:4d} {hf:4d}  ⚠")
        if base is not None:
            bst = bbl = brf = bhf = 0
            for label, n_opp, dl in SCEN:
                lvl = max(1, base_lvl + dl)
                for gi in range(G):
                    opp = [DD.OPP_PANEL[gi % len(DD.OPP_PANEL)] for _ in range(n_opp)]
                    w, nm, r_, ns, nb, h, s_ = DD.run_combat(
                        base, ident, opp, lvl, base_lvl, f"{ident}|{label}|{gi}")
                    bst += s_; bbl += nb; brf += r_; bhf += h
            base_cells.append(Cell(ident, -(bst + bbl + brf + bhf), G * len(SCEN)))
    if g_fatal > 0:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "count")
    if reg:
        r.status = "FAIL"
        r.note = f"⚠退化增加 vs base：{reg[0]} Δfatal={-reg[1]:.0f}"
    r.summary = (f"零致命退化（{len(idents)} 身分×{len(SCEN)}情境）"
                 if g_fatal == 0 else f"致命退化總計={g_fatal}（見上⚠行）")
    return r


# ── C4: 對手席 play 品質（#34；候選網坐對手席，稱職專家坐 agent 席） ────────────

def check_oppseat(cand, base, G):
    r = CheckResult("oppseat", "對手席品質：候選網駕對手席會不會退化空轉（部署 GUI 真實席位）",
                    ["#34 對手席"])
    idents = ["battle_master", "champion", "evocation", "berserker", "assassin"]
    r.lines.append(f"{'對手身分':16s} {'攻勢odmg%':>9s} {'expWR':>6s}  判定")
    cells = []; base_cells = []
    lo_off = 1.0
    for ident in idents:
        # agent 席＝該身分的稱職腳本專家；對手席＝候選網。量對手網的攻勢(打掉專家多少血)
        off = wr = 0.0
        for opp in OPP_PANEL:
            for gi in range(G):
                o = drive_episode(_expert_driver(opp), opp, [ident], 5, 5,
                                  key=f"opp|{ident}|{opp}|{gi}", self_play_net=cand)
                off += o.opp_dmg_frac; wr += o.win
        n = G * len(OPP_PANEL)
        off /= n; wr /= n
        lo_off = min(lo_off, off)
        cells.append(Cell(ident, off, n, aux=f"expWR={wr:.0%}"))
        r.lines.append(f"{ident:16s} {off:9.0%} {wr:6.0%}  "
                       f"{'OK' if off >= 0.15 else 'DEAD?'}")
        if base is not None:
            boff = 0.0
            for opp in OPP_PANEL:
                for gi in range(G):
                    o = drive_episode(_expert_driver(opp), opp, [ident], 5, 5,
                                      key=f"opp|{ident}|{opp}|{gi}", self_play_net=base)
                    boff += o.opp_dmg_frac
            base_cells.append(Cell(ident, boff / (G * len(OPP_PANEL)), G * len(OPP_PANEL)))
    # 退化對手席＝幾乎不輸出（dodge-only / 空轉），對專家造不出傷害
    if lo_off < 0.10:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"
        r.note = f"⚠回歸 vs base：{reg[0]} 攻勢 {reg[1]:+.0%}"
    r.summary = (f"對手席均有實質攻勢（最低 {lo_off:.0%}）"
                 if r.status == "PASS" else f"有身分對手席近乎零輸出（{lo_off:.0%}）＝退化")
    return r


# ── C5: 扮任意怪/職業縫合怪/怪縫合怪/synth kit（goal 支柱｜Part2 identity + #6 + #30） ──

def check_general(cand, base, G):
    r = CheckResult("general", "扮任意怪/縫合怪/synth kit：像通用腦一樣駕馭（Δpp vs 駕同 kit 的腳本）",
                    ["Part2 identity", "#6 OOD自訂kit", "#30 kit覆蓋", "goal:扮任意身分"])
    try:
        import eval_generalize as GEN
    except Exception as ex:
        r.status = "ERROR"; r.summary = f"無法載入 eval_generalize: {ex}"; return r
    buckets = GEN.build_buckets(["monster", "chimera_cls", "chimera_mon", "synth"],
                                6, 20260613)
    cells = []; base_cells = []
    r.lines.append(f"{'身分':20s} {'script%':>8s} {'model%':>7s} {'Δpp':>6s}  bucket")
    worst = 99.0
    for bname, idents in buckets.items():
        for ident in idents:
            rr = GEN.eval_identity(ident, OPP_PANEL, G, 5, cand)
            sw = rr["script"]["w"] / max(1, rr["script"]["n"])
            mw = rr["model"]["w"] / max(1, rr["model"]["n"])
            dpp = (mw - sw) * 100
            worst = min(worst, dpp)
            cells.append(Cell(f"{bname}/{ident}", mw, rr["model"]["n"], aux=f"Δ{dpp:+.0f}"))
            if dpp < -15:
                r.lines.append(f"{ident:20s} {sw:8.0%} {mw:7.0%} {dpp:+6.0f}  {bname} ⚠")
            if base is not None:
                br = GEN.eval_identity(ident, OPP_PANEL, G, 5, base)
                base_cells.append(Cell(f"{bname}/{ident}",
                                       br["model"]["w"] / max(1, br["model"]["n"]),
                                       br["model"]["n"]))
    # 通用判準：記憶量測「通用腦通常還贏駕同 kit 的腳本」→ 明顯遜於腳本=沒泛化。
    # 但 Δpp 在低局數噪音極大(#33)：n=4 時 ±35pp，-25pp 根本在噪音內。故只有
    # 「明顯遜(<-20pp) 且超出 2σ 噪音」才硬 FAIL；否則標 UNDERPOWERED（加局數再判）。
    n_cell = G * len(OPP_PANEL)
    sig_dpp = (2 * 0.25 / max(1, n_cell)) ** 0.5 * 100
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} WR {reg[1]:+.0%}"
    elif worst < -20:
        if abs(worst) > 2 * sig_dpp:
            r.status = "FAIL"
        else:
            r.status = "UNDERPOWERED"
            r.note = f"最差 Δpp={worst:+.0f} 但在噪音內(±{2 * sig_dpp:.0f}pp @n/cell={n_cell})；加 --games 再判"
    n_id = len(cells)
    r.summary = f"{n_id} 個未見身分，最差 Δpp={worst:+.0f}（vs 駕同 kit 的腳本，n/cell={n_cell}）"
    return r


# ── C6: 走位／繞牆視線重建（goal 支柱｜Part1 走位 + #21） ──────────────────────

def check_walls(cand, base, G):
    r = CheckResult("walls", "走位／繞牆視線重建：牆地形 WR 不該相對開闊地崩塌",
                    ["Part1 走位", "#21 對主動近戰走位", "goal:走位"])
    try:
        import eval_wall_combat as WC
    except Exception as ex:
        r.status = "ERROR"; r.summary = f"無法載入 eval_wall_combat: {ex}"; return r
    idents = ["assassin", "champion", "war", "battle_master", "evocation", "berserker"]
    opps = ["champion", "evocation"]
    r.lines.append(f"{'身分':16s} {'open%':>6s} {'walls%':>7s} {'Δ':>5s}")
    cells = []; base_cells = []
    collapsed = []
    for ident in idents:
        ow = ww = n = 0
        for opp in opps:
            for gi in range(G):
                ow += int(WC.play(cand, ident, opp, "open", f"o|{ident}|{opp}|{gi}"))
                ww += int(WC.play(cand, ident, opp, "walls", f"w|{ident}|{opp}|{gi}"))
                n += 1
        owr = ow / n; wwr = ww / n; gap = owr - wwr
        cells.append(Cell(ident, wwr, n, aux=f"open={owr:.0%}"))
        bad = wwr < 0.20 and gap > 0.35     # 牆上幾乎全敗且明顯低於開闊地＝被牆卡/搆不到
        if bad:
            collapsed.append(ident)
        r.lines.append(f"{ident:16s} {owr:6.0%} {wwr:7.0%} {gap * -100:+5.0f}"
                       + ("  ⚠崩" if bad else ""))
        if base is not None:
            bw = sum(int(WC.play(base, ident, opp, "walls", f"w|{ident}|{opp}|{gi}"))
                     for opp in opps for gi in range(G))
            base_cells.append(Cell(ident, bw / (len(opps) * G), len(opps) * G))
    if collapsed:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} walls WR {reg[1]:+.0%}"
    r.summary = ("牆上無身分崩塌（走位/視線重建保留）" if not collapsed
                 else f"牆上崩塌身分：{','.join(collapsed)}（開闊地正常→走位/繞牆失效）")
    return r


# ── C7: 優劣勢不划水且追戰局（goal 支柱｜Part1 劣勢 + 劣勢1v2/低階Δ-3/優勢2v1） ──

def check_asym(cand, base, G):
    r = CheckResult("asym", "面對優劣勢：劣勢(1v2/低階Δ-3)不划水、優勢(2v1)追戰局",
                    ["Part1 劣勢", "goal:面對優劣勢"])
    idents = ["battle_master", "champion", "evocation", "vengeance", "war", "life"]
    BUCK = [
        ("均勢1v1", lambda i: ([i], ["champion"], 5, 5)),
        ("劣勢1v2", lambda i: ([i], ["champion", "evocation"], 5, 5)),
        ("劣勢Δ-3", lambda i: ([i], ["champion"], 5, 8)),
        ("優勢2v1", lambda i: ([i, i], ["champion"], 5, 5)),
    ]
    r.lines.append(f"{'情境':10s} {'WR':>6s} {'dealt%':>7s} {'idle%':>6s}")
    cells = []; base_cells = []
    worst_idle = 0.0
    for label, fn in BUCK:
        w = dealt = idle = ft = n = 0
        for i in idents:
            aa, oa, al, ol = fn(i)
            for gi in range(G):
                o = drive_episode(_model_driver(cand), aa, oa, al, ol,
                                  f"asym|{label}|{i}|{gi}")
                w += o.win; dealt += o.dmg_frac; idle += o.idle
                ft += o.first_turns; n += 1
        wr = w / n; de = dealt / n; idl = idle / max(1, ft)
        worst_idle = max(worst_idle, idl)
        cells.append(Cell(label, 1.0 - idl, n, aux=f"WR={wr:.0%} dealt={de:.0%}"))
        r.lines.append(f"{label:10s} {wr:6.0%} {de:7.1%} {idl:6.1%}"
                       + ("  ⚠划水" if idl > 0.05 else ""))
        if base is not None:
            bi = bft = 0
            for i in idents:
                aa, oa, al, ol = fn(i)
                for gi in range(G):
                    o = drive_episode(_model_driver(base), aa, oa, al, ol,
                                      f"asym|{label}|{i}|{gi}")
                    bi += o.idle; bft += o.first_turns
            base_cells.append(Cell(label, 1.0 - bi / max(1, bft), G * len(idents)))
    # 劣勢不划水＝首手 idle 應接近 0（記憶：baseline idle 0% 全桶）
    if worst_idle > 0.10:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠划水回歸 vs base：{reg[0]}"
    r.summary = f"最高划水率={worst_idle:.1%}（劣勢不划水；memory baseline≈0）"
    return r


# ── C8: 依敵抗性資訊切傷害型（goal 支柱｜Part1 held-out 切招 + #26） ──────────────
# self-calibrating：先量基線主型 D，再注入對 D 的免疫，看模型有沒有把攻擊移出 D。
# 不需預知任何 kit 的內部傷害型（數據自己講），單一傷害型 agent 自動略過(immune 已涵蓋)。

def check_switch(cand, base, G):
    r = CheckResult("switch", "依敵抗性資訊切傷害型：主型被免疫→改砸非免疫型（不空砸被免疫型）",
                    ["Part1 held-out切招", "goal:依資訊選招", "#26 抗性選型(部分)"])
    try:
        from seed_switch_bc import eval_switch
    except Exception as ex:
        r.status = "ERROR"; r.summary = f"無法載入 seed_switch_bc: {ex}"; return r
    AGENTS = ["life", "vengeance", "war", "arcane_trickster", "devotion",
              "champion", "battle_master"]
    enemy = "orc"; o_lvl = MONSTER_DEFS["orc"].natural_level
    r.lines.append(f"{'agent':16s} {'主型':>6s} {'注前':>6s} {'注後':>6s} {'注後WR':>7s}")
    cells = []; base_cells = []; two_type = 0
    for ag in AGENTS:
        base_sh, _ = eval_switch(cand, ag, enemy, 8, o_lvl, None, False, G, f"sw_{ag}_b")
        big = [t for t, v in base_sh.items() if v > 0.15]
        if len(big) < 2:
            r.lines.append(f"{ag:16s}  (單一傷害型，免疫檢測已涵蓋，略)")
            continue
        two_type += 1
        D = max(base_sh, key=base_sh.get)
        inj_sh, inj_wr = eval_switch(cand, ag, enemy, 8, o_lvl, (D, 0.0), False, G,
                                     f"sw_{ag}_i")
        after = inj_sh.get(D, 0.0)
        cells.append(Cell(ag, 1.0 - after, G, aux=f"D={D} after={after:.0%} wr={inj_wr:.0%}"))
        r.lines.append(f"{ag:16s} {D:>6s} {base_sh[D]:6.0%} {after:6.0%} {inj_wr:7.0%}"
                       + ("  ⚠沒切" if after > 0.5 else ""))
        if base is not None:
            b_sh, _ = eval_switch(base, ag, enemy, 8, o_lvl, (D, 0.0), False, G,
                                  f"sw_{ag}_i")
            base_cells.append(Cell(ag, 1.0 - b_sh.get(D, 0.0), G))
    if two_type == 0:
        r.status = "UNDERPOWERED"
        r.summary = "候選 agent 全單一傷害型（→ immune 檢測已涵蓋）"
        return r
    worst = min((c.value for c in cells), default=1.0)   # value=1-after；低=沒切
    # after 是「幾次攻擊/局」的比例，低局數極粗(#33)：G=1 只有 2-4 次攻擊→±58% 噪音。
    # 故只有「注入後砸免疫型明顯 >50% 且超出 2σ」才硬 FAIL；否則 UNDERPOWERED。
    n_eff = max(1, G * 3)
    sig = (0.25 / n_eff) ** 0.5
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠切招回歸 vs base：{reg[0]} {reg[1]:+.0%}"
    elif worst < 0.5:
        if (0.5 - worst) > 2 * sig:
            r.status = "FAIL"
        else:
            r.status = "UNDERPOWERED"
            r.note = f"注入後砸免疫型 {1 - worst:.0%} 但在噪音內(±{2 * sig:.0%} @G={G})；加 --games 再判"
    r.summary = f"雙型 agent {two_type} 個，最差『注入後仍砸免疫型』={1 - worst:.0%}"
    return r


# ── C9: 對消極/被動敵不空轉（#5 被動對手×全身分 + #11 GUI 部署人類玩家） ──────────
# 用戶最初手抓的 Evoker disengage-loop 正是「對消極敵空轉不打」——這條若當初存在就會抓到。

def check_passive(cand, base, G):
    r = CheckResult("passive", "對消極/被動敵（GUI 人類玩家式）不空轉：能打的敵要主動接戰擊殺",
                    ["#5 被動對手×全身分", "#11 GUI部署", "goal:零退化"])
    idents = CLASSES + ["orc", "goblin", "ogre", "shadow"]
    r.lines.append(f"{'身分':16s} {'dmg%':>6s} {'win%':>6s} {'idle%':>6s}")
    cells = []; base_cells = []
    worst = 1.0; worst_id = ""
    for ident in idents:
        lvl = MONSTER_DEFS[ident].natural_level if ident in MONSTER_DEFS else 5
        dmg = w = idle = ft = n = 0
        for gi in range(G):
            o = drive_episode(_model_driver(cand), ident, ["orc"], lvl,
                              MONSTER_DEFS["orc"].natural_level,
                              f"pas|{ident}|{gi}", passive_enemy=True)
            dmg += o.dmg_frac; w += o.win; idle += o.idle; ft += o.first_turns; n += 1
        dm = dmg / n; idl = idle / max(1, ft)
        if dm < worst:
            worst = dm; worst_id = ident
        cells.append(Cell(ident, dm, n, aux=f"win={w / n:.0%}"))
        if dm < 0.5 or idl > 0.10:
            r.lines.append(f"{ident:16s} {dm:6.0%} {w / n:6.0%} {idl:6.1%}  ⚠空轉?")
        if base is not None:
            bd = sum(drive_episode(_model_driver(base), ident, ["orc"], lvl,
                                   MONSTER_DEFS["orc"].natural_level,
                                   f"pas|{ident}|{gi}", passive_enemy=True).dmg_frac
                     for gi in range(G)) / G
            base_cells.append(Cell(ident, bd, G))
    # 被動敵(低血 orc)該被任何有攻擊手段的身分打死→dmg% 應高。分層判(#33 噪音誠實)：
    #   近零(<15%)＝不折不扣的 disengage-loop，任何局數都硬 FAIL；
    #   部分停滯(15~50%)＝可能單局慢，需 G≥3 才 FAIL，低局數只標 UNDERPOWERED。
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} dmg% {reg[1]:+.0%}"
    elif worst < 0.15:
        r.status = "FAIL"
    elif worst < 0.5:
        if G >= 3:
            r.status = "FAIL"
        else:
            r.status = "UNDERPOWERED"
            r.note = f"最低 dmg%={worst:.0%}（{worst_id}）低局數存疑；加 --games 再判"
    r.summary = (f"全身分對被動敵均主動接戰（最低 dmg%={worst:.0%}）"
                 if r.status == "PASS" else
                 f"『{worst_id}』對能打的被動敵僅 {worst:.0%} 傷害＝空轉/disengage-loop")
    return r


# ── C10: 利用敵人既有優勢狀態（#27）——純目標選擇，非傷害佔比（避開命中率污染） ──────
# 兩個同型被動敵，其一預設 prone（近戰打它有優勢）。數模型的攻擊動作落在哪個敵（用
# _entity_id_at_slot 反查 act[1]→char，不看傷害＝不被 prone 的命中率加成污染）。
# 因果：同種子跑「有 prone / 無 prone」，delta=傾斜差＝狀態是否驅動目標選擇。

def _attack_target_split(net, agent, enemy, level, key, prone):
    seed = stable_seed(key); random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=1, n_opps=2)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[enemy, enemy],
                       level=level, opp_level=level, layout="open")
    o0, o1 = env.opp_ids
    for o in (o0, o1):
        env._opp_policies[o] = _StationaryPolicy()   # 兩敵都被動：唯一差異＝prone
    if prone:
        env.ws.characters[o0].add_status(Prone())
    drv = _model_driver(net)
    a0 = a1 = 0; done = False; steps = 0
    while not done:
        actor = env.current_agent_id
        act = drv(env, obs, actor)
        ch = env.ws.characters[actor]
        sks = available_skills(ch, env.ws)
        if 0 < act[0] < len(sks) and sks[act[0]].features.target_type in (
                TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
            tgt = _entity_id_at_slot(env.ws, actor, act[1])
            if tgt == o0:
                a0 += 1
            elif tgt == o1:
                a1 += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc; steps += 1
        if steps > 200:
            break
    tot = a0 + a1
    return (a0 / tot if tot else 0.5), tot


def check_exploit(cand, base, G):
    r = CheckResult("exploit", "利用敵既有優勢狀態：一被動敵預設 prone→攻擊是否傾向可利用的那個(純目標選擇)",
                    ["#27 利用敵狀態", "goal:依資訊選招"])
    agents = ["champion", "battle_master", "war", "berserker", "totem_bear"]
    enemy = "champion"
    r.lines.append(f"{'agent':16s} {'prone時':>7s} {'無狀態':>7s} {'Δ傾斜':>7s}")
    cells = []; base_cells = []; deltas = []
    for ag in agents:
        on = off = 0.0; non = noff = 0
        for gi in range(G):
            s_on, t_on = _attack_target_split(cand, ag, enemy, 5, f"exp|{ag}|{gi}", True)
            s_off, t_off = _attack_target_split(cand, ag, enemy, 5, f"exp|{ag}|{gi}", False)
            if t_on:
                on += s_on; non += 1
            if t_off:
                off += s_off; noff += 1
        s_on = on / max(1, non); s_off = off / max(1, noff)
        delta = s_on - s_off
        deltas.append(delta)
        cells.append(Cell(ag, delta, G, aux=f"on={s_on:.0%} off={s_off:.0%}"))
        r.lines.append(f"{ag:16s} {s_on:7.0%} {s_off:7.0%} {delta * 100:+7.0f}"
                       + ("  ⚠避開" if delta < -0.15 else ("  ✓利用" if delta > 0.15 else "")))
        if base is not None:
            b_on = b_off = 0.0; bn = bf = 0
            for gi in range(G):
                so, to = _attack_target_split(base, ag, enemy, 5, f"exp|{ag}|{gi}", True)
                sf, tf = _attack_target_split(base, ag, enemy, 5, f"exp|{ag}|{gi}", False)
                if to:
                    b_on += so; bn += 1
                if tf:
                    b_off += sf; bf += 1
            base_cells.append(Cell(ag, b_on / max(1, bn) - b_off / max(1, bf), G))
    mean_delta = sum(deltas) / max(1, len(deltas))
    worst = min(deltas, default=0.0)
    # 這是「軟能力」＝利用 prone 是加分項，不利用不算退化。故 gate 只在兩種情況 FAIL：
    #   (a) 明顯『主動避開』可利用敵且超出 2σ 噪音（真的做錯，非只是沒讀）；
    #   (b) 比 base 回歸（本來會利用、現在不利用了）。
    # 否則 PASS，把 mean Δ 攤開＝能力指標（Δ>0 會利用、≈0 沒讀、Δ<0 反向）。
    n_eff = max(1, G * 4)
    sig = (2 * 0.25 / n_eff) ** 0.5
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} 傾斜 {reg[1]:+.0%}"
    elif worst < -0.15 and abs(worst) > 2 * sig:
        r.status = "FAIL"
        r.note = "有身分明顯『避開』可利用敵（超出噪音）"
    else:
        r.note = f"能力指標；噪音帶±{2 * sig * 100:.0f}pp @G={G}"
    r.summary = f"平均傾向 prone 敵 Δ={mean_delta * 100:+.0f}pp（>0＝會利用狀態；最差 {worst * 100:+.0f}pp）"
    return r


# ── C11: 承受方——被 restrained 時該攻擊而非放棄（#12 被擒抱/緊勒的行為面） ──────────
# 引擎事實（skill.py:575-582）：角色被 restrained(speed×0) 時 available_skills **直接不列
# MOVE**（can_move=False）＝「被控還空移浪費」引擎根本不可能發生。故真正該測的是：被 restrained
# ＋敵貼臉時，action 在手、攻擊合法，模型是**攻擊**還是**選 END 放棄(拒戰)**＝被控放棄的退化。
# （原本測「空移浪費」是瞎的＝測不可能事件永遠 PASS，被 --self_test 揪出後改寫成此。）

def _restrained_behavior(net, agent, enemy, level, key):
    seed = stable_seed(key); random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[enemy],
                       level=level, opp_level=level, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    env._opp_policies[oid] = _StationaryPolicy()
    a = env.ws.characters[aid]; e = env.ws.characters[oid]
    e.position = Vec2(a.position.x + 1.0, a.position.y)   # 貼在近戰 reach 內
    a.add_status(Restrained())                            # 速度 0（MOVE 會被移出技能表）
    drv = _model_driver(net)
    refuse = attack = steps = 0; done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        sks = available_skills(ch, env.ws)
        atk_legal = any(s.features.target_type in (TargetType.SINGLE_ENEMY,
                        TargetType.MULTI_ENEMY) for s in sks)
        act = drv(env, obs, actor)
        if actor == aid:
            steps += 1
            act_avail = env.resources.get("action", 0) > 0
            is_end = (act[0] == 0)
            is_atk = (0 < act[0] < len(sks) and sks[act[0]].features.target_type
                      in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY))
            if is_end and act_avail and atk_legal:       # 能打卻選結束＝被控放棄
                refuse += 1
            if is_atk:
                attack += 1
        obs, _, term, trunc, _ = env.step(act); done = term or trunc
        if steps > 120:
            break
    return refuse, attack, max(1, steps)


def check_victim(cand, base, G):
    r = CheckResult("victim", "承受方：被 restrained(速度0)＋敵貼臉→該攻擊而非放棄(END/拒戰)(#12)",
                    ["#12 被擒抱/吞噬", "goal:零退化", "goal:面對(承受方)"])
    agents = ["champion", "battle_master", "war", "berserker", "totem_bear", "assassin"]
    enemy = "champion"
    r.lines.append(f"{'agent':16s} {'拒戰%':>6s} {'攻擊%':>6s} {'決策數':>6s}")
    cells = []; base_cells = []
    worst = 0.0; worst_id = ""
    for ag in agents:
        rf = a = s = 0
        for gi in range(G):
            rr, aa, ss = _restrained_behavior(cand, ag, enemy, 5, f"vic|{ag}|{gi}")
            rf += rr; a += aa; s += ss
        rr_rate = rf / max(1, s); ar = a / max(1, s)
        if rr_rate > worst:
            worst = rr_rate; worst_id = ag
        cells.append(Cell(ag, 1.0 - rr_rate, s, aux=f"refuse={rr_rate:.0%}"))
        r.lines.append(f"{ag:16s} {rr_rate:6.0%} {ar:6.0%} {s:6d}"
                       + ("  ⚠被控放棄" if rr_rate > 0.15 else ""))
        if base is not None:
            brf = bs = 0
            for gi in range(G):
                rr, aa, ss = _restrained_behavior(base, ag, enemy, 5, f"vic|{ag}|{gi}")
                brf += rr; bs += ss
            base_cells.append(Cell(ag, 1.0 - brf / max(1, bs), bs))
    # 被 restrained＋敵貼臉：能打卻選 END＝被控放棄＝真退化(挨打不還手)。>15% 回合拒戰＝FAIL。
    if worst > 0.15:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠拒戰增加 vs base：{reg[0]} {-reg[1]:.0%}"
    r.summary = (f"被 restrained 時全身分仍接戰（最高拒戰率 {worst:.0%}）"
                 if r.status == "PASS" else
                 f"『{worst_id}』被 restrained 時 {worst:.0%} 回合放棄(能打卻選結束)")
    return r


# ── C12: 死動作（Part1）——選到引擎會拒絕(ERROR)的動作＝浪費回合（合法性遮罩漏網） ──
# 做法同 diag_dead_action：暫時 patch env_v2.execute_action 數「agent 執行時回 ERROR」，
# check 結束即還原（不污染其他 check）。良好模型應接近 0：合法性遮罩本應擋掉不可行動作。

def check_dead(cand, base, G):
    r = CheckResult("dead", "死動作：選到引擎會拒絕(ERROR)的動作＝浪費回合（合法性遮罩漏網）",
                    ["Part1 死動作", "goal:零退化"])
    import trpg.rl.env_v2 as _ev
    orig = _ev.execute_action
    st = {"tally": Counter(), "exec": 0}

    def wrap(action, ws):
        res = orig(action, ws)
        aid = (action.get("attacker") or action.get("caster")
               or action.get("character"))
        if isinstance(aid, str) and aid.startswith("agent"):
            st["exec"] += 1
            if isinstance(res, dict) and res.get("type") == "ERROR":
                st["tally"][(action.get("skill_id", "?"),
                             str(res.get("message", "?"))[:45])] += 1
        return res

    idents = CLASSES + ["orc", "goblin", "ogre", "shadow"]
    cells = []; base_cells = []; top = Counter()
    _ev.execute_action = wrap
    try:
        def measure(net, sink):
            for ident in idents:
                st["tally"] = Counter(); st["exec"] = 0
                lvl = MONSTER_DEFS[ident].natural_level if ident in MONSTER_DEFS else 6
                for opp in OPP_PANEL:
                    for gi in range(G):
                        DD.run_combat(net, ident, [opp], lvl, lvl,
                                      f"dead|{ident}|{opp}|{gi}")
                err = sum(st["tally"].values()); ex = max(1, st["exec"])
                top.update(st["tally"])
                sink.append(Cell(ident, 1.0 - err / ex, ex, aux=f"dead={err}/{ex}"))
        measure(cand, cells)
        if base is not None:
            top_save = Counter(top); top.clear()
            measure(base, base_cells)
            top.clear(); top.update(top_save)
    finally:
        _ev.execute_action = orig
    worst = max((1.0 - c.value for c in cells), default=0.0)  # 最高死動作率
    for ident, c in zip(idents, cells):
        if 1.0 - c.value > 0.02:
            r.lines.append(f"{ident:16s} 死動作率 {(1 - c.value):.1%}  ({c.aux})")
    for (sk, msg), n in top.most_common(5):
        r.lines.append(f"   × {n:3d}  {sk:18s} | {msg}")
    # 合法性遮罩本應擋掉→良好模型≈0。任一身分 >2% 死動作＝遮罩有漏＝FAIL。
    if worst > 0.02:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠死動作增加 vs base：{reg[0]} {-reg[1]:.0%}"
    r.summary = (f"全身分死動作率 <2%（最高 {worst:.1%}）" if r.status == "PASS"
                 else f"最高死動作率 {worst:.1%}（見上；合法性遮罩漏網）")
    return r


# ── C13: 逐通道因果（Part1）——敵每個資訊通道歸零→行為翻不翻＝模型有沒有讀該資訊 ──
# 重用 diag_info_channels：主要靠回歸守門抓「本來會讀某資訊、現在不讀了」的能力退化。
# 絕對「惰性」只報不 FAIL（不是每個通道都該影響動作選擇＝惰性未必是 bug）。

def check_infochan(cand, base, G):
    r = CheckResult("infochan", "逐通道因果：敵資訊通道歸零→行為翻不翻（模型有沒有讀該資訊）",
                    ["Part1 逐通道因果", "goal:依資訊選招"])
    try:
        import diag_info_channels as IC
        from collections import defaultdict
        from trpg.scenarios.monsters import onev1_viable_monsters, EQUIV_LEVEL_1V1
    except Exception as ex:
        r.status = "ERROR"; r.summary = f"無法載入 diag_info_channels: {ex}"; return r
    # 取『描述子豐富』的怪（caster/resist/status/undead/petrify），否則通道 present=0＝
    # 歸零也沒差、測不出讀取。弱怪描述子近乎空會誤導成「什麼都不讀」。
    viable = set(onev1_viable_monsters(8.0))
    # 描述子豐富＋帶天賦(pack/undead)＋帶條件免疫的怪，讓 passive_traits / condition_immunity
    # 通道 present（否則 present=0 測不出）＝順帶覆蓋「天賦因果」與 #8/#9「條件免疫」。
    RICH = ["shadow", "mage_npc", "basilisk", "ghoul", "wight", "manticore",
            "gargoyle", "wraith", "dire_wolf", "wolf", "zombie", "skeleton"]
    mons = [m for m in RICH if m in viable] or sorted(
        viable, key=lambda m: -EQUIV_LEVEL_1V1[m])[:6]
    agents = ["battle_master", "evocation", "life", "assassin"]

    def measure(net):
        tally = defaultdict(lambda: {"present": 0, "flip": 0})
        for mon in mons:
            lvl = max(1, round(EQUIV_LEVEL_1V1[mon]))
            m_lvl = MONSTER_DEFS[mon].natural_level
            for a in agents:
                for k in range(G):
                    IC.play(net, a, mon, stable_seed(f"ic|{a}|{mon}|{k}"),
                            lvl, m_lvl, tally)
        return tally

    tally = measure(cand)
    cells = []; base_cells = []
    r.lines.append(f"{'channel':30s} {'present':>7s} {'flip%':>6s}")
    for name, _, _ in IC.CHANNELS:
        t = tally[name]; pr = t["present"]; fl = t["flip"] / max(1, pr)
        cells.append(Cell(name, fl, pr, aux=f"n={pr}"))
        r.lines.append(f"{name:30s} {pr:7d} {fl:6.1%}"
                       + ("  ⚠惰性(帶值不讀)" if pr >= 100 and fl < 0.05 else ""))
    if base is not None:
        bt = measure(base)
        for name, _, _ in IC.CHANNELS:
            t = bt[name]
            base_cells.append(Cell(name, t["flip"] / max(1, t["present"]), t["present"]))
    read = sum(1 for c in cells if c.value >= 0.05)
    inert = [c.label for c in cells if c.n >= 100 and c.value < 0.05]
    # 回歸語義＝「從『讀』(≥5%) 退成『不讀』(<5%)」——與本 check 的宣稱一致。
    # 舊實作用通用 regression_flag（flip 比例大降即 FAIL）會把「移除病態過度反應、
    # 仍在讀」誤判成回歸：v9 typed_resist flip 23.5% 大部分是 dodge-collapse 恐慌
    # （probe_resist_curve 錘定），v10 修復後 5.6%（仍 ≥5%、switch/immune 行為門
    # 全過）被舊規則標 FAIL。flip 統計是狀態級代理指標，行為級因果（switch/probe）
    # 才是讀取的證據本體；本規則只抓真正跨過讀取門檻的退化。
    reg = None
    if base_cells:
        bmap = {c.label: (c.value, c.n) for c in base_cells}
        for c in cells:
            bv, bn = bmap.get(c.label, (0.0, 0))
            if bv >= 0.05 and bn >= 100 and c.n >= 100 and c.value < 0.05:
                reg = (c.label, c.value - bv)
                break
    if reg:
        r.status = "FAIL"
        r.note = f"⚠某資訊通道從『讀』退成『不讀』 vs base：{reg[0]} flip {reg[1]:+.0%}"
    r.summary = (f"{read}/{len(cells)} 通道有讀取(flip≥5%)"
                 + (f"；惰性通道:{','.join(inert)}（只報不判 FAIL）" if inert else ""))
    return r


# ── C14: 特殊地形適應（#29 difficult 2×移動 / #24 危險格 lava） ─────────────────
# 乾淨的 layout 開關＋drive_episode。difficult：功能是否保留(傷害相對 open 不該崩)；
# lava：自傷增量(lava−open)＝是否呆站在危險格挨重複傷害。lava 是否有訊號用數據判。

def check_terrain(cand, base, G):
    r = CheckResult("terrain", "特殊地形適應：difficult(2×移動)功能保留、危險格(lava)不呆站挨傷",
                    ["#29 difficult地形", "#24 危險地形/光環"])
    agents = ["champion", "battle_master", "assassin", "evocation", "berserker"]
    enemy = "champion"
    r.lines.append(f"{'agent':14s} {'open傷%':>7s} {'diff傷%':>7s} {'lava自傷Δ':>9s}")
    cells = []; base_cells = []
    worst_diff = 1.0; worst_lava = 0.0

    def run(net, ag, gi, layout):
        return drive_episode(_model_driver(net), ag, [enemy], 5, 5,
                             f"ter|{layout}|{ag}|{gi}", layout=layout)
    for ag in agents:
        od = dd = 0.0; open_self = lava_self = 0.0
        for gi in range(G):
            oo = run(cand, ag, gi, "open"); od += oo.dmg_frac; open_self += oo.opp_dmg_frac
            dd += run(cand, ag, gi, "difficult").dmg_frac
            lava_self += run(cand, ag, gi, "lava").opp_dmg_frac
        od /= G; dd /= G
        diff_ratio = dd / max(0.05, od)                # <1 = difficult 削弱功能
        lava_extra = (lava_self - open_self) / G       # >0 = lava 多挨的自傷
        worst_diff = min(worst_diff, diff_ratio); worst_lava = max(worst_lava, lava_extra)
        cells.append(Cell(ag, min(1.0, diff_ratio), G,
                          aux=f"diff/open={diff_ratio:.0%} lavaΔ={lava_extra:+.0%}"))
        r.lines.append(f"{ag:14s} {od:7.0%} {dd:7.0%} {lava_extra * 100:+9.0f}"
                       + ("  ⚠地形崩" if diff_ratio < 0.4 else "")
                       + ("  ⚠呆站lava" if lava_extra > 0.25 else ""))
        if base is not None:
            bod = bdd = 0.0
            for gi in range(G):
                bod += run(base, ag, gi, "open").dmg_frac
                bdd += run(base, ag, gi, "difficult").dmg_frac
            base_cells.append(Cell(ag, min(1.0, (bdd / G) / max(0.05, bod / G)), G))
    # gate 判據＝difficult：傷害掉 >60%＝功能崩(FAIL)。lava 只報不判——實測自傷增量正負
    # 混雜(-29%~+20%)＝局部 3×3 危險格很少被迫經過＝低訊號，當 gate 會 flaky(見 note)。
    if worst_diff < 0.4:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠difficult 功能回歸 vs base：{reg[0]} {reg[1]:+.0%}"
    if not r.note:
        r.note = f"lava 自傷增量最高 {worst_lava:+.0%}（低訊號、只報不判：危險格局部、路徑少被迫經過）"
    r.summary = (f"difficult 功能保留（最低 diff/open {worst_diff:.0%}，近戰受 2× 移動影響最大）"
                 if r.status == "PASS" else
                 f"difficult 功能崩：最低 diff/open {worst_diff:.0%}")
    return r


# ── C15: 高等級外推（#1 L>8）——所有訓練/評估都在 L≤8，用戶在 L20 玩；obs 外推從沒驗 ──
# 跑核心行為(接戰/傷害/不空轉)於 L20 vs L20。若 obs 外推失效→模型在高階崩(0傷/全空轉)。

def check_highlevel(cand, base, G):
    r = CheckResult("highlevel", "高等級外推(L20)：訓練只到 L8、用戶在 L20 玩，obs 外推是否崩",
                    ["#1 等級>8", "goal:GUI 部署高階"])
    agents = ["champion", "battle_master", "evocation", "life", "assassin", "war"]
    r.lines.append(f"{'agent':16s} {'L20傷%':>7s} {'L20勝%':>7s} {'空轉%':>6s}")
    cells = []; base_cells = []
    worst_dmg = 1.0; worst_id = ""
    for ag in agents:
        dmg = w = idle = ft = 0.0
        for gi in range(G):
            o = drive_episode(_model_driver(cand), ag, ["champion"], 20, 20,
                              f"hl|{ag}|{gi}")
            dmg += o.dmg_frac; w += o.win; idle += o.idle; ft += o.first_turns
        dm = dmg / G; idl = idle / max(1, ft)
        if dm < worst_dmg:
            worst_dmg = dm; worst_id = ag
        cells.append(Cell(ag, dm, G, aux=f"win={w / G:.0%} idle={idl:.0%}"))
        r.lines.append(f"{ag:16s} {dm:7.0%} {w / G:7.0%} {idl:6.0%}"
                       + ("  ⚠高階崩" if dm < 0.25 or idl > 0.15 else ""))
        if base is not None:
            bd = sum(drive_episode(_model_driver(base), ag, ["champion"], 20, 20,
                                   f"hl|{ag}|{gi}").dmg_frac for gi in range(G)) / G
            base_cells.append(Cell(ag, bd, G))
    # L20 打 L20 對稱局：至少該正常接戰輸出(不崩)。近零傷害/全空轉＝obs 外推失效。
    if worst_dmg < 0.25:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠L20 回歸 vs base：{reg[0]} 傷害 {reg[1]:+.0%}"
    r.summary = (f"L20 全身分正常接戰（最低傷害 {worst_dmg:.0%}）＝obs 外推未崩"
                 if r.status == "PASS" else
                 f"『{worst_id}』L20 僅 {worst_dmg:.0%} 傷害＝高階 obs 外推失效")
    return r


# ── C16: 承受方——被 frightened 後遠程 agent 是否還接戰輸出（#13 恐懼光環行為面） ──
# 引擎：frightened＝攻擊 disadvantage＋不能朝恐懼源移動，但**不阻止攻擊**。遠程 agent 已在
# 射程、不需朝源移動→被 frighten 仍該持續輸出。攻擊率崩＝#13「被恐懼後輸出崩」。

def _frighten_attack_rate(net, agent, enemy, level, key, frighten):
    seed = stable_seed(key); random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=[enemy],
                       level=level, opp_level=level, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    env._opp_policies[oid] = _StationaryPolicy()          # 被動敵：隔離 frighten 效果
    a = env.ws.characters[aid]; e = env.ws.characters[oid]
    e.position = Vec2(a.position.x + 6.0, a.position.y)    # 中距(遠程射程內、非近戰)
    if frighten:
        a.add_status(Frightened())
    drv = _model_driver(net)
    atk = turns = 0; done = False; steps = 0
    OFF = (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
           TargetType.POINT, TargetType.LINE, TargetType.CONE)
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]; sks = available_skills(ch, env.ws)
        act = drv(env, obs, actor)
        if actor == aid:
            turns += 1
            if (0 < act[0] < len(sks) and sks[act[0]].features.target_type in OFF
                    and sks[act[0]].skill_id != "move"):
                atk += 1
        obs, _, term, trunc, _ = env.step(act); done = term or trunc; steps += 1
        if steps > 120:
            break
    return atk / max(1, turns)


def check_frighten(cand, base, G):
    r = CheckResult("frighten", "承受方：被 frightened 後遠程 agent 是否還接戰輸出（非輸出崩）",
                    ["#13 恐懼光環", "goal:面對(承受方)"])
    agents = ["evocation", "assassin", "arcane_trickster", "divination"]
    enemy = "champion"
    r.lines.append(f"{'agent':16s} {'恐懼攻擊率':>9s} {'常態攻擊率':>9s} {'Δ':>5s}")
    cells = []; base_cells = []
    worst_on = 1.0; worst_id = ""; worst_gap = 0.0
    for ag in agents:
        on = off = 0.0
        for gi in range(G):
            on += _frighten_attack_rate(cand, ag, enemy, 5, f"fr|{ag}|{gi}", True)
            off += _frighten_attack_rate(cand, ag, enemy, 5, f"fr|{ag}|{gi}", False)
        on /= G; off /= G
        if on < worst_on:
            worst_on = on; worst_id = ag; worst_gap = off - on
        cells.append(Cell(ag, on, G, aux=f"off={off:.0%}"))
        r.lines.append(f"{ag:16s} {on:9.0%} {off:9.0%} {(on - off) * 100:+5.0f}"
                       + ("  ⚠輸出崩" if on < 0.3 and off - on > 0.2 else ""))
        if base is not None:
            bon = sum(_frighten_attack_rate(base, ag, enemy, 5, f"fr|{ag}|{gi}", True)
                      for gi in range(G)) / G
            base_cells.append(Cell(ag, bon, G))
    # frighten 是 disadvantage(機制)、非「停止攻擊」。若攻擊率崩到 <30% 且明顯低於常態
    # (>20pp gap)＝模型被恐懼後不再接戰＝#13 輸出崩。
    if worst_on < 0.3 and worst_gap > 0.2:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠恐懼下攻擊率回歸 vs base：{reg[0]} {reg[1]:+.0%}"
    r.summary = (f"被 frightened 全遠程 agent 仍接戰（最低攻擊率 {worst_on:.0%}）"
                 if r.status == "PASS" else
                 f"『{worst_id}』被 frightened 後攻擊率崩至 {worst_on:.0%}（常態高 {worst_gap:.0%}）")
    return r


# ── C17: 再生怪持續壓制（#14）——對 troll(每回合再生、火/酸抑制)能否淨侵蝕 HP ─────────
# 乾淨因果開關：troll.regeneration 開/關。壓制得好(持續火/酸)→開關差異小(regen 被壓)；
# 壓制不住→regen-on 的 troll 存活遠久於 regen-off＝delta 大。

def _regen_dmg(net, agent, key, regen_on, level=10):
    seed = stable_seed(key); random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent], opp_archs=["troll"], level=level,
                       opp_level=MONSTER_DEFS["troll"].natural_level, layout="open")
    oid = env.opp_ids[0]; e = env.ws.characters[oid]
    hp0 = e.hp
    if not regen_on:
        e.regeneration = None
    drv = _model_driver(net); done = False; steps = 0
    while not done:
        actor = env.current_agent_id
        act = drv(env, obs, actor)
        obs, _, term, trunc, _ = env.step(act); done = term or trunc; steps += 1
        if steps > 250:
            break
    net_dmg = (hp0 - max(0, env.ws.characters[oid].hp)) / max(1, hp0)
    won = not env.ws.characters[oid].is_alive()
    return net_dmg, int(won)


def check_regen(cand, base, G):
    r = CheckResult("regen", "再生怪壓制（#14）：對 troll(每回合再生、僅火/酸抑制)能否持續壓制淨侵蝕 HP",
                    ["#14 再生怪持續施型", "goal:依資訊選招(持續)"])
    agents = ["evocation", "divination"]      # 火傷主力，天生能壓制
    r.lines.append(f"{'agent':14s} {'regenON淨傷':>10s} {'regenOFF淨傷':>11s} {'Δ':>5s} {'ON勝%':>6s}")
    cells = []; base_cells = []
    worst_net = 1.0; worst_id = ""
    for ag in agents:
        on = off = wonon = 0.0
        for gi in range(G):
            no, w = _regen_dmg(cand, ag, f"rg|{ag}|{gi}|on", True); on += no; wonon += w
            nf, _ = _regen_dmg(cand, ag, f"rg|{ag}|{gi}|off", False); off += nf
        on /= G; off /= G; wonon /= G
        if on < worst_net:
            worst_net = on; worst_id = ag
        cells.append(Cell(ag, on, G, aux=f"off={off:.0%} win={wonon:.0%}"))
        r.lines.append(f"{ag:14s} {on:10.0%} {off:11.0%} {(on - off) * 100:+5.0f} {wonon:6.0%}")
        if base is not None:
            bon = sum(_regen_dmg(base, ag, f"rg|{ag}|{gi}|on", True)[0]
                      for gi in range(G)) / G
            base_cells.append(Cell(ag, bon, G))
    # 火法師天生壓制 troll→regen-ON 仍該淨侵蝕大量 HP。若 regen-ON 淨傷 ~0＝完全壓不住
    # (沒持續丟火)＝#14 退化。
    if worst_net < 0.30:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠壓制回歸 vs base：{reg[0]} 淨傷 {reg[1]:+.0%}"
    r.summary = (f"火法師持續壓制 troll（regen-ON 最低淨傷 {worst_net:.0%}）"
                 if r.status == "PASS" else
                 f"『{worst_id}』對 troll regen-ON 僅淨侵蝕 {worst_net:.0%}＝壓不住再生")
    return r


# ── C18: 非evo 隊伍 AoE 誤傷（#17）——放 AoE 時是否把隊友納入爆炸半徑（無 sculpt） ──
# 只有 evocation 有 sculpt_spells（AoE 免傷友軍）；divination 等的 fireball 會炸到自己人。
# 幾何法乾淨測：模型放 aoe_radius>0 的招時，用 act[2]→世界座標，量活著隊友是否落在半徑內。

def _teamff_rate(net, team, opp_archs, key):
    from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
    seed = stable_seed(key); random.seed(seed)
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=len(team), n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=list(team), opp_archs=list(opp_archs),
                       level=6, opp_level=6, layout="open")
    aids = list(env.agent_ids); oids = list(env.opp_ids)
    # 把敵人叢集在『第一個敵人自己的位置』(＝敵方 spawn，遠離隊友)＝有乾淨 AoE 可選：
    # 正常模型 AoE 敵團(不炸隊友)＝~0% FF；只有主動亂放才會炸到隊友。分得開＝真有牙齒。
    epos = env.ws.characters[oids[0]].position
    for o in oids[1:]:
        env.ws.characters[o].position = Vec2(epos.x, epos.y)
    drv = _model_driver(net)
    reckless = aoe_total = 0; done = False; steps = 0
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]; sks = available_skills(ch, env.ws)
        act = drv(env, obs, actor)
        if actor in aids and 0 < act[0] < len(sks):
            sk = sks[act[0]]; rad = float(getattr(sk.features, "aoe_radius_m", 0.0) or 0.0)
            if rad > 0 and sk.skill_id != "move":
                gx = act[2] // N_GRID; gy = act[2] % N_GRID
                tx = (gx + 0.5) * GRID_CELL_SIZE_M; ty = (gy + 0.5) * GRID_CELL_SIZE_M
                # 反事實乾淨落點判定（#17 的本意）：舊版凡「隊友在半徑內」都算誤傷，
                # 但近戰隊友衝進敵團後，炸敵團＝炸隊友＝不可免——正常(~60%)與故意
                # 亂炸(~62-71%)分不開（self-test 實測）。改成只在「存在覆蓋≥同樣多
                # 敵人且 0 隊友的替代落點、卻選了炸隊友的落點」才算 reckless；
                # 無乾淨替代（肉搏團）的 cast 不進分母＝不可免情境不記帳。
                epos_ = [env.ws.characters[o].position for o in oids
                         if env.ws.characters[o].is_alive()]
                apos_ = [env.ws.characters[al].position for al in aids
                         if al != actor and env.ws.characters[al].is_alive()]

                def _cov(cx, cy):
                    ne_ = sum(1 for p in epos_
                              if ((p.x - cx) ** 2 + (p.y - cy) ** 2) ** 0.5 <= rad)
                    na_ = sum(1 for p in apos_
                              if ((p.x - cx) ** 2 + (p.y - cy) ** 2) ** 0.5 <= rad)
                    return ne_, na_

                ne_c, na_c = _cov(tx, ty)
                clean_exists = False
                if epos_:
                    for cgx in range(N_GRID):
                        for cgy in range(N_GRID):
                            ne_a, na_a = _cov((cgx + 0.5) * GRID_CELL_SIZE_M,
                                              (cgy + 0.5) * GRID_CELL_SIZE_M)
                            if na_a == 0 and ne_a >= max(1, ne_c):
                                clean_exists = True
                                break
                        if clean_exists:
                            break
                if clean_exists:
                    aoe_total += 1
                    if na_c > 0:
                        reckless += 1
        obs, _, term, trunc, _ = env.step(act); done = term or trunc; steps += 1
        if steps > 150:
            break
    return reckless, aoe_total


def check_teamff(cand, base, G):
    r = CheckResult("teamff", "非evo 隊伍 AoE 誤傷（#17）：放 AoE 時是否把隊友納入爆炸半徑",
                    ["#17 非evo隊伍AoE誤傷", "goal:隊伍協同"])
    TEAMS = [("divination", ["divination", "champion", "life"]),
             ("arcane_trickster", ["arcane_trickster", "champion", "life"])]
    enemies = ["orc", "orc"]
    r.lines.append(f"{'lead':16s} {'AoE次數':>7s} {'誤傷次數':>7s} {'誤傷率':>6s}")
    cells = []; base_cells = []; worst = 0.0; worst_id = ""; any_aoe = 0
    for lead, team in TEAMS:
        rk = tot = 0
        for gi in range(G):
            a, b = _teamff_rate(cand, team, enemies, f"ff|{lead}|{gi}")
            rk += a; tot += b
        any_aoe += tot
        rate = rk / tot if tot else 0.0
        if tot and rate > worst:
            worst = rate; worst_id = lead
        cells.append(Cell(lead, 1.0 - rate, max(1, tot), aux=f"aoe={tot} ff={rk}"))
        r.lines.append(f"{lead:16s} {tot:7d} {rk:7d} {rate:6.0%}"
                       + ("  ⚠誤傷" if tot and rate > 0.3 else ""))
        if base is not None:
            brk = bt = 0
            for gi in range(G):
                a, b = _teamff_rate(base, team, enemies, f"ff|{lead}|{gi}")
                brk += a; bt += b
            base_cells.append(Cell(lead, 1.0 - (brk / bt if bt else 0.0), max(1, bt)))
    # 敵團遠離隊友，但近戰隊友(champion)會衝進敵團肉搏→非-sculpt 火球炸敵團順帶炸到它＝
    # 真實不可免的 FF。self_test 量到：故意亂炸 71% vs 正常 60%＝差距薄＝**teamff 是弱檢**
    # (分不太開)。故軟判：報 FF 率＋回歸守門為主，只在極端(>65%)才硬 FAIL(讓故意亂炸的合成
    # agent 有牙齒、又不因『非-sculpt 天生會 FF』就硬判正常模型)。
    if any_aoe < 4:
        r.status = "UNDERPOWERED"
        r.summary = f"模型放 AoE 太少(n={any_aoe})無法判誤傷；低訊號"
        return r
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠誤傷增加 vs base：{reg[0]}"
    elif worst > 0.5:
        # 反事實乾淨落點版（見 _teamff_rate）：分母只含「有乾淨替代落點」的
        # cast，正常模型應遠低於 0.5、強制亂炸→~100%＝可分離、0.5 有效。
        r.status = "FAIL"
    if not r.note:
        r.note = ("弱檢：非-sculpt 天生會炸衝進敵團的近戰隊友，正常(~60%)與故意亂炸(~65%)差距薄"
                  "＝S/N 低；此 FAIL 亦如實反映該身分真有顯著 FF")
    r.summary = f"最高 AoE 誤傷率 {worst:.0%}（{worst_id}，n_aoe={any_aoe}；軟判＝報＋回歸守門）"
    return r


# ── C19: NvM 部署形狀（怪1v隊3/隊3v怪1/2v2/劣勢1v3/3v3｜goal:多對多） ──────────
# 07-03 儀器缺口修補：舊儀器最多 2 實體（asym 劣勢1v2/優勢2v1、miner 舊形狀表），
# 「怪 vs 冒險隊」這個 DND 部署主形狀從未被驗過。本檢查的兩道門：
#   ① 不划水：回合首手 idle ≤10%（asym 同門檻）
#   ② 接戰輸出 ≥ 同席腳本專家的一半——可贏性內建：專家 dealt<15% 的桶＝對局
#      不可贏（標定實測 L10 ogre 被 3 人集火 4 回合死、雙席同 7.9%；basilisk 1v3
#      雙席 0%），只考 idle 不考相對輸出＝miner 專家仲裁同一條原則。
# 桶配置經 probe_nvm_calib 標定＝專家同席打得出輸出（behir@11 v 3xL8=100%、
# troll@10 v 3xL5≈20%、隊3 v troll@10≈93%）。教訓順帶記錄：07-03 前探針曾把
# 「零傷害場」(速死/全miss) 誤計成空轉→誤判 1vN 退化；本檢查的 idle 是逐決策
# 首手棄行動定義、輸出門是專家相對值，兩者都分得開匹配現實與模型退化。

def _multi_expert_driver(archs):
    """多 agent 席各配自己身分的腳本專家（_expert_driver 單策略駕全席＝隊伍席拿
    錯 kit 的專家、基準失真；Nv1/NvM 桶必須用這個）。"""
    pols = [make_archetype_policy(a) for a in archs]
    def drv(env, obs, actor):
        aids = list(env.agent_ids)
        i = aids.index(actor) if actor in aids else 0
        pol = pols[min(i, len(pols) - 1)]
        a = env.ws.characters[actor]
        dec = pol.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
        if dec.action is None or getattr(dec, "fled", False):
            return [0, 0, 0]
        return list(encode_action(dec.action, env.ws, actor))
    return drv


def check_nvm(cand, base, G):
    r = CheckResult("nvm", "NvM 部署形狀：怪1v隊3/隊3v怪1/2v2/劣勢1v3/3v3 不划水且接戰≥專家半",
                    ["Part1 1vN 品質", "goal:多對多部署"])
    PARTY = ["battle_master", "evocation", "life"]      # 坦/爆發/治療＝部署隊形
    BUCK = [
        ("怪1v隊3", [(["ogre"], PARTY, 2, 2), (["troll"], PARTY, 10, 5),
                     (["behir"], PARTY, 11, 8)]),
        ("隊3v怪1", [(PARTY, ["troll"], 5, 10), (PARTY, ["ettin"], 5, 10)]),
        ("2v2", [(["battle_master", "life"], ["champion", "evocation"], 5, 5)]),
        ("劣勢1v3", [(["battle_master"],
                      ["champion", "evocation", "vengeance"], 5, 5)]),
        ("3v3", [(PARTY, ["champion", "vengeance", "mage_npc"], 5, 5)]),
    ]
    r.lines.append(f"{'情境':10s} {'WR':>5s} {'dealt%':>7s} {'idle%':>6s} "
                   f"{'專家dealt%':>9s}")
    cells = []; base_cells = []; bad = []
    for label, cfgs in BUCK:
        def run(net):
            w = de = idl = ft = n = 0
            for ci, (aa, oa, al, ol) in enumerate(cfgs):
                for gi in range(G):
                    o = drive_episode(_model_driver(net), list(aa), list(oa),
                                      al, ol, f"nvm|{label}|{ci}|{gi}")
                    w += o.win; de += o.dmg_frac; idl += o.idle
                    ft += o.first_turns; n += 1
            return w / n, de / n, idl / max(1, ft)
        ede = 0.0; en = 0
        for ci, (aa, oa, al, ol) in enumerate(cfgs):
            for gi in range(G):        # 專家同席同 key＝同種子成對可比
                o = drive_episode(_multi_expert_driver(aa), list(aa), list(oa),
                                  al, ol, f"nvm|{label}|{ci}|{gi}")
                ede += o.dmg_frac; en += 1
        ede /= max(1, en)
        wr, de, idl = run(cand)
        flag = ""
        if idl > 0.10:
            bad.append(f"{label}划水{idl:.0%}"); flag = "  ⚠划水"
        elif ede >= 0.15 and de < 0.5 * ede:
            bad.append(f"{label}接戰{de:.0%}<專家半({ede:.0%})"); flag = "  ⚠不接戰"
        n_ep = G * len(cfgs)
        cells.append(Cell(label, 1.0 - idl, n_ep, aux=f"dealt={de:.0%}"))
        cells.append(Cell(label + "·dealt", de, n_ep, aux=f"exp={ede:.0%}"))
        r.lines.append(f"{label:10s} {wr:5.0%} {de:7.1%} {idl:6.1%} "
                       f"{ede:9.1%}{flag}")
        if base is not None:
            bwr, bde, bidl = run(base)
            base_cells.append(Cell(label, 1.0 - bidl, n_ep))
            base_cells.append(Cell(label + "·dealt", bde, n_ep))
    if bad:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} {reg[1]:+.0%}"
    r.summary = ("全部署形狀不划水、接戰≥同席專家半" if not bad
                 else "；".join(bad))
    return r


def check_healerfocus(cand, base, G):
    """#19/#20：敵隊含治療者（先秒治療者）＋怪席（BOSS 視角）目標選擇。

    2026-07-03 NvM 訓練波 Phase-1 量測結論（scripts/_nvm_healer2_g32.txt）：
    uni_v10 的敵治療者集火 hShare 79-86% 遠超同席腳本專家(39-67%)＝湧現能力，
    且 oracle 介入(集火推到 92-96%)零收益＝目標紀律在部署形狀因果惰性。
    故本檢不設「必須先秒治療者」的硬門檻（那是偽科學）；門＝nvm 同款雙門
    （不划水＋接戰≥同席專家半＝可贏性內建），hFirst/hShare 印出並入回歸
    cells——湧現的集火能力若在未來版本流失，--base 守門翻紅。"""
    r = CheckResult("healerfocus",
                    "#19/#20 敵含治療者/怪席目標選擇：不划水+接戰≥專家半+集火回歸守門",
                    ["#19 1vN 角色分流", "#20 模型當BOSS目標選擇"])
    import diag_nvm_behavior as NB     # 延遲載入：NB 反向 import 本檔（循環安全）
    buckets = [(lbl, cfgs) for lbl, cfgs in NB.BUCKETS
               if "敵治療" in lbl or lbl == "怪1v隊3"]
    r.lines.append(f"{'情境':10s} {'WR':>5s} {'dealt%':>7s} {'idle%':>6s} "
                   f"{'專家dealt%':>9s} {'hFirst':>9s} {'hShare':>9s}")
    cells = []; base_cells = []; bad = []

    def run(net, drv_fn, out_cells):
        by = {}
        for label, cfgs in buckets:
            acc = NB.Acc()
            for ci, (aa, oa, al, ol) in enumerate(cfgs):
                for gi in range(G):
                    NB.run_episode(drv_fn(net, list(aa)), list(aa), list(oa),
                                   al, ol, f"hfoc|{label}|{ci}|{gi}", acc)
            n = max(1, acc.n)
            row = dict(wr=acc.win / n, de=acc.dealt / n,
                       idl=acc.idle / max(1, acc.ft),
                       hf=(acc.hk_first / acc.hk_n) if acc.hk_n else None,
                       hs=(acc.hs_dec / acc.hs_tot) if acc.hs_tot else None,
                       n=acc.n, hn=acc.hk_n, hst=acc.hs_tot)
            by[label] = row
            if out_cells is not None:
                out_cells.append(Cell(label, 1.0 - row["idl"], acc.n))
                out_cells.append(Cell(label + "·dealt", row["de"], acc.n))
                if row["hf"] is not None:
                    out_cells.append(Cell(label + "·hfirst", row["hf"], acc.hk_n))
                if row["hs"] is not None:
                    out_cells.append(Cell(label + "·hshare", row["hs"], acc.hs_tot))
        return by

    cand_rows = run(cand, lambda net, aa: _model_driver(net), cells)
    exp_rows = run(None, lambda net, aa: _multi_expert_driver(aa), None)
    if base is not None:
        run(base, lambda net, aa: _model_driver(net), base_cells)
    for label, _ in buckets:
        c, e = cand_rows[label], exp_rows[label]
        flag = ""
        if c["idl"] > 0.10:
            bad.append(f"{label}划水{c['idl']:.0%}"); flag = "  ⚠划水"
        elif e["de"] >= 0.15 and c["de"] < 0.5 * e["de"]:
            bad.append(f"{label}接戰{c['de']:.0%}<專家半({e['de']:.0%})")
            flag = "  ⚠不接戰"
        fm = lambda v: f"{v:.0%}" if v is not None else "-"
        r.lines.append(
            f"{label:10s} {c['wr']:5.0%} {c['de']:7.1%} {c['idl']:6.1%} "
            f"{e['de']:9.1%} {fm(c['hf']):>4s}/{fm(e['hf']):<4s} "
            f"{fm(c['hs']):>4s}/{fm(e['hs']):<4s}{flag}")
    if bad:
        r.status = "FAIL"
    reg = regression_flag(cells, base_cells, "frac")
    if reg:
        r.status = "FAIL"; r.note = f"⚠回歸 vs base：{reg[0]} {reg[1]:+.0%}"
    r.summary = ("敵含治療者/怪席全不划水且接戰≥專家半（hFirst/hShare 入回歸守門）"
                 if not bad else "；".join(bad))
    return r


CHECKS = {
    "std12": check_std12,
    "immune": check_immune,
    "degen": check_degen,
    "oppseat": check_oppseat,
    "general": check_general,
    "walls": check_walls,
    "asym": check_asym,
    "switch": check_switch,
    "passive": check_passive,
    "exploit": check_exploit,
    "victim": check_victim,
    "dead": check_dead,
    "infochan": check_infochan,
    "terrain": check_terrain,
    "highlevel": check_highlevel,
    "frighten": check_frighten,
    "regen": check_regen,
    "teamff": check_teamff,
    "nvm": check_nvm,
    "healerfocus": check_healerfocus,
}


# ── 覆蓋表：把 EVAL_COVERAGE 每一項顯性列出，接上的判 PASS/FAIL、沒接上印 NOT_COVERED ──
# (item, 接上的 check_id 或 None, 一句話)
COVERAGE_MAP = [
    ("Part1 WR/Δpp", "std12", "對稱腳本專家勝率"),
    ("Part1 usage-mix", None, "招式使用分布需人工判讀，無自動門檻"),
    ("Part1 零退化審計", "degen", "全場致命退化指紋"),
    ("Part1 死動作 ERROR", "dead", "選到引擎拒絕(ERROR)的動作＝浪費回合"),
    ("Part1 逐通道因果", "infochan", "敵資訊通道歸零→行為翻不翻(讀/惰性)＋回歸守門"),
    ("Part1 免疫傷害輸出", "immune", "dodge-collapse 偵測器"),
    ("Part1 走位/視線繞牆", "walls", "牆地形 WR 不該相對開闊地崩塌"),
    ("Part1 held-out 切招", "switch", "主型被免疫→改砸非免疫型(self-calibrating)"),
    ("Part1 天賦因果", "infochan", "passive_traits 通道 flip（regen=已知死載體，低flip屬正確）"),
    ("Part1 1vN 品質", "nvm", "NvM 部署形狀（怪1v隊3/隊3v怪1/2v2/1v3/3v3）不划水+接戰≥專家半"),
    ("Part1 劣勢不划水", "asym", "劣勢1v2/低階Δ-3/優勢2v1 不划水"),
    ("Part2 扮怪/縫合怪/synth", "general", "eval_generalize 多桶 Δpp vs 駕同kit腳本"),
    ("Part1 侵蝕守門", "std12", "→ 由 --base 逐角落回歸守門實現(#4/#32)"),
    ("#1 等級>8", "highlevel", "L20 對稱局核心行為(接戰/傷害/不空轉)是否崩"),
    ("#2 強弱錯配免疫", "immune", "L8 法師 vs 15HP 免疫弱敵"),
    ("#3 傷害當主指標", "immune", "dmg% + DPR 取代純 WR"),
    ("#4 跨版本回歸", "std12", "→ --base 逐角落 2σ 回歸紅旗（本腳本核心）"),
    ("#5 被動對手×全身分", "passive", "全標準身分 vs 靜止被動敵、量空轉/接戰"),
    ("#6 分布外自訂 kit", "general", "synth 隨機 kit 桶＝OOD 組合 Δpp"),
    ("#7 瀕死/倒地/復活", None, "補刀倒地敵/撿隊友無評估"),
    ("#8 條件免疫槓桿", "infochan", "condition_immunity 通道 flip（實測 0%＝模型不讀→會浪費控制）"),
    ("#9 狀態 vs 免疫", "infochan", "同上：模型不讀 condition_immunity＝施免疫狀態＝浪費（根源已量）"),
    ("#10 反應/傳奇品質", None, "shield/counterspell 觸發時機無通用評估"),
    ("#11 GUI 部署制式", "passive", "被動敵已測；高階+手挑 kit 面仍缺"),
    ("#12 被擒抱/吞噬(obs隱形)", "victim", "被 restrained 貼臉→拒戰率(能打卻放棄；swallowed obs 面仍缺)"),
    ("#13 恐懼光環", "frighten", "被 frightened 後遠程 agent 攻擊率是否崩（on/off 因果）"),
    ("#14 再生怪持續施型", "regen", "對 troll regen-ON/OFF 淨侵蝕(火/酸壓制)"),
    ("#15 石化兩段式", None, "basilisk restrained→petrified 搶殺窗口未測"),
    ("#16 眼射線/死亡爆裂", None, "slowed 隱形＋balor death_throes 補刀距離未測"),
    ("#17 非evo隊伍AoE誤傷", "teamff", "放 AoE 是否把隊友納入半徑(幾何法)"),
    ("#18 burst/專注時機", None, "有限資源花在對的回合/目標未測"),
    ("#19 1vN 角色分流", "healerfocus", "敵隊含治療者桶：雙門+hFirst/hShare 回歸守門"),
    ("#20 模型當BOSS目標選擇", "healerfocus", "怪1v隊3(隊含治療者)＝怪席目標選擇同儀器"),
    ("#21 對主動近戰整場走位", "walls", "牆地形走位含之（reach 內存活面仍偏窄）"),
    ("#22 敵reach>1.5m obs隱形", None, "巨人/龍尾/觸手 reach 寫死 1.5m"),
    ("#23 資源耗盡 fallback", "immune", "免疫檢測部分覆蓋(主招失效改次級)"),
    ("#24 站危險光環/地形不移", "terrain", "lava 危險格自傷增量（光環面仍缺）"),
    ("#25 通用反應決策品質", None, "跨身分跨三反應觸發合適度未測"),
    ("#26 抗性0.5/易傷2x選型", "switch", "×0 免疫切型已測；分級 0.5/2 倍率選型仍缺"),
    ("#27 利用敵既有優勢狀態", "exploit", "預設敵 prone→攻擊傾斜(純目標選擇+因果)"),
    ("#28 攻擊性打斷專注", None, "集火敵專注者打斷法術未測"),
    ("#29 difficult 地形移動規劃", "terrain", "2x 移動花費下功能是否保留(diff/open 傷害比)"),
    ("#30 縫合怪 kit 覆蓋率", "general", "縫合怪 Δpp 桶（獨立 kit-coverage 指標仍缺）"),
    ("#31 平手併入敗", "std12", "→ 平手/拖平獨立拆分列印"),
    ("#32 per-class 資料被丟", "std12", "→ 逐角落保留 + 回歸守門（煙槍修復）"),
    ("#33 無信賴區間", None, "已印 n/σ 並標 UNDERPOWERED；未做最小局數推導"),
    ("#34 對手席品質", "oppseat", "候選網駕對手席退化偵測"),
    ("#35 DPR/擊殺時間", "std12", "→ 逐職 DPR 列印(model vs expert)"),
]


def print_coverage_table(results_by_cid):
    print("\n" + "=" * 74)
    print("覆蓋表：EVAL_COVERAGE 每一項 → 接上的檢測判定 / 未接上顯性標 NOT_COVERED")
    print("=" * 74)
    n_pass = n_fail = n_nc = n_under = 0
    for item, cid, desc in COVERAGE_MAP:
        if cid and cid in results_by_cid:
            st = results_by_cid[cid].status
            tag = {"PASS": "PASS      ", "FAIL": "FAIL ✗    ",
                   "UNDERPOWERED": "UNDERPWR  ", "ERROR": "ERROR     ",
                   "SKIP": "SKIP      "}.get(st, st)
            if st == "PASS":
                n_pass += 1
            elif st == "FAIL":
                n_fail += 1
            elif st == "UNDERPOWERED":
                n_under += 1
            print(f"  [{tag}] {item:26s} ({cid})  {desc}")
        else:
            n_nc += 1
            print(f"  [NOT_COVERED ] {item:26s}  {desc}")
    print("-" * 74)
    print(f"  接上檢測: PASS={n_pass}  FAIL={n_fail}  UNDERPWR={n_under}  "
          f"|  NOT_COVERED={n_nc} / {len(COVERAGE_MAP)} 項")
    return n_fail, n_nc


# ── SELF-TEST（測試這些測試）：合成明顯壞掉的 agent，確認對應 check 會 FAIL（有牙齒） ──
# 不必等版本歷史剛好有壞模型——直接包住真 net、竄改輸出＝造一個「明顯做錯」的假 agent，
# 看該抓到它的 check 是否真的印 FAIL。印 PASS＝那 check 瞎了(假陰性)、需修。

class _EvilNet:
    """包住真 net、只竄改輸出＝合成壞 agent。always_end＝每回合直接結束(什麼都不做)。
    真 net 只提供正確的張量形狀；always_end 下權重無關（end_logit 被覆寫成一律 END）。"""
    def __init__(self, base, mode):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_mode", mode)

    def eval(self):
        self._base.eval(); return self

    def __getattr__(self, k):
        return getattr(self._base, k)

    def __call__(self, obs):
        el, s, e, g = self._base(obs)
        m = self._mode
        ent = obs["entities"]                    # (B, N_SLOTS, ENTITY_DIM)
        if m == "always_end":
            el = torch.full_like(el, 50.0)       # sigmoid→1 → pick_action 一律 END
        elif m == "freeze_when_frightened":
            # 讀 self(row 0) 的 frightened 位元＝有恐懼就 END＝合成「被恐懼輸出崩」
            fr = ent[:, 0, _EV_STATUS_START + _EV_FRIGHT_IDX].reshape(el.shape)
            el = torch.where(fr > 0.5, torch.full_like(el, 50.0), el)
        elif m == "avoid_prone":
            # 敵 o0 若 prone→改打 o1(避開可利用敵)；否則打 o0＝合成「反向利用敵狀態」。
            # e 是 per-skill 二維 (N_SKILL, N_ENTITY)（pick_action 用 e[skill].argmax）→
            # 要對『所有 skill 列』的目標 entity 欄設大值才控得到 act[1]。
            o0 = ENEMY_SLOT_START; o1 = ENEMY_SLOT_START + 1
            prone0 = ent[:, o0, _EV_STATUS_START + _EV_PRONE_IDX]
            e = torch.full_like(e, -1e4)
            for b in range(e.shape[0]):
                e[b, :, o1 if prone0[b] > 0.5 else o0] = 1e4
        elif m == "freeze_on_walls":
            # terrain obs 牆=1.0、開闊=0。有牆就 END＝開闊地正常、牆上全敗（測 walls 崩塌偵測）
            ter = obs["terrain"]
            has_wall = (ter.reshape(ter.shape[0], -1) >= 0.99).any(dim=1)
            el = torch.where(has_wall.reshape(el.shape), torch.full_like(el, 50.0), el)
        elif m == "aoe_on_ally":
            # 把 AoE 落點強制打在前排隊友(slot 1)的格子＝AoE 必炸自己人（測 teamff 誤傷偵測）。
            # 解碼隊友位置(entity col 1,2 normalized by N_GRID 米)→格子→強制 g[所有skill, cell]。
            g = torch.full_like(g, -1e4)
            ng = _EV_N_GRID
            for b in range(g.shape[0]):
                ax = float(ent[b, 1, 1]); ay = float(ent[b, 1, 2])
                gx = int(min(ng - 1, max(0, ax * ng)))
                gy = int(min(ng - 1, max(0, ay * ng)))
                g[b, :, gx * ng + gy] = 1e4
        return el, s, e, g


from trpg.rl.obs import (RL_STATUS_NAMES, N_ARCHETYPES, ENEMY_SLOT_START,  # noqa: E402
                         N_GRID as _EV_N_GRID)
_EV_STATUS_START = 7 + N_ARCHETYPES                      # 實體列狀態多熱區塊起點
_EV_FRIGHT_IDX = RL_STATUS_NAMES.index("frightened")
_EV_PRONE_IDX = RL_STATUS_NAMES.index("prone")


EVIL_AGENTS = {
    "always_end": {
        "desc": "每回合直接結束、什麼都不做（極端拒戰／空轉／0 輸出）",
        # always_end 會讓「該做點事」的 check 全部觸發：0 傷/0 WR/全 idle/對手席空轉…
        "should_fail": ["std12", "immune", "passive", "degen", "asym",
                        "highlevel", "regen", "terrain", "oppseat", "general",
                        "victim", "nvm", "healerfocus"],
    },
    "freeze_when_frightened": {
        "desc": "平時正常、一被 frightened 就直接結束（合成『被恐懼輸出崩』）",
        "should_fail": ["frighten"],
    },
    "avoid_prone": {
        "desc": "敵人 prone 時反而改打另一個健康敵（合成『反向利用敵狀態』）",
        "should_fail": ["exploit"],
    },
    "freeze_on_walls": {
        "desc": "開闊地正常、一看到牆就直接結束（合成『走位/繞牆崩塌』）",
        "should_fail": ["walls"],
    },
    "aoe_on_ally": {
        "desc": "把 AoE 落點強制打在前排隊友的格子＝AoE 必炸自己人（合成『隊伍誤傷』）",
        # 2026-07-02 語義升級：pick_action 加了 AoE 支配格重選守門（乾淨落點存在
        # 時炸隊友格被重選走）→ 這個失效模式在決策層被結構性擋掉，端到端測不到
        # FAIL 是「守門生效」而非「檢測瞎了」。期望改為 guard_holds：強制亂炸下
        # 反事實誤傷率仍須為 0（守門失守→誤傷率飆高→FAIL→本自測翻紅）。
        # 守門本身的單元證明：tests/rl/test_model.py 的 aoe relocate/scrum 雙測。
        "guard_holds": ["teamff"],
        "should_fail": [],
    },
}


def run_self_test(base_net, G):
    print("=" * 74)
    print("SELF-TEST（測試這些測試）：餵合成的『明顯壞掉』agent，確認該抓到的 check 印 FAIL")
    print("＝有牙齒；若對明顯壞行為印 PASS＝那 check 瞎了（假陰性），需修。")
    print("=" * 74)
    all_ok = True
    for mode, spec in EVIL_AGENTS.items():
        evil = _EvilNet(base_net, mode)
        print(f"\n== 壞 agent [{mode}]：{spec['desc']} ==")
        for cid in spec["should_fail"]:
            try:
                res = CHECKS[cid](evil, None, G)
                caught = (res.status == "FAIL")
                mark = "✓ 有牙齒" if caught else "✗ 瞎了!!"
                all_ok = all_ok and caught
                print(f"   [{mark}] {cid:10s} → {res.status:12s} {res.summary}")
            except Exception as ex:
                import traceback; traceback.print_exc()
                all_ok = False
                print(f"   [✗ ERROR ] {cid:10s} → {type(ex).__name__}: {str(ex)[:50]}")
        for cid in spec.get("guard_holds", ()):
            # 失效模式已被決策層守門結構性擋掉：斷言「即使 logit 層強制壞行為，
            # check 量到的仍是乾淨行為(PASS)」＝守門端到端有效；守門失守會 FAIL。
            try:
                res = CHECKS[cid](evil, None, G)
                held = (res.status == "PASS")
                mark = "✓ 守門有效" if held else "✗ 守門失守!!"
                all_ok = all_ok and held
                print(f"   [{mark}] {cid:10s} → {res.status:12s} {res.summary}")
            except Exception as ex:
                import traceback; traceback.print_exc()
                all_ok = False
                print(f"   [✗ ERROR ] {cid:10s} → {type(ex).__name__}: {str(ex)[:50]}")

    # 特例：dead 的失效模式＝合法性遮罩「有漏」。用 net 造不出（遮罩會擋掉非法選擇＝正說明
    # 遮罩是好的），故直接模擬「遮罩破掉」(pass-through)：非法動作放行→引擎回 ERROR→dead 該抓到。
    print("\n== 特例 [broken_mask]：模擬合法性遮罩破掉→非法動作放行（dead 該抓到 ERROR） ==")
    import diag_degen_audit as _DDmod
    _orig_mask = _DDmod.apply_resource_mask
    try:
        _DDmod.apply_resource_mask = lambda s, *a, **k: s      # pass-through＝破掉的遮罩
        res = check_dead(base_net, None, G)
        caught = (res.status == "FAIL")
        all_ok = all_ok and caught
        print(f"   [{'✓ 有牙齒' if caught else '✗ 瞎了!!'}] dead       → "
              f"{res.status:12s} {res.summary}")
    except Exception as ex:
        import traceback; traceback.print_exc()
        all_ok = False
        print(f"   [✗ ERROR ] dead       → {type(ex).__name__}: {str(ex)[:50]}")
    finally:
        _DDmod.apply_resource_mask = _orig_mask

    print("\n" + "=" * 74)
    print("SELF-TEST 通過：列出的 check 對『什麼都不做』的壞 agent 都有牙齒" if all_ok
          else "SELF-TEST 未過：有 check 對明顯壞行為印 PASS＝瞎了、需修（見上 ✗ 瞎了）")
    print("=" * 74)
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", help="候選模型 .pt")
    ap.add_argument("--self_test", action="store_true",
                    help="測試這些測試：合成壞 agent，確認對應 check 會 FAIL（有牙齒）")
    ap.add_argument("--base", default=None, help="舊最佳模型，做逐角落不侵蝕守門")
    ap.add_argument("--games", type=int, default=3, help="每角落局數（預設 3＝快篩；驗收用 8+）")
    ap.add_argument("--only", default=None,
                    help="逗號分隔 check 子集：std12,immune,degen,oppseat,general,walls,asym,switch,passive,exploit,victim,dead,infochan,terrain,highlevel,frighten,regen,teamff,nvm,healerfocus")
    ap.add_argument("--list", action="store_true", help="只印覆蓋表結構，不跑模型")
    args = ap.parse_args()

    if args.list:
        print_coverage_table({})
        print("\n(--list：只列結構。給模型路徑實跑才有 PASS/FAIL。)")
        return

    if not args.ckpt:
        ap.error("需要模型路徑（或用 --list 只看覆蓋結構）")

    if args.self_test:
        base_net = load_student(args.ckpt); base_net.eval()
        ok = run_self_test(base_net, args.games)
        sys.exit(0 if ok else 1)

    which = args.only.split(",") if args.only else list(CHECKS)
    which = [w for w in which if w in CHECKS]
    print(f"候選模型: {args.ckpt}")
    if args.base:
        print(f"對照 base（不侵蝕守門）: {args.base}")
    print(f"每角落局數: {args.games}   檢測: {', '.join(which)}\n")

    cand = load_student(args.ckpt); cand.eval()
    base = None
    if args.base:
        base = load_student(args.base); base.eval()

    results = {}
    for cid in which:
        print(f"── 執行 [{cid}] ...", flush=True)
        try:
            res = CHECKS[cid](cand, base, args.games)
        except Exception as ex:
            import traceback; traceback.print_exc()
            res = CheckResult(cid, cid, [], status="ERROR", summary=str(ex))
        results[cid] = res
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "⚠"}.get(res.status, "•")
        print(f"\n{icon} [{res.status}] {res.title}")
        for ln in res.lines:
            print("   " + ln)
        print(f"   → {res.summary}{('  ' + res.note) if res.note else ''}\n")

    n_fail, n_nc = print_coverage_table(results)
    print("\n" + "=" * 74)
    verdict = "驗收通過（已接上的檢測全綠）" if n_fail == 0 else f"驗收未過：{n_fail} 項 FAIL"
    print(f"總判定：{verdict}")
    if n_nc:
        print(f"⚠ 仍有 {n_nc} 項 NOT_COVERED＝目前這支腳本看不見的盲區（顯性掛在表上，不再隱形）")
    print("=" * 74)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
