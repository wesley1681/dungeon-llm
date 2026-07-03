"""NvM 行為差距診斷（diag_nvm_behavior）：模型 vs 同席腳本專家，成對種子高局數。

動機（2026-07-03 NvM 訓練波 Phase 1）：eval_gate 第19檢 G=3 只能守「不划水/接戰≥半」，
3v3 80.2 vs 專家 83.1 這種 2.9pp 差距在 G=3 全是噪音。訓練波開刀前必須先量：
  (a) 差距是否真實（高局數 WR / dealt / recv）
  (b) 差距長在哪個行為維度（目標選擇？集火？補刀？換目標？治療分配？）
模型與專家用同一把尺（成對種子、同席位、同解碼路徑 available_skills+
_entity_id_at_slot 取「意圖目標」）＝差值可信，不吃 HP 差分的光環/再生噪音。

指標（全部只在「單體傷害技」決策上計，AoE 另計次數）：
  low_pick   有 ≥2 活敵且最低血 <40% 時，選中最低血敵的比率（補刀/易殺優先）
  switch     上一個傷害目標還活著卻換打別人的比率（集火紀律，越低越專注）
  downed     打已倒地敵（還有站立敵時）的比率（過殺傾向）
  HHI        整場傷害決策在敵人間的集中度（1/M=均攤, 1=全集火）
  first_kill 第一個敵人死亡時的我方累積決策數（越小=集火轉化越快）
  kill_conv  敵人首次 <40% 到死亡之間的我方決策數（收尾速度）
  heal@full  治療目標血量 ≥90% 的比率（治療浪費）

用法:
  python scripts/diag_nvm_behavior.py models/unified/uni_v10.pt --games 24
  python scripts/diag_nvm_behavior.py <m> --games 8 --buckets 3v3,2v2
"""
from __future__ import annotations
import sys, os, argparse, random, math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import _entity_id_at_slot
from trpg.rl.obs import partition_entities, ENEMY_SLOT_START
from trpg.rl.model import _enemy_target_legal
from trpg.engine.skill import available_skills, TargetType

from distill_routed import load_student
from eval_routed import stable_seed
from eval_gate import _model_driver, _multi_expert_driver

LOW = 0.40          # 「易殺」門檻：血量佔比
TIE = 0.02          # 最低血並列容差

# 與 eval_gate check_nvm 同一組部署形狀（key 前綴不同＝獨立種子面）
PARTY = ["battle_master", "evocation", "life"]
BUCKETS = [
    ("怪1v隊3", [(["ogre"], PARTY, 2, 2), (["troll"], PARTY, 10, 5),
                 (["behir"], PARTY, 11, 8)]),
    ("隊3v怪1", [(PARTY, ["troll"], 5, 10), (PARTY, ["ettin"], 5, 10)]),
    ("2v2", [(["battle_master", "life"], ["champion", "evocation"], 5, 5)]),
    ("劣勢1v3", [(["battle_master"],
                  ["champion", "evocation", "vengeance"], 5, 5)]),
    ("3v3", [(PARTY, ["champion", "vengeance", "mage_npc"], 5, 5)]),
    # 敵隊含治療者＝目標選擇才真正撬動勝負的情境（覆蓋表 #19 先秒治療者）
    ("1v2敵治療", [(["berserker"], ["champion", "life"], 5, 5)]),
    ("2v2敵治療", [(["battle_master", "evocation"], ["champion", "life"], 5, 5)]),
    ("3v3敵治療", [(PARTY, ["champion", "life", "evocation"], 5, 5)]),
]


def _healer_ids(ws, opp_ids):
    """能治「隊友」的敵人集合（特徵判定、無硬寫名單）。

    必須限定 ally-target 治療：second_wind 這類自療技 expected_healing>0
    但目標型是 SELF，把 champion 也算成「治療者」＝全敵皆治療者、指標失義
    （第一版就踩了這個，hShare 恆 100%）。"""
    out = set()
    for o in opp_ids:
        c = ws.characters.get(o)
        if c is None:
            continue
        try:
            if any(sk.features.expected_healing > 0
                   and sk.features.target_type in (TargetType.SINGLE_ALLY,
                                                   TargetType.MULTI_ALLY)
                   for sk in available_skills(c, ws)):
                out.add(o)
        except Exception:
            pass
    return out


def _intervene_driver(net, mode):
    """介入臂（純儀器，非交付）：量「若目標紀律完美，成果差多少」的因果上界。

    mode="low"    模型選了單體傷害技但沒打最低血(<LOW)敵 → 改寫成打最低血合法敵
    mode="sticky" 上一個傷害目標還活著且合法 → 黏住不換（集火紀律上界）
    mode="healer" 有活的敵治療者且沒在打它 → 改打最低血的合法敵治療者（#19 上界）
    只改寫 entity 槽、不改技能choice；改寫前用 _enemy_target_legal 驗證
    （非法就不動＝永不製造引擎 ERROR）。"""
    base = _model_driver(net)
    last = {}                                    # seat -> 上一個傷害意圖目標
    def drv(env, obs, actor):
        act = list(base(env, obs, actor))
        si, ei = int(act[0]), int(act[1])
        ws = env.ws
        agent = ws.characters[actor]
        sks = available_skills(agent, ws)
        if not (0 <= si < len(sks)) or sks[si].skill_id == "end":
            return act
        sk = sks[si]; f = sk.features
        if not (f.expected_damage > 0
                and f.target_type == TargetType.SINGLE_ENEMY):
            return act
        _, enemies = partition_entities(ws, actor)
        live = [(e, ws.characters[e].hp / max(1.0, ws.characters[e].max_hp))
                for e in enemies
                if e in ws.characters and ws.characters[e].is_alive()]
        cur = _entity_id_at_slot(ws, actor, ei)
        bf = ws.combat.battlefield if ws.combat else None
        want = None
        if mode == "low" and len(live) >= 2:
            live.sort(key=lambda t: t[1])
            low_id, low_fr = live[0]
            if low_fr < LOW and cur != low_id:
                want = low_id
        elif mode == "sticky":
            prev = last.get(actor)
            alive_ids = {e for e, _ in live}
            if prev in alive_ids and cur != prev:
                want = prev
        elif mode == "healer" and len(live) >= 2:
            hs = _healer_ids(ws, [e for e, _ in live])
            if hs and cur not in hs:
                cand = sorted((fr, e) for e, fr in live if e in hs)
                want = cand[0][1]
        if want is not None and _enemy_target_legal(agent, actor, sk, want,
                                                    ws, bf):
            act[1] = ENEMY_SLOT_START + enemies.index(want)
            cur = want
        last[actor] = cur
        return act
    return drv


class Acc:
    """一個 (bucket, 駕駛) 的累積器。"""
    def __init__(self):
        self.n = 0; self.win = 0; self.dealt = 0.0; self.recv = 0.0
        self.dmg_dec = 0          # 單體傷害決策數
        self.aoe_dec = 0
        self.low_opp = 0; self.low_hit = 0
        self.sw_opp = 0; self.sw_hit = 0
        self.down_opp = 0; self.down_hit = 0
        self.hhi_sum = 0.0; self.hhi_n = 0
        self.fk_sum = 0; self.fk_n = 0
        self.kc_sum = 0; self.kc_n = 0
        self.heal_n = 0; self.heal_full = 0; self.heal_low = 0
        self.hk_n = 0; self.hk_first = 0      # 首殺=治療者（有治療敵且有擊殺的場）
        self.hs_dec = 0; self.hs_tot = 0      # 治療者活著時，傷害決策打治療者的佔比
        self.idle = 0; self.ft = 0            # 回合首手划水（gate 雙門用，同 drive_episode 語義）

    def row(self):
        f = lambda h, o: f"{h/o:5.0%}" if o else "    -"
        return (f"{self.win/max(1,self.n):4.0%} {self.dealt/max(1,self.n):6.1%} "
                f"{self.recv/max(1,self.n):6.1%} "
                f"{f(self.low_hit, self.low_opp)}({self.low_opp:4d}) "
                f"{f(self.sw_hit, self.sw_opp)}({self.sw_opp:4d}) "
                f"{f(self.down_hit, self.down_opp)} "
                f"{self.hhi_sum/max(1,self.hhi_n):5.2f} "
                f"{self.fk_sum/max(1,self.fk_n):5.1f} "
                f"{self.kc_sum/max(1,self.kc_n):5.1f} "
                f"{f(self.heal_full, self.heal_n)} "
                f"{f(self.hk_first, self.hk_n)} "
                f"{f(self.hs_dec, self.hs_tot)}")


def run_episode(driver, aa, oa, al, ol, key, acc: Acc):
    seed = stable_seed(key)
    random.seed(seed)                      # 全域引擎骰（記憶教訓：成對可比）
    env = CombatEnvV2(seed=seed ^ 0x5A5A5A, n_agents=len(aa), n_opps=len(oa))
    obs, _ = env.reset(agent_archs=list(aa), opp_archs=list(oa),
                       level=al, opp_level=ol, layout="open")
    aids = list(env.agent_ids)
    opp_set = set(env.opp_ids)
    healers = _healer_ids(env.ws, env.opp_ids)
    hp0 = {o: env.ws.characters[o].hp for o in env.opp_ids}
    mx = {o: max(1.0, env.ws.characters[o].max_hp) for o in env.opp_ids}
    team_hp0 = sum(env.ws.characters[a].hp for a in aids)
    team_max = max(1.0, sum(env.ws.characters[a].max_hp for a in aids))

    last_tgt = {}            # seat -> 上一個單體傷害意圖目標
    dmg_at = Counter()       # oid -> 團隊傷害決策數（HHI 用）
    seen_low = {}            # oid -> 首次 <LOW 的團隊決策序號
    death_at = {}            # oid -> 死亡時的團隊決策序號
    dec_i = 0                # 團隊決策序號（agent 席步數）
    turns = 0; done = False
    prev_actor = None
    while not done:
        actor = env.current_agent_id
        first = actor != prev_actor
        prev_actor = actor
        had_action = env.resources.get("action", 0) > 0
        act = driver(env, obs, actor)
        if first:
            acc.ft += 1
            if int(act[0]) == 0 and had_action:
                acc.idle += 1
        si, ei = int(act[0]), int(act[1])
        # ── 意圖解碼（step 前的世界狀態）──
        agent = env.ws.characters[actor]
        sks = available_skills(agent, env.ws)
        alive = [o for o in env.opp_ids if env.ws.characters[o].is_alive()]
        frac = {o: env.ws.characters[o].hp / mx[o] for o in alive}
        if 0 <= si < len(sks) and sks[si].skill_id != "end":
            f = sks[si].features
            tt = f.target_type
            tgt = _entity_id_at_slot(env.ws, actor, ei)
            if f.expected_damage > 0 and tt == TargetType.SINGLE_ENEMY \
                    and tgt in opp_set:
                acc.dmg_dec += 1
                dmg_at[tgt] += 1
                t_alive = env.ws.characters[tgt].is_alive()
                if any(h in frac for h in healers):     # 有治療敵活著
                    acc.hs_tot += 1
                    if tgt in healers:
                        acc.hs_dec += 1
                # downed：目標已倒、場上還有站立敵
                if alive:
                    standing = [o for o in alive if o != tgt]
                    if not t_alive and standing:
                        acc.down_opp += 1; acc.down_hit += 1
                    elif t_alive and not standing:
                        pass               # 只剩它，無選擇題
                # low-pick：≥2 活敵且最弱 <LOW
                if t_alive and len(alive) >= 2:
                    lo = min(frac.values())
                    if lo < LOW:
                        acc.low_opp += 1
                        if frac.get(tgt, 1.0) <= lo + TIE:
                            acc.low_hit += 1
                # switch：上個目標還活著卻換人
                prev = last_tgt.get(actor)
                if prev is not None and prev in frac:   # prev 還活著
                    acc.sw_opp += 1
                    if tgt != prev:
                        acc.sw_hit += 1
                last_tgt[actor] = tgt
            elif f.expected_damage > 0 and tt in (TargetType.POINT,
                                                  TargetType.LINE,
                                                  TargetType.CONE,
                                                  TargetType.MULTI_ENEMY):
                acc.aoe_dec += 1
            elif f.expected_healing > 0 and tgt is not None \
                    and tgt not in opp_set:
                hc = env.ws.characters.get(tgt)
                if hc is not None:
                    acc.heal_n += 1
                    hf = hc.hp / max(1.0, hc.max_hp)
                    if hf >= 0.90:
                        acc.heal_full += 1
                    elif hf < 0.50:
                        acc.heal_low += 1
        dec_i += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
        # ── step 後：死亡/低血事件記錄 ──
        for o in env.opp_ids:
            c = env.ws.characters[o]
            if o not in death_at and not c.is_alive():
                death_at[o] = dec_i
            elif o not in seen_low and c.is_alive() \
                    and c.hp / mx[o] < LOW:
                seen_low[o] = dec_i
        turns += 1
        if turns > 400:
            break

    acc.n += 1
    if healers and death_at:
        acc.hk_n += 1
        if min(death_at, key=death_at.get) in healers:
            acc.hk_first += 1
    team_alive = any(env.ws.characters[a].is_alive() for a in aids)
    opp_alive = any(env.ws.characters[o].is_alive() for o in env.opp_ids)
    acc.win += int(team_alive and not opp_alive)
    lost = sum(hp0[o] - max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    acc.dealt += lost / max(1.0, sum(hp0.values()))
    team_left = sum(max(0, env.ws.characters[a].hp) for a in aids)
    acc.recv += (team_hp0 - team_left) / team_max
    if death_at:
        acc.fk_sum += min(death_at.values()); acc.fk_n += 1
    for o, d in death_at.items():
        if o in seen_low and d >= seen_low[o]:
            acc.kc_sum += d - seen_low[o]; acc.kc_n += 1
    if sum(dmg_at.values()) >= 3 and len(env.opp_ids) >= 2:
        tot = sum(dmg_at.values())
        acc.hhi_sum += sum((v / tot) ** 2 for v in dmg_at.values())
        acc.hhi_n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--games", type=int, default=24, help="每 config 局數")
    ap.add_argument("--buckets", default=None, help="逗號過濾，如 3v3,2v2")
    ap.add_argument("--arms", default="model,expert",
                    help="逗號組合 model,expert,low,sticky（low/sticky=介入臂）")
    args = ap.parse_args()
    torch.set_num_threads(1)               # 驗收決定論（MKL 教訓）
    net = load_student(args.ckpt); net.eval()

    picks = set(args.buckets.split(",")) if args.buckets else None
    hdr = (f"{'bucket':10s} {'駕駛':6s} {'WR':>4s} {'dealt':>6s} {'recv':>6s} "
           f"{'low_pick(n)':>12s} {'switch(n)':>11s} {'downed':>6s} "
           f"{'HHI':>5s} {'firstK':>5s} {'kConv':>5s} {'heal@full':>9s} "
           f"{'hFirst':>6s} {'hShare':>6s}")
    print(f"模型: {args.ckpt}   G/config={args.games}   LOW={LOW}")
    print(hdr); print("-" * len(hdr))
    ARMS = {
        "model": lambda aa: _model_driver(net),
        "expert": lambda aa: _multi_expert_driver(aa),
        "low": lambda aa: _intervene_driver(net, "low"),
        "sticky": lambda aa: _intervene_driver(net, "sticky"),
        "healer": lambda aa: _intervene_driver(net, "healer"),
    }
    arm_names = [a for a in args.arms.split(",") if a in ARMS]
    for label, cfgs in BUCKETS:
        if picks and label not in picks:
            continue
        accs = {a: Acc() for a in arm_names}
        for ci, (aa, oa, al, ol) in enumerate(cfgs):
            for gi in range(args.games):
                key = f"nvmdiag|{label}|{ci}|{gi}"
                for a in arm_names:
                    run_episode(ARMS[a](aa), aa, oa, al, ol, key, accs[a])
        for i, a in enumerate(arm_names):
            print(f"{label if i == 0 else '':10s} {a:6s} {accs[a].row()}")
        if "model" in accs:
            am = accs["model"]
            for a in arm_names:
                if a == "model":
                    continue
                ax = accs[a]
                dw = ax.win / max(1, ax.n) - am.win / max(1, am.n)
                dd = ax.dealt / max(1, ax.n) - am.dealt / max(1, am.n)
                sig = math.sqrt(0.25 / max(1, am.n))
                print(f"{'':10s} Δ({a}-model) WR{dw:+.0%} dealt{dd:+.1%} "
                      f"(±2σ={2*sig:.0%} @n={am.n})")
        print()


if __name__ == "__main__":
    main()
