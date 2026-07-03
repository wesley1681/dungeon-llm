"""DAgger-engage：修「接戰承諾失敗」家族（門口 dodge / 貼臉戒備蹲 / 2m 站樁）。

家族錘定（seed666/seed3 miner＋逐決策軌跡，全 v9 位元級同值＝既存）：
  變體 A（berserker）：已進 reach(0.9m)、攻擊合法、血量健康，卻每輪 dodge
    ——法師每輪蹭刀＝壓力存在＝repeat-dodge 守門正確放行（坦克合法性）。
  變體 B（chimera_omni×火免疫）：站 2.0m（劍被距離遮罩＝正當）、行動花在
    自療（under fire＝正當單步）、9m 移動全程不花、永不跨 0.5m 進 reach。
  兩變體單步都「正當」＝引擎真值守門無從剪枝 → policy 級訓練是唯一治本。
  觸發域=mirror（神經對神經）施法者對局；腳本施法者誘發不了（照打獲勝）。
  敵無抗性條目＝resist 載體零梯度＝v10 凍結手術結構性管不到。

方法＝train_dagger 的驗證配方（uni_v1 前例）：滾學生（部署行為＝含守門）、
腳本專家逐狀態標籤、全聚合 buffer、非焦點錨每步交錯防蝕；本腳本只換採樣域
（FOCUS×施法者×mirror-heavy）。守門=每輪 battery（家族雙變體+保持面），
終局 eval_gate --base uni_v10 + miner 雙種子。

用法:
  python scripts/train_engage_dagger.py --warm models/unified/uni_v10.pt \
      --out_dir models/engage_dagger --rounds 4
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from pathlib import Path
import numpy as np
import torch

import train_dagger as TD
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.train_bc import bc_loss_step
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy
from distill_routed import load_student
from eval_routed import stable_seed
from train_population import blind_np_single
try:
    from chimera_defs import register_chimeras
    register_chimeras()
except Exception:
    pass

FOCUS = ["berserker", "chimera_omni"]
CASTERS = ["mage_npc", "evocation", "divination", "life"]


class _StationaryPolicy:
    def decide(self, opp_id, opp, ws, resources, round_number):
        from types import SimpleNamespace
        return SimpleNamespace(action=None, fled=False, ended=True)


def collect_coverage_anchor(net, n_states, seed):
    """覆蓋錨＝驗收分布錨（run3/4 教訓的架構性收束：逐洞補錨是打地鼠——
    evocation 被動逃角、frightened 拒戰各是一個「錨定空洞」。驗收分布＝
    miner 的情境生成器（身分×突變×地形×對手×等級），直接用它採 warm net
    的貪婪行為當錨：BC 在整個驗收分布上都被拉住，除了刻意留白的待修區
    （FOCUS×mirror＝engage 家族本體）。"""
    import bug_miner as bm
    net.eval()
    O, A, T = [], [], []
    scens = bm.sample_scenarios(400, seed)
    for sc in scens:
        if len(A) >= n_states:
            break
        if sc.pair_role != "main":
            continue
        if sc.agents[0] in FOCUS and sc.opp_kind == "mirror":
            continue                     # 待修區留白，其餘全錨
        try:
            random.seed(bm.stable_seed(sc.pair_id or sc.key))
            env = CombatEnvV2(seed=bm.stable_seed(sc.pair_id or sc.key)
                              ^ 0x5A5A5A, n_agents=len(sc.agents),
                              n_opps=len(sc.opps))
            if sc.opp_kind == "mirror":
                env.use_self_play_opponent(net, blind=True)
            obs, _ = env.reset(agent_archs=sc.agents, opp_archs=sc.opps,
                               level=sc.lvl, opp_level=sc.opp_lvl,
                               layout=sc.layout)
            if sc.opp_kind == "passive":
                for o in env.opp_ids:
                    env._opp_policies[o] = _StationaryPolicy()
            bm._apply_mutation(env, sc)
        except Exception:
            continue
        aids = set(env.agent_ids)
        done = False; guard = 0
        while not done and guard < 250:
            guard += 1
            actor = env.current_agent_id
            act, ob = TD._student_action(net, env, actor, obs)
            if actor in aids:
                tt = -1
                if act[0] != 0:
                    sks = available_skills(env.ws.characters[actor], env.ws)
                    tt = int(sks[act[0]].features.target_type) \
                        if act[0] < len(sks) else -1
                O.append(ob); A.append(act); T.append(tt)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


def collect_engage_states(net, n_states, seed, rng):
    """滾學生（含守門＝部署行為），專家逐狀態標籤。60% mirror 施法者
    /20% 腳本施法者/20% 標準對手（防過擬合觸發域）。"""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        ident = rng.choice(FOCUS)
        r = rng.random()
        passive = False
        if r < 0.6:
            opp = rng.choice(CASTERS)
            mirror = r < 0.45
        elif r < 0.8:
            # run1 教訓：專家在 out-of-reach 狀態合法回 END 標籤，BC 把「END」
            # 過度泛化到被動敵情境（miner refuse×1355）——被動敵格子的專家標籤
            # ＝攻擊＝直接反壓
            opp = rng.choice(list(TD.STANDARD_IDS))
            mirror = False
            passive = True
        else:
            opp = rng.choice(list(TD.STANDARD_IDS))
            mirror = False
        a_lvl = rng.randint(4, 8)
        o_lvl = min(12, a_lvl + rng.choice([0, 1, 1, 2]))
        env = CombatEnvV2(seed=seed * 1_000_003 + ep, n_agents=1, n_opps=1)
        if mirror:
            env.use_self_play_opponent(net, blind=True)
        obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                           level=a_lvl, opp_level=o_lvl)
        if passive:
            for o in env.opp_ids:
                env._opp_policies[o] = _StationaryPolicy()
        aid = env.agent_ids[0]
        try:
            script = make_archetype_policy(ident)
        except Exception:
            ep += 1
            continue
        done = False; guard = 0
        while not done and guard < 300:
            guard += 1
            actor = env.current_agent_id
            s_act, ob = TD._student_action(net, env, actor, obs)
            if actor == aid:
                e_enc, tt = TD._expert_label(script, env, actor)
                # run1/2 教訓：專家 decide()=None 的狀態全標 END [0,0,0]，BC 在
                # 這些標籤上把「提前收手」泛化到被動/prone 敵（miner refuse 疫情
                # ×1355→仍漏 2/3）。接戰修復的監督訊號全在攻擊/移動標籤；END
                # 樣本由錨定批（warm net 自身的 END）提供平衡，不需要 DAgger 教
                # →過濾 END 標籤，只學正向動作。
                if e_enc != [0, 0, 0]:
                    O.append(ob); A.append(e_enc); T.append(tt)
            obs, _, term, trunc, _ = env.step(s_act)
            done = term or trunc
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    return obs_np, np.array(A, np.int64), np.array(T, np.int64)


# ── battery：家族雙變體＋保持面 ─────────────────────────────────────────────────

def _drive(net, ident, opp, lvl, olvl, mirror, key, mut_fire=False, cap=200,
           passive=False):
    random.seed(stable_seed(key))
    env = CombatEnvV2(seed=stable_seed(key) ^ 0x5A5A5A, n_agents=1, n_opps=1)
    if mirror:
        env.use_self_play_opponent(net, blind=True)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp], level=lvl,
                       opp_level=olvl, layout="open")
    if passive:
        for o in env.opp_ids:
            env._opp_policies[o] = _StationaryPolicy()
    if mut_fire:
        for o in env.opp_ids:
            env.ws.characters[o].damage_multipliers["火"] = 0.0
    aid = env.agent_ids[0]
    dmg0 = {o: env.ws.characters[o].hp for o in env.opp_ids}
    atk = 0; done = False; steps = 0
    while not done and steps < cap:
        steps += 1
        actor = env.current_agent_id
        if actor != aid:
            obs, _, t_, tr_, _ = env.step([0, 0, 0])
            done = t_ or tr_
            continue
        s_act, _ = TD._student_action(net, env, actor, obs)
        sks = available_skills(env.ws.characters[actor], env.ws)
        sk = sks[s_act[0]] if 0 < s_act[0] < len(sks) else None
        if sk is not None and getattr(sk.features, "expected_damage", 0) > 0:
            atk += 1
        obs, _, t_, tr_, _ = env.step(s_act)
        done = t_ or tr_
    dmg = sum(dmg0[o] - max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    win = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    return atk, dmg, win


def battery(net, tag, games=4):
    rows = []
    # 變體 A：berserker vs mirror 施法者——必須出手
    a_atk = a_dmg = a_w = 0
    for gi in range(games):
        atk, dmg, w = _drive(net, "berserker", "mage_npc", 5, 6, True,
                             f"eng_bat|A|{gi}")
        a_atk += (atk > 0); a_dmg += (dmg > 0); a_w += w
    rows.append(f"    A berserker/mirror法師: 出手={a_atk}/{games} "
                f"有傷={a_dmg}/{games} WR={a_w}/{games}")
    # 變體 B：chimera_omni vs 腳本法師×火免疫——必須跨進去輸出
    b_dmg = b_w = 0
    for gi in range(games):
        atk, dmg, w = _drive(net, "chimera_omni", "mage_npc", 7, 6, False,
                             f"eng_bat|B|{gi}", mut_fire=True)
        b_dmg += (dmg > 0); b_w += w
    rows.append(f"    B omni/火免疫腳本法師: 有傷={b_dmg}/{games} WR={b_w}/{games}")
    # 保持面 1：焦點外職業 vs 腳本專家（蝕檢快篩）
    keep_w = keep_n = 0
    for ident in ("champion", "battle_master", "evocation", "war"):
        for gi in range(2):
            _, _, w = _drive(net, ident, ident, 5, 5, False,
                             f"eng_keep|{ident}|{gi}")
            keep_w += w; keep_n += 1
    rows.append(f"    保持 4職 vs 同職腳本: WR={keep_w}/{keep_n}")
    # 保持面 2：resist 語義（totem_bear vs 天然 shadow 照打獲勝）
    atk, dmg, w = _drive(net, "totem_bear", "shadow", 10, 1, False,
                         "eng_keep|resist")
    rows.append(f"    保持 resist(totem/shadow): 出手={int(atk>0)} win={int(w)}")
    # 保持面 3：對被動敵不 refuse（run1 侵蝕面：END 泛化）——全體必須有傷且贏
    p_dmg = p_w = 0
    for ident in ("berserker", "champion", "evocation"):
        atk2, dmg2, w2 = _drive(net, ident, "orc", 6, 6, False,
                                f"eng_keep|passive|{ident}", passive=True)
        p_dmg += (dmg2 > 0); p_w += w2
    rows.append(f"    保持 被動敵接戰: 有傷={p_dmg}/3 WR={p_w}/3")
    score = (a_atk + a_dmg + a_w + b_dmg + b_w) / (5.0 * games) \
        + keep_w / keep_n + int(w)
    print(f"  [{tag}]\n" + "\n".join(rows), flush=True)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm", default="models/unified/uni_v10.pt")
    p.add_argument("--out_dir", default="models/engage_dagger")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--roll_states", type=int, default=4000)
    p.add_argument("--steps_per_round", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--anchor_states", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    TD.FOCUS = list(FOCUS)          # 非焦點錨池＝標準身分減去 FOCUS
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    net = load_student(args.warm)
    print("=== 非焦點錨（warm net 貪婪、防蝕）===", flush=True)
    s_obs, s_act, s_tt = TD.collect_nonfocus_anchor(
        net, args.anchor_states, args.seed * 104729 + 7, rng)
    print("=== 覆蓋錨（miner 情境生成器＝驗收分布；待修區留白）===", flush=True)
    p_obs, p_act, p_tt = collect_coverage_anchor(
        net, args.anchor_states, args.seed * 15485863 + 3)
    s_obs = {k: np.concatenate([s_obs[k], p_obs[k]], 0) for k in s_obs}
    s_act = np.concatenate([s_act, p_act], 0)
    s_tt = np.concatenate([s_tt, p_tt], 0)
    print(f"anchor={len(s_act)}（含覆蓋錨 {len(p_act)}） FOCUS={FOCUS} "
          f"casters={CASTERS}", flush=True)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("baseline:", flush=True)
    battery(net, "r0")
    buf = []
    for rd in range(1, args.rounds + 1):
        o_np, a_np, t_np = collect_engage_states(
            net, args.roll_states, args.seed * 7919 + rd, rng)
        buf.append((o_np, a_np, t_np))          # 全聚合（keep_rounds=∞）
        keys = buf[0][0].keys()
        agg_o = {k: np.concatenate([b[0][k] for b in buf], 0) for k in keys}
        agg_a = np.concatenate([b[1] for b in buf], 0)
        agg_t = np.concatenate([b[2] for b in buf], 0)
        net.train()
        for step in range(1, args.steps_per_round + 1):
            sel = np.random.randint(0, len(agg_a), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in agg_o.items()}
            acts = torch.from_numpy(agg_a[sel])
            with torch.no_grad():
                _, sl, _, _ = net(ob)
            w = 1.0 + 3.0 * (sl.argmax(-1) != acts[:, 0]).float()
            bc_loss_step(net, ob, acts, torch.from_numpy(agg_t[sel]), optim,
                         skill_sample_w=w)
            sel = np.random.randint(0, len(s_act), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in s_obs.items()}
            bc_loss_step(net, ob, torch.from_numpy(s_act[sel]),
                         torch.from_numpy(s_tt[sel]), optim)
        net.eval()
        torch.save(net.state_dict(), out_dir / f"eng_r{rd:02d}.pt")
        print(f"round {rd}: {len(a_np)} states (agg={len(agg_a)}) "
              f"→ eng_r{rd:02d}.pt", flush=True)
        battery(net, f"r{rd}")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
