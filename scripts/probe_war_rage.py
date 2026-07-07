"""後續探針:把狂暴(rage)嫁接到戰爭牧師,看訓練好的 general5 網會不會用。

背景:exp_scratch_general5 的網訓練時**已經當狂戰士學過狂暴**(rage 是狂戰士技能,該職
87% WR)。技能用 66 維機制向量編碼(非 ID),codeword 論點預測「狂暴的用處」該跨 kit 轉移。
所以這是乾淨的**技能層 zero-shot transfer 探針**:戰爭牧訓練時 pool 裡沒有 rage,現在
runtime 嫁接(char.known_abilities.append('rage'),available_skills 就列出),不重訓,問
訓練好的網會不會在戰爭牧席位開狂暴。

機制查證:我們的 Raging(status.py:104)只有 +2 物理傷害/物理抗性/10 回合,**不擋施法**
(異於 5e),所以沒有「拒用才是正解」的陷阱。戰爭牧有長劍(物理武器,狂暴加成)。狂暴的
代價=搶 bonus action(靈魂武器/治療語)、抗性只剋物理攻擊者。

量測(greedy,n≥48/格):
  use%   = 該局曾開狂暴的比例
  open%  = 第一個 agent 動作就開狂暴的比例
  p(rage)= 狂暴合法時,它在技能 softmax 裡的平均機率(區分「沒考慮」vs「考慮但略遜」)
  WR     = 勝率;對照 = 同種子同對手、**不嫁接**狂暴的戰爭牧 WR(看嫁接有沒有改變結果)
對照組:狂戰士自己的狂暴 use%(證明網本來就會開狂暴 → 戰爭牧不開=情境抑制,非不會用)。

用法:python scripts/probe_war_rage.py --ckpt models/exp_general5/g5_u0060.pt --n 48
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TRPG_WASTED_MOVE_COST", "0")

import argparse, random
import numpy as np
import torch

import exp_scratch_general5 as G
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills

ARCHS = G.ARCHS
NAME = G.ARCH_NAME


def _graft_rage(char):
    ka = list(char.known_abilities or [])
    if "rage" not in ka:
        ka.append("rage")
    char.known_abilities = ka


def eval_rage(net, agent_arch, opp, n, level, graft, seed0=70_000):
    """跑 n 局 greedy,回傳狂暴使用/開場/機率/WR。graft=True 才把 rage 嫁接上去。"""
    net.eval()
    use = open_ = wins = 0
    rprobs = []
    for gi in range(n):
        seed = seed0 + gi
        random.seed(seed)
        env = G.make_env(seed, agent_arch, opp, level)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        if graft:
            _graft_rage(env.ws.characters[aid])
        used = False; first = True; done = False; steps = 0
        while not done and steps < 200:
            steps += 1; actor = env.current_agent_id
            ob = build_obs(env.ws, actor, env.resources)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            sks = available_skills(env.ws.characters[actor], env.ws)
            if actor == aid:
                ridx = next((i for i, sk in enumerate(sks) if sk.skill_id == "rage"), None)
                if ridx is not None and ridx < s.shape[1] and s[0, ridx].item() > -1e8:
                    rprobs.append(torch.softmax(s[0], dim=-1)[ridx].item())
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            if actor == aid and 0 < act[0] < len(sks) and sks[act[0]].skill_id == "rage":
                used = True
                if first:
                    open_ += 1
            if actor == aid:
                first = False
            _, _, term, trunc, _ = env.step(act); done = term or trunc
        wins += (not env.ws.characters[oid].is_alive()
                 and env.ws.characters[aid].is_alive())
        use += used
    return dict(use=use / n, open=open_ / n, wr=wins / n,
                p=float(np.mean(rprobs)) if rprobs else 0.0,
                nturns=len(rprobs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/exp_general5/g5_u0060.pt")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--level", type=int, default=5)
    args = p.parse_args()

    net = G.build_net()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()
    print(f"ckpt={args.ckpt} n={args.n}/格 L{args.level}", flush=True)

    print("\n── 對照:狂戰士自己開狂暴的頻率(證明網本來就會開) ──", flush=True)
    for opp in ARCHS:
        r = eval_rage(net, "berserker", opp, args.n, args.level, graft=False)
        print(f"  狂戰士 vs {NAME[opp]}: 開狂暴use={r['use']:4.0%} open={r['open']:4.0%} "
              f"p(rage合法時)={r['p']:4.0%} WR={r['wr']:4.0%}", flush=True)

    print("\n── 主測:戰爭牧【嫁接狂暴】會不會用 + WR對照【未嫁接】 ──", flush=True)
    print(f"{'對手':<8}{'use%':>7}{'open%':>7}{'p(rage)':>9}{'WR(有rage)':>11}{'WR(無rage)':>11}{'ΔWR':>7}",
          flush=True)
    agg = {"use": [], "wr_g": [], "wr_b": []}
    for opp in ARCHS:
        rg = eval_rage(net, "war", opp, args.n, args.level, graft=True)
        rb = eval_rage(net, "war", opp, args.n, args.level, graft=False)
        d = rg["wr"] - rb["wr"]
        agg["use"].append(rg["use"]); agg["wr_g"].append(rg["wr"]); agg["wr_b"].append(rb["wr"])
        print(f"{NAME[opp]:<8}{rg['use']:7.0%}{rg['open']:7.0%}{rg['p']:9.0%}"
              f"{rg['wr']:11.0%}{rb['wr']:11.0%}{d:+7.0%}", flush=True)
    print(f"{'均':<8}{np.mean(agg['use']):7.0%}{'':>7}{'':>9}"
          f"{np.mean(agg['wr_g']):11.0%}{np.mean(agg['wr_b']):11.0%}"
          f"{np.mean(agg['wr_g'])-np.mean(agg['wr_b']):+7.0%}", flush=True)


if __name__ == "__main__":
    main()
