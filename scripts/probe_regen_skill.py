"""再生是否該翻轉「近戰 vs 遠程」技能選擇？（regen→melee BC-seed 的 headroom 量測）

邏輯：近戰（巨棒）貼臉吃反擊、遠程（擲岩）拉開較安全；再生把反擊的血補回 →
再生高時近戰（高傷+補得回）該勝過遠程。冠軍預設 kite（擲岩，見 probe_skill_edit）。

量測（hill_giant vs 校準隊伍，注入 regen）：
  forced-melee  把擲岩傷害壓到 1d1 → 模型只能用巨棒（近戰）
  forced-ranged 把巨棒傷害壓到 1d1 → 模型只能用擲岩（遠程）
  各在 regen∈{0,30} 量 WR。若 WR(melee) 隨 regen 升並超過 WR(ranged) →
  最佳技能是「再生閘控」的 → regen→melee oracle 成立、BC-seed 有意義。
也量「未強制」冠軍的近戰率隨 regen 是否平（盲＝有 headroom）。

Usage: python scripts/probe_regen_skill.py --games 40
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from collections import Counter
from zlib import crc32

import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.engine.items import WEAPON_DEFS
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single

CID = "hill_giant"
NAT = MONSTER_DEFS[CID].natural_level
PARTY = ("battle_master", "life", "evocation")
MELEE, RANGED = "weapon:巨人巨棒", "weapon:擲岩"
BLOCKED = ("火", "強酸")


def episode(opp_lvl, ep_key, net, regen):
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(PARTY))
    obs, _ = env.reset(agent_archs=[CID], opp_archs=list(PARTY),
                       level=NAT, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    ch = env.ws.characters[aid]
    ch.regeneration = None if regen <= 0 else {"amount": int(regen),
                                               "blocked_by": BLOCKED}
    usage = Counter(); turns = 0
    done = False
    while not done:
        actor = env.current_agent_id
        c = env.ws.characters[actor]
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        if actor == aid and act[0] > 0:
            sks = available_skills(c, env.ws)
            if act[0] < len(sks):
                usage[sks[act[0]].skill_id] += 1
        turns += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (env.ws.characters[aid].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, usage, turns


def measure(net, opp_lvl, games, regen):
    w = 0; usage = Counter(); turns = 0
    for gi in range(games):
        won, u, t = episode(opp_lvl, f"prs|{regen}|{gi}", net, regen)
        w += int(won); usage += u; turns += t
    return w / games, usage.get(MELEE, 0) / max(1, turns), \
        usage.get(RANGED, 0) / max(1, turns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--opp_lvl", type=int, default=4)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    base_m = WEAPON_DEFS["巨人巨棒"].damage_dice
    base_r = WEAPON_DEFS["擲岩"].damage_dice
    print(f"ckpt={args.ckpt}  {CID} vs {PARTY}@L{args.opp_lvl}, {args.games} g/cell\n")

    print("== 未強制：冠軍近戰率是否隨 regen 變？（盲＝平）==")
    for rg in (0, 30):
        wr, m, r = measure(net, args.opp_lvl, args.games, rg)
        print(f"  regen={rg:>2}: WR={wr:.0%}  巨棒×{m:.2f}/回合  擲岩×{r:.2f}/回合")

    print("\n== forced-melee（擲岩→1d1）vs forced-ranged（巨棒→1d1）的 WR ==")
    print(f"  {'regen':>6} | {'melee WR':>9} {'ranged WR':>10} {'Δ(m-r)':>8}")
    for rg in (0, 30):
        WEAPON_DEFS["擲岩"].damage_dice = "1d1"
        wr_m, *_ = measure(net, args.opp_lvl, args.games, rg)
        WEAPON_DEFS["擲岩"].damage_dice = base_r
        WEAPON_DEFS["巨人巨棒"].damage_dice = "1d1"
        wr_r, *_ = measure(net, args.opp_lvl, args.games, rg)
        WEAPON_DEFS["巨人巨棒"].damage_dice = base_m
        print(f"  {rg:>6} | {wr_m:>8.0%} {wr_r:>9.0%} {(wr_m-wr_r)*100:>+7.0f}")
    print("\n判讀：Δ(m-r) 在 regen=0 為負（遠程較好）、regen=30 轉正（近戰較好）"
          "＝再生翻轉最佳技能＝oracle『regen→melee』成立。")


if __name__ == "__main__":
    main()
