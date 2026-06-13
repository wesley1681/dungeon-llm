"""regen→melee 因果驗證（obs v5 目標的決定性交付）。

對 hill_giant（dual-weapon bruiser），量三條件下的近戰率與 WR：
  ON   引擎注入再生、obs I_TRAIT_REGEN 誠實顯示（=該近戰、且模型看得到）
  OFF  無再生、通道誠實=0（=該遠程、模型也知道）
  ABL  引擎注入再生、但 obs I_TRAIT_REGEN「歸零」（再生仍在、但蒙住模型）

三條件共用同一場種骰，唯一差異就是天賦/通道。判讀：
  • melee(ON) ≫ melee(OFF)  → 模型依天賦改技能選擇
  • melee(ON) ≫ melee(ABL)  → 行為是「讀通道」造成的（蒙住即退回遠程）＝因果
  • WR(ON) > WR(ABL)         → 讀到再生→近戰→更高勝率（此資訊必定帶來增益）

Usage: python scripts/verify_regen_causal.py --ckpt models/regen_v5/regen_final.pt --games 60
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from collections import Counter
from zlib import crc32
import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import I_TRAIT_REGEN
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from seed_regen_bc import weapon_skills, PARTY, BLOCKED

# hill_giant = the ONLY creature in the BC demos; the rest are held-out
# dual-weapon creatures (melee+ranged). The model is BLIND to self-identity, so a
# learned "regen channel → melee" rule must transfer to them = trait-rule, not
# hill_giant-memorisation.
TRAINED = {"hill_giant"}
CREATURES = ("hill_giant", "manticore", "skeleton", "bandit", "assassin")


def episode(creature, opp_lvl, ep_key, net, regen_on, ablate):
    k = crc32(ep_key.encode())
    random.seed(k)
    lvl = MONSTER_DEFS[creature].natural_level if creature in MONSTER_DEFS else 5
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(PARTY))
    obs, _ = env.reset(agent_archs=[creature], opp_archs=list(PARTY),
                       level=lvl, opp_level=opp_lvl)
    aid0 = env.agent_ids[0]
    env.ws.characters[aid0].regeneration = (
        {"amount": 30, "blocked_by": BLOCKED} if regen_on else None)
    mel_sk, ran_sk = weapon_skills(env.ws, aid0)
    mel_id = mel_sk.skill_id if mel_sk else None
    ran_id = ran_sk.skill_id if ran_sk else None
    melee = ranged = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        if ablate:
            ot["entities"][:, :, I_TRAIT_REGEN] = 0.0   # blind model to its regen
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        if actor == aid0 and act[0] > 0:
            sks = available_skills(env.ws.characters[actor], env.ws)
            if act[0] < len(sks):
                sid = sks[act[0]].skill_id
                if sid == mel_id:
                    melee += 1
                elif sid == ran_id:
                    ranged += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (env.ws.characters[aid0].is_alive()
           and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))
    return won, melee, ranged


def run(net, creature, opp_lvl, games, regen_on, ablate):
    w = mel = ran = 0
    for g in range(games):
        key = f"vrc|{creature}|{opp_lvl}|{g}"   # SAME key across conditions
        won, m, r = episode(creature, opp_lvl, key, net, regen_on, ablate)
        w += int(won); mel += m; ran += r
    melee_rate = mel / max(1, mel + ran)
    return w / games, melee_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/regen_v5/regen_final.pt")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--opp_lvl", type=int, default=4)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  vs {PARTY}@L{args.opp_lvl}, {args.games} games/cell")
    print(f"{'creature':<12} {'role':<9} {'meleeON':>8} {'meleeOFF':>9} {'meleeABL':>9}  "
          f"{'WR_ON':>6} {'WR_OFF':>7} {'WR_ABL':>7}")
    for c in CREATURES:
        role = "trained" if c in TRAINED else "held-out"
        wr_on, m_on = run(net, c, args.opp_lvl, args.games, True, False)
        wr_off, m_off = run(net, c, args.opp_lvl, args.games, False, False)
        wr_abl, m_abl = run(net, c, args.opp_lvl, args.games, True, True)
        flag = "  <- causal" if (
            m_on - m_off > 0.15 and m_on - m_abl > 0.15) else ""
        print(f"{c:<12} {('('+role+')'):<9} {m_on:>7.1%} {m_off:>8.1%} {m_abl:>8.1%}  "
              f"{wr_on:>5.1%} {wr_off:>6.1%} {wr_abl:>6.1%}{flag}")
    print("\n判讀：meleeON≫meleeOFF＝依天賦改技能；meleeON≫meleeABL＝讀通道驅動"
          "（蒙住退回遠程）＝因果；WR_ON>WR_ABL＝再生資訊必定帶來增益。")


if __name__ == "__main__":
    main()
