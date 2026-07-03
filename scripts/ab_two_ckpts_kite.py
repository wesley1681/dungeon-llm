"""比較兩個 checkpoint 在遠程身分的『自然』WR(無 gate),驗證 kiting DAgger 是否真改善勝率。
同批 seed 配對。用法: python scripts/ab_two_ckpts_kite.py <ckptA> <ckptB> [--games N]
"""
from __future__ import annotations
import sys, os, argparse, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import torch
from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single
try:
    from chimera_defs import register_chimeras; register_chimeras()
except Exception:
    pass


def load(path):
    from distill_routed import load_student
    return load_student(path)


def run(net, ident, opp_archs, lvl, opp_lvl, ep_key):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]; done = False
    while not done:
        actor = env.current_agent_id
        if actor in env.agent_ids:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    return (env.ws.characters[aid].is_alive()
            and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckptA"); ap.add_argument("ckptB")
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()
    nA, nB = load(args.ckptA), load(args.ckptB)
    IDENTS = ["evocation", "divination", "arcane_trickster", "assassin",
              "mage_npc", "manticore", "kobold"]
    OPP = ["champion", "berserker", "totem_bear", "vengeance"]
    print(f"A={os.path.basename(args.ckptA)}  B={os.path.basename(args.ckptB)}  games={args.games}\n")
    print(f"{'身分':16s} | A | B | Δ")
    print("-" * 40)
    tA = tB = tn = 0
    for ident in IDENTS:
        eq = EQUIV_LEVEL_1V1.get(ident, 5)
        lvl = 8 if eq == float("inf") else max(1, int(round(eq)))
        a = b = 0
        for gi in range(args.games):
            opp = [OPP[gi % len(OPP)]]; key = f"{ident}|{gi}"
            a += int(run(nA, ident, opp, lvl, lvl, key))
            b += int(run(nB, ident, opp, lvl, lvl, key))
        tA += a; tB += b; tn += args.games
        d = b - a; flag = " ⚠" if d else ""
        print(f"{ident:16s} | {a:2d}/{args.games} | {b:2d}/{args.games} | {d:+d}{flag}")
    print("-" * 40)
    print(f"{'總計':16s} | {tA}/{tn} | {tB}/{tn} | {tB-tA:+d}  (A={tA/tn*100:.1f}% B={tB/tn*100:.1f}%)")


if __name__ == "__main__":
    main()
