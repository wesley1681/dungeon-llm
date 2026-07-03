"""WR across team ARRANGEMENTS — 1v1 / 1vN / Nv1 / NvN.

The goal requires the model to handle all three arrangements. pop_u0005 was
trained n_agents=1 only, so this measures whether multi-seat training (p_team)
gave it the missing capability — and guards that 1v1 didn't regress.

The model controls EVERY agent seat (shared policy); opponents are scripted
class experts. Paired global-RNG seeds so checkpoints face identical dice.
Agent + opponent identities are standard classes (fixed per cell for a clean
comparison). WR = the agent team wipes the opponents while ≥1 agent lives.

Usage:
  python scripts/eval_team_arrangements.py --ckpts models/pop_mon/pop_u0005.pt \
         models/pop_integ1/pop_u0050.pt --games 12
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single

# (n_agents, n_opps) arrangements spanning the goal's three cases.
ARRANGEMENTS = [(1, 1), (2, 2), (3, 3), (2, 1), (3, 2), (1, 2), (1, 3), (3, 1)]
# rotating identity panels (deterministic per seat)
A_POOL = ["champion", "battle_master", "evocation"]
O_POOL = ["champion", "war", "assassin"]


def play(net, na, no_, layout, ep_key, level=5):
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x7, n_agents=na, n_opps=no_)
    a_archs = [A_POOL[i % len(A_POOL)] for i in range(na)]
    o_archs = [O_POOL[i % len(O_POOL)] for i in range(no_)]
    obs, _ = env.reset(agent_archs=a_archs, opp_archs=o_archs,
                       level=level, opp_level=level, layout=layout)
    done = False
    while not done:
        actor = env.current_agent_id
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    agents_alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    return agents_alive and opps_dead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--layout", default="open")
    args = p.parse_args()

    nets = {c: load_student(c) for c in args.ckpts}
    for n in nets.values():
        n.eval()

    print(f"layout={args.layout}  {args.games} games/arrangement  paired seeds")
    print(f"{'arrange':<10}" + "".join(
        f"{os.path.basename(c)[:18]:>20}" for c in args.ckpts))
    tot = {c: [0, 0] for c in args.ckpts}
    for (na, no_) in ARRANGEMENTS:
        row = f"{na}v{no_:<8}"
        for c in args.ckpts:
            w = sum(play(nets[c], na, no_, args.layout, f"{na}v{no_}|{g}")
                    for g in range(args.games))
            tot[c][0] += w; tot[c][1] += args.games
            row += f"{w / args.games:>19.0%} "
        print(row, flush=True)
    row = f"{'MEAN':<10}"
    for c in args.ckpts:
        row += f"{tot[c][0] / tot[c][1]:>19.0%} "
    print(row)


if __name__ == "__main__":
    main()
