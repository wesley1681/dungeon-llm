"""Realistic wall-combat WR: does the LoS-channel model engage better than the
base when there are walls on the board (moving enemy, normal obstacles)?

diag_wall_los is an adversarial extreme (stationary enemy behind a 6m pillar,
flanking INCREASES distance). The goal-relevant question is ordinary combat on
the `walls` layout: agent vs a scripted-expert opponent that moves, obstacles
2m. We compare checkpoints on WR in `walls` vs `open` (open = regression
guard), paired global-RNG seeds so the two checkpoints face identical dice.

Usage:
  python scripts/eval_wall_combat.py --ckpts models/pop_mon/pop_u0005.pt \
         models/pop_wall2/pop_u0040.pt --idents assassin champion war \
         --games 12
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


def play(net, ident, opp, layout, ep_key, max_steps_mult=25):
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x9, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                       level=5, opp_level=5, layout=layout)
    aid = env.agent_ids[0]
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
    return (env.ws.characters[aid].is_alive()
            and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--idents", nargs="*",
                   default=["assassin", "champion", "war", "battle_master",
                            "evocation", "berserker"])
    p.add_argument("--opps", nargs="*", default=["champion", "evocation"])
    p.add_argument("--games", type=int, default=12)
    args = p.parse_args()

    nets = {c: load_student(c) for c in args.ckpts}
    for n in nets.values():
        n.eval()

    for layout in ("walls", "open"):
        print(f"\n==== layout={layout}  ({args.games} games/(ident,opp), "
              f"paired seeds) ====")
        head = f"{'identity':<16}" + "".join(
            f"{os.path.basename(c)[:18]:>20}" for c in args.ckpts)
        print(head)
        tot = {c: [0, 0] for c in args.ckpts}
        for ident in args.idents:
            row = f"{ident:<16}"
            for c in args.ckpts:
                w = n = 0
                for opp in args.opps:
                    for g in range(args.games):
                        key = f"{layout}|{ident}|{opp}|{g}"
                        w += int(play(nets[c], ident, opp, layout, key))
                        n += 1
                tot[c][0] += w; tot[c][1] += n
                row += f"{w / n:>19.0%} "
            print(row, flush=True)
        row = f"{'MEAN':<16}"
        for c in args.ckpts:
            row += f"{tot[c][0] / tot[c][1]:>19.0%} "
        print(row)


if __name__ == "__main__":
    main()
