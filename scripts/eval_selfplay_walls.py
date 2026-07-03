"""Model-vs-MODEL wall behaviour — the real deployment condition.

The freeze-behind-wall failure is a property of the MODEL, so in deployment
(model plays BOTH sides) the side that must flank stalls and the fight becomes
a mutual dithering stalemate. Scripted-opponent evals hide this. Here BOTH
seats are the same net (use_self_play_opponent). Signals:
    trunc%   episodes that hit the step budget with nobody dead = stalemate
             (the freeze tax — should spike in walls vs open if it's real)
    turns    mean episode length (drags out when neither side can engage)
    decisive% episodes that ended in a kill (1 - trunc on living draws)

Compare base vs the LoS-channel/flanking model across open vs walls/pillar.

Usage:
  python scripts/eval_selfplay_walls.py --ckpts models/pop_mon/pop_u0005.pt \
         models/pop_wall2/pop_u0032.pt --games 30
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
from synth_identity import STANDARD_IDS


def play_selfplay(net, ident, opp, layout, ep_key):
    """Both seats driven by `net`. Returns (truncated, turns, decisive)."""
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x11, n_agents=1, n_opps=1)
    env.use_self_play_opponent(net)               # opponent = same model
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                       level=5, opp_level=5, layout=layout)
    turns = 0
    term = trunc = False
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
        turns += 1
        done = term or trunc
    return trunc, turns, term


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--idents", nargs="*",
                   default=["champion", "battle_master", "assassin", "war",
                            "berserker", "evocation"])
    p.add_argument("--games", type=int, default=30)
    args = p.parse_args()

    nets = {c: load_student(c) for c in args.ckpts}
    for n in nets.values():
        n.eval()

    for layout in ("open", "walls", "pillar"):
        print(f"\n==== layout={layout}  model-vs-model  "
              f"{args.games} games/ident ====")
        print(f"{'metric':<12}" + "".join(
            f"{os.path.basename(c)[:18]:>20}" for c in args.ckpts))
        agg = {c: {"trunc": 0, "turns": 0, "n": 0} for c in args.ckpts}
        for ident in args.idents:
            for c in args.ckpts:
                for g in range(args.games):
                    opp = ident   # mirror match (same kit both sides)
                    tr, t, dec = play_selfplay(nets[c], ident, opp, layout,
                                               f"{layout}|{ident}|{g}")
                    agg[c]["trunc"] += int(tr); agg[c]["turns"] += t
                    agg[c]["n"] += 1
        r1 = f"{'trunc%(freeze)':<12}"
        r2 = f"{'mean turns':<12}"
        for c in args.ckpts:
            a = agg[c]
            r1 += f"{a['trunc'] / a['n']:>19.0%} "
            r2 += f"{a['turns'] / a['n']:>19.1f} "
        print(r1, flush=True)
        print(r2, flush=True)


if __name__ == "__main__":
    main()
