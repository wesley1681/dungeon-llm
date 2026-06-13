"""Ensemble eval: average action logits from multiple models, pick_action on the mean.

Each model contributes equally (uniform weights). Reuses eval_v2's per-archetype
matrix output so results are apples-to-apples with single-model runs.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                            apply_entity_mask, pick_action)


def load(path: str, hidden: int = 128) -> CombatPolicyNet:
    net = CombatPolicyNet(hidden=hidden)
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    net.eval()
    return net


def ensemble_forward(nets, obs_t):
    """Average logits across nets. All must have matching output shapes."""
    end_ls, sks, ents, grids = [], [], [], []
    for net in nets:
        e, s, en, g = net(obs_t)
        end_ls.append(e); sks.append(s); ents.append(en); grids.append(g)
    return (torch.stack(end_ls).mean(0),
            torch.stack(sks).mean(0),
            torch.stack(ents).mean(0),
            torch.stack(grids).mean(0))


def play_one(nets, agent_arch, opp_arch, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor_id = env.current_agent_id
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            end_l, s, e, g = ensemble_forward(nets, obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, actor_id)
        e = apply_entity_mask(e, obs_t)
        action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor_id))
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
             and env.ws.characters[aid].is_alive())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="paths to checkpoints to ensemble (≥2)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--label", type=str, default="ENSEMBLE")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    nets = [load(p, args.hidden) for p in args.models]
    print(f"Ensembling {len(nets)} models:")
    for p in args.models:
        print(f"  {p}")

    archs = list(ARCHETYPE_LIST)
    result = {}
    total = len(archs) ** 2
    done = 0
    for a in archs:
        result[a] = {}
        for o in archs:
            base = hash(f"{a}_{o}") & 0xFFFFFF
            wins = sum(1 for i in range(args.games) if play_one(nets, a, o, base + i))
            result[a][o] = {"win_rate": wins / args.games, "n": args.games}
            done += 1
            if done % 24 == 0:
                print(f"  {done}/{total}")

    print(f"\n=== {args.label} ===")
    print(f"{'arch':20s}  {'wr':>5s}")
    for a in archs:
        wr = np.mean([result[a][o]["win_rate"] for o in archs])
        print(f"{a:20s}  {wr:>4.0%}")
    overall = np.mean([result[a][o]["win_rate"] for a in archs for o in archs])
    print(f"{'OVERALL':20s}  {overall:>4.0%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "models": args.models,
                       "games_per_matchup": args.games, "result": result}, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
