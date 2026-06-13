"""Sweep all ppo_*.pt snapshots in a dir, eval each vs all 12 opps (eval_v2 win
condition), print sorted. Selects the TRUE best by the real metric instead of
the noisy internal training eval.

Usage: python scripts/sweep_snapshots.py <dir> <arch> [games] [hidden]
"""
from __future__ import annotations
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)

d = sys.argv[1]; arch = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 20
hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 128


def wr_for(net):
    wrs = []
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        w = 0
        for i in range(games):
            env = CombatEnvV2(seed=base + i, n_agents=1, n_opps=1)
            obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
            aid = env.agent_ids[0]; oid = env.opp_ids[0]
            done = False
            while not done:
                actor = env.current_agent_id
                ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
                obs, _, term, trunc, _ = env.step(act)
                done = term or trunc
            if (not env.ws.characters[oid].is_alive()) and env.ws.characters[aid].is_alive():
                w += 1
        wrs.append(w / games)
    return float(np.mean(wrs))


paths = sorted(glob.glob(os.path.join(d, "ppo_u*.pt"))) \
        + [os.path.join(d, "ppo_final.pt")]
results = []
for p in paths:
    if not os.path.exists(p):
        continue
    net = CombatPolicyNet(hidden=hidden)
    sd = torch.load(p, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    msd = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in msd and v.shape != msd[k].shape)}
    net.load_state_dict(sd, strict=False); net.eval()
    wr = wr_for(net)
    results.append((wr, os.path.basename(p)))
    print(f"  {os.path.basename(p):16s}  WR={wr:.0%}")

results.sort(reverse=True)
print(f"\nBEST for {arch}: {results[0][1]} = {results[0][0]:.0%}  ({games} games/opp)")
