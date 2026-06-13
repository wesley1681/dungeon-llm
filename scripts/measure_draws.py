"""Measure the OUTCOME BREAKDOWN of eval games, not just win/not-win.

eval_v2 collapses everything that isn't (opp dead AND agent alive) into 'loss'.
That hides draws. This script classifies every game into:
    agent_win  : opp dead, agent alive
    opp_win    : agent dead, opp alive
    both_dead  : both dead same round (mutual kill)
    timeout    : neither dead at step cap (truncation)
and reports the population breakdown. A high draw rate (both_dead + timeout)
structurally caps win_rate well below 50% for evenly-matched play.

Usage: python scripts/measure_draws.py [model_path] [games_per_matchup]
       (no model_path => scripted expert as agent)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                            apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action

model_path = None
games = 12
hidden = 128
args = [a for a in sys.argv[1:]]
if args and not args[0].isdigit():
    model_path = args.pop(0)
if args:
    games = int(args.pop(0))
if "--h256" in sys.argv:
    hidden = 256

net = None
if model_path:
    net = CombatPolicyNet(hidden=hidden)
    sd = torch.load(model_path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"agent = model {model_path} (h={hidden})")
else:
    print("agent = scripted expert")


def play(agent_arch, opp_arch, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert_pol = make_archetype_policy(agent_arch) if net is None else None
    done = False; trunc = False
    while not done:
        actor_id = env.current_agent_id
        if net is not None:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            s = apply_resource_mask(s, env.resources, env.ws, actor_id)
            e = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                      ws=env.ws, agent_id=actor_id))
        else:
            actor = env.ws.characters[actor_id]
            decision = expert_pol.decide(actor_id, actor, env.ws, env.resources,
                                         env.ws.combat.round_number)
            if decision.action is None or decision.fled:
                action = [0, 0, 0]
            else:
                action = list(encode_action(decision.action, env.ws, actor_id))
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    a_alive = env.ws.characters[aid].is_alive()
    o_alive = env.ws.characters[oid].is_alive()
    if not o_alive and a_alive:
        return "agent_win"
    if not a_alive and o_alive:
        return "opp_win"
    if not a_alive and not o_alive:
        return "both_dead"
    return "timeout"   # trunc / both alive


archs = list(ARCHETYPE_LIST)
counts = {"agent_win": 0, "opp_win": 0, "both_dead": 0, "timeout": 0}
per_arch = {a: [0, 0] for a in archs}   # [agent_win, total]
total = 0
for a in archs:
    for o in archs:
        base = hash(f"{a}_{o}") & 0xFFFFFF
        for i in range(games):
            r = play(a, o, base + i)
            counts[r] += 1
            per_arch[a][1] += 1
            if r == "agent_win":
                per_arch[a][0] += 1
            total += 1

print("\nper agent-archetype win rate:")
for a in archs:
    w, t = per_arch[a]
    print(f"  {a:18s} {w/max(1,t):5.1%}")

print(f"\ntotal games: {total}")
for k in ("agent_win", "opp_win", "both_dead", "timeout"):
    print(f"  {k:10s}: {counts[k]:5d}  {counts[k]/total:6.1%}")
draws = counts["both_dead"] + counts["timeout"]
print(f"  {'DRAWS':10s}: {draws:5d}  {draws/total:6.1%}  (both_dead + timeout)")
print(f"\nwin_rate (agent_win/total)         = {counts['agent_win']/total:.1%}")
decisive = counts["agent_win"] + counts["opp_win"]
if decisive:
    print(f"win-among-decisive (excl. draws)   = {counts['agent_win']/decisive:.1%}")
