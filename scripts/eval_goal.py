"""Goal check: for EVERY archetype, does the routed model tie/beat the fair
env.step expert? Plays each class's routed checkpoint vs all opponents and the
scripted expert vs the same opponents/seeds, reports WR side by side.

Usage: python scripts/eval_goal.py [games]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import DEFAULT_ROUTING, DEFAULT_HIDDEN, load_net

games = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def play(arch, net, expert, opp, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        if net is not None:
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        else:
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


cache = {}
print(f"{'archetype':18s} {'model':>6s} {'expert':>7s} {'diff':>6s}  verdict")
print("-" * 52)
all_ok = True
rows = []
for arch in ARCHETYPE_LIST:
    path = DEFAULT_ROUTING[arch]
    if path not in cache:
        cache[path] = load_net(path, DEFAULT_HIDDEN[arch])
    net = cache[path]
    expert = make_archetype_policy(arch)
    mw = ew = 0
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        for i in range(games):
            mw += int(play(arch, net, None, opp, base + i))
            ew += int(play(arch, None, expert, opp, base + i))
    n = games * len(ARCHETYPE_LIST)
    mwr, ewr = mw / n, ew / n
    diff = mwr - ewr
    # tie threshold: within 1-sigma binomial noise at this n
    sigma = (0.25 / n) ** 0.5
    ok = diff >= -2 * sigma
    all_ok = all_ok and ok
    rows.append((arch, mwr, ewr, diff, ok))
    print(f"{arch:18s} {mwr:6.0%} {ewr:7.0%} {diff:+6.0%}  "
          f"{'OK' if ok else 'SHORT'}")

print("-" * 52)
print(f"2-sigma tie band at n={games*12}: +/-{2*(0.25/(games*12))**0.5:.0%}")
print("GOAL MET" if all_ok else "GOAL NOT MET — short classes above")
