"""Diagnose WHY the model underperforms on a specific archetype.

Plays the MODEL as <arch>; at every state also queries the scripted expert.
Logs both action distributions (skill_id) + end-rate + sub-actions/turn +
outcome, so we can see exactly where the model diverges from the expert
(e.g. wrong skill, never attacks, never action-surges, bad end timing).

Usage: python scripts/diag_class.py <model.pt> <arch> [games] [--h256]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                            apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.rl.action import encode_action, decode_action
from trpg.engine.skill import available_skills
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single

model_path = sys.argv[1]
arch = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 20
hidden = 256 if "--h256" in sys.argv else 128

net = CombatPolicyNet(hidden=hidden)
sd = torch.load(model_path, map_location="cpu")
sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
net.load_state_dict(sd, strict=False)
net.eval()

model_acts = Counter()
expert_acts = Counter()
model_end = 0; expert_end = 0; steps = 0
wins = 0; opp_archs = list(ARCHETYPE_LIST)


def skill_name(ws, aid, idx):
    sk = available_skills(ws.characters[aid], ws)
    return sk[idx].skill_id if 0 <= idx < len(sk) else f"oob{idx}"


for gi in range(games):
    opp = opp_archs[gi % len(opp_archs)]
    env = CombatEnvV2(seed=1000 + gi, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert = make_archetype_policy(arch)
    done = False
    while not done:
        actor = env.current_agent_id
        agent = env.ws.characters[actor]
        # expert's action at this state (for comparison)
        dec = expert.decide(actor, agent, env.ws, env.resources,
                            env.ws.combat.round_number)
        if dec.action is None or dec.fled:
            expert_end += 1; expert_acts["END"] += 1
        else:
            enc = encode_action(dec.action, env.ws, actor)
            expert_acts[skill_name(env.ws, actor, enc[0]) if enc[0] >= 0 else "?"] += 1
        # model's action. BLIND the self-identity one-hot: the unified students
        # are trained with it zeroed, so feeding the live obs corrupts them.
        ob_b = blind_np_single(obs)
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob_b.items()}
        with torch.no_grad():
            end_l, s, e, g = net(obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, obs_t)
        act = list(pick_action(end_l[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        if act[0] == 0:
            model_end += 1; model_acts["END"] += 1
        else:
            model_acts[skill_name(env.ws, actor, act[0])] += 1
        steps += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    if not env.ws.characters[oid].is_alive() and env.ws.characters[aid].is_alive():
        wins += 1

print(f"=== {arch} : model {model_path} ({games} games, win={wins/games:.0%}) ===")
print(f"steps={steps}  model_end={model_end/steps:.0%}  expert_end={expert_end/steps:.0%}")
print(f"\n{'skill_id':28s} {'MODEL':>8s} {'EXPERT':>8s}")
keys = sorted(set(model_acts) | set(expert_acts),
              key=lambda k: -(model_acts[k] + expert_acts[k]))
for k in keys[:16]:
    m = model_acts[k] / max(1, steps)
    x = expert_acts[k] / max(1, steps)
    flag = "  <<< DIVERGE" if abs(m - x) > 0.10 else ""
    print(f"{k:28s} {m:8.1%} {x:8.1%}{flag}")
