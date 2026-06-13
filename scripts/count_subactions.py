"""Count sub-actions per AGENT turn under the (now fixed) env.

If a policy uses full turns (move/bonus after its action), it averages >1
executed sub-action per turn. If it ends right after its action, it averages
~1. Compares a model checkpoint against the scripted expert so we can see
whether the model actually exploits the un-truncated turn the env fix enables.
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

model_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
games = 3
hidden = 128
net = None
if model_path:
    net = CombatPolicyNet(hidden=hidden)
    sd = torch.load(model_path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"policy = model {model_path}")
else:
    print("policy = scripted expert")


def play(agent_arch, opp_arch, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=5)
    expert_pol = make_archetype_policy(agent_arch) if net is None else None
    done = False
    # turn -> number of non-END executed sub-actions
    turn_actions = []
    cur_turn_count = 0
    prev_sub = 0
    while not done:
        actor_id = env.current_agent_id
        sub = getattr(env, "_turn_sub_actions", 0)
        if sub < prev_sub or sub == 0:
            # new turn began
            if cur_turn_count > 0:
                turn_actions.append(cur_turn_count)
            cur_turn_count = 0
        prev_sub = sub
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
        is_end = (action[0] == 0)
        if not is_end:
            cur_turn_count += 1
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    if cur_turn_count > 0:
        turn_actions.append(cur_turn_count)
    return turn_actions


all_counts = []
multi = 0
for ag in ARCHETYPE_LIST:
    for op in ARCHETYPE_LIST:
        base = hash(f"{ag}_{op}") & 0xFFFFFF
        for i in range(games):
            tc = play(ag, op, base + i)
            all_counts.extend(tc)

arr = np.array(all_counts) if all_counts else np.array([0])
print(f"agent turns measured : {len(arr)}")
print(f"avg sub-actions/turn  : {arr.mean():.2f}")
print(f"turns with >=2 actions: {(arr >= 2).mean():.1%}")
print(f"turns with ==1 action : {(arr == 1).mean():.1%}")
print(f"distribution          : {np.bincount(arr)[:6].tolist()} (index=#actions)")
