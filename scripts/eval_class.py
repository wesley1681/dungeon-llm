"""Fast single-class eval: model plays ONE archetype vs all 12 opponents.

Mirrors eval_v2's agent path (pick_action + masks via env.step) but only the
given agent archetype, so a specialist can be scored in ~1/12 the time.

Usage: python scripts/eval_class.py <model.pt> <arch> [games] [hidden]
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

model_path = sys.argv[1]      # path to .pt OR "expert"
arch = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 128

IS_EXPERT = (model_path.lower() == "expert")
if IS_EXPERT:
    from trpg.engine.combat_policy import make_archetype_policy
    from trpg.rl.action import encode_action
    net = None
else:
    net = CombatPolicyNet(hidden=hidden)
    sd = torch.load(model_path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    # Drop shape-mismatched keys (e.g. the critic input changed dim across model
    # versions). Inference never uses the critic, so this is safe.
    msd = net.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in msd and v.shape != msd[k].shape)}
    net.load_state_dict(sd, strict=False)
    net.eval()


def play(opp, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert = make_archetype_policy(arch) if IS_EXPERT else None
    done = False
    while not done:
        actor = env.current_agent_id
        if IS_EXPERT:
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources,
                                env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        else:
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()) and env.ws.characters[aid].is_alive()


wrs = []
for opp in ARCHETYPE_LIST:
    base = hash(f"{arch}_{opp}") & 0xFFFFFF
    w = sum(1 for i in range(games) if play(opp, base + i))
    wrs.append(w / games)
print(f"{arch:18s} ({os.path.basename(os.path.dirname(model_path))}/"
      f"{os.path.basename(model_path)}): WR = {np.mean(wrs):.0%}")
