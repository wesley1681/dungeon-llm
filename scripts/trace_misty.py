"""Test the premise: does the WIZARD waste misty_step the way champion wasted
action_surge? Expert fires misty_step only when cornered (enemy <=3m). Count
model vs expert misty_step fires and how many are fired while NOT cornered
(enemy >3m) -> wasted teleport / wasted slot, a time the expert never fires.

Usage: python scripts/trace_misty.py <arch> <model.pt> [games]
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
from trpg.engine.skill import available_skills

arch = sys.argv[1]
model_path = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SKILL = "misty_step"
CORNER = 3.0

net = CombatPolicyNet()
sd = torch.load(model_path, map_location="cpu")
sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
_msd = net.state_dict()
sd = {k: v for k, v in sd.items() if not (k in _msd and v.shape != _msd[k].shape)}
net.load_state_dict(sd, strict=False); net.eval()


def play(driver, opp, seed, st):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert = make_archetype_policy(arch) if driver == "expert" else None
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]; opp_c = env.ws.characters[oid]
        skills = available_skills(ag, env.ws)
        if driver == "model":
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        else:
            dec = expert.decide(actor, ag, env.ws, env.resources, env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        sid = skills[act[0]].skill_id if 0 <= act[0] < len(skills) else None
        if actor == aid and sid == SKILL:
            st["fire"] += 1
            d = ag.position.distance_to(opp_c.position) if opp_c.is_alive() else 99
            if d > CORNER:
                st["not_cornered"] += 1
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    st["games"] += 1


for driver in ("model", "expert"):
    st = dict(games=0, fire=0, not_cornered=0)
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        for i in range(games):
            play(driver, opp, base + i, st)
    n = st["games"]
    print(f"{driver:7s}: {SKILL}/game={st['fire']/n:4.2f}  "
          f"fired-while-NOT-cornered(>{CORNER}m)/game={st['not_cornered']/n:4.2f}")
