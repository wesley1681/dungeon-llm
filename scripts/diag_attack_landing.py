"""Localize the champion/BM damage-output deficit: is the model WHIFFING attacks
(attacking out of reach / missing) vs the expert?

For a class, play model and expert (both env.step) and for every AGENT
weapon-attack sub-action record the opponent HP drop it caused. Reports:
  - attacks/game, avg damage per attack, % of attacks dealing 0 (whiff/no-op),
    and avg dmg-dealt — so we see if the model attacks AS MUCH but lands LESS.

Usage: python scripts/diag_attack_landing.py <arch> <model.pt> [games]
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
games = int(sys.argv[3]) if len(sys.argv) > 3 else 12

net = CombatPolicyNet()
sd = torch.load(model_path, map_location="cpu")
sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
net.load_state_dict(sd, strict=False); net.eval()


def is_weapon_attack(act, ws, actor):
    if act[0] <= 0:
        return False
    sk = available_skills(ws.characters[actor], ws)
    if not (0 <= act[0] < len(sk)):
        return False
    return sk[act[0]].skill_id.startswith("weapon:")


def play(driver, opp, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[opp], level=5)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    expert = make_archetype_policy(arch) if driver == "expert" else None
    atk_dmgs = []
    done = False
    while not done:
        actor = env.current_agent_id
        if driver == "model":
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        else:
            a = env.ws.characters[actor]
            dec = expert.decide(actor, a, env.ws, env.resources, env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        track = (actor == aid and is_weapon_attack(act, env.ws, actor))
        hp_before = env.ws.characters[oid].hp if track else None
        obs, _, term, trunc, _ = env.step(act)
        if track:
            dealt = max(0.0, hp_before - env.ws.characters[oid].hp)
            atk_dmgs.append(dealt)
        done = term or trunc
    return atk_dmgs


for driver in ("model", "expert"):
    alld = []
    for opp in ARCHETYPE_LIST:
        base = hash(f"{arch}_{opp}") & 0xFFFFFF
        for i in range(games):
            alld.extend(play(driver, opp, base + i))
    a = np.array(alld) if alld else np.array([0.0])
    n_games = games * len(ARCHETYPE_LIST)
    print(f"{driver:7s}: weapon-attacks/game={len(a)/n_games:4.2f}  "
          f"avg_dmg/attack={a.mean():5.1f}  "
          f"whiff(0dmg)={np.mean(a == 0):4.0%}  "
          f"total_atk_dmg/game={a.sum()/n_games:5.1f}")
