"""Why do wall-fine-tuned models crater on symmetric 2v2 (58%->8%)?

Action mix is identical to base (attack 43% both), so it's not over-moving.
Next hypothesis: TARGET SELECTION / focus-fire. Two agents that split fire
(each chips a different enemy) lose the action-economy race vs two that focus
one enemy down (removing an attacker a turn sooner). This traces, per round,
which enemy each living agent attacks, and reports:
    focus%     rounds where all living agents that attacked hit the SAME enemy
    t_firstkill mean turns until the agent side scores its first kill
    lead%      episodes where the agent side gets first blood
A base>fine-tuned gap on focus%/first-kill timing would be the mechanism.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import random
from collections import defaultdict
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import decode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single


def trace_2v2(net, games=24):
    foc_num = foc_den = 0
    t_kills, leads, n = [], 0, 0
    for g in range(games):
        k = crc32(f"2v2|{g}".encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 7, n_agents=2, n_opps=2)
        env.reset(agent_archs=["champion", "battle_master"],
                  opp_archs=["champion", "war"], level=5, opp_level=5,
                  layout="open")
        aids = set(env.agent_ids); oids = list(env.opp_ids)
        round_targets = defaultdict(list)   # round -> [enemy_id hit by an agent]
        first_kill_turn = None
        first_blood = None
        turns = 0
        done = False
        while not done:
            actor = env.current_agent_id
            rnd = env.ws.combat.round_number
            n_opp_alive_before = sum(env.ws.characters[o].is_alive() for o in oids)
            n_ag_alive_before = sum(env.ws.characters[a].is_alive() for a in aids)
            obs = build_obs(env.ws, actor, env.resources)
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g2 = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g2[0], ws=env.ws, agent_id=actor))
            # decode target enemy (if this is an attack on an enemy)
            if actor in aids and act[0] > 0:
                ad = decode_action(act, env.ws, actor)
                if ad is not None:
                    tgt = (ad.get("target") or ad.get("character")
                           or ad.get("target_id"))
                    if tgt in oids:
                        round_targets[rnd].append(tgt)
            env.step(act)
            turns += 1
            n_opp_alive = sum(env.ws.characters[o].is_alive() for o in oids)
            n_ag_alive = sum(env.ws.characters[a].is_alive() for a in aids)
            if first_kill_turn is None and n_opp_alive < n_opp_alive_before:
                first_kill_turn = turns; first_blood = "agent"
            if first_blood is None and n_ag_alive < n_ag_alive_before:
                first_blood = "opp"
            done = env.ws.combat is None or not env.ws.combat.active or turns > 120
            if (all(not env.ws.characters[o].is_alive() for o in oids)
                    or all(not env.ws.characters[a].is_alive() for a in aids)):
                done = True
        # focus metric: rounds with >=2 agent attacks where all hit same enemy
        for rnd, tgts in round_targets.items():
            if len(tgts) >= 2:
                foc_den += 1
                foc_num += int(len(set(tgts)) == 1)
        if first_kill_turn is not None:
            t_kills.append(first_kill_turn)
        if first_blood == "agent":
            leads += 1
        n += 1
    focus = foc_num / foc_den if foc_den else float("nan")
    tk = sum(t_kills) / len(t_kills) if t_kills else float("nan")
    return focus, foc_den, tk, leads / n


def main():
    for name, ck in [("base", "models/pop_mon/pop_u0005.pt"),
                     ("u0032(wall)", "models/pop_wall2/pop_u0032.pt"),
                     ("u0060(clean)", "models/pop_los_clean/pop_u0060.pt")]:
        net = load_student(ck); net.eval()
        focus, den, tk, lead = trace_2v2(net, games=24)
        print(f"{name:14s}  focus-fire={focus:.0%} (n={den})  "
              f"t_firstkill={tk:.1f}  first-blood%={lead:.0%}", flush=True)


if __name__ == "__main__":
    main()
