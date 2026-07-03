"""Does the model KNOW when to use the high-value actions it skips, or has it
never learned them? Drive the seat with the SCRIPTED EXPERT; at each expert
decision compute the MODEL's masked-softmax probability of the SAME skill slot.

  knows but won't pick (credit/exploration gap): model assigns meaningful prob
     to the expert's high-value action but its greedy argmax is something else.
  doesn't know (BC/knowledge gap): model prob ~0 on the expert's action.

Reports, per class, the model's mean prob on the expert's chosen skill, broken
out for the DROPPED high-value skills (menacing_attack, spiritual_weapon,
cure_wounds, sleep, melee shortsword, spirit_guardians) vs everything else.
Same seats/dice, 1v2 weak.

Usage: python scripts/probe_bcfit.py --ckpt models/pop_los_final/pop_u0044.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from collections import defaultdict
from zlib import crc32
import torch
import torch.nn.functional as F

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS

HIGH_VALUE = {"menacing_attack", "spiritual_weapon", "spiritual_weapon_attack_war",
              "cure_wounds", "sleep", "spirit_guardians", "weapon:短劍",
              "distracting_strike"}


def run(net, ident, games):
    # skill_id -> [sum_model_prob, n, sum_is_argmax]
    stat = defaultdict(lambda: [0.0, 0, 0])
    for gi in range(games):
        key = f"{ident}|2|{gi}"; k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(2)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]
        script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id; ch = env.ws.characters[actor]
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]
            if actor == aid and act[0] > 0:
                sks = available_skills(ch, env.ws)
                if act[0] < len(sks):
                    sid = sks[act[0]].skill_id
                    # model's masked-softmax prob of THIS skill slot at THIS state
                    ob = blind_np_single(obs)
                    ot = {kk: torch.from_numpy(v).unsqueeze(0)
                          for kk, v in ob.items()}
                    with torch.no_grad():
                        _, s, _, _ = net(ot)
                    s = apply_resource_mask(s, env.resources, env.ws, actor)
                    probs = F.softmax(s[0], dim=-1)
                    p = float(probs[act[0]])
                    is_arg = int(int(probs.argmax()) == act[0])
                    rec = stat[sid]
                    rec[0] += p; rec[1] += 1; rec[2] += is_arg
            obs, _, term, trunc, _ = env.step(act); done = term or trunc
    return stat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--idents", nargs="*",
                   default=["battle_master", "war", "arcane_trickster", "assassin"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  expert drives seat; model prob of expert's action")
    print("p = model masked-softmax prob on expert's chosen skill; "
          "arg% = expert's action is model's greedy pick.  [HV]=high-value\n")
    for ident in args.idents:
        stat = run(net, ident, args.games)
        print(f"== {ident} ==")
        for sid in sorted(stat, key=lambda s: -stat[s][1]):
            tot, n, arg = stat[sid]
            if n < 3:
                continue
            mark = " [HV]" if sid in HIGH_VALUE else ""
            print(f"  {sid:<28} n={n:>3}  model_p={tot/n:>5.1%}  "
                  f"arg%={arg/n:>4.0%}{mark}", flush=True)
        print()


if __name__ == "__main__":
    main()
