"""What does the EXPERT do that the model doesn't? The 1v2 gap is survival
(agent ends at half the expert's HP). This dumps the per-turn skill-usage
histogram for both arms so we can see which DEFENSIVE / mitigation options
(dodge, second wind, parry, shield, cure, disengage, uncanny dodge...) the
expert uses and the model skips. Same seats/dice, 1v2 weak.

Usage: python scripts/probe_usage.py --ckpt models/pop_los_final/pop_u0044.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from collections import Counter
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import decode_action, encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def run(net, ident, games, n_opp=2):
    usage = Counter(); n_act = 0; wins = 0
    for gi in range(games):
        key = f"{ident}|{n_opp}|{gi}"; k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=n_opp)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n_opp)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]; oids = list(env.opp_ids)
        script = None if net is not None else make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id; ch = env.ws.characters[actor]
            if net is not None:
                ob = blind_np_single(obs)
                ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            else:
                dec = script.decide(actor, ch, env.ws, env.resources,
                                    env.ws.combat.round_number)
                act = [0, 0, 0] if (dec.action is None or getattr(dec, "fled", False)) \
                    else (list(encode_action(dec.action, env.ws, actor))
                          if _try(dec, env, actor) else [0, 0, 0])
            if actor == aid and act[0] > 0:
                sks = available_skills(ch, env.ws)
                if act[0] < len(sks):
                    usage[sks[act[0]].skill_id] += 1; n_act += 1
            obs, _, term, trunc, _ = env.step(act); done = term or trunc
        if env.ws.characters[aid].is_alive() and all(
                not env.ws.characters[o].is_alive() for o in oids):
            wins += 1
    return usage, n_act, wins / games


def _try(dec, env, actor):
    try:
        encode_action(dec.action, env.ws, actor); return True
    except Exception:
        return False


def top(usage, n, k=8):
    if not n:
        return "-"
    return "  ".join(f"{sid.replace('weapon:','w:')}={c/n:.0%}"
                     for sid, c in usage.most_common(k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--n_opp", type=int, default=2)
    p.add_argument("--idents", nargs="*",
                   default=["champion", "battle_master", "war", "assassin",
                            "arcane_trickster"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  1v{args.n_opp} weak  (share of the arm's non-end actions)\n")
    for ident in args.idents:
        mu, mn, mw = run(net, ident, args.games, args.n_opp)
        xu, xn, xw = run(None, ident, args.games, args.n_opp)
        print(f"== {ident}  (model WR {mw:.0%} / expert WR {xw:.0%}) ==")
        print(f"  MODEL : {top(mu, mn)}")
        print(f"  EXPERT: {top(xu, xn)}\n", flush=True)


if __name__ == "__main__":
    main()
