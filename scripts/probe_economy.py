"""Is the WR gap an ATTACK-ECONOMY gap? Move-cell probe showed the base model's
adjacent moves are post-attack leftovers (attack already spent) — harmless to
WR. So the gap vs the expert must be DAMAGE OUTPUT per turn. This probe measures,
model vs scripted expert, same seats/dice:

  atk/turn       ATTACK sub-actions resolved per agent turn
  end-w/action   share of agent turns ENDED while an action was still available
                 (left an attack on the table)
  end-w/bonus    share ended while a bonus action was still available
  dmg/game       total enemy HP fraction removed

If the model's atk/turn << expert's and/or end-w/action >> expert's, the gap is
attack economy (under-using actions / bonus actions / extra attacks), not
movement. Paired global-RNG dice (seeded per episode).

Usage:
  python scripts/probe_economy.py --ckpt models/pop_los_final/pop_u0044.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
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


def _is_attack(act, ws, aid):
    ad = decode_action(act, ws, aid)
    return ad is not None and "ATTACK" in str(ad.get("type", "")).upper()


def run(net, ident, n_opp, games):
    atks = turns = end_act = end_bonus = 0
    dmg_sum = 0.0
    for gi in range(games):
        key = f"{ident}|{n_opp}|{gi}"
        k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=n_opp)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n_opp)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]
        oids = list(env.opp_ids)
        tot_max = sum(env.ws.characters[o].max_hp for o in oids)
        script = None if net is not None else make_archetype_policy(ident)
        prev_actor = None
        done = False
        while not done:
            actor = env.current_agent_id
            ch = env.ws.characters[actor]
            if net is not None:
                ob = blind_np_single(obs)
                ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
                with torch.no_grad():
                    el, s, e, g = net(ot)
                s = apply_resource_mask(s, env.resources, env.ws, actor)
                e = apply_entity_mask(e, ot, env.ws, actor)
                act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                       agent_id=actor))
            else:
                dec = script.decide(actor, ch, env.ws, env.resources,
                                    env.ws.combat.round_number)
                if dec.action is None or getattr(dec, "fled", False):
                    act = [0, 0, 0]
                else:
                    try:
                        act = list(encode_action(dec.action, env.ws, actor))
                    except Exception:
                        act = [0, 0, 0]
            if actor == aid:
                if actor != prev_actor:
                    turns += 1
                ad = decode_action(act, env.ws, aid)
                if ad is None:    # END decision
                    end_act += int(env.resources.get("action", 0) > 0)
                    end_bonus += int(env.resources.get("bonus_action", 0) > 0)
                elif "ATTACK" in str(ad.get("type", "")).upper():
                    atks += 1
            prev_actor = actor
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
        hp_left = sum(max(0, env.ws.characters[o].hp) for o in oids)
        dmg_sum += (1.0 - hp_left / tot_max) if tot_max else 0.0
    t = turns or 1
    return dict(atk_turn=atks / t, end_act=end_act / t, end_bonus=end_bonus / t,
                dmg=dmg_sum / games, turns=turns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--idents", nargs="*",
                   default=["champion", "battle_master", "war", "assassin",
                            "arcane_trickster"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  (model vs scripted expert, same seats/dice)")
    print("atk/turn = ATTACK sub-actions per agent turn; end-act = turns ended "
          "with an ACTION still available (attack left on table)\n")
    for n_opp in (1, 2):
        print(f"== 1v{n_opp} weak ==")
        for ident in args.idents:
            m = run(net, ident, n_opp, args.games)
            x = run(None, ident, n_opp, args.games)
            print(f"  {ident:<17} MODEL atk/turn={m['atk_turn']:.2f} "
                  f"end-act={m['end_act']:.0%} end-bonus={m['end_bonus']:.0%} "
                  f"dmg={m['dmg']:.0%}  ||  EXPERT atk/turn={x['atk_turn']:.2f} "
                  f"end-act={x['end_act']:.0%} end-bonus={x['end_bonus']:.0%} "
                  f"dmg={x['dmg']:.0%}", flush=True)
        print()


if __name__ == "__main__":
    main()
