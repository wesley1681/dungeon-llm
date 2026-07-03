"""1v2 focus-fire test. Economy probe showed: model attacks as much as the
expert and deals ~similar total damage, but loses more in 1v2. Hypothesis: it
SPREADS damage instead of focus-killing one enemy first, so both enemies keep
attacking → more incoming damage → it dies.

Measured, model vs scripted expert, same seats/dice (1v2 weak):
  spread@1stkill  when the FIRST enemy dies, the fraction of the SECOND enemy's
                  HP already removed. HIGH = spread damage (bad: both were kept
                  alive together). LOW = clean focus-kill. (nan if never gets a
                  kill.)
  agent_end_hp    agent HP fraction at game end (higher = took less incoming).
  ttfk            agent decisions until the first enemy dies (lower = faster).
  WR              win rate.

Usage: python scripts/probe_focus.py --ckpt models/pop_los_final/pop_u0044.pt
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
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def run(net, ident, games, n_opp=2, opp_level=2):
    spreads, end_hps, ttfks, wins = [], [], [], 0
    for gi in range(games):
        key = f"{ident}|{n_opp}|{opp_level}|{gi}"
        k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=n_opp)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n_opp)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=opp_level, layout="open")
        aid = env.agent_ids[0]
        oids = list(env.opp_ids)
        maxhp = {o: env.ws.characters[o].max_hp for o in oids}
        script = None if net is not None else make_archetype_policy(ident)
        dec_count = 0
        first_kill_dec = None
        spread_at_kill = None
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
                dec_count += 1
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
            # detect first enemy death
            if first_kill_dec is None:
                dead = [o for o in oids if not env.ws.characters[o].is_alive()]
                if dead:
                    first_kill_dec = dec_count
                    others = [o for o in oids if o not in dead]
                    if others:
                        o = others[0]
                        spread_at_kill = 1.0 - max(0, env.ws.characters[o].hp) / maxhp[o]
                    else:
                        spread_at_kill = 0.0   # both died same step
        if first_kill_dec is not None:
            ttfks.append(first_kill_dec)
            if spread_at_kill is not None:
                spreads.append(spread_at_kill)
        end_hps.append(max(0, env.ws.characters[aid].hp) / env.ws.characters[aid].max_hp)
        if env.ws.characters[aid].is_alive() and all(
                not env.ws.characters[o].is_alive() for o in oids):
            wins += 1
    f = lambda L: (sum(L) / len(L)) if L else float("nan")
    return dict(spread=f(spreads), end_hp=f(end_hps), ttfk=f(ttfks),
                wr=wins / games)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--idents", nargs="*",
                   default=["champion", "battle_master", "war", "assassin",
                            "arcane_trickster"])
    p.add_argument("--n_opp", type=int, default=2)
    p.add_argument("--opp_level", type=int, default=2)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  {1}v{args.n_opp} opp_lvl={args.opp_level}, "
          f"model vs expert, same dice")
    print("spread@1stkill: 2nd enemy HP already gone when 1st dies "
          "(HIGH=spreading=bad). end_hp: agent survival.\n")
    for ident in args.idents:
        m = run(net, ident, args.games, args.n_opp, args.opp_level)
        x = run(None, ident, args.games, args.n_opp, args.opp_level)
        print(f"  {ident:<17} MODEL spread={m['spread']:>4.0%} end_hp={m['end_hp']:>4.0%} "
              f"ttfk={m['ttfk']:>4.1f} WR={m['wr']:>4.0%}  ||  EXPERT "
              f"spread={x['spread']:>4.0%} end_hp={x['end_hp']:>4.0%} "
              f"ttfk={x['ttfk']:>4.1f} WR={x['wr']:>4.0%}", flush=True)


if __name__ == "__main__":
    main()
