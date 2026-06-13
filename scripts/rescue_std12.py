"""Post-PPO BC rescue of std12 (扮演怪物 wave — gate repair).

The u0032 monster-actor champion learned the monster seats at a −6.6pp std12
cost (Pareto-optimal but a real erosion of standard-class self-play; §10).
The seed-wave precedent: pure BC against greedy self-distill anchors pulls a
metric back without erasing learned behavior — IF the behavior you want to
KEEP is itself anchored (zero-desc anchors can't pin behavior in the space
that actually plays — the v2 −5.6pp lesson).

So three greedy-self-distill anchors, 1:1:1, pure BC, no PPO:
  A  std12 standard-seat states  → the metric being rescued (collected here
     from the WARM seed_s1600 net so the labels target the GOOD std12 play)
  B  monster-seat states          → lock the PPO-learned monster play
     (collected from u0032 itself — NOT a script, so this is self-distill,
     not imitation; the /goal bar is about how the play was LEARNED, and it
     was learned by PPO)
  C  switch_demos                  → keep the 12j conditional switching

If std12 climbs back toward 44 while seat-boss/seat-1v1 hold, that snapshot
is the complete champion. If the rescue trades monster play away point-for-
point, the trade-off is unsolvable here and u0032 stands as-is.

Usage: python scripts/rescue_std12.py --steps 500 --out_dir models/mon_rescue
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     PARTY3_EQUIV_LEVEL, register_monsters,
                                     onev1_viable_monsters)
register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single, standard_probe, monster_opp_probe
from train_monster_actor import (seat_probe, SEAT_1V1, SEAT_BOSS, train_pools,
                                 sample_seat)


def collect_greedy(net, n_states, seed, rng, seat_sampler):
    """Greedy self-distill: walk episodes whose seat is chosen by
    seat_sampler(rng, ep) -> (agent, a_lvl, opps, o_lvl); record the net's
    GREEDY action at each blinded state (END pairs incl., TT=-1)."""
    net.eval()
    O, A, T = [], [], []
    ep = 0
    while len(A) < n_states:
        agent, a_lvl, opps, o_lvl = seat_sampler(rng, ep)
        env = CombatEnvV2(seed=seed * 1_000_003 + ep,
                          n_agents=1, n_opps=len(opps))
        env.reset(agent_archs=[agent], opp_archs=list(opps),
                  level=a_lvl, opp_level=o_lvl)
        obs = blind_np_single(build_obs(env.ws, env.current_agent_id,
                                        env.resources))
        done = False
        while not done:
            aid = env.current_agent_id
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, aid)
            e = apply_entity_mask(e, ot, env.ws, aid)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=aid))
            tt = -1
            if act[0] != 0:
                sks = available_skills(env.ws.characters[aid], env.ws)
                tt = (int(sks[act[0]].features.target_type)
                      if act[0] < len(sks) else -1)
            O.append(obs); A.append(act); T.append(tt)
            obs2, _, term, trunc, _ = env.step(act)
            done = term or trunc
            obs = blind_np_single(obs2) if not done else obs2
        ep += 1
        if ep % 200 == 0:
            print(f"    {len(A)}/{n_states} ({ep} eps)", flush=True)
    return ({k: np.stack([o[k] for o in O]) for k in O[0]},
            np.array(A, np.int64), np.array(T, np.int64))


def std_sampler(rng, ep):
    """Standard class vs standard/monster opponent (the std12 seat)."""
    a_lvl = rng.randint(5, 8)
    agent = rng.choice(list(STANDARD_ARCHETYPES))
    if rng.random() < 0.3:
        opp = rng.choice(onev1_viable_monsters(8.0))
        eq = EQUIV_LEVEL_1V1[opp]
        a_lvl = int(min(8, max(3, round(eq))))
        o_lvl = MONSTER_DEFS[opp].natural_level
    else:
        opp = rng.choice(list(STANDARD_ARCHETYPES)); o_lvl = a_lvl
    return agent, a_lvl, [opp], o_lvl


def mon_sampler(t1, tb):
    def s(rng, ep):
        kind, agent, a_lvl, opps, o_lvl, tag = sample_seat(
            rng, t1, tb, p_boss=0.4, p_mon_agent=0.6, p_synth=0.0,
            p_mon_opp=0.0, ep=ep)
        return agent, a_lvl, opps, o_lvl
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champ", default="models/mon_actor1/ma_u0032.pt")
    p.add_argument("--warm_std", default="models/seed_dtype1/seed_s1600.pt",
                   help="net whose GREEDY std12 play is the rescue target")
    p.add_argument("--switch_demos",
                   default="models/seed_dtype1/switch_demos.npz")
    p.add_argument("--out_dir", default="models/mon_rescue")
    p.add_argument("--std_states", type=int, default=6000)
    p.add_argument("--mon_states", type=int, default=6000)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "BLIND").write_text("std12 BC rescue of monster-actor champ\n")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t1, tb = train_pools()

    champ = load_student(args.champ)
    warm = load_student(args.warm_std)

    print(f"A: collecting std-seat greedy self-distill from WARM "
          f"({args.std_states})…", flush=True)
    A_obs, A_act, A_tt = collect_greedy(warm, args.std_states,
                                        args.seed * 7919 + 1, rng, std_sampler)
    print(f"B: collecting monster-seat greedy self-distill from CHAMP "
          f"({args.mon_states})…", flush=True)
    B_obs, B_act, B_tt = collect_greedy(champ, args.mon_states,
                                        args.seed * 7919 + 2, rng,
                                        mon_sampler(t1, tb))
    sw = np.load(args.switch_demos)
    C_obs = {k[4:]: sw[k] for k in sw.files if k.startswith("obs_")}
    C_act, C_tt = sw["actions"], sw["target_types"]
    print(f"C: switch_demos {len(C_act)}", flush=True)

    net = champ
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    def evalp(tag):
        net.eval()
        std, _ = standard_probe(net, games=2)
        mon, _ = monster_opp_probe(net, t1, games=2)
        seats = seat_probe(net, games_1v1=4, games_boss=12)
        s1 = sum(seats[m] for m in SEAT_1V1) / len(SEAT_1V1)
        sb = sum(seats[f"boss:{m}@L{l}"] for m, l in SEAT_BOSS) / len(SEAT_BOSS)
        print(f"  [{tag}] std12={std:.1%} vs-mon={mon:.1%} "
              f"seat-1v1={s1:.1%} seat-boss={sb:.1%}", flush=True)
        net.train()
        return std, sb

    print("\nbaseline (champ, pre-rescue):", flush=True)
    evalp("u0032")
    sources = [(A_obs, A_act, A_tt), (B_obs, B_act, B_tt),
               (C_obs, C_act, C_tt)]
    for step in range(1, args.steps + 1):
        for so, sa, st in sources:          # A, B, C each one minibatch
            sel = np.random.randint(0, len(sa), size=args.batch)
            ob = {k: torch.from_numpy(v[sel]) for k, v in so.items()}
            bc_loss_step(net, ob, torch.from_numpy(sa[sel]),
                         torch.from_numpy(st[sel]), optim)
        if step % args.eval_every == 0:
            torch.save(net.state_dict(), out_dir / f"rescue_s{step:04d}.pt")
            evalp(f"s{step}")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
