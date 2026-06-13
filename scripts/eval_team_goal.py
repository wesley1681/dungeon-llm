"""Team-mode (3v3) goal eval — same fair standard as eval_goal, at team level.

Both sides of the comparison drive the AGENT TEAM through env.step against
expert-controlled opponent teams, on identical matchups and seeds:

  routed : each agent routed to its archetype's best checkpoint (eval_routed)
  expert : each agent driven by its archetype's scripted expert policy

Teams are composed from the role template (1 front + 1 striker + 1 support,
roles from ARCHETYPE_ROLES) so every party is sensible — 48 possible comps.
Opponent comps are drawn deterministically from the same template population.

Win = all opponents dead while at least one agent alive (same rule both
drivers). GOAL: routed WR exceeds expert WR beyond the 2-sigma noise band.

Usage: python scripts/eval_team_goal.py [games_per_matchup] [n_opp_comps] [limit_agent_comps]
       defaults: 4 games x 6 opp comps x all 48 agent comps  (1152 eps/driver)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_opp_comps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

FRONT   = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front")
STRIKER = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker")
SUPPORT = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support")
ALL_COMPS = [list(c) for c in itertools.product(FRONT, STRIKER, SUPPORT)]

AGENT_COMPS = ALL_COMPS[:limit] if limit else ALL_COMPS
# Deterministic opponent comps per agent comp (varied, reproducible).
_rng = random.Random(20260610)
OPP_TABLE = {i: _rng.sample(range(len(ALL_COMPS)), n_opp_comps)
             for i in range(len(AGENT_COMPS))}


def play(driver: str, arch_to_net, agent_comp, opp_comp, seed) -> str:
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    experts = ({aid: make_archetype_policy(a) for aid, a in arch_of.items()}
               if driver == "expert" else None)
    done = False
    while not done:
        actor = env.current_agent_id
        if driver == "expert":
            ag = env.ws.characters[actor]
            dec = experts[actor].decide(actor, ag, env.ws, env.resources,
                                        env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        else:
            net = arch_to_net[arch_of[actor]]
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    opps_dead = all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
    team_alive = any(env.ws.characters[aid].is_alive() for aid in env.agent_ids)
    if opps_dead and team_alive:
        return "win"
    if not team_alive:
        return "loss"
    return "trunc"


def run_driver(driver: str, arch_to_net) -> dict:
    per_comp = {}
    total = len(AGENT_COMPS)
    for i, comp in enumerate(AGENT_COMPS):
        w = n = tr = 0
        for k in OPP_TABLE[i]:
            opp_comp = ALL_COMPS[k]
            base = stable_seed(f"{'+'.join(comp)}_vs_{'+'.join(opp_comp)}")
            for g in range(games):
                out = play(driver, arch_to_net, comp, opp_comp, base + g)
                n += 1
                w += (out == "win")
                tr += (out == "trunc")
        per_comp[tuple(comp)] = (w, n, tr)
        if (i + 1) % 8 == 0:
            ww = sum(v[0] for v in per_comp.values())
            nn = sum(v[1] for v in per_comp.values())
            print(f"  [{driver}] {i+1}/{total} comps  running WR={ww/nn:.0%}",
                  flush=True)
    return per_comp


def main():
    print(f"comps: {len(AGENT_COMPS)} agent x {n_opp_comps} opp x {games} games "
          f"= {len(AGENT_COMPS)*n_opp_comps*games} episodes per driver")
    print("loading routed checkpoints...")
    cache = {}
    arch_to_net = {}
    for arch, path in DEFAULT_ROUTING.items():
        if path not in cache:
            cache[path] = load_net(path)
        arch_to_net[arch] = cache[path]

    results = {}
    for driver in ("routed", "expert"):
        print(f"\n--- driver: {driver} ---", flush=True)
        results[driver] = run_driver(driver, arch_to_net)

    n_total = sum(v[1] for v in results["routed"].values())
    wr_m = sum(v[0] for v in results["routed"].values()) / n_total
    wr_e = sum(v[0] for v in results["expert"].values()) / n_total
    tr_m = sum(v[2] for v in results["routed"].values()) / n_total
    tr_e = sum(v[2] for v in results["expert"].values()) / n_total
    sigma = (wr_m*(1-wr_m)/n_total + wr_e*(1-wr_e)/n_total) ** 0.5
    diff = wr_m - wr_e

    print(f"\n=== TEAM GOAL EVAL (3v3, {n_total} eps/driver) ===")
    print(f"routed model team : WR={wr_m:6.1%}  (trunc {tr_m:.0%})")
    print(f"expert team       : WR={wr_e:6.1%}  (trunc {tr_e:.0%})")
    print(f"diff = {diff:+.1%}  (2-sigma band = ±{2*sigma:.1%})")
    if diff > 2 * sigma:
        print("VERDICT: BEAT — model team wins more than expert team beyond noise")
    elif diff >= -2 * sigma:
        print("VERDICT: TIE (within noise)")
    else:
        print("VERDICT: BEHIND")

    print(f"\nper-comp diff (model - expert), worst 12:")
    rows = []
    for comp in results["routed"]:
        wm, nm, _ = results["routed"][comp]
        we, ne, _ = results["expert"][comp]
        rows.append((wm/nm - we/ne, comp, wm, we, nm))
    rows.sort()
    for d, comp, wm, we, nm in rows[:12]:
        print(f"  {'+'.join(comp):44s} model {wm:2d}/{nm}  expert {we:2d}/{nm}  {d:+.0%}")
    print(f"\nper-comp diff, best 6:")
    for d, comp, wm, we, nm in rows[-6:]:
        print(f"  {'+'.join(comp):44s} model {wm:2d}/{nm}  expert {we:2d}/{nm}  {d:+.0%}")


if __name__ == "__main__":
    main()
