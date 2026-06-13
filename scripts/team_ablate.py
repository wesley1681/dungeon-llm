"""Per-slot ablation on team comps: which member causes the model-expert gap?

For each comp, plays the same matchups/seeds with driver assignments:
  MMM (all model), EMM / MEM / MME (one slot expert), EEE (all expert)
Slot order = (front, striker, support). If swapping one slot to expert
recovers most of the MMM->EEE gap, that slot's policy is the team-mode
weak link.

Usage: python scripts/team_ablate.py [games] [n_opp_comps] [comp ...]
  comp as front+striker+support, e.g. battle_master+assassin+war
  (no comps given -> a default set of the worst-12 from team_baseline_v1)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_opp_comps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
comps_arg = sys.argv[3:]

DEFAULT_COMPS = [
    "battle_master+assassin+war",
    "devotion+arcane_trickster+war",
    "devotion+assassin+life",
    "battle_master+arcane_trickster+life",
    "berserker+arcane_trickster+life",
    "champion+arcane_trickster+war",
    "totem_bear+assassin+life",
]
COMPS = [c.split("+") for c in (comps_arg or DEFAULT_COMPS)]

FRONT   = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front")
STRIKER = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker")
SUPPORT = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support")
ALL_COMPS = [list(c) for c in itertools.product(FRONT, STRIKER, SUPPORT)]
_rng = random.Random(20260610)
OPP_SET = [ALL_COMPS[i] for i in _rng.sample(range(len(ALL_COMPS)), n_opp_comps)]

print("loading routed checkpoints...")
cache = {}
ARCH_TO_NET = {}
for arch, path in DEFAULT_ROUTING.items():
    if path not in cache:
        cache[path] = load_net(path)
    ARCH_TO_NET[arch] = cache[path]


def play(assign: str, agent_comp, opp_comp, seed) -> bool:
    """assign: string of M/E per slot, e.g. 'MEM'."""
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    slot_of = {aid: i for i, aid in enumerate(env.agent_ids)}
    experts = {aid: make_archetype_policy(a) for aid, a in arch_of.items()}
    done = False
    while not done:
        actor = env.current_agent_id
        if assign[slot_of[actor]] == "E":
            ag = env.ws.characters[actor]
            dec = experts[actor].decide(actor, ag, env.ws, env.resources,
                                        env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
        else:
            net = ARCH_TO_NET[arch_of[actor]]
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
    return opps_dead and team_alive


ASSIGNS = ("MMM", "EMM", "MEM", "MME", "EEE")
print(f"{games} games x {n_opp_comps} opp comps per assign "
      f"({games*n_opp_comps} eps per cell)\n")
print(f"{'comp':44s}  " + "  ".join(f"{a:>5s}" for a in ASSIGNS))
tot = {a: [0, 0] for a in ASSIGNS}
for comp in COMPS:
    row = {}
    for assign in ASSIGNS:
        w = n = 0
        for opp_comp in OPP_SET:
            base = stable_seed(f"{'+'.join(comp)}_vs_{'+'.join(opp_comp)}")
            for g in range(games):
                n += 1
                w += play(assign, comp, opp_comp, base + g)
        row[assign] = (w, n)
        tot[assign][0] += w
        tot[assign][1] += n
    print(f"{'+'.join(comp):44s}  "
          + "  ".join(f"{row[a][0]/row[a][1]:5.0%}" for a in ASSIGNS), flush=True)

print(f"\n{'TOTAL':44s}  "
      + "  ".join(f"{tot[a][0]/tot[a][1]:5.0%}" for a in ASSIGNS))
print("\nslot legend: M=model E=expert, order = front/striker/support")
print("EMM=expert front, MEM=expert striker, MME=expert support")
