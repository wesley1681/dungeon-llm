"""Post-revive-mechanic balance probe.

PART A — 3v3 expert-vs-expert with life-cleric supports: how often does the
    new pick-up (revive) actually fire, and what does it do to win rate /
    front-death? Agent side are PCs (death saves + revivable); opponents are
    NPCs (die at 0) — the standard D&D PC-vs-monster asymmetry.

PART B — 1v1 expert-vs-expert vs life/war cleric opponents: the
    _find_heal_target side-relativity fix means opponent clerics now heal
    THEMSELVES (before: they scanned the enemy team and healed nobody — or,
    in touch range, the enemy). Their 1v1 strength shifts; this re-baselines
    expert-vs-expert WR for the eval_goal criterion.

Usage: python scripts/diag_revive.py [games_3v3] [games_1v1]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
from collections import defaultdict

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import stable_seed

games_3v3 = int(sys.argv[1]) if len(sys.argv) > 1 else 16
games_1v1 = int(sys.argv[2]) if len(sys.argv) > 2 else 120

FRONTS   = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "front")
STRIKERS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "striker")
SUPPORTS = sorted(a for a, r in ARCHETYPE_ROLES.items() if r == "support")
ALL_COMPS = [list(c) for c in itertools.product(FRONTS, STRIKERS, SUPPORTS)]
_rng = random.Random(20260610)
OPP_SET = [ALL_COMPS[i] for i in _rng.sample(range(len(ALL_COMPS)), 6)]
LIFE_COMPS = [[f, s, "life"] for f in FRONTS for s in ("evocation", "divination")]


def play_team(agent_comp, opp_comp, seed, st):
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    experts = {aid: make_archetype_policy(a)
               for aid, a in zip(env.agent_ids, env.agent_archs)}
    front = env.agent_ids[0]
    done = False
    downs_seen = set()
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        rnd = env.ws.combat.round_number
        dec = experts[actor].decide(actor, ag, env.ws, env.resources, rnd)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        if act[0] < 0:
            act = [0, 0, 0]
        _, _, term, trunc, info = env.step(act)
        done = term or trunc
        res = (info or {}).get("action_result") or {}
        if res.get("type") in ("HEAL", "LAY_ON_HANDS") and res.get("revived"):
            st["revives"] += 1
        if res.get("type") == "ERROR":
            st["errors"] += 1
        for aid in env.agent_ids:
            if env.ws.characters[aid].is_dying():
                downs_seen.add(aid)
    st["downs"] += len(downs_seen)
    st["rounds"] += env.ws.combat.round_number
    st["games"] += 1
    st["front_dead"] += int(env.ws.characters[front].is_dead())
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    st["wins"] += int(opps_dead and alive)


def play_1v1(agent_arch, opp_arch, seed, st):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=5)
    expert = make_archetype_policy(agent_arch)
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        dec = expert.decide(actor, ag, env.ws, env.resources,
                            env.ws.combat.round_number)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        if act[0] < 0:
            act = [0, 0, 0]
        _, _, term, trunc, _ = env.step(act)
        done = term or trunc
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    st["games"] += 1
    st["wins"] += int(opps_dead and alive)


print(f"=== PART A: 3v3 expert vs expert, life comps × {len(OPP_SET)} opps × {games_3v3}g ===")
st = defaultdict(float)
for comp in LIFE_COMPS:
    for opp in OPP_SET:
        base = stable_seed(f"revive_{'+'.join(comp)}_{'+'.join(opp)}")
        for g in range(games_3v3):
            play_team(comp, opp, base + g, st)
n = st["games"]
wr = st["wins"] / n
sig = (wr * (1 - wr) / n) ** 0.5
print(f"  WR={wr:5.1%} (±{2*sig:.1%})  downs/g={st['downs']/n:4.2f}  "
      f"revives/g={st['revives']/n:4.2f}  front-DEAD={st['front_dead']/n:3.0%}  "
      f"rounds/g={st['rounds']/n:3.1f}  expert-ERRORs/g={st['errors']/n:4.2f}",
      flush=True)

print(f"\n=== PART B: 1v1 expert vs life/war cleric opponents × {games_1v1}g ===")
ALL_ARCH = sorted(ARCHETYPE_ROLES.keys())
for opp_arch in ("life", "war"):
    print(f"  [vs {opp_arch}]")
    for arch in ALL_ARCH:
        st = defaultdict(float)
        base = stable_seed(f"revive1v1_{arch}_{opp_arch}")
        for g in range(games_1v1):
            play_1v1(arch, opp_arch, base + g, st)
        wr = st["wins"] / st["games"]
        sig = (wr * (1 - wr) / st["games"]) ** 0.5
        print(f"    {arch:12s} WR={wr:5.1%} (±{2*sig:.1%})", flush=True)
