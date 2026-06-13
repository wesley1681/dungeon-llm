"""Asymmetric-encounter gap map: does the routed model hold up (and not slack
off) when the fight is unbalanced?

Three axes, all same-standard (routed model team vs per-agent expert policies,
identical matchups/seeds, both driven through env.step):

  A. 1v1 level gap   — mirror archetype, agent L5 vs opp L2..L8 (ΔL −3..+3)
  B. 3v3 level gap   — mirror comp, agent side L5 vs opp side L2..L8
  C. 3v3 number gap  — both sides L5, (3v2 / 2v3 / 3v1 / 1v3), template comps
                       truncated front-first ([front] / [front,striker] / full)

Per bucket and per driver:
  WR        — win = all opponents dead AND >=1 agent alive
  dealt     — fraction of the enemy side's total HP pool removed by episode end
              (the "effort" metric: a slacking policy stops dealing damage in
              hopeless buckets; an expert keeps trading to the end)
  idle      — fraction of agent TURNS that ended on their first sub-action with
              the action budget still unspent (direct give-up/end-spam marker)
  rounds    — avg combat rounds

"擺爛" verdict per bucket = model meaningfully below expert on dealt/idle in
the disadvantage buckets even where both WRs are ~0.

Usage: python scripts/eval_asym.py [games_1v1] [games_3v3]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from collections import defaultdict

import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

games_1v1 = int(sys.argv[1]) if len(sys.argv) > 1 else 30
games_3v3 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
# argv[3]: optional bucket filter, e.g. "B+1,B+2,C2v3".
# argv[4:]: optional routing overrides "arch=checkpoint" (candidate testing).
FILTER = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
for _a in sys.argv[4:]:
    _k, _v = _a.split("=", 1)
    DEFAULT_ROUTING[_k] = _v


def want(tag: str) -> bool:
    return FILTER is None or tag in FILTER

SPREAD_COMPS = [c.split("+") for c in (
    "battle_master+evocation+life",  "battle_master+assassin+war",
    "champion+divination+life",      "champion+arcane_trickster+war",
    "totem_bear+evocation+war",      "totem_bear+assassin+life",
    "berserker+divination+war",      "berserker+arcane_trickster+life",
    "devotion+evocation+life",       "devotion+assassin+war",
    "vengeance+divination+life",     "vengeance+arcane_trickster+war",
)]
DELTAS = (-3, -2, -1, 0, 1, 2, 3)
SIZE_CONFIGS = ((3, 2), (2, 3), (3, 1), (1, 3))

print("loading routed checkpoints...")
_cache: dict = {}
ARCH_TO_NET: dict = {}
for arch, path in DEFAULT_ROUTING.items():
    if path not in _cache:
        _cache[path] = load_net(path)
    ARCH_TO_NET[arch] = _cache[path]


def play(driver, agent_comp, opp_comp, n_agents, n_opps,
         agent_level, opp_level, seed, st) -> None:
    env = CombatEnvV2(seed=seed, n_agents=n_agents, n_opps=n_opps)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp,
                       level=agent_level, opp_level=opp_level)
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    experts = ({aid: make_archetype_policy(a) for aid, a in arch_of.items()}
               if driver == "expert" else None)
    opp_pool = sum(env.ws.characters[o].max_hp for o in env.opp_ids)

    done = False
    prev_actor = None
    first_step_of_turn = True
    while not done:
        actor = env.current_agent_id
        first_step_of_turn = (actor != prev_actor)
        prev_actor = actor
        ag = env.ws.characters[actor]
        had_action = env.resources.get("action", 0) > 0
        if experts is not None:
            dec = experts[actor].decide(actor, ag, env.ws, env.resources,
                                        env.ws.combat.round_number)
            act = ([0, 0, 0] if (dec.action is None or dec.fled)
                   else list(encode_action(dec.action, env.ws, actor)))
            if act[0] < 0:
                act = [0, 0, 0]
        else:
            net = ARCH_TO_NET[arch_of[actor]]
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0],
                                   ws=env.ws, agent_id=actor))
        if first_step_of_turn:
            st["turns"] += 1
            if act[0] == 0 and had_action:
                st["idle_turns"] += 1   # gave the turn away with action unspent
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc

    st["games"] += 1
    st["trunc"] += int(trunc)
    st["rounds"] += env.ws.combat.round_number
    left = sum(max(0, env.ws.characters[o].hp) for o in env.opp_ids)
    st["dealt"] += 1.0 - left / max(1, opp_pool)
    opps_dead = all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
    alive = any(env.ws.characters[a].is_alive() for a in env.agent_ids)
    st["wins"] += int(opps_dead and alive)


def fmt(st) -> str:
    n = max(1, st["games"])
    return (f"WR={st['wins']/n:5.1%} dealt={st['dealt']/n:5.1%} "
            f"idle={st['idle_turns']/max(1, st['turns']):5.1%} "
            f"rounds={st['rounds']/n:4.1f} trunc={st['trunc']/n:4.0%}")


def run_bucket(tag, episodes) -> None:
    """episodes: list of (agent_comp, opp_comp, nA, nO, lvlA, lvlO, seed)."""
    sts = {d: defaultdict(float) for d in ("model", "expert")}
    for ep in episodes:
        for d in ("model", "expert"):
            play(d, *ep, sts[d])
    m, e = sts["model"], sts["expert"]
    n = m["games"]
    wm, we = m["wins"]/n, e["wins"]/n
    sig2 = 2 * ((wm*(1-wm) + we*(1-we)) / n) ** 0.5
    print(f"  {tag:8s} model[{fmt(m)}]  expert[{fmt(e)}]  "
          f"ΔWR={wm-we:+5.1%}(±{sig2:.1%})", flush=True)


print(f"\n=== A. 1v1 mirror, agent L5 vs opp L5+ΔL — 12 archs × {games_1v1}g/bucket ===")
for dl in DELTAS:
    if not want(f"A{dl:+d}"):
        continue
    eps = []
    for arch in ARCHETYPE_LIST:
        base = stable_seed(f"asymA_{arch}_{dl}")
        for i in range(games_1v1):
            eps.append(([arch], [arch], 1, 1, 5, 5 + dl, base + i))
    run_bucket(f"ΔL={dl:+d}", eps)

print(f"\n=== B. 3v3 mirror comp, agent L5 vs opp L5+ΔL — 12 comps × {games_3v3}g/bucket ===")
for dl in DELTAS:
    if not want(f"B{dl:+d}"):
        continue
    eps = []
    for comp in SPREAD_COMPS:
        base = stable_seed(f"asymB_{'+'.join(comp)}_{dl}")
        for i in range(games_3v3):
            eps.append((comp, comp, 3, 3, 5, 5 + dl, base + i))
    run_bucket(f"ΔL={dl:+d}", eps)

print(f"\n=== C. number gap at L5 — 12 comps × {games_3v3}g/config (comps truncated front-first) ===")
for nA, nO in SIZE_CONFIGS:
    if not want(f"C{nA}v{nO}"):
        continue
    eps = []
    for comp in SPREAD_COMPS:
        base = stable_seed(f"asymC_{'+'.join(comp)}_{nA}v{nO}")
        for i in range(games_3v3):
            eps.append((comp[:nA], comp[:nO], nA, nO, 5, 5, base + i))
    run_bucket(f"{nA}v{nO}", eps)
