"""Team-cooperation trace: does the routed model team actually cooperate?

Plays full model teams (MMM) and full expert teams (EEE) against the same
expert opponent comps/seeds and measures cooperation markers per driver:

  heals       — heal events (any team member's HP rising during a member's
                step): on ALLY vs on SELF, and the target's HP fraction at
                cast time (healing the wounded = cooperation)
  focus fire  — fraction of rounds where >=2 distinct team members damaged
                the SAME enemy; avg distinct enemies damaged per round
  friendly fx — ally damage attributed from the engine's own AoE cast result
                (target_results), NOT from HP diffs: hostile-aura/terrain
                ticks of the NEXT character fire inside the previous actor's
                env.step and used to pollute this number (the old 64.7/game
                was ticks, not wizard AoE). Ticks are reported separately.
  positioning — avg distance to nearest enemy per role slot (front should be
                close, striker/support behind)

Usage: python scripts/trace_coop.py [games] [n_opp_comps] [comp ...]
       default comps: a 12-comp spread (every front once, mixed strikers).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import itertools
import random
from collections import defaultdict
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import (apply_resource_mask, apply_entity_mask, pick_action)
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import ARCHETYPE_ROLES

from eval_routed import DEFAULT_ROUTING, load_net, stable_seed

games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_opp_comps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# Args with "=" are routing overrides (arch=checkpoint_path) — lets the
# behavior trace compare candidate checkpoints before they are routed.
comps_arg = []
for _a in sys.argv[3:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        DEFAULT_ROUTING[_k] = _v
    else:
        comps_arg.append(_a)

DEFAULT_COMPS = [
    "battle_master+evocation+life",
    "battle_master+assassin+war",
    "champion+divination+life",
    "champion+arcane_trickster+war",
    "totem_bear+evocation+war",
    "totem_bear+assassin+life",
    "berserker+divination+war",
    "berserker+arcane_trickster+life",
    "devotion+evocation+life",
    "devotion+assassin+war",
    "vengeance+divination+life",
    "vengeance+arcane_trickster+war",
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


def nearest_enemy_dist(env, agent):
    best = None
    for oid in env.opp_ids:
        o = env.ws.characters.get(oid)
        if o is not None and o.is_alive():
            d = agent.position.distance_to(o.position)
            if best is None or d < best:
                best = d
    return best


def play(driver: str, agent_comp, opp_comp, seed, st) -> None:
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    # Unique display names so SPELL target_results map back to char ids
    # (factory default names can collide across the two teams).
    for cid, c in env.ws.characters.items():
        c.name = f"{c.name}#{cid}"
    name2cid = {c.name: cid for cid, c in env.ws.characters.items()}
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    slot_of = {aid: i for i, aid in enumerate(env.agent_ids)}
    experts = ({aid: make_archetype_policy(a) for aid, a in arch_of.items()}
               if driver == "expert" else None)
    # (round, enemy_id) -> set of team actors who damaged it
    dmg_by: dict[tuple, set] = defaultdict(set)
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        rnd = env.ws.combat.round_number
        if driver == "expert":
            dec = experts[actor].decide(actor, ag, env.ws, env.resources, rnd)
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

        slot = slot_of[actor]

        # positioning sample (alive actors only, at decision time)
        d = nearest_enemy_dist(env, ag)
        if d is not None and ag.is_alive():
            st[f"dist_sum_{slot}"] += d
            st[f"dist_n_{slot}"] += 1

        hp_before = {cid: c.hp for cid, c in env.ws.characters.items()}
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
        res = (info or {}).get("action_result") or {}

        # heal accounting: any team member whose HP rose during this actor's
        # step (covers every heal ability without naming them)
        for cid in env.agent_ids:
            if env.ws.characters[cid].hp > hp_before[cid]:
                st["heal_casts"] += 1
                st["heal_on_ally" if cid != actor else "heal_on_self"] += 1
                frac = hp_before[cid] / max(1, env.ws.characters[cid].max_hp)
                st["heal_target_hpfrac_sum"] += frac
        # damage to enemies during this actor's step -> focus-fire bookkeeping
        for oid in env.opp_ids:
            if env.ws.characters[oid].hp < hp_before[oid]:
                dmg_by[(rnd, oid)].add(actor)
        # friendly fire: ONLY damage the engine attributes to this actor's own
        # AoE cast (target_results). HP-diff-based counting also caught the
        # NEXT character's hostile-aura/terrain ticks (they fire inside this
        # step when the turn advances) — that inflated FF to ~64/game for
        # BOTH drivers. Ticks are tallied separately below.
        aoe_ally = 0.0
        if res.get("type") == "SPELL":
            for tr in res.get("target_results", ()):
                cid = name2cid.get(tr.get("target_name"))
                dmg = float(tr.get("damage", 0) or 0)
                if cid in slot_of and dmg > 0:
                    aoe_ally += dmg
        st["friendly_dmg"] += aoe_ally
        same_drop = sum(hp_before[cid] - env.ws.characters[cid].hp
                        for cid in env.agent_ids
                        if env.ws.characters[cid].hp < hp_before[cid])
        st["tick_dmg"] += max(0.0, same_drop - aoe_ally)

    rounds = defaultdict(set)      # round -> enemies damaged
    for (rnd, oid), actors in dmg_by.items():
        rounds[rnd].add(oid)
        if len(actors) >= 2:
            st["ff_rounds"] += 1   # >=2 distinct members damaged same enemy
    st["atk_rounds"] += len(rounds)
    st["targets_per_round_sum"] += sum(len(v) for v in rounds.values())

    st["games"] += 1
    opps_dead = all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
    team_alive = any(env.ws.characters[aid].is_alive() for aid in env.agent_ids)
    st["wins"] += int(opps_dead and team_alive)


for driver in ("model", "expert"):
    st = defaultdict(float)
    for comp in COMPS:
        for opp_comp in OPP_SET:
            base = stable_seed(f"coop_{'+'.join(comp)}_{'+'.join(opp_comp)}")
            for g in range(games):
                play(driver, comp, opp_comp, base + g, st)
    n = st["games"]
    heals = max(1, st["heal_casts"])
    print(f"\n=== {driver} team (eps={n:.0f}, WR={st['wins']/n:.0%}) ===")
    print(f"  heals/game={st['heal_casts']/n:4.2f}  on-ally={st['heal_on_ally']/heals:4.0%} "
          f"on-self={st['heal_on_self']/heals:4.0%}  "
          f"target hp% at cast={st['heal_target_hpfrac_sum']/heals:4.0%}")
    print(f"  focus-fire rounds (>=2 members hit same enemy): "
          f"{st['ff_rounds']/max(1,st['atk_rounds']):4.0%}   "
          f"distinct enemies damaged/round={st['targets_per_round_sum']/max(1,st['atk_rounds']):3.1f}")
    print(f"  AoE friendly-fire dmg/game={st['friendly_dmg']/n:4.1f}   "
          f"(non-attributed tick dmg/game={st['tick_dmg']/n:5.1f})")
    print(f"  avg dist to nearest enemy: front={st['dist_sum_0']/max(1,st['dist_n_0']):4.1f}m  "
          f"striker={st['dist_sum_1']/max(1,st['dist_n_1']):4.1f}m  "
          f"support={st['dist_sum_2']/max(1,st['dist_n_2']):4.1f}m")
