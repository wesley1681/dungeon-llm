"""Behavior trace of the FRONT slot in 3v3: model front vs expert front,
with IDENTICAL model teammates (striker+support) on the same comps/seeds.

Stats collected for the front agent only:
  - deaths, died_first (front died before both teammates)
  - dmg dealt (weapon) / taken
  - sub-actions: move/melee/ranged/surge/heal/other
  - moves categorized vs nearest enemy: closing / away / lateral
  - target focus: distinct enemies weapon-attacked per game,
    attacks on the NEAREST living enemy vs others
  - avg distance to nearest enemy at decision time

Usage: python scripts/trace_team.py [games] [n_opp_comps] [comp ...]
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
from trpg.engine.skill import available_skills
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


def nearest_enemy(env, agent):
    best_id, best_d = None, None
    for oid in env.opp_ids:
        o = env.ws.characters.get(oid)
        if o is not None and o.is_alive():
            d = agent.position.distance_to(o.position)
            if best_d is None or d < best_d:
                best_id, best_d = oid, d
    return best_id, best_d


def play(front_driver: str, agent_comp, opp_comp, seed, st) -> None:
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)
    obs, _ = env.reset(agent_archs=agent_comp, opp_archs=opp_comp, level=5)
    arch_of = dict(zip(env.agent_ids, env.agent_archs))
    front_id = env.agent_ids[0]
    mates = [a for a in env.agent_ids if a != front_id]
    experts = {aid: make_archetype_policy(a) for aid, a in arch_of.items()}
    targets_hit: set = set()
    front_died_at: int | None = None
    mate_dead_when_front_died = 0
    done = False
    while not done:
        actor = env.current_agent_id
        ag = env.ws.characters[actor]
        use_expert = (front_driver == "expert" and actor == front_id)
        if use_expert:
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

        is_front = (actor == front_id)
        sid = None
        target_slot_id = None
        if is_front:
            skills = available_skills(ag, env.ws)
            if 0 <= act[0] < len(skills):
                sid = skills[act[0]].skill_id
            ne_id, ne_d = nearest_enemy(env, ag)
            if ne_d is not None:
                st["dist_sum"] += ne_d
                st["dist_n"] += 1
            if sid and sid.startswith("weapon:"):
                # which enemy slot the entity head picked
                from trpg.rl.obs import partition_entities
                _, enemy_ids = partition_entities(env.ws, actor)
                slot = act[1] - 3
                if 0 <= slot < len(enemy_ids):
                    target_slot_id = enemy_ids[slot]
                elif use_expert:
                    target_slot_id = None  # resolved below from hp delta
        hp_before = {oid: env.ws.characters[oid].hp for oid in env.opp_ids}
        front_hp_before = env.ws.characters[front_id].hp
        front_pos_before = env.ws.characters[front_id].position
        ne_before = nearest_enemy(env, env.ws.characters[front_id])[1] \
            if env.ws.characters[front_id].is_alive() else None

        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc

        if is_front and sid:
            if sid == "move":
                st["c_move"] += 1
                ne_after = nearest_enemy(env, env.ws.characters[front_id])[1]
                if ne_before is not None and ne_after is not None:
                    dd = ne_after - ne_before
                    if dd < -0.5:
                        st["move_closing"] += 1
                    elif dd > 0.5:
                        st["move_away"] += 1
                    else:
                        st["move_lateral"] += 1
            elif sid.startswith("weapon:"):
                st["c_attack"] += 1
                dealt = sum(max(0, hp_before[oid] - env.ws.characters[oid].hp)
                            for oid in env.opp_ids)
                st["dmg_dealt"] += dealt
                # attribute target: explicit slot if model, else hp-delta
                tgt = target_slot_id
                if tgt is None:
                    deltas = [(max(0, hp_before[oid] - env.ws.characters[oid].hp), oid)
                              for oid in env.opp_ids]
                    deltas.sort(reverse=True)
                    tgt = deltas[0][1] if deltas[0][0] > 0 else None
                if tgt:
                    targets_hit.add(tgt)
                    ne_id, _ = nearest_enemy(env, env.ws.characters[front_id])
                    st["atk_nearest" if tgt == ne_id else "atk_other"] += 1
            elif sid in ("second_wind", "cure_wounds", "lay_on_hands_ability"):
                st["c_heal"] += 1
            elif sid == "action_surge":
                st["c_surge"] += 1
            elif sid != "end":
                st["c_other"] += 1

        # front damage taken (any step)
        fc = env.ws.characters[front_id]
        st["dmg_taken"] += max(0, front_hp_before - fc.hp)
        if front_died_at is None and not fc.is_alive():
            front_died_at = env.ws.combat.round_number
            mate_dead_when_front_died = sum(
                1 for m in mates if not env.ws.characters[m].is_alive())

    st["games"] += 1
    opps_dead = all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
    team_alive = any(env.ws.characters[aid].is_alive() for aid in env.agent_ids)
    st["wins"] += int(opps_dead and team_alive)
    if front_died_at is not None:
        st["front_deaths"] += 1
        if mate_dead_when_front_died == 0:
            st["front_died_first"] += 1
    st["targets_per_game"] += len(targets_hit)


for front_driver in ("model", "expert"):
    st = dict(games=0, wins=0, front_deaths=0, front_died_first=0,
              dmg_dealt=0.0, dmg_taken=0.0, dist_sum=0.0, dist_n=0,
              c_move=0, c_attack=0, c_heal=0, c_surge=0, c_other=0,
              move_closing=0, move_away=0, move_lateral=0,
              atk_nearest=0, atk_other=0, targets_per_game=0)
    for comp in COMPS:
        for opp_comp in OPP_SET:
            base = stable_seed(f"{'+'.join(comp)}_vs_{'+'.join(opp_comp)}")
            for g in range(games):
                play(front_driver, comp, opp_comp, base + g, st)
    n = st["games"]
    atk_tot = max(1, st["atk_nearest"] + st["atk_other"])
    print(f"\nfront={front_driver:6s} (teammates always model)  "
          f"WR={st['wins']/n:4.0%}  eps={n}")
    print(f"  front: deaths={st['front_deaths']/n:4.0%}  "
          f"died-first={st['front_died_first']/n:4.0%}  "
          f"dmg_dealt={st['dmg_dealt']/n:5.1f}  dmg_taken={st['dmg_taken']/n:5.1f}")
    print(f"  per-game: move={st['c_move']/n:4.1f} attack={st['c_attack']/n:4.1f} "
          f"surge={st['c_surge']/n:4.2f} heal={st['c_heal']/n:4.2f} "
          f"other={st['c_other']/n:4.1f}")
    print(f"  moves: closing={st['move_closing']/n:4.2f} "
          f"away={st['move_away']/n:4.2f} lateral={st['move_lateral']/n:4.2f}   "
          f"avg_dist_to_nearest={st['dist_sum']/max(1,st['dist_n']):4.1f}m")
    print(f"  attacks on nearest enemy: {st['atk_nearest']/atk_tot:4.0%}   "
          f"distinct targets/game: {st['targets_per_game']/n:3.1f}")
