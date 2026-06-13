"""Has the trained B-net learned to READ the enemy descriptor in the one case
where it is actionable? — fire-caster vs fire-immune.

evocation's best option is fireball (火, ~28 base); its non-fire fallback is
magic_missile (力場). Against a fire-IMMUNE enemy (fire_elemental,
young_red_dragon) fireball deals 0 — the descriptor's typed_resist[火]=immune is
the ONLY signal that should make the policy switch to magic_missile.

We run the net as evocation, greedy, blinded self one-hot, and tally the damage
type of every damaging action under two arms:
  desc-on  : normal v4 obs
  desc-off : enemy descriptor zeroed
If the net reads the immunity, desc-on shows a LOWER fire share than desc-off
against fire-immune enemies (and ~equal against a non-immune control).

Damage type is read from the built action dict (weapon / spell / auto-damage) —
no hardcoded skill-name lists.

Usage: python scripts/probe_fire_switch.py models/pop_mon/pop_u0005.pt --games 30
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
from collections import Counter
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.spells import SPELLS
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from eval_routed import stable_seed
from probe_descriptor_ab import blind_self_one_hot, zero_enemy_descriptor


def action_damage_type(act, actor):
    """Return the damage type of an action dict, or None if non-damaging."""
    if not act:
        return None
    t = act.get("type")
    if t in ("ATTACK", "MULTI_ATTACK"):
        w = actor.get_weapon(act.get("weapon", ""))
        return getattr(w, "damage_type", None) if w else None
    if t == "AUTO_DAMAGE":
        return act.get("damage_type")
    if t == "SPELL":
        sp = SPELLS.get(act.get("spell_name", ""))
        return sp.damage_type if (sp and sp.damage_dice) else None
    return None


def run(net, enemy, level, opp_level, games, kill_desc):
    types = Counter()
    n_dmg = 0
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"fire_{enemy}_{gi}"),
                          n_agents=1, n_opps=1)
        obs, _ = env.reset(agent_archs=["evocation"], opp_archs=[enemy],
                           level=level, opp_level=opp_level)
        aids = set(env.agent_ids)
        done = False
        while not done:
            actor_id = env.current_agent_id
            ob = blind_self_one_hot(obs)
            if kill_desc:
                ob = zero_enemy_descriptor(ob)
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor_id)
            e = apply_entity_mask(e, ot, env.ws, actor_id)
            act_idx = pick_action(el[0], s[0], e[0], g[0],
                                  ws=env.ws, agent_id=actor_id)
            if actor_id in aids:
                actor = env.ws.characters[actor_id]
                sks = available_skills(actor, env.ws)
                si = act_idx[0]
                if 0 < si < len(sks):
                    sk = sks[si]
                    oid = env.opp_ids[0]
                    o = env.ws.characters[oid]
                    built = sk.build_action(actor_id, oid,
                                            (o.position.x, o.position.y))
                    dt = action_damage_type(built, actor)
                    if dt:
                        types[dt] += 1
                        n_dmg += 1
            obs, _, term, trunc, _ = env.step(list(act_idx))
            done = term or trunc
    return types, n_dmg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=30)
    args = p.parse_args()
    register_monsters()
    net = load_student(args.ckpt); net.eval()

    # (enemy, agent_level, note). fire_elemental/dragon = immune; ogre = control.
    cases = [
        ("fire_elemental", 8, "火免疫"),
        ("young_red_dragon", 10, "火免疫"),
        ("ogre", 6, "控制組(火有效)"),
    ]
    print(f"ckpt={args.ckpt}  games={args.games}  agent=evocation\n")
    print(f"{'enemy':17s} {'arm':8s} {'火share':>7s} {'力場share':>9s} "
          f"{'dmg-acts':>8s}   note")
    for enemy, lvl, note in cases:
        olvl = MONSTER_DEFS[enemy].natural_level
        for kill in (False, True):
            types, n = run(net, enemy, lvl, olvl, args.games, kill)
            fire = types.get("火", 0) / max(1, n)
            force = types.get("力場", 0) / max(1, n)
            arm = "desc-off" if kill else "desc-on"
            print(f"{enemy:17s} {arm:8s} {fire:7.0%} {force:9.0%} "
                  f"{n:8d}   {note}")
        print()


if __name__ == "__main__":
    main()
