"""Does the enemy typed_resist descriptor carry ACTIONABLE info? — direct test.

A life/war cleric has two damage options of DIFFERENT types:
  長劍 (斬擊/slashing)  and  神聖光輝 sacred_flame (光耀/radiant).
Monsters' typed_resist (the v4 descriptor's 13-dim block) flips which is best:
  shadow:   斬擊 ×0.5 (resisted),  光耀 ×2 (vulnerable)  -> radiant wins 4×
  commoner: neutral                                      -> slashing wins (higher base)

We fire each option at each monster MANY times through the real engine (to-hit,
saves, damage multipliers all applied) and measure mean HP removed. Then:
  blind  = always the highest-BASE-damage option (ignores resist) = 長劍
  oracle = the option with the highest ACTUAL damage vs THIS enemy (reads resist)
  info_gain = oracle_dmg - blind_dmg   (per swing; >0 means the descriptor pays)

No model, no training — this is the information-value ceiling. If it is large,
the descriptor info is real and actionable; whether a given policy captures it
is a separate (learning) question.

Usage:  python scripts/probe_resist_choice.py --agent life --trials 400
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import numpy as np

from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                     onev1_viable_monsters)
from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.combat import execute_action
from trpg.engine.skill import available_skills


def _find_skill(ws, aid, predicate):
    for s in available_skills(ws.characters[aid], ws):
        if predicate(s.skill_id):
            return s
    return None


def _place_adjacent(ws, aid, oid):
    """Put the agent 1.2 m from the opponent so melee is in range."""
    a = ws.characters[aid]; o = ws.characters[oid]
    op = o.position
    # Position supports .x/.y; nudge agent next to the opponent.
    try:
        a.position = type(op)(op.x + 1.2, op.y)
    except Exception:
        pass


def mean_damage(env_factory, aid, oid, skill_pred, trials):
    """Re-execute one skill `trials` times against a full-HP opponent; return
    mean HP removed (>=0). Fresh world each trial so action economy/positions
    are clean."""
    dmgs = []
    for _ in range(trials):
        ws = env_factory()
        _place_adjacent(ws, aid, oid)
        sk = _find_skill(ws, aid, skill_pred)
        if sk is None:
            return None, "skill unavailable"
        o = ws.characters[oid]
        o.hp = o.max_hp
        before = o.hp
        # pass the enemy position as target_coord so AoE spells (fireball)
        # actually center on the target instead of defaulting to [0,0].
        act = sk.build_action(aid, oid, (o.position.x, o.position.y))
        if act is None:
            return None, "build_action None"
        res = execute_action(act, ws)
        if res.get("type") == "ERROR":
            return None, res.get("message", "ERR")
        dmgs.append(max(0, before - o.hp))
    return float(np.mean(dmgs)), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="life")
    p.add_argument("--trials", type=int, default=300)
    # two damage options by skill_id; "weapon" = the basic weapon attack
    p.add_argument("--opt_a", default="weapon", help="skill_id or 'weapon'")
    p.add_argument("--opt_b", default="sacred_flame")
    p.add_argument("--type_a", default="斬擊"); p.add_argument("--type_b", default="光耀")
    args = p.parse_args()
    register_monsters()

    def pred(sid_arg):
        if sid_arg == "weapon":
            return lambda sid: sid.startswith("weapon:")
        return lambda sid: sid == sid_arg
    pa, pb = pred(args.opt_a), pred(args.opt_b)

    pool = list(MONSTER_DEFS)   # ALL monsters — pure damage probe, fairness N/A
    # (onev1_viable filters inf-marked 1vN brutes; here we WANT fire_elemental
    #  / young_red_dragon — the fire-immune cases that create the ranking flip)
    print(f"agent={args.agent}  trials={args.trials}  "
          f"A={args.opt_a}({args.type_a})  B={args.opt_b}({args.type_b})\n",
          flush=True)
    print(f"{'monster':16s} {'A.dmg':>6s} {'B.dmg':>6s} {'best':>5s} "
          f"{'regretA':>7s} {'regretB':>7s}  resist(A/B)")
    rA = rB = 0.0; a_best = b_best = 0
    for mon in pool:
        mlvl = MONSTER_DEFS[mon].natural_level

        def factory(mon=mon, mlvl=mlvl):
            env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
            env.reset(agent_archs=[args.agent], opp_archs=[mon],
                      level=6, opp_level=mlvl)
            return env.ws
        env0 = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
        env0.reset(agent_archs=[args.agent], opp_archs=[mon],
                   level=6, opp_level=mlvl)
        aid = env0.agent_ids[0]; oid = env0.opp_ids[0]
        omult = env0.ws.characters[oid].damage_multipliers or {}

        da, e1 = mean_damage(factory, aid, oid, pa, args.trials)
        db, e2 = mean_damage(factory, aid, oid, pb, args.trials)
        if da is None or db is None:
            print(f"{mon:16s} ERR a={e1} b={e2}")
            continue
        best = max(da, db)
        rA += best - da; rB += best - db
        who = "A" if da >= db else "B"
        if who == "A": a_best += 1
        else: b_best += 1
        rs = f"{args.type_a}×{omult.get(args.type_a,1)} {args.type_b}×{omult.get(args.type_b,1)}"
        print(f"{mon:16s} {da:6.1f} {db:6.1f} {who:>5s} "
              f"{best-da:7.1f} {best-db:7.1f}  {rs}")
    print("-" * 72)
    print(f"A best on {a_best} monsters, B best on {b_best} "
          f"(ranking FLIPS across enemies = descriptor actionable iff both >0)")
    print(f"regret of FIXED-A (always A) = {rA:.1f} HP total")
    print(f"regret of FIXED-B (always B) = {rB:.1f} HP total")
    print(f"==> descriptor info-gain over BEST fixed policy = "
          f"min(regretA,regretB) = {min(rA,rB):.1f} HP "
          f"(this is what reading the enemy descriptor actually saves)")


if __name__ == "__main__":
    main()
