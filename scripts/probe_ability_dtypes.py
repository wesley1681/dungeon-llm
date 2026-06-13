"""Enumerate the ENGINE damage type of every reachable skill (data-entry audit).

For the skill-dtype schema surgery (MONSTER_CATALOG §6.6, 2026-06-12i) every
SkillFeatures template needs a `damage_type`. This probe derives the ground
truth the same way the oracle does — build each skill's action in a live env
and read the damage data off the action dict / weapon / spell def — across
all standard archetypes AND all monsters at several levels, then prints:

    skill_id -> {observed damage types} | action type | expected_damage

Skills whose features claim expected_damage > 0 but yield NO dtype are
flagged MISSING (a schema hole), and skills observed with >1 dtype across
characters are flagged @WEAPON (weapon-riding — dtype must resolve from the
wielder's weapon at materialize time, not from a static template).

Usage: python scripts/probe_ability_dtypes.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from collections import defaultdict

from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.skill import available_skills
from trpg.scenarios.archetypes import STANDARD_ARCHETYPES
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_fire_switch import action_damage_type

LEVELS = (1, 3, 5, 8, 11, 17, 20)

seen: dict[str, set[str]] = defaultdict(set)          # skill_id -> dtypes
act_types: dict[str, set[str]] = defaultdict(set)     # skill_id -> action types
exp_dmg: dict[str, float] = defaultdict(float)        # skill_id -> max expected_damage
act_shapes: dict[str, str] = {}                       # skill_id -> sample action keys
carriers: dict[str, set[str]] = defaultdict(set)      # skill_id -> who has it


def scan_char(ws, aid, oid, who):
    a = ws.characters[aid]
    o = ws.characters[oid]
    for sk in available_skills(a, ws):
        carriers[sk.skill_id].add(who)
        exp_dmg[sk.skill_id] = max(exp_dmg[sk.skill_id], sk.features.expected_damage)
        try:
            act = sk.build_action(aid, oid, (o.position.x, o.position.y))
        except Exception:
            act = None
        if act is None:
            continue
        act_types[sk.skill_id].add(act.get("type", "?"))
        if sk.skill_id not in act_shapes:
            act_shapes[sk.skill_id] = ",".join(sorted(act.keys()))
        dt = action_damage_type(act, a)
        if dt:
            seen[sk.skill_id].add(dt)


def main():
    register_monsters()
    pools = list(STANDARD_ARCHETYPES) + sorted(MONSTER_DEFS.keys())
    for arch in pools:
        is_mon = arch in MONSTER_DEFS
        levels = (MONSTER_DEFS[arch].natural_level,) if is_mon else LEVELS
        for lvl in levels:
            try:
                env = CombatEnvV2(seed=11, n_agents=1, n_opps=1)
                env.reset(agent_archs=[arch], opp_archs=["orc"],
                          level=lvl, opp_level=3)
            except Exception as e:
                print(f"  !! reset failed {arch} L{lvl}: {e}")
                continue
            scan_char(env.ws, env.agent_ids[0], env.opp_ids[0], arch)

    print(f"{'skill_id':32s} {'dtypes':14s} {'maxED':>6s} {'action':14s} flag")
    print("-" * 96)
    for sid in sorted(set(carriers) | set(seen)):
        dts = seen.get(sid, set())
        ed = exp_dmg[sid]
        at = "/".join(sorted(act_types.get(sid, ()))) or "-"
        flag = ""
        if ed > 0 and not dts:
            flag = "MISSING"
        elif len(dts) > 1:
            flag = "@WEAPON?"
        elif ed == 0 and dts:
            flag = "dtype-but-no-ED"
        print(f"{sid:32s} {'/'.join(sorted(dts)) or '-':14s} {ed:6.1f} {at:14s} {flag}")
    print()
    print("=== sample action shapes (damaging skills) ===")
    for sid in sorted(seen):
        print(f"  {sid:32s} [{act_shapes.get(sid, '')}]")


if __name__ == "__main__":
    main()
