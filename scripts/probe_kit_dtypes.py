"""Inventory: which archetypes have >=2 distinct DAMAGE TYPES in their kit?

The typed-resist switch behavior is only expressible for agents that have an
alternative damage type to switch TO (info-gain law, MONSTER_CATALOG §6.6).
This probe enumerates each standard archetype's available damaging skills at
several levels and prints the damage-type set — discovered from engine data
(build_action + action_damage_type), no hardcoded skill/class lists. The seed
pipeline imports `agent_damage_types` / `multi_dtype_archetypes` from here.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_fire_switch import action_damage_type


def agent_damage_types(ws, aid, oid) -> dict[str, list[str]]:
    """{damage_type: [skill_ids]} over the agent's currently-available skills.

    Since 12i the damage type comes from the MATERIALIZED features
    (damage_types soft one-hot — the same data the obs dtype tail and the
    oracle EV read; primary = largest share). Buildability vs the live
    opponent still gates entry. This also classifies rider-typed rows
    correctly: divine_smite (2d8 光耀 rider on a 斬擊 swing) counts as 光耀,
    so paladins join the multi-dtype pool.
    """
    a = ws.characters[aid]
    o = ws.characters[oid]
    out: dict[str, list[str]] = defaultdict(list)
    for sk in available_skills(a, ws):
        if sk.features.expected_damage <= 0:
            continue
        pairs = list(sk.features.iter_damage_types())
        if not pairs:
            continue
        try:
            act = sk.build_action(aid, oid, (o.position.x, o.position.y))
        except Exception:
            continue
        if act is None:
            continue
        dt = max(pairs, key=lambda p: p[1])[0]
        out[dt].append(sk.skill_id)
    return dict(out)


def multi_dtype_archetypes(level: int) -> dict[str, dict[str, list[str]]]:
    """Archetypes with >=2 distinct damage types at `level` (fresh env each)."""
    out = {}
    for arch in STANDARD_ARCHETYPES:
        env = CombatEnvV2(seed=7, n_agents=1, n_opps=1)
        env.reset(agent_archs=[arch], opp_archs=[arch], level=level)
        dts = agent_damage_types(env.ws, env.agent_ids[0], env.opp_ids[0])
        if len(dts) >= 2:
            out[arch] = dts
    return out


def main():
    for level in (5, 6, 8):
        print(f"=== level {level} ===")
        for arch in STANDARD_ARCHETYPES:
            env = CombatEnvV2(seed=7, n_agents=1, n_opps=1)
            env.reset(agent_archs=[arch], opp_archs=[arch], level=level)
            dts = agent_damage_types(env.ws, env.agent_ids[0], env.opp_ids[0])
            tag = " <== multi" if len(dts) >= 2 else ""
            parts = "; ".join(f"{dt}:{','.join(ids)}" for dt, ids in dts.items())
            print(f"  {arch:16s} {len(dts)} types  {parts}{tag}")
        print()


if __name__ == "__main__":
    main()
