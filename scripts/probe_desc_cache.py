"""Scope-check: does an AFTER-reset injected fire-immunity actually reach the
v4 capability descriptor the model reads? capability_descriptor() caches on
char._cap_desc_cache keyed by (level, n_abilities, n_weapons) — none of which a
damage_multipliers injection changes. reset() builds an obs (env_v2.py:246),
which would populate the orc's cache with the NON-immune descriptor BEFORE the
fire-lab's _inject_and_obs runs. If so, the injected immunity never reaches the
obs and the whole fire-lab compared two identical observations.

Prints the enemy row's fire-resist descriptor cell under three conditions:
  A. plain orc (control)                          -> expect 0.0
  B. inject 火->0 AFTER reset, rebuild obs         -> 0.0 == cache bug confirmed
  C. inject 火->0 then BUST the cache, rebuild     -> expect -1.0 (true immune)
Also D: a REAL fire-immune monster (fire_elemental) read straight from reset.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.obs import (build_obs, ENT_DESC_START, ENEMY_SLOT_START,
                         N_STATUS_SLOTS, N_SAVE_STATS)
from trpg.engine.damage import DAMAGE_TYPES
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

FIRE_I = DAMAGE_TYPES.index("火")
FIRE_COL = ENT_DESC_START + 9 + N_STATUS_SLOTS + N_SAVE_STATS + FIRE_I


def fire_cell(obs):
    return float(obs["entities"][ENEMY_SLOT_START, FIRE_COL])


def main():
    register_monsters()
    print(f"FIRE_COL={FIRE_COL}  (DAMAGE_TYPES[{FIRE_I}]='火')\n")

    # A: plain orc
    env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=["evocation"], opp_archs=["orc"],
                       level=6, opp_level=MONSTER_DEFS["orc"].natural_level)
    print(f"A plain orc                       fire-cell = {fire_cell(obs):+.2f}")

    # B: inject AFTER reset, rebuild (exactly what fire-lab does)
    oid = env.opp_ids[0]
    env.ws.characters[oid].damage_multipliers["火"] = 0.0
    obsB = build_obs(env.ws, env.current_agent_id, env.resources)
    print(f"B inject->rebuild (fire-lab path) fire-cell = {fire_cell(obsB):+.2f}  "
          f"<- 0.00 means the CACHE ate the injection")

    # C: inject, bust the cache, rebuild
    if hasattr(env.ws.characters[oid], "_cap_desc_cache"):
        del env.ws.characters[oid]._cap_desc_cache
    obsC = build_obs(env.ws, env.current_agent_id, env.resources)
    print(f"C inject + cache-bust + rebuild   fire-cell = {fire_cell(obsC):+.2f}  "
          f"<- -1.00 = true immune signal")

    # D: a REAL fire-immune monster straight from reset (no injection)
    env2 = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
    obsD, _ = env2.reset(agent_archs=["evocation"], opp_archs=["fire_elemental"],
                         level=8,
                         opp_level=MONSTER_DEFS["fire_elemental"].natural_level)
    print(f"D real fire_elemental @reset      fire-cell = {fire_cell(obsD):+.2f}  "
          f"<- real monsters get it right (immunity present at creation)")


if __name__ == "__main__":
    main()
