"""Diagnostic: can CORRECT play (magic_missile, 力場) actually WIN vs a fire-
immune enemy? If not, the fire-lab's immune half always loses regardless of the
spell chosen, the dense-reward edge of magic_missile over fireball is swamped by
the terminal loss, and the 50/50 mix just converges to "always fireball" — a
REGIME weakness, not proof the model can't learn.

Scripts the agent to cast a fixed skill at the enemy every turn via the real RL
action encoding (env auto-runs the opponent). Reports WR across agent levels.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
from trpg.rl.env_v2 import CombatEnvV2
from trpg.engine.skill import available_skills
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import stable_seed
from trpg.rl.action import _entity_id_at_slot


def _enemy_slot(ws, aid, oid):
    for idx in range(10):
        if _entity_id_at_slot(ws, aid, idx) == oid:
            return idx
    return 4


def scripted_wr(enemy, level, opp_level, forced_sid, seed, inject_immune=False):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=["evocation"], opp_archs=[enemy],
              level=level, opp_level=opp_level)
    aid = env.agent_ids[0]; oid = env.opp_ids[0]; ws = env.ws
    if inject_immune:
        ws.characters[oid].damage_multipliers["火"] = 0.0
    done = False
    while not done:
        a = ws.characters[aid]
        sks = available_skills(a, ws)
        si = next((i for i, s in enumerate(sks) if s.skill_id == forced_sid), None)
        eslot = _enemy_slot(ws, aid, oid)
        if si is not None:
            action = [si, eslot, 0]
        else:                                    # skill gone (no slots) -> end
            action = [len(sks), eslot, 0]
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return ws.characters[oid].is_dead() and ws.characters[aid].is_alive()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--levels", type=int, nargs="+", default=[8, 12, 16, 20])
    p.add_argument("--enemies", nargs="+",
                   default=["fire_elemental", "young_red_dragon"])
    p.add_argument("--inject_immune", action="store_true",
                   help="inject 火→0 onto the (normally non-immune) enemy")
    args = p.parse_args()
    register_monsters()
    inj = args.inject_immune
    for enemy in args.enemies:
        olvl = MONSTER_DEFS[enemy].natural_level
        tag = " +injected-fire-immune" if inj else ""
        print(f"\n=== {enemy} (natL{olvl}){tag} ===")
        print(f"{'agent_L':>7s} {'always-fireball':>15s} {'always-magic_missile':>21s}")
        for lvl in args.levels:
            wf = wm = 0
            for g in range(args.games):
                s = stable_seed(f"win_{enemy}_{lvl}_{g}")
                wf += int(scripted_wr(enemy, lvl, olvl, "fireball_ev", s, inj))
                wm += int(scripted_wr(enemy, lvl, olvl, "magic_missile", s, inj))
            print(f"{lvl:7d} {wf/args.games:15.0%} {wm/args.games:21.0%}")


if __name__ == "__main__":
    main()
