"""1vN Boss 對局標定 (MONSTER_CATALOG.md Wave 3).

The CR≥4 monsters have been "1vN party content" by measurement since Wave 0
(equiv inf) — this script finally MEASURES that band: a scripted 3-expert
party (front/heal/nuke = battle_master + life + evocation) fights each 1vN
monster at party levels 1..8, monster at natural_level, via the env's 3v1
team config. Legendary actions / frightful presence / legendary resistance
all run live through the normal driver wiring.

隊伍等效等級 = the party level where WR crosses 50% (linear interpolation).
Bosses with no crossing through 3×L8 are flagged >8 — content for either
larger parties or higher-level registries (engine classes cap at L8).

Seeds are crc32-stable. Engine dice are global RNG — only compare numbers
within one run.

Usage: python scripts/calibrate_boss.py [games_per_level] [monster_id ...]
       (no ids = every monster whose 1v1 equiv is inf, by CR)
"""
from __future__ import annotations
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from zlib import crc32

from trpg.scenarios.monsters import (
    MONSTER_DEFS, EQUIV_LEVEL_1V1, register_monsters,
)

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy

PARTY = ("battle_master", "life", "evocation")   # front / heal / nuke
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ONLY = set(sys.argv[2:])
if ONLY:
    unknown = ONLY - set(MONSTER_DEFS)
    assert not unknown, f"unknown monster ids: {unknown}"
LEVELS = list(range(1, 9))


def play(lvl: int, mid: str, seed: int) -> tuple[bool, int]:
    """One 3v1 episode. Returns (party_won, legendary_actions_spent_proxy)."""
    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=1)
    env.reset(agent_archs=list(PARTY), opp_archs=[mid],
              level=lvl, opp_level=MONSTER_DEFS[mid].natural_level)
    experts = {aid: make_archetype_policy(arch)
               for aid, arch in zip(env.agent_ids, env.agent_archs)}
    boss_id = env.opp_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        a = env.ws.characters[actor]
        dec = experts[actor].decide(actor, a, env.ws, env.resources,
                                    env.ws.combat.round_number)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        _, _, term, trunc, _ = env.step(act)
        done = term or trunc
    boss = env.ws.characters[boss_id]
    won = (not boss.is_alive()
           and any(env.ws.characters[a].is_alive() for a in env.agent_ids))
    return won, 0


def equiv_level(wrs: dict[int, float]) -> str:
    lvls = sorted(wrs)
    if wrs[lvls[0]] >= 0.5:
        return "<1" if wrs[lvls[0]] > 0.5 else "1.0"
    prev = lvls[0]
    for l in lvls[1:]:
        if wrs[l] >= 0.5:
            lo, hi = wrs[prev], wrs[l]
            frac = (0.5 - lo) / (hi - lo) if hi > lo else 1.0
            return f"{prev + frac * (l - prev):.1f}"
        prev = l
    return ">8"


def main():
    targets = sorted(
        (m for m in MONSTER_DEFS
         if (not ONLY or m in ONLY)
         and (ONLY or math.isinf(EQUIV_LEVEL_1V1[m]))),
        key=lambda m: MONSTER_DEFS[m].cr)
    print(f"1vN boss calibration — party {'+'.join(PARTY)} × {GAMES} "
          f"games/level, monster at natural_level")
    print(f"{'monster':<20} {'CR':>5} {'natL':>4} " +
          " ".join(f"L{l:<4}" for l in LEVELS) + f" {'partyL':>7}")
    print("-" * 100)
    summary = []
    for mid in targets:
        md = MONSTER_DEFS[mid]
        wrs: dict[int, float] = {}
        cells = []
        for lvl in LEVELS:
            base = crc32(f"boss_{mid}_{lvl}".encode()) & 0xFFFFFF
            w = sum(int(play(lvl, mid, base + i)[0]) for i in range(GAMES))
            wrs[lvl] = w / GAMES
            cells.append(f"{wrs[lvl]:4.0%}")
            if wrs[lvl] >= 0.75 and any(v >= 0.5 for v in wrs.values()):
                break
        eq = equiv_level(wrs)
        cells += ["  - "] * (len(LEVELS) - len(cells))
        print(f"{mid:<20} {md.cr:>5.2f} {md.natural_level:>4} " +
              " ".join(cells) + f" {eq:>7}", flush=True)
        summary.append((mid, md.cr, eq))
    print("-" * 100)
    print("CR→隊伍(3人)等效等級（寫回 MONSTER_CATALOG §6.8）：")
    for mid, cr, eq in summary:
        print(f"  CR {cr:>5.2f}  {mid:<20} → 隊伍 L{eq}")


if __name__ == "__main__":
    main()
