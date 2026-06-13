"""CR→等效對位等級標定 (MONSTER_CATALOG.md Wave 0).

For each Wave 0 monster: the scripted standard-12 expert panel plays the
agent side (eval_goal protocol) at levels 1..8 vs the monster on the env
opponent side at its natural_level (driven by GenericMonsterPolicy).

等效等級 = the agent level where panel WR crosses 50% (linear interpolation
between the bracketing levels). Reported alongside a crude analytic power
score (HP × best-option damage-per-round vs the panel's median AC) as a
sanity cross-check, NOT as the calibration itself.

Seeds are crc32-stable (cross-process comparable). Engine dice are global
RNG — only compare numbers within one run.

Usage: python scripts/calibrate_cr.py [games_per_arch_per_level] [monster_id ...]
       (no monster ids = full registry)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from zlib import crc32

from trpg.scenarios.archetypes import STANDARD_ARCHETYPES
from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters, make_monster

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.engine.combat_policy import make_archetype_policy

# Panel = the frozen standard 12, explicitly. (First version used
# env_v2.ARCHETYPE_LIST, which at the time snapshotted the LIVE registry —
# the monsters joined their own calibration panel. Fixed at the source, but
# stay explicit here.)
PANEL = STANDARD_ARCHETYPES

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ONLY = set(sys.argv[2:])          # optional monster-id filter (Wave 2+ deltas)
if ONLY:
    unknown = ONLY - set(MONSTER_DEFS)
    assert not unknown, f"unknown monster ids: {unknown}"
LEVELS = list(range(1, 9))


def play(arch: str, lvl: int, mid: str, seed: int) -> bool:
    """One episode: scripted `arch` expert at `lvl` vs monster `mid`.
    Returns True when the class side wins (monster dead, class alive)."""
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[arch], opp_archs=[mid],
              level=lvl, opp_level=MONSTER_DEFS[mid].natural_level)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    expert = make_archetype_policy(arch)
    done = False
    while not done:
        actor = env.current_agent_id
        a = env.ws.characters[actor]
        dec = expert.decide(actor, a, env.ws, env.resources,
                            env.ws.combat.round_number)
        act = ([0, 0, 0] if (dec.action is None or dec.fled)
               else list(encode_action(dec.action, env.ws, actor)))
        _, _, term, trunc, _ = env.step(act)
        done = term or trunc
    return (not env.ws.characters[oid].is_alive()
            and env.ws.characters[aid].is_alive())


def _avg_dice(notation: str) -> float:
    n, s = notation.lower().split("d")
    return int(n) * (int(s) + 1) / 2.0


def power_score(mid: str, ref_ac: int = 15) -> float:
    """HP × best damage-per-round vs ref AC — analytic cross-check only."""
    c = make_monster(mid)
    n_atk = getattr(c, "attacks_per_action", 1) or 1
    best = 0.0
    for w in c.weapons:
        stat = ("DEX" if (w.range_type == "遠程" or
                          ("精巧" in w.properties
                           and c.stats.modifier("DEX") > c.stats.modifier("STR")))
                else "STR")
        mod = c.stats.modifier(stat)
        p_hit = max(0.05, min(0.95,
                    (21 + mod + c.proficiency_bonus - ref_ac) / 20.0))
        best = max(best, n_atk * (_avg_dice(w.damage_dice) + mod) * p_hit)
    from trpg.engine.abilities import ABILITY_REGISTRY
    for sid in c.known_abilities:
        ab = ABILITY_REGISTRY.get(sid)
        if ab is not None and ab.features.expected_damage > 0:
            best = max(best, ab.features.expected_damage * 0.55)
    return c.max_hp * best


def equiv_level(wrs: dict[int, float]) -> str:
    """First 50% crossing with linear interpolation; <1 / >8 at the edges."""
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
    n_per_level = GAMES * len(PANEL)
    print(f"CR calibration — panel: 12 experts × {GAMES} games/level "
          f"(n={n_per_level}/level), monster at natural_level")
    print(f"{'monster':<12} {'CR':>5} {'natL':>4} " +
          " ".join(f"L{l:<4}" for l in LEVELS) + f" {'equivL':>7} {'power':>8}")
    print("-" * 100)
    summary = []
    targets = [m for m in sorted(MONSTER_DEFS, key=lambda m: MONSTER_DEFS[m].cr)
               if not ONLY or m in ONLY]
    for mid in targets:
        md = MONSTER_DEFS[mid]
        wrs: dict[int, float] = {}
        cells = []
        for lvl in LEVELS:
            w = 0
            for arch in PANEL:
                base = crc32(f"cr_{mid}_{arch}_{lvl}".encode()) & 0xFFFFFF
                for i in range(GAMES):
                    w += int(play(arch, lvl, mid, base + i))
            wrs[lvl] = w / n_per_level
            cells.append(f"{wrs[lvl]:4.0%}")
            # Early stop once well past the crossing — saves pointless stomps.
            if wrs[lvl] >= 0.75 and any(v >= 0.5 for v in wrs.values()):
                break
        eq = equiv_level(wrs)
        cells += ["  - "] * (len(LEVELS) - len(cells))
        print(f"{mid:<12} {md.cr:>5.2f} {md.natural_level:>4} " +
              " ".join(cells) + f" {eq:>7} {power_score(mid):>8.0f}")
        summary.append((mid, md.cr, eq))
    print("-" * 100)
    print("CR→等效等級（寫回 MONSTER_CATALOG §Wave0）：")
    for mid, cr, eq in summary:
        print(f"  CR {cr:>5.2f}  {mid:<12} → 等效 L{eq}")


if __name__ == "__main__":
    main()
