"""EVAL-ONLY chimera monsters (扮演怪物 wave generalization instruments).

Three novel kits assembled from existing primitives. They are registered
ONLY at eval time — train_monster_actor.py never imports this module, so
the model has seen these identities on NEITHER side of any fight. Zero-shot
seat play on them is the strongest 通用化 evidence (the class-side analogue
is chimera_defs.py, the 12i/12j held-out probes).

Axes covered:
  frost_troll   melee BOSS seat — troll regen chassis × inverted element
                profile (冰 immune / 火 vulnerable / 霜爪 cold claws):
                novel descriptor combo on the seat the baseline craters on.
  storm_ogre    breath timing — ogre chassis + behir's lightning_breath
                (LINE, uses=1, recharge 5-6): the only in-band breath user;
                tests limited-use AoE aiming/timing from the seat.
  plague_wight  1v1 rider CHOICE — wight chassis + ghoul's 鬼爪 alongside
                生命吸取: two near-tie-EV weapons with different riders
                (paralyze-fish vs maxHP-drain). ghoul is quarantined from
                the wave, so 鬼爪 itself is unseen.

Equiv levels are MEASURED (this module's __main__ runs the calibration
sweeps with the script arm of eval_monster_actor.run_episode) and frozen
below — the same measured-constant pattern as EQUIV_LEVEL_1V1.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from trpg.engine.items import Weapon
from trpg.scenarios.archetypes import SkillGrant, TraitGrant
from trpg.scenarios.monsters import (MonsterDef, MONSTER_DEFS, MONSTER_WEAPONS,
                                     EQUIV_LEVEL_1V1, PARTY3_EQUIV_LEVEL,
                                     register_monsters, _PHYS_RESIST)

CHIMERA_MON_WEAPONS = {
    "霜爪": Weapon("霜爪", "2d6", "冰", "近戰", range_normal=1.5),
}

CHIMERA_MON_DEFS = [
    MonsterDef(
        archetype_id="frost_troll", default_name="霜巨魔",
        class_display="怪物", role="front", cr=5.0, natural_level=5,
        stat_block=dict(STR=18, DEX=13, CON=20, INT=7, WIS=9, CHA=7),
        hp_base=84, hp_per_level=0, ac=15,
        weapons=("霜爪",),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 3}),
            TraitGrant("regeneration",
                       params={"amount": 10, "blocked_by": ["火"]}),
            TraitGrant("damage_table", params={"multipliers": {
                "冰": 0.0, "火": 2.0}}),
        ),
    ),
    MonsterDef(
        archetype_id="storm_ogre", default_name="風暴食人魔",
        class_display="怪物", role="front", cr=4.0, natural_level=4,
        stat_block=dict(STR=19, DEX=8, CON=16, INT=5, WIS=7, CHA=7),
        hp_base=59, hp_per_level=0, ac=11,
        weapons=("巨棒",),
        skills=(SkillGrant("lightning_breath"),),
        traits=(
            TraitGrant("uses_flat",
                       params={"skill": "lightning_breath", "uses": 1}),
            TraitGrant("recharge",
                       params={"skill": "lightning_breath", "on": 5}),
            TraitGrant("damage_table", params={"multipliers": {"閃電": 0.5}}),
        ),
    ),
    MonsterDef(
        archetype_id="plague_wight", default_name="瘟疫屍妖",
        class_display="怪物", role="front", cr=3.0, natural_level=3,
        stat_block=dict(STR=15, DEX=14, CON=16, INT=10, WIS=13, CHA=15),
        hp_base=45, hp_per_level=0, ac=14,
        weapons=("生命吸取", "鬼爪"),
        traits=(
            TraitGrant("extra_attack", params={"attacks": 2}),
            TraitGrant("damage_table", params={"multipliers": {
                **_PHYS_RESIST, "黯蝕": 0.5, "毒": 0.0}}),
            TraitGrant("condition_immunity",
                       params={"conditions": ["poisoned"]}),
        ),
    ),
]

# MEASURED by this module's __main__ (script arm, single process, n=30/level;
# engine dice are global RNG → within-run crossings, same protocol as
# calibrate_cr/calibrate_boss). None = not yet measured (fail-loud).
# eval_results/chimera_mon_calibration.txt (2026-06-13, n=24/level):
CHIMERA_EQUIV_1V1: dict[str, float | None] = {
    "frost_troll": math.inf,     # class WR ≤17% through L8 — regen law again
    "storm_ogre": math.inf,      # breath deletes same-level duelists
    "plague_wight": 4.9,         # crossing L4→L5 (54%)
}
CHIMERA_PARTY3_EQUIV: dict[str, float | None] = {
    "frost_troll": 4.2,          # party 29%@L3 / 38%@L4 / 100%@L5
    "storm_ogre": 2.7,           # party 4%@L2 / 71%@L3 (recharge swing)
    "plague_wight": 1.6,         # 1v1 instrument — party sweep informational
}

_registered = False


def register_chimera_monsters() -> tuple[str, ...]:
    """Mirror register_monsters() for the eval-only chimeras. Idempotent.
    Also injects the measured equiv levels into the global fair-pairing
    tables so eval_monster_actor resolves levels without special cases."""
    global _registered
    register_monsters()                      # obs freeze + base registries
    from trpg.engine.items import WEAPON_DEFS
    from trpg.engine.combat_policy import (ARCHETYPE_POLICIES,
                                           GenericMonsterPolicy)
    from trpg.scenarios.archetypes import (CLASS_DEFS, ARCHETYPE_FACTORIES,
                                           ARCHETYPE_ROLES, _factory)
    if _registered:
        return tuple(md.archetype_id for md in CHIMERA_MON_DEFS)
    for wname, w in CHIMERA_MON_WEAPONS.items():
        WEAPON_DEFS.setdefault(wname, w)
        MONSTER_WEAPONS.setdefault(wname, w)
    for md in CHIMERA_MON_DEFS:
        mid = md.archetype_id
        MONSTER_DEFS[mid] = md
        CLASS_DEFS[mid] = md
        ARCHETYPE_FACTORIES[mid] = _factory(mid)
        ARCHETYPE_ROLES[mid] = md.role
        ARCHETYPE_POLICIES[mid] = GenericMonsterPolicy
        eq1 = CHIMERA_EQUIV_1V1[mid]
        eqp = CHIMERA_PARTY3_EQUIV[mid]
        assert eq1 is not None, f"{mid}: run the 1v1 calibration first"
        EQUIV_LEVEL_1V1[mid] = eq1
        if eqp is not None:
            PARTY3_EQUIV_LEVEL[mid] = eqp
    _registered = True
    return tuple(md.archetype_id for md in CHIMERA_MON_DEFS)


# ── Calibration sweeps (script arm only — no model involved) ────────────────

def _calibrate(games: int = 30):
    # bypass the equiv asserts during measurement
    for md in CHIMERA_MON_DEFS:
        CHIMERA_EQUIV_1V1.setdefault(md.archetype_id, None)
        if CHIMERA_EQUIV_1V1[md.archetype_id] is None:
            CHIMERA_EQUIV_1V1[md.archetype_id] = math.inf
        if md.archetype_id not in CHIMERA_PARTY3_EQUIV or \
                CHIMERA_PARTY3_EQUIV[md.archetype_id] is None:
            CHIMERA_PARTY3_EQUIV[md.archetype_id] = math.inf
    register_chimera_monsters()
    from eval_monster_actor import run_episode, PARTY
    from synth_identity import STANDARD_IDS

    def crossing(wrs):
        lvls = sorted(wrs)
        if wrs[lvls[0]] >= 0.5:
            return "<1"
        prev = lvls[0]
        for l in lvls[1:]:
            if wrs[l] >= 0.5:
                lo, hi = wrs[prev], wrs[l]
                frac = (0.5 - lo) / (hi - lo) if hi > lo else 1.0
                return f"{prev + frac * (l - prev):.1f}"
            prev = l
        return ">8"

    for md in CHIMERA_MON_DEFS:
        mid = md.archetype_id
        print(f"\n== {mid} 1v1 sweep (class WR, n={games}/lvl) ==", flush=True)
        wrs = {}
        for lvl in range(1, 9):
            w = n = 0
            for i, cls in enumerate(STANDARD_IDS):
                for k in range(max(1, games // len(STANDARD_IDS))):
                    key = f"chimcal1|{mid}|{cls}|{lvl}|{k}"
                    won, *_ = run_episode(mid, [cls], md.natural_level, lvl,
                                          key, None)
                    w += int(not won); n += 1     # class-side WR
            wrs[lvl] = w / n
            print(f"  L{lvl}: class {wrs[lvl]:.0%}", flush=True)
            if wrs[lvl] >= 0.75:
                break
        print(f"  -> 1v1 equiv = {crossing(wrs)}")
        print(f"\n== {mid} party sweep (party WR, n={games}/lvl) ==",
              flush=True)
        wrs = {}
        for lvl in range(1, 9):
            w = n = 0
            for k in range(games):
                key = f"chimcalb|{mid}|{lvl}|{k}"
                won, *_ = run_episode(mid, list(PARTY), md.natural_level,
                                      lvl, key, None)
                w += int(not won); n += 1         # party-side WR
            wrs[lvl] = w / n
            print(f"  L{lvl}: party {wrs[lvl]:.0%}", flush=True)
            if wrs[lvl] >= 0.75:
                break
        print(f"  -> party3 equiv = {crossing(wrs)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings; warnings.filterwarnings("ignore")
    _calibrate(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
