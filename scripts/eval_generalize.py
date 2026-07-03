"""Generalization GRID — one checkpoint, the whole {identity × kit} space.

The 通用模型 goal restated by the user: ONE net that, dropped onto an
arbitrary identity carrying an arbitrary skill kit, picks sensible actions
and beats the enemy. This instrument measures how close a checkpoint is to
that, and — more importantly — draws the FAILURE MAP: which kind of identity
it craters on, and (via the skill-usage mix) which skill SEMANTIC it fails
to wield.

Every cell is the paired model-vs-script measurement from eval_monster_actor
(arm-independent global-RNG seeding, same env orientation / action lattice,
within-run Δ only). The script arm is the "appropriate action" oracle for
that kit:
    standard class      -> its hand-written expert policy
    synth / class-chimera-> HeuristicCombatPolicy   (generic EV/greedy brain)
    monster / mon-chimera-> GenericMonsterPolicy
so Δpp = model% − script% answers exactly "did the net figure out how to
PLAY this kit, the way a sane generic brain would?". A large negative Δ on a
bucket the script handles fine = the net cannot wield that kit.

Identity buckets (the any-identity × any-kit axes):
  std12        the 12 standard classes                  (in-distribution floor)
  synth        random legal kits on random chassis      (novel COMBINATIONS of
               seen components — feature-acting vs kit-memorization)
  chimera_cls  held-out class chimeras                  (broken panel↔kit corr)
  monster      in-band monster bodies                   (the body axis)
  chimera_mon  held-out monster kits                    (storm_ogre breath, …)

Opponents are the standard class panel at a fair level (classes: same level
both sides; monsters: EQUIV_LEVEL_1V1 convention, monster at natural_level).

Usage:
  python scripts/eval_generalize.py --ckpt models/pop_mon/pop_u0005.pt \
         --buckets std12 synth chimera_cls monster chimera_mon \
         --games 2 --synth_n 8 --level 5
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from collections import Counter

from trpg.scenarios.monsters import (MONSTER_DEFS, EQUIV_LEVEL_1V1,
                                     register_monsters, onev1_viable_monsters)

register_monsters()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_monster_actor import run_episode, fmt_usage
from distill_routed import load_student
from synth_identity import STANDARD_IDS, fresh_synth
from chimera_defs import register_chimeras, CHIMERA_IDS
from chimera_monsters import register_chimera_monsters, CHIMERA_EQUIV_1V1


def is_monster(ident: str) -> bool:
    return ident in MONSTER_DEFS


def resolve_levels(ident: str, class_level: int) -> tuple[int, int]:
    """(agent_level, opp_level). Classes: symmetric. Monsters: equiv table."""
    if is_monster(ident):
        m_lvl = MONSTER_DEFS[ident].natural_level
        o_lvl = max(1, min(8, round(EQUIV_LEVEL_1V1[ident])))
        return m_lvl, o_lvl
    return class_level, class_level


def build_buckets(names: list[str], synth_n: int, synth_seed: int) -> dict:
    """ident-id -> bucket name. Registers held-out identities as needed."""
    buckets: dict[str, list[str]] = {}
    if "std12" in names:
        buckets["std12"] = list(STANDARD_IDS)
    if "synth" in names:
        # fixed seed => a frozen, reproducible held-out combination panel.
        rng = random.Random(synth_seed)
        ids = []
        for s in range(synth_n):
            ids.append(fresh_synth(rng, slot=s))
        buckets["synth"] = ids
    if "chimera_cls" in names:
        register_chimeras()
        buckets["chimera_cls"] = list(CHIMERA_IDS)
    if "monster" in names:
        buckets["monster"] = sorted(onev1_viable_monsters(8.0),
                                    key=lambda m: EQUIV_LEVEL_1V1[m])
    if "chimera_mon" in names:
        register_chimera_monsters()
        # only the 1v1-viable ones (finite equiv) belong on the 1v1 grid
        buckets["chimera_mon"] = [m for m, eq in CHIMERA_EQUIV_1V1.items()
                                  if eq != float("inf")]
    return buckets


def eval_identity(ident: str, opps: list[str], games: int, class_level: int,
                  net) -> dict:
    """Paired model/script over (opponent × games). Returns per-arm tallies."""
    m_lvl, o_lvl = resolve_levels(ident, class_level)
    res = {arm: {"w": 0, "n": 0, "usage": Counter(), "turns": 0,
                 "fails": Counter()} for arm in ("model", "script")}
    for opp in opps:
        for k in range(games):
            key = f"gen|{ident}|{opp}|{m_lvl}v{o_lvl}|{k}"
            for arm, drv in (("model", net), ("script", None)):
                w, u, t, f = run_episode(ident, [opp], m_lvl, o_lvl, key, drv)
                r = res[arm]
                r["w"] += int(w); r["n"] += 1
                r["usage"] += u; r["turns"] += t; r["fails"] += f
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--buckets", nargs="*",
                   default=["std12", "synth", "chimera_cls", "monster",
                            "chimera_mon"])
    p.add_argument("--games", type=int, default=2,
                   help="games per (identity, opponent-class) per arm")
    p.add_argument("--synth_n", type=int, default=8)
    p.add_argument("--synth_seed", type=int, default=20260613)
    p.add_argument("--level", type=int, default=5,
                   help="class-vs-class fight level (monsters use equiv table)")
    p.add_argument("--opps", nargs="*", default=None,
                   help="opponent class panel (default: all 12 standard)")
    args = p.parse_args()

    opps = args.opps or list(STANDARD_IDS)
    buckets = build_buckets(args.buckets, args.synth_n, args.synth_seed)
    net = load_student(args.ckpt)
    net.eval()

    print(f"ckpt={args.ckpt}")
    print(f"opponents={len(opps)} classes  games/pairing={args.games}  "
          f"class-level={args.level}  (paired seeds, within-run Δ only)\n")

    grand = {arm: [0, 0] for arm in ("model", "script")}
    bucket_rows: list[tuple] = []
    worst: list[tuple] = []   # (delta, bucket, ident, sw, mw)

    for bname, idents in buckets.items():
        print(f"== bucket: {bname}  ({len(idents)} identities × {len(opps)} "
              f"opp × {args.games}g) ==")
        print(f"{'identity':<18} {'lvl':>5} {'script%':>8} {'model%':>7} "
              f"{'Δpp':>6}  {'model-mix':<30} {'script-mix':<30}")
        btot = {arm: [0, 0] for arm in ("model", "script")}
        for ident in idents:
            r = eval_identity(ident, opps, args.games, args.level, net)
            sw = r["script"]["w"] / max(1, r["script"]["n"])
            mw = r["model"]["w"] / max(1, r["model"]["n"])
            m_lvl, o_lvl = resolve_levels(ident, args.level)
            lvl_s = f"{m_lvl}v{o_lvl}"
            for arm in ("model", "script"):
                btot[arm][0] += r[arm]["w"]; btot[arm][1] += r[arm]["n"]
                grand[arm][0] += r[arm]["w"]; grand[arm][1] += r[arm]["n"]
            print(f"{ident:<18} {lvl_s:>5} {sw:>8.1%} {mw:>7.1%} "
                  f"{(mw - sw) * 100:>+6.1f}  "
                  f"{fmt_usage(r['model']['usage'], r['model']['turns']):<30} "
                  f"{fmt_usage(r['script']['usage'], r['script']['turns']):<30}",
                  flush=True)
            for arm in ("model", "script"):
                if r[arm]["fails"]:
                    print(f"    !! {arm} encode/exec fails: "
                          f"{dict(r[arm]['fails'])}")
            worst.append(((mw - sw) * 100, bname, ident, sw, mw))
        bsw = btot["script"][0] / max(1, btot["script"][1])
        bmw = btot["model"][0] / max(1, btot["model"][1])
        print(f"{'  BUCKET MEAN':<18} {'':>5} {bsw:>8.1%} {bmw:>7.1%} "
              f"{(bmw - bsw) * 100:>+6.1f}   (n={btot['model'][1]}/arm)\n")
        bucket_rows.append((bname, bsw, bmw, btot["model"][1]))

    print("================ SUMMARY ================")
    print(f"{'bucket':<14} {'script%':>8} {'model%':>7} {'Δpp':>6} {'n/arm':>7}")
    for bname, bsw, bmw, n in bucket_rows:
        print(f"{bname:<14} {bsw:>8.1%} {bmw:>7.1%} {(bmw - bsw) * 100:>+6.1f} "
              f"{n:>7}")
    gsw = grand["script"][0] / max(1, grand["script"][1])
    gmw = grand["model"][0] / max(1, grand["model"][1])
    print(f"{'OVERALL':<14} {gsw:>8.1%} {gmw:>7.1%} {(gmw - gsw) * 100:>+6.1f} "
          f"{grand['model'][1]:>7}")

    print("\n---- worst 12 cells (model − script) ----")
    for d, b, ident, sw, mw in sorted(worst)[:12]:
        print(f"  {d:>+6.1f}pp  [{b}] {ident:<18} script {sw:.0%} -> "
              f"model {mw:.0%}")


if __name__ == "__main__":
    main()
