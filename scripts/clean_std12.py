"""Seeded std12 — the CLEAN instrument (扮演怪物 wave gate fix).

train_population.standard_probe does NOT random.seed() the global engine dice
per game, so warm-then-candidate in one process still face DIFFERENT dice
streams (warm's episodes advance the global RNG before candidate runs). That
is exactly the documented trap ("种全域骰 or the experiment is invalid"),
and it makes the std12 deltas it reports noisy: the same warm-vs-u0032 gap
measured −6.6 / −6.6 / −1.9 across three processes.

eval_monster_actor.run_episode already random.seed(crc32(key)) per game, so
reusing it with a STANDARD class in the agent seat gives a seeded std12 where
warm and candidate face bit-identical dice at every (a,o,k) cell — the gap is
then pure policy difference. Same protocol as the W3 fingerprint proof.

Usage: python scripts/clean_std12.py --games 4
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse

from trpg.scenarios.monsters import register_monsters
register_monsters()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from synth_identity import STANDARD_IDS
from eval_monster_actor import run_episode

SNAPSHOTS = [
    ("warm actor1/u0032", "models/mon_actor1/ma_u0032.pt"),   # regen-BC base = baseline
    ("regen_v5/s0400", "models/regen_v5/regen_s0400.pt"),
    ("regen_v5/final", "models/regen_v5/regen_final.pt"),
]


def std12_seeded(net, games, level=5):
    """12×12 greedy matrix, every game random.seed'd so all snapshots face
    identical dice at each cell. Returns (mean_wr, per_class dict)."""
    wins = n = 0
    per = {}
    for a in STANDARD_IDS:
        wa = na = 0
        for o in STANDARD_IDS:
            for k in range(games):
                key = f"cleanstd|{a}|{o}|{k}"
                won, *_ = run_episode(a, [o], level, level, key, net)
                wa += int(won); na += 1
        per[a] = wa / na
        wins += wa; n += na
    return wins / n, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=4)
    args = p.parse_args()
    print(f"seeded std12 (level 5, {args.games} games/cell, "
          f"identical dice across snapshots)\n")
    base = None
    print(f"{'snapshot':<18} {'std12':>6} {'Δ vs warm':>10}")
    print("-" * 36)
    for name, path in SNAPSHOTS:
        net = load_student(path); net.eval()
        std, per = std12_seeded(net, args.games)
        if base is None:
            base = std
        print(f"{name:<18} {std:>6.1%} {(std - base) * 100:>+9.1f}",
              flush=True)
    print(f"\n(n={len(STANDARD_IDS)**2 * args.games} games/snapshot, "
          f"same dice → Δ is pure policy)")


if __name__ == "__main__":
    main()
