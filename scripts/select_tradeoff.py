"""std12-erosion vs monster-seat-gain Pareto sweep (扮演怪物 wave gate).

probe_seed_accept showed the u0032 champion erodes std12 −6.6pp / vs-mon
−7.2pp vs the seed_s1600 warm (same-process, real). That is a real cost of
the population-style PPO (記憶: 「u5 後 PPO 侵蝕 std12」). Before accepting a
champion we must find the Pareto knee: the snapshot whose monster-seat gain
is worth its std12 cost. All snapshots scored in ONE process so std12 deltas
are trustworthy (cross-process ±4.5pp does not apply within a run).

Monster-seat gain here = mean model-arm WR over the probe seats (the cheap
within-run trend metric from training); the chosen knee then gets the full
paired high-n eval.

Usage: python scripts/select_tradeoff.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from trpg.scenarios.monsters import register_monsters, onev1_viable_monsters
register_monsters()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import standard_probe, monster_opp_probe
from train_monster_actor import seat_probe, SEAT_1V1, SEAT_BOSS

SNAPSHOTS = [
    ("warm seed_s1600", "models/seed_dtype1/seed_s1600.pt"),
    ("actor1/u0032", "models/mon_actor1/ma_u0032.pt"),
    ("rescue/s0100", "models/mon_rescue/rescue_s0100.pt"),
    ("rescue/s0400", "models/mon_rescue/rescue_s0400.pt"),
]

import os as _os
STD_GAMES = int(_os.environ.get("STD_GAMES", "2"))
SEAT_BOSS_GAMES = int(_os.environ.get("SEAT_BOSS_GAMES", "12"))
SEAT_1V1_GAMES = int(_os.environ.get("SEAT_1V1_GAMES", "4"))


def main():
    mon_pool = onev1_viable_monsters(8.0)
    print(f"{'snapshot':<18} {'std12':>6} {'vs-mon':>7} "
          f"{'seat-1v1':>9} {'seat-boss':>10}")
    print("-" * 56)
    for name, path in SNAPSHOTS:
        net = load_student(path); net.eval()
        std, _ = standard_probe(net, games=STD_GAMES)
        mon, _ = monster_opp_probe(net, mon_pool, games=STD_GAMES)
        seats = seat_probe(net, games_1v1=SEAT_1V1_GAMES,
                           games_boss=SEAT_BOSS_GAMES)
        s1 = sum(seats[m] for m in SEAT_1V1) / len(SEAT_1V1)
        sb = sum(seats[f"boss:{m}@L{l}"] for m, l in SEAT_BOSS) / len(SEAT_BOSS)
        print(f"{name:<18} {std:>6.1%} {mon:>7.1%} {s1:>9.1%} {sb:>10.1%}",
              flush=True)


if __name__ == "__main__":
    main()
