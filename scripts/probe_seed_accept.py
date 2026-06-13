"""FINAL acceptance for the switch-seed wave — high-n, single-process.

Why a dedicated script: the std12 probe measured the SAME frozen warm net at
43.1 / 42.4 / 36.8 across three different processes — engine dice draw from
the GLOBAL RNG stream, so prior workload shifts every roll (documented:
only trust within-run deltas). Final verdicts therefore need (a) warm and
candidate gated back-to-back in ONE process, (b) more games than the in-
training batteries (battery arms 60 games, std12 x2 = 288 games, monsters x2).

Usage: python scripts/probe_seed_accept.py models/seed_switch4/seed_s1600.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse

from trpg.scenarios.monsters import (register_monsters, MONSTER_DEFS,
                                     onev1_viable_monsters)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import standard_probe, monster_opp_probe
from seed_switch_bc import battery, dominant_dtype


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--warm", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--std_games", type=int, default=2)
    p.add_argument("--mon_games", type=int, default=2)
    args = p.parse_args()
    register_monsters()

    warm = load_student(args.warm); warm.eval()
    cand = load_student(args.ckpt); cand.eval()

    dom2 = dominant_dtype(warm, "life", "orc", 6,
                          MONSTER_DEFS["orc"].natural_level, 12, "life")
    onat = MONSTER_DEFS["orc"].natural_level
    arms = [
        ("evo/orc+火immune",  "evocation", "orc", 6, onat, ("火", 0.0), "火"),
        ("evo/orc normal",    "evocation", "orc", 6, onat, None,        "火"),
        (f"life/orc+{dom2}immune", "life", "orc", 6, onat, (dom2, 0.0), dom2),
        ("evo/fire_elemental", "evocation", "fire_elemental", 8,
         MONSTER_DEFS["fire_elemental"].natural_level, None, "火"),
        ("life/orc normal",   "life", "orc", 6, onat, None, dom2),
    ]
    for name, net in (("WARM", warm), ("CANDIDATE", cand)):
        print(f"\n=== {name} battery ({args.games} games/arm) ===", flush=True)
        battery(net, args.games, arms, name)

    print(f"\n=== gates (within-process, std x{args.std_games} "
          f"mon x{args.mon_games}) ===", flush=True)
    mon_pool = onev1_viable_monsters(8.0)
    for name, net in (("WARM", warm), ("CANDIDATE", cand)):
        std, per = standard_probe(net, games=args.std_games)
        mon, _ = monster_opp_probe(net, mon_pool, games=args.mon_games)
        worst = sorted(per.items(), key=lambda kv: kv[1])[:3]
        print(f"  {name:9s} std12={std:.1%}  vs-mon={mon:.1%}  "
              f"worst3={[(a, f'{v:.0%}') for a, v in worst]}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
