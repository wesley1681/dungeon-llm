"""Capability-descriptor information probe (MONSTER_CATALOG §4.1 驗證 1).

Question: does the v4 descriptor carry identity-equivalent information for
the standard 12? Method: build descriptors for all 12 archetypes at levels
3..8; classify held-out levels by nearest neighbour in descriptor space
(train levels 3/5/7 → test levels 4/6/8). 100% = the descriptor alone
recovers the archetype; that is the information the enemy one-hot used to
be the only carrier of — and monsters get it for free.

Also prints each Wave 0 monster's nearest standard-class descriptor — a
sanity read of "what does the net think this stranger resembles".

Usage: python scripts/probe_descriptor_info.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from trpg.scenarios.archetypes import STANDARD_ARCHETYPES, make_character
from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters, make_monster
from trpg.rl.obs import capability_descriptor

register_monsters()

TRAIN_LEVELS = (3, 5, 7)
TEST_LEVELS = (4, 6, 8)


def desc(arch: str, level: int) -> np.ndarray:
    return capability_descriptor(make_character(arch, level=level))


def main():
    train = [(arch, lvl, desc(arch, lvl))
             for arch in STANDARD_ARCHETYPES for lvl in TRAIN_LEVELS]

    print("=== 1-NN archetype recovery (train L3/5/7 → test L4/6/8) ===")
    correct = total = 0
    errors = []
    for arch in STANDARD_ARCHETYPES:
        for lvl in TEST_LEVELS:
            d = desc(arch, lvl)
            pred, _ = min(((a, np.linalg.norm(d - v)) for a, _, v in train),
                          key=lambda kv: kv[1])
            total += 1
            if pred == arch:
                correct += 1
            else:
                errors.append((arch, lvl, pred))
    print(f"accuracy: {correct}/{total} = {correct/total:.0%}")
    for arch, lvl, pred in errors:
        print(f"  MISS {arch} L{lvl} -> {pred}")

    print("\n=== pairwise separation at L5 (min distance between classes) ===")
    d5 = {a: desc(a, 5) for a in STANDARD_ARCHETYPES}
    pairs = [(a, b, float(np.linalg.norm(d5[a] - d5[b])))
             for i, a in enumerate(STANDARD_ARCHETYPES)
             for b in list(STANDARD_ARCHETYPES)[i + 1:]]
    pairs.sort(key=lambda p: p[2])
    for a, b, dist in pairs[:5]:
        print(f"  {a:<18s} vs {b:<18s} dist={dist:.3f}")
    print(f"  (min={pairs[0][2]:.3f} — 0.0 would mean two classes are "
          f"indistinguishable by kit)")

    print("\n=== Wave 0 monsters: nearest standard-class descriptor ===")
    for mid in sorted(MONSTER_DEFS, key=lambda m: MONSTER_DEFS[m].cr):
        dm = capability_descriptor(make_monster(mid))
        ranked = sorted(((a, float(np.linalg.norm(dm - v))) for a, v in d5.items()),
                        key=lambda kv: kv[1])
        a1, dist1 = ranked[0]
        print(f"  {mid:<12} ~ {a1:<18s} (dist {dist1:.3f})")


if __name__ == "__main__":
    main()
