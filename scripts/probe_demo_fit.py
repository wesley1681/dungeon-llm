"""Audit: WHY is seed-training's switch-acc stuck at ~= the confirm share?

Loads the saved switch_demos.npz + a checkpoint and measures, per bucket:
  - bucket = does the ENEMY's typed_resist descriptor show a negative cell for
    a given damage type (resist/immune injected or natural)?
  - label composition inside the bucket (proxy: is the labeled slot an
    auto_hit skill = magic_missile-like? what target_type? POINT = fireball-like)
  - masked-argmax fit rate (the number that matters at eval time)

If fit-rate ~0 on resist buckets while labels look correct -> signal too thin
/ slow (fix = density). If labels look WRONG in-bucket -> collection bug.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch

from trpg.rl.obs import (ENT_DESC_START, ENEMY_SLOT_START, N_STATUS_SLOTS,
                         N_SAVE_STATS)
from trpg.engine.damage import DAMAGE_TYPES
from trpg.engine.skill import TargetType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student

RESIST_BASE = ENT_DESC_START + 9 + N_STATUS_SLOTS + N_SAVE_STATS
AUTO_HIT_F = 12          # scalar index of auto_hit in SkillFeatures.as_vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="models/seed_switch/switch_demos.npz")
    p.add_argument("--ckpt", default="models/seed_switch2/seed_s0400.pt")
    args = p.parse_args()

    z = np.load(args.demos)
    act = z["actions"]; tt = z["target_types"]
    obs = {k[len("obs_"):]: z[k] for k in z.files if k.startswith("obs_")}
    N = len(act)
    print(f"demos={args.demos}  N={N}")

    net = load_student(args.ckpt); net.eval()
    # batched masked argmax
    preds = np.zeros(N, np.int64)
    B = 512
    with torch.no_grad():
        for i in range(0, N, B):
            ob = {k: torch.from_numpy(v[i:i + B]) for k, v in obs.items()}
            _, sl, _, _ = net(ob)
            sl = sl.masked_fill(ob["skill_mask"] < 0.5, -1e9)
            sl[:, 0] = -1e9
            preds[i:i + B] = sl.argmax(-1).numpy()
    fit = preds == act[:, 0]
    print(f"overall masked fit = {fit.mean():.2%}\n")

    ents = obs["entities"]            # [N, slots, dim]
    skl = obs["skills"]               # [N, n_slots, F]
    lab_auto = skl[np.arange(N), act[:, 0], AUTO_HIT_F] > 0.5
    lab_point = tt == int(TargetType.POINT)

    # enemy resist cells split IMMUNE (<=-0.9) vs RESIST (-0.9..-0.01):
    # oracle SHOULD switch on immune; on resist it often correctly keeps the
    # dominant option — lumping them hides whether labels make sense.
    print(f"{'bucket':24s} {'n':>6s} {'fit':>6s} {'lab=autohit':>11s} "
          f"{'lab=POINT':>9s}")
    for di, dt in enumerate(DAMAGE_TYPES):
        col = RESIST_BASE + di
        cells = ents[:, ENEMY_SLOT_START:, col]
        for name, m in ((f"immune[{dt}]", (cells <= -0.9).any(axis=1)),
                        (f"resist[{dt}]",
                         ((cells < -0.01) & (cells > -0.9)).any(axis=1))):
            if m.sum() == 0:
                continue
            print(f"{name:24s} {m.sum():6d} {fit[m].mean():6.1%} "
                  f"{lab_auto[m].mean():11.1%} {lab_point[m].mean():9.1%}")
    none = ~np.any(
        ents[:, ENEMY_SLOT_START:, RESIST_BASE:RESIST_BASE + len(DAMAGE_TYPES)]
        < -0.01, axis=(1, 2))
    print(f"{'no-resist':24s} {none.sum():6d} {fit[none].mean():6.1%} "
          f"{lab_auto[none].mean():11.1%} {lab_point[none].mean():9.1%}")

    # Direct label-conflict scan: identical obs bytes -> different skill label.
    # Any such group is UNFITTABLE mass no optimizer can remove.
    sig = {}
    for i in range(N):
        h = hash((ents[i].tobytes(), skl[i].tobytes(),
                  obs["resources"][i].tobytes()))
        sig.setdefault(h, []).append(i)
    dup_groups = {h: ix for h, ix in sig.items() if len(ix) > 1}
    n_dup = sum(len(ix) for ix in dup_groups.values())
    conf = [ix for ix in dup_groups.values()
            if len(set(act[ix, 0].tolist())) > 1]
    n_conf = sum(len(ix) for ix in conf)
    print(f"\nexact-dup obs: {n_dup} samples in {len(dup_groups)} groups; "
          f"CONFLICTING labels: {n_conf} samples in {len(conf)} groups "
          f"({n_conf / N:.1%} of dataset)")
    if conf:
        ix = conf[0]
        print(f"  example group: idx={ix[:6]} labels={act[ix[:6], 0].tolist()}")


if __name__ == "__main__":
    main()
