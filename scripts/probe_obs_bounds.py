"""obs 邊界實測 (MONSTER_CATALOG.md §4 #1, Wave 0 deliverable).

Three measurements against the B-net champion (blind student):

  1. FEATURE AUDIT — for every Wave 0 monster, the actual enemy-row v3-tail
     values (level / max_hp / ac); flags anything outside [0, 1].
  2. SENSITIVITY SCAN — real obs snapshots from hill_giant fights, enemy
     max_hp feature synthetically set to in-band and out-of-band magnitudes
     (ogre 0.59 … tarrasque 6.76); reports total-variation distance of the
     raw skill softmax vs the snapshot's own baseline. Measures extrapolation
     behaviour without needing engine support for huge HP.
  3. BEHAVIOUR A/B — same-seed games vs the out-of-band Wave 0 monsters,
     arm A raw obs vs arm B with the enemy max_hp feature clipped to 1.0.
     ΔWR ≈ 0 ⇒ today's 5% overshoot is harmless; the scan tells us where
     it stops being harmless.

Usage: python scripts/probe_obs_bounds.py [ckpt] [games]
       (default ckpt: models/pop_b2/pop_u0005.pt — B-net champion)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
from zlib import crc32

import numpy as np
import torch

from trpg.scenarios.monsters import MONSTER_DEFS, register_monsters

register_monsters()

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl import obs as obs_mod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student, blind_self_identity_t

CKPT = sys.argv[1] if len(sys.argv) > 1 else "models/pop_b2/pop_u0005.pt"
GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ARCHS = ["battle_master", "totem_bear", "evocation", "life"]

# Named column indices (obs v4 appended a descriptor after the v3 tail —
# negative-from-end offsets are exactly the bug class this prevents).
I_LEVEL, I_MAXHP, I_AC = obs_mod.I_ENT_LEVEL, obs_mod.I_ENT_MAXHP, obs_mod.I_ENT_AC
E0 = obs_mod.ENEMY_SLOT_START


def fresh_env(arch, mid, seed):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[arch], opp_archs=[mid],
                       level=5, opp_level=MONSTER_DEFS[mid].natural_level)
    return env, obs


def play(net, arch, mid, seed, clip_maxhp):
    env, obs = fresh_env(arch, mid, seed)
    aid, oid = env.agent_ids[0], env.opp_ids[0]
    done = False
    snaps = []
    while not done:
        actor = env.current_agent_id
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        ot = blind_self_identity_t(ot)
        if clip_maxhp:
            ot["entities"][..., I_MAXHP] = ot["entities"][..., I_MAXHP].clamp(max=1.0)
        snaps.append(ot)
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    won = (not env.ws.characters[oid].is_alive()
           and env.ws.characters[aid].is_alive())
    return won, snaps


def main():
    net = load_student(CKPT)
    print(f"ckpt={CKPT} games/arm-cell={GAMES} archs={ARCHS}")

    # ── 1. Feature audit ────────────────────────────────────────────────────
    print("\n=== 1. enemy-row v3-tail feature audit (level/max_hp/ac) ===")
    out_of_band = []
    for mid in sorted(MONSTER_DEFS, key=lambda m: MONSTER_DEFS[m].cr):
        _, obs = fresh_env("battle_master", mid, 0)
        row = obs["entities"][E0]
        lv, mh, ac = float(row[I_LEVEL]), float(row[I_MAXHP]), float(row[I_AC])
        flags = [n for n, v in (("level", lv), ("max_hp", mh), ("ac", ac))
                 if not (0.0 <= v <= 1.0)]
        if flags:
            out_of_band.append(mid)
        print(f"  {mid:<12} level={lv:5.2f} max_hp={mh:5.2f} ac={ac:5.2f}"
              f"   {'OUT: ' + ','.join(flags) if flags else 'in band'}")

    # ── 2. Sensitivity scan ─────────────────────────────────────────────────
    print("\n=== 2. skill-softmax TV distance vs synthetic enemy max_hp ===")
    print("(snapshots from real hill_giant fights; baseline = snapshot's own value)")
    _, snaps = play(net, "battle_master", "hill_giant", 12345, clip_maxhp=False)
    snaps = snaps[:80]
    scales = [0.59, 1.00, 1.05, 2.00, 4.00, 6.76]
    base_probs = []
    for ot in snaps:
        with torch.no_grad():
            _, s, _, _ = net(ot)
        base_probs.append(torch.softmax(s[0, :, 0] if s.dim() == 3 else s[0], dim=-1))
    for sc in scales:
        tvs = []
        for ot, bp in zip(snaps, base_probs):
            ot2 = {k: v.clone() for k, v in ot.items()}
            ot2["entities"][0, E0, I_MAXHP] = sc
            with torch.no_grad():
                _, s, _, _ = net(ot2)
            p = torch.softmax(s[0, :, 0] if s.dim() == 3 else s[0], dim=-1)
            tvs.append(0.5 * float((p - bp).abs().sum()))
        print(f"  max_hp={sc:5.2f}  ({sc*100:4.0f}HP)  "
              f"mean TV={np.mean(tvs):.4f}  max TV={np.max(tvs):.4f}")

    # ── 3. Behaviour A/B (same seeds, raw vs clipped) ───────────────────────
    print("\n=== 3. same-seed WR: raw obs vs max_hp clipped to 1.0 ===")
    targets = [m for m in out_of_band] or ["hill_giant"]
    print(f"targets (out-of-band today): {targets}")
    for mid in targets:
        wa = wb = 0
        for arch in ARCHS:
            base = crc32(f"obsb_{mid}_{arch}".encode()) & 0xFFFFFF
            for i in range(GAMES):
                wa += int(play(net, arch, mid, base + i, clip_maxhp=False)[0])
                wb += int(play(net, arch, mid, base + i, clip_maxhp=True)[0])
        n = GAMES * len(ARCHS)
        print(f"  {mid:<12} raw {wa/n:5.1%}  clipped {wb/n:5.1%}  "
              f"Δ={(wa-wb)/n:+.1%}  (n={n}/arm)")


if __name__ == "__main__":
    main()
