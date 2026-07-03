"""obs v6 (condition-immunity) migration diagnostic — the diag_obsv*_shift.py of
the 2026-07-01 wave. Proves a PRE-v6 checkpoint, loaded through the migration
chain, is BIT-EXACT on real combat states despite the obs now carrying the
condition-immunity descriptor + the skill↔enemy cimmun join.

Mechanism: on a migrated pre-v6 net the entity_mlp cimmun columns AND the head
cimmun-join column are all zero, so the new feature contributes exactly 0. We
verify that directly: for every real rollout step, forward() with the cimmun obs
columns PRESENT must equal forward() with them ZEROED — max |Δ| == 0 across all
four policy heads AND the critic. A per-matchup counter reports how many steps
actually carried a non-zero immunity descriptor, so the test is provably
non-trivial (undead/elemental/boss matchups populate it).

Usage:  python scripts/diag_cimmun_shift.py [--ckpt models/unified/uni_v9.pt] [--steps 300]
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.obs import build_obs, I_DESC_CIMMUN, N_V6_CIMMUN
from trpg.rl.model import CombatPolicyNet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single

# Matchups deliberately loaded with condition-immune monsters on BOTH seats so
# the cimmun descriptor is genuinely populated (undead poison/charm immunity,
# elemental paralyze/prone, lich/tarrasque big-control immunity).
MATCHUPS = [
    ("divination", "lich", 8, 8), ("life", "skeleton", 5, 5),
    ("evocation", "ghoul", 5, 3), ("champion", "zombie", 5, 5),
    ("vengeance", "shadow", 6, 4), ("war", "gargoyle", 6, 5),
    ("wight", "champion", 3, 5), ("lich", "divination", 8, 8),
    ("battle_master", "balor", 8, 8),
    ("assassin", "wight", 6, 5),
]


def load(path):
    net = CombatPolicyNet(hidden=128, n_head_groups=1)
    sd = torch.load(path, map_location="cpu")
    net.load_state_dict(CombatPolicyNet.adapt_state_dict_for_obs(sd))
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/unified/uni_v9.pt")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    net = load(args.ckpt)

    worst = 0.0
    n_steps = 0
    n_active = 0     # steps where the cimmun descriptor was non-zero
    ep = 0
    while n_steps < args.steps:
        ag, opp, lvl, olvl = MATCHUPS[ep % len(MATCHUPS)]
        ep += 1
        env = CombatEnvV2(seed=ep * 2654435761 % (2**31), n_agents=1, n_opps=1)
        try:
            env.reset(agent_archs=[ag], opp_archs=[opp], level=lvl,
                      opp_level=olvl, layout="open")
        except Exception:
            continue
        for _ in range(6):
            aid = env.current_agent_id
            if not aid or aid not in env.ws.characters:
                break
            ob = blind_np_single(build_obs(env.ws, aid, env.resources))
            t = {k: torch.tensor(ob[k]).unsqueeze(0) for k in ob}
            t0 = {k: v.clone() for k, v in t.items()}
            t0["entities"][:, :, I_DESC_CIMMUN:I_DESC_CIMMUN + N_V6_CIMMUN] = 0.0
            with torch.no_grad():
                o1 = net(t); o0 = net(t0)
                v1 = net.value(t); v0 = net.value(t0)
            d = max(float((a - b).abs().max()) for a, b in zip(o1, o0))
            d = max(d, float((v1 - v0).abs().max()))
            worst = max(worst, d)
            if float(t["entities"][:, :, I_DESC_CIMMUN:I_DESC_CIMMUN
                                  + N_V6_CIMMUN].sum()) > 0:
                n_active += 1
            n_steps += 1
            # advance one turn (just end, to walk the state forward)
            try:
                _, _, term, trunc, _ = env.step((0, 0, 0))
            except Exception:
                break
            if term or trunc or n_steps >= args.steps:
                break

    print(f"ckpt={args.ckpt}")
    print(f"steps={n_steps}  cimmun-active steps={n_active} "
          f"({100*n_active/max(1,n_steps):.0f}%)")
    print(f"worst |Δ| (cimmun present vs zeroed, all heads + critic) = {worst:.2e}")
    ok = worst == 0.0 and n_active > 0
    print("BIT-EXACT ✓ (migration inert, test non-trivial)" if ok else
          "FAIL — cimmun perturbs a migrated net" if worst > 0 else
          "INCONCLUSIVE — no cimmun-active states sampled")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
