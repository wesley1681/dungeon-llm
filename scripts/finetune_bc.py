"""Finetune an existing BC checkpoint with extra emphasis on weak archetypes.

Usage:
    python scripts/finetune_bc.py --base models/bc_v13.pt \
        --focus vengeance devotion --focus_episodes 100 \
        --other_episodes 20 --epochs 10 --lr 1e-4 \
        --out models/bc_v13_ft.pt

Strategy: collect a mixed dataset where the weak archetypes are over-
represented (focus_episodes per arch) but all archetypes still appear
(other_episodes per arch). This prevents catastrophic forgetting while
giving the model many more gradient steps on the weak ones.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from trpg.rl.bc_collect import collect_bc_rollout
from trpg.rl.env_v2 import ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet
from trpg.rl.train_bc import bc_loss_step
from trpg.rl.skill_probe import probe_and_report


def collect_weighted(focus: list[str], focus_episodes: int,
                     other_episodes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_pairs: list[tuple[dict, tuple, int]] = []
    for arch in ARCHETYPE_LIST:
        n_ep = focus_episodes if arch in focus else other_episodes
        for _ in range(n_ep):
            opp = ARCHETYPE_LIST[rng.integers(len(ARCHETYPE_LIST))]
            level = int(rng.integers(3, 9))
            ep_seed = int(rng.integers(0, 2**31))
            all_pairs.extend(collect_bc_rollout(arch, opp, level, ep_seed))
    obs_keys = list(all_pairs[0][0].keys())
    return {
        "obs":           {k: np.stack([p[0][k] for p in all_pairs], axis=0)
                          for k in obs_keys},
        "actions":       np.array([p[1] for p in all_pairs], dtype=np.int64),
        "target_types":  np.array([p[2] for p in all_pairs], dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Existing checkpoint to load")
    ap.add_argument("--focus", nargs="+", default=["vengeance", "devotion"])
    ap.add_argument("--focus_episodes", type=int, default=100)
    ap.add_argument("--other_episodes", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base:   {args.base}")
    print(f"Focus:  {args.focus} ({args.focus_episodes} ep), "
          f"others ({args.other_episodes} ep)")

    print("\n== Collecting weighted dataset ==")
    t0 = time.time()
    ds = collect_weighted(args.focus, args.focus_episodes,
                          args.other_episodes, args.seed)
    n = ds["actions"].shape[0]
    elapsed = time.time() - t0
    print(f"   {n:,} pairs in {elapsed:.1f}s")
    arch_counts = {}
    # quick sanity: which archetype each pair belongs to is implicit in
    # collection order — skip detailed split, just print first/last action.

    print(f"\n== Loading base checkpoint: {args.base} ==")
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.base, map_location=device,
                                    weights_only=True))
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    actions = torch.from_numpy(ds["actions"]).long()
    target_types = torch.from_numpy(ds["target_types"]).long()
    idx = np.arange(n)

    from trpg.rl.obs import N_SKILL_SLOTS
    act_actions = actions[actions[:, 0] > 0, 0]
    counts = torch.bincount(act_actions, minlength=N_SKILL_SLOTS).float()
    n_active = (counts > 0).sum().clamp(min=1)
    n_act = counts.sum().clamp(min=1)
    raw_w = n_act / (n_active * counts.clamp(min=1))
    skill_weight = torch.where(counts > 0, raw_w.sqrt(), torch.zeros_like(counts)).to(device)
    print(f"   skill_weight computed ({int(n_active)} active slots)")

    print(f"\n== Finetune: {args.epochs} epochs, batch={args.batch}, "
          f"lr={args.lr} ==")
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        ep_losses, ep_accs = [], {"end": [], "skill": [], "entity": [], "grid": []}
        for start in range(0, n, args.batch):
            sel = idx[start:start + args.batch]
            obs_b = {k: torch.from_numpy(v[sel]).to(device)
                     for k, v in ds["obs"].items()}
            loss, acc = bc_loss_step(net, obs_b,
                                      actions[sel].to(device),
                                      target_types[sel].to(device),
                                      optim,
                                      skill_weight=skill_weight)
            ep_losses.append(float(loss))
            for k in ep_accs:
                if acc[k] == acc[k]:
                    ep_accs[k].append(acc[k])
        print(f"   epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={np.mean(ep_losses):.3f}  "
              f"acc end={np.mean(ep_accs['end']):.3f} "
              f"skill={np.mean(ep_accs['skill']):.3f} "
              f"entity={np.mean(ep_accs['entity']):.3f} "
              f"grid={np.mean(ep_accs['grid']):.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(out_path))
    print(f"\nSaved: {out_path}")

    # Quick probe on focus archetypes only
    print("\n== Probe (focus archetypes only) ==")
    net.eval()
    from trpg.rl.skill_probe import run_model_episodes, run_expert_episodes, _format_breakdown
    for arch in args.focus:
        print(f"=== {arch} ===")
        m = run_model_episodes(net, arch, 8, device, 98765)
        e = run_expert_episodes(arch, 8, 98765)
        for line in _format_breakdown("MODEL ", m):
            print(line)
        for line in _format_breakdown("EXPERT", e):
            print(line)
        print()


if __name__ == "__main__":
    main()
