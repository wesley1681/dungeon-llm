"""BC training script.

Usage:
    python scripts/train_bc.py [--episodes N] [--epochs N] [--out PATH]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="每個 archetype 的 episode 數（調小 CPU 不忙）")
    parser.add_argument("--epochs",   type=int, default=20)
    parser.add_argument("--batch",    type=int, default=128)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--out",      type=str, default="models/bc_v1.pt")
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: 資料收集 ─────────────────────────────────────────────────────
    from trpg.rl.bc_collect import collect_bc_dataset
    from trpg.rl.env_v2 import ARCHETYPE_LIST

    print(f"\n== 資料收集：{len(ARCHETYPE_LIST)} 個 archetype × {args.episodes} episode ==")
    t0 = time.time()
    ds = collect_bc_dataset(n_episodes_per_arch=args.episodes, seed=args.seed)
    n_pairs = ds["actions"].shape[0]
    elapsed = time.time() - t0
    print(f"   收集了 {n_pairs:,} 筆 (obs, action)，耗時 {elapsed:.1f}s")

    action_dist = np.bincount(ds["actions"][:, 0], minlength=20)
    print(f"   技能使用分布 (top 5): {sorted(enumerate(action_dist), key=lambda x: -x[1])[:5]}")

    # ── Phase 2: BC 訓練 ──────────────────────────────────────────────────────
    from trpg.rl.train_bc import train_bc

    print(f"\n== BC 訓練：{args.epochs} epochs, batch={args.batch}, lr={args.lr} ==")
    hist = train_bc(
        ds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=device,
        save_path=str(out_path),
    )
    losses = hist["losses"]
    print(f"   Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    if losses[-1] < losses[0]:
        print("   OK Loss 下降，訓練正常")
    else:
        print("   NG Loss 沒有下降，請檢查資料或學習率")

    # ── Phase 3: 評估 ─────────────────────────────────────────────────────────
    from trpg.rl.env_v2 import CombatEnvV2
    from trpg.rl.model import CombatPolicyNet

    print("\n== 評估：20 episode vs 隨機對手 ==")
    net = hist["model"]
    net.to(device).eval()
    env = CombatEnvV2(seed=99999)
    wins = 0
    n_eval = 20
    for ep in range(n_eval):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            from trpg.rl.model import apply_resource_mask, pick_action
            from trpg.rl.env_v2 import _AGENT_ID
            s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                       ws=env.ws, agent_id=_AGENT_ID))
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        if not env.ws.characters["opponent"].is_alive():
            wins += 1
    win_rate = wins / n_eval
    print(f"   Win rate: {win_rate:.0%}  ({wins}/{n_eval})")

    # ── Phase 4: 行為診斷 ────────────────────────────────────────────────────
    # Aggregate win-rate hides per-archetype collapses (wizard never casts,
    # everyone forgets to end turn, etc.). Run the skill-distribution probe
    # so those failure modes show up before the model is shipped.
    from trpg.rl.skill_probe import probe_and_report
    print("\n== 行為診斷：per-archetype 技能分布 vs expert ==")
    flags = probe_and_report(net, n_episodes=8, device=device,
                              seed=98765, verbose=True)
    if flags:
        print(f"\n⚠️  {len(flags)} 個 category 崩潰 — 模型在這些情境沒模仿好 expert。")
    else:
        print("\n✓ 無 category 崩潰。")

    print(f"\nModel 存到: {out_path}")


if __name__ == "__main__":
    main()
