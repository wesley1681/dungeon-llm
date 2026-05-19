"""PPO fine-tuning from a BC checkpoint.

Usage:
    python scripts/train_ppo.py [--model PATH] [--updates N] [--steps N]

Loads a BC-trained model and continues training with PPO.
Saves checkpoints by win-rate (low/mid/boss thresholds).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch
import numpy as np

from trpg.rl.model import CombatPolicyNet, apply_resource_mask
from trpg.rl.train_ppo import collect_ppo_rollout, ppo_update
from trpg.rl.env_v2 import CombatEnvV2


CHECKPOINT_THRESHOLDS = {"low": 0.40, "mid": 0.55, "boss": 0.70}


def evaluate(net, n_episodes=50, device="cuda"):
    net.to(device).eval()
    env = CombatEnvV2(seed=88888)
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                s, e, g = net(obs_t)
            from trpg.rl.env_v2 import _AGENT_ID
            s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
            action = [int(s[0].argmax()), int(e.argmax()), int(g.argmax())]
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        # Only count as win if opponent actually died (truncation = draw, not win)
        if not env.ws.characters["opponent"].is_alive():
            wins += 1
    return wins / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="models/bc_v1.pt",
                        help="BC checkpoint to warm-start from")
    parser.add_argument("--updates", type=int, default=100,
                        help="PPO update iterations")
    parser.add_argument("--steps",   type=int, default=1024,
                        help="steps per rollout")
    parser.add_argument("--epochs",  type=int, default=4,
                        help="minibatch epochs per update")
    parser.add_argument("--batch",   type=int, default=256)
    parser.add_argument("--n_envs",   type=int, default=8,
                        help="parallel rollout workers (use CPU cores)")
    parser.add_argument("--ent_coef",     type=float, default=0.02,
                        help="entropy bonus coefficient")
    parser.add_argument("--curriculum",   type=int,   default=50,
                        help="first N updates use HeuristicCombatPolicy opponents (easier)")
    parser.add_argument("--lr",       type=float, default=3e-5,
                        help="use smaller LR than BC to avoid forgetting")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--eval_every", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded: {args.model}  device={device}")

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    saved = set()

    # Initial eval
    wr = evaluate(net, n_episodes=50, device=device)
    print(f"\nStart win-rate: {wr:.0%}")

    for update in range(1, args.updates + 1):
        use_heuristic = (update <= args.curriculum)
        if update == args.curriculum + 1:
            print(f"  [curriculum] switching to expert opponents at update {update}")
        batch = collect_ppo_rollout(net, n_steps=args.steps,
                                    n_envs=args.n_envs,
                                    seed=update, device=device,
                                    use_heuristic_opponent=use_heuristic)
        info = ppo_update(net, batch, optim,
                          n_epochs=args.epochs, batch_size=args.batch,
                          ent_coef=args.ent_coef,
                          device=device)

        print(f"Update {update:4d}/{args.updates}  "
              f"policy={info['policy_loss']:+.4f}  "
              f"value={info['value_loss']:.4f}  "
              f"entropy={info['entropy']:.3f}")

        if update % args.eval_every == 0:
            wr = evaluate(net, n_episodes=50, device=device)
            print(f"  => win-rate: {wr:.0%}")

            for label, threshold in CHECKPOINT_THRESHOLDS.items():
                if wr >= threshold and label not in saved:
                    path = out_dir / f"ppo_{label}.pt"
                    torch.save(net.state_dict(), str(path))
                    saved.add(label)
                    print(f"  => checkpoint saved: {path}")

    # Final save
    torch.save(net.state_dict(), str(out_dir / "ppo_final.pt"))
    print(f"\nFinal model: {out_dir}/ppo_final.pt")


if __name__ == "__main__":
    main()
