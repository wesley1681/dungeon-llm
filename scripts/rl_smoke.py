"""End-to-end smoke test of the Phase 2 RL pipeline.

  1. Collect a small BC dataset (5 episodes per archetype)
  2. Train BC for 5 epochs — verify loss goes down
  3. Run 3 PPO updates — verify they don't crash
  4. Print win rate against random opponent over 20 evaluation episodes
"""
import numpy as np
import torch
from trpg.rl.bc_collect import collect_bc_dataset
from trpg.rl.train_bc import train_bc
from trpg.rl.train_ppo import collect_ppo_rollout, ppo_update
from trpg.rl.env_v2 import CombatEnvV2


def evaluate(net, n_episodes: int = 20, device: str = "cpu") -> float:
    """Win rate vs randomly-archetype opponents."""
    net.to(device).eval()
    env = CombatEnvV2(seed=12345)
    wins = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
            with torch.no_grad():
                _, s, e, g = net(obs_t)
            action = [int(s.argmax(-1)), int(e.argmax(-1)), int(g.argmax(-1))]
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        if env.ws.characters["agent"].is_alive():
            wins += 1
    return wins / n_episodes


def main():
    print("== Phase 1: BC data collection ==")
    ds = collect_bc_dataset(n_episodes_per_arch=5, seed=0)
    print(f"  collected {ds['actions'].shape[0]} (obs, action) pairs")

    print("== Phase 2: BC training ==")
    hist = train_bc(ds, epochs=5, batch_size=64, lr=3e-4, device="cuda")
    print(f"  losses per epoch: {[round(l, 3) for l in hist['losses']]}")
    assert hist["losses"][-1] < hist["losses"][0], "BC loss did not decrease"

    net = hist["model"]
    pre_ppo = evaluate(net, n_episodes=20)
    print(f"  win rate after BC: {pre_ppo:.0%}")

    print("== Phase 3: PPO updates ==")
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    for i in range(3):
        batch = collect_ppo_rollout(net, n_steps=256, seed=i+1)
        info = ppo_update(net, batch, optim, n_epochs=2, batch_size=64)
        print(f"  update {i+1}: policy_loss={info['policy_loss']:.4f}")

    post_ppo = evaluate(net, n_episodes=20)
    print(f"  win rate after PPO: {post_ppo:.0%}")


if __name__ == "__main__":
    main()
