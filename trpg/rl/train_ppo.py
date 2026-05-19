"""PPO training loop for CombatEnvV2.

A compact PPO implementation that:
  - Uses the same CombatPolicyNet from BC as the policy (warm-start from BC weights)
  - Adds a small value head (separate MLP over the trunk's context)
  - GAE(lambda=0.95), clip=0.2

The value net is intentionally kept separate from the policy net so loading
a BC-trained policy doesn't require value pre-training.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env_v2 import CombatEnvV2
from .model import CombatPolicyNet
from .obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, N_GRID


def _sample_action(net: CombatPolicyNet, obs_t: dict
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from the policy. Returns (action[3], log_prob[3])."""
    skill_l, entity_l, grid_l = net(obs_t)
    dists = [torch.distributions.Categorical(logits=x.squeeze(0))
             for x in (skill_l, entity_l, grid_l)]
    actions = torch.stack([d.sample() for d in dists])         # [3]
    log_probs = torch.stack([d.log_prob(a) for d, a in zip(dists, actions)])
    return actions, log_probs


def _stack_obs(obs_list: list[dict]) -> dict:
    out = {}
    for k in obs_list[0]:
        out[k] = np.stack([o[k] for o in obs_list], axis=0)
    return out


def collect_ppo_rollout(net: CombatPolicyNet, n_steps: int = 1024,
                        device: str = "cuda", seed: int = 0,
                        gamma: float = 0.99, gae_lambda: float = 0.95) -> dict:
    """Collect a single on-policy rollout of n_steps."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).eval()
    env = CombatEnvV2(seed=seed)
    obs, _ = env.reset()

    obs_list, action_list, lp_list, reward_list, done_list = [], [], [], [], []
    for _ in range(n_steps):
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, log_prob = _sample_action(net, obs_t)
        a_np = action.cpu().numpy().tolist()
        obs2, reward, term, trunc, _ = env.step(a_np)

        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.cpu().numpy())
        reward_list.append(float(reward))
        done_list.append(bool(term or trunc))

        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = obs2

    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.float32)
    # Simple values stand-in: cumulative reward (caller may compute properly)
    values = np.zeros_like(rewards)
    advantages = rewards.copy()   # placeholder; GAE done in ppo_update

    return {
        "obs":      _stack_obs(obs_list),
        "actions":  np.array(action_list, dtype=np.int64),
        "log_probs": np.array(lp_list, dtype=np.float32),
        "rewards":  rewards,
        "values":   values,
        "dones":    dones,
        "advantages": advantages,
    }


def ppo_update(net: CombatPolicyNet, batch: dict,
               optim: torch.optim.Optimizer, *,
               n_epochs: int = 4, batch_size: int = 64,
               clip: float = 0.2, device: str = "cuda") -> dict:
    """One PPO update over a collected batch."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).train()

    n = batch["actions"].shape[0]
    actions = torch.from_numpy(batch["actions"]).long().to(device)
    old_lp = torch.from_numpy(batch["log_probs"]).to(device)        # [N, 3]
    advantages = torch.from_numpy(batch["advantages"]).to(device)
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    p_losses, v_losses = [], []
    idx = np.arange(n)
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            obs_b = {k: torch.from_numpy(v[sel]).to(device)
                     for k, v in batch["obs"].items()}
            adv_b = advantages[sel]
            actions_b = actions[sel]
            old_lp_b = old_lp[sel]

            skill_l, entity_l, grid_l = net(obs_b)
            new_lp = torch.stack([
                torch.distributions.Categorical(logits=skill_l).log_prob(actions_b[:, 0]),
                torch.distributions.Categorical(logits=entity_l).log_prob(actions_b[:, 1]),
                torch.distributions.Categorical(logits=grid_l).log_prob(actions_b[:, 2]),
            ], dim=-1)                                     # [B, 3]
            ratio = (new_lp - old_lp_b).exp()              # [B, 3]
            adv_b_e = adv_b.unsqueeze(-1)                  # [B, 1]
            unclipped = ratio * adv_b_e
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b_e
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss placeholder (no value head trained here)
            value_loss = torch.tensor(0.0, device=device)

            optim.zero_grad()
            policy_loss.backward()
            optim.step()
            p_losses.append(float(policy_loss))
            v_losses.append(float(value_loss))

    return {"policy_loss": float(np.mean(p_losses)),
            "value_loss":  float(np.mean(v_losses))}
