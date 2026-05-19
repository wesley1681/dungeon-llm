"""PPO training loop for CombatEnvV2.

Full PPO with:
  - Shared-encoder value head (critic) for TD bootstrapping
  - GAE(gamma=0.99, lambda=0.95) advantage estimation
  - Clipped surrogate objective (clip=0.2)
  - Value loss (MSE) + entropy bonus
  - Gradient clipping (max_norm=0.5)

Warm-start from a BC-trained checkpoint via load_state_dict.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .env_v2 import CombatEnvV2
from .model import CombatPolicyNet


def _sample_action(net: CombatPolicyNet, obs_t: dict,
                   resources: dict | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Sample action, return (action[3], log_prob[3], value_scalar)."""
    from .model import apply_resource_mask
    skill_l, entity_l, grid_l = net(obs_t)
    if resources is not None:
        skill_l = apply_resource_mask(skill_l, resources)
    val = net.value(obs_t).item()
    dists = [torch.distributions.Categorical(logits=x.squeeze(0))
             for x in (skill_l, entity_l, grid_l)]
    actions  = torch.stack([d.sample()          for d     in dists])
    log_probs = torch.stack([d.log_prob(a)       for d, a  in zip(dists, actions)])
    return actions, log_probs, val


def _stack_obs(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in obs_list[0]}


def _compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                 next_value: float, gamma: float = 0.99,
                 gae_lambda: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """GAE advantage + bootstrapped returns."""
    n          = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae   = 0.0
    for t in reversed(range(n)):
        nv     = next_value if t == n - 1 else values[t + 1]
        delta  = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_ppo_rollout(net: CombatPolicyNet, n_steps: int = 1024,
                        device: str = "cuda", seed: int = 0,
                        gamma: float = 0.99,
                        gae_lambda: float = 0.95) -> dict:
    """Collect an on-policy rollout and compute GAE advantages."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).eval()
    env = CombatEnvV2(seed=seed)
    obs, _ = env.reset()

    obs_list, action_list, lp_list = [], [], []
    reward_list, value_list, done_list = [], [], []

    for _ in range(n_steps):
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, log_prob, val = _sample_action(net, obs_t, env.resources)
        a_np = action.cpu().numpy().tolist()
        obs2, reward, term, trunc, _ = env.step(a_np)

        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.cpu().numpy())
        reward_list.append(float(reward))
        value_list.append(val)
        done_list.append(bool(term or trunc))

        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = obs2

    # Bootstrap the final value
    with torch.no_grad():
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        next_value = net.value(obs_t).item() if not done_list[-1] else 0.0

    rewards    = np.array(reward_list, dtype=np.float32)
    values     = np.array(value_list,  dtype=np.float32)
    dones      = np.array(done_list,   dtype=np.float32)
    advantages, returns = _compute_gae(rewards, values, dones, next_value,
                                       gamma, gae_lambda)
    return {
        "obs":        _stack_obs(obs_list),
        "actions":    np.array(action_list, dtype=np.int64),
        "log_probs":  np.array(lp_list,     dtype=np.float32),
        "rewards":    rewards,
        "values":     values,
        "returns":    returns,
        "advantages": advantages,
        "dones":      dones,
    }


def ppo_update(net: CombatPolicyNet, batch: dict,
               optim: torch.optim.Optimizer, *,
               n_epochs: int = 4, batch_size: int = 64,
               clip: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.01, max_grad_norm: float = 0.5,
               device: str = "cuda") -> dict:
    """One PPO minibatch update with value loss + entropy bonus."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).train()

    n        = batch["actions"].shape[0]
    actions  = torch.from_numpy(batch["actions"]).long().to(device)
    old_lp   = torch.from_numpy(batch["log_probs"]).to(device)   # [N, 3]
    returns  = torch.from_numpy(batch["returns"]).to(device)
    adv      = torch.from_numpy(batch["advantages"]).to(device)
    adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

    p_losses, v_losses, entropies = [], [], []
    idx = np.arange(n)
    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            sel    = idx[start:start + batch_size]
            obs_b  = {k: torch.from_numpy(v[sel]).to(device)
                      for k, v in batch["obs"].items()}
            adv_b  = adv[sel]
            acts_b = actions[sel]
            old_b  = old_lp[sel]
            ret_b  = returns[sel]

            skill_l, entity_l, grid_l = net(obs_b)
            dists = [torch.distributions.Categorical(logits=x)
                     for x in (skill_l, entity_l, grid_l)]

            new_lp = torch.stack([
                d.log_prob(acts_b[:, i]) for i, d in enumerate(dists)
            ], dim=-1)                                       # [B, 3]
            entropy = torch.stack([d.entropy() for d in dists], dim=-1).mean()

            ratio     = (new_lp - old_b).exp()              # [B, 3]
            adv_exp   = adv_b.unsqueeze(-1)
            unclipped = ratio * adv_exp
            clipped   = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_exp
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_pred  = net.value(obs_b)                  # [B]
            value_loss  = F.mse_loss(value_pred, ret_b)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optim.step()

            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            entropies.append(entropy.item())

    return {
        "policy_loss": float(np.mean(p_losses)),
        "value_loss":  float(np.mean(v_losses)),
        "entropy":     float(np.mean(entropies)),
    }
