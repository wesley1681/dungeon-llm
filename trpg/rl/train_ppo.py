"""PPO training loop for CombatEnvV2.

Full PPO with:
  - Shared-encoder value head (critic) for TD bootstrapping
  - GAE(gamma=0.99, lambda=0.95) advantage estimation
  - Clipped surrogate objective (clip=0.2)
  - Value loss (MSE) + entropy bonus
  - Gradient clipping (max_norm=0.5)
  - Parallel rollout collection via multiprocessing (n_envs workers)

Warm-start from a BC-trained checkpoint via load_state_dict.
"""
from __future__ import annotations
import io
import numpy as np
import torch
import torch.nn.functional as F

from .env_v2 import CombatEnvV2, _AGENT_ID
from .model import CombatPolicyNet, apply_resource_mask, apply_entity_mask


def _sample_action(net: CombatPolicyNet, obs_t: dict,
                   resources: dict | None = None,
                   ws=None, agent_id: str = ""
                   ) -> tuple[torch.Tensor, torch.Tensor, float,
                              "torch.Tensor", "torch.Tensor"]:
    """Sample action with the same factored policy that pick_action uses.

    Action factorisation (matches inference):
      1. end / not-end via Bernoulli(sigmoid(end_logit))
      2. if not end: skill via Categorical(skill_logits) WITH SLOT 0 MASKED
                     (slot 0 = end is decided at step 1)
      3. entity / grid conditional on the chosen skill

    Returns log_probs as a 4-vector [end_lp, skill_lp, entity_lp, grid_lp]
    so PPO can compute per-component ratios. When the actor ends the turn
    (either chosen or forced), skill / entity / grid log_probs are
    placeholders (0.0) — they had no policy decision to influence.

    Aligning sampling with pick_action is critical for PPO correctness:
    without this, the policy PPO trains differs from the policy that
    gets evaluated (the prior failure mode that destroyed bc_v19's win
    rate within 20 updates).
    """
    end_l, skill_l, entity_l, grid_l = net(obs_t)
    # entity_l: [B, N_SKILL, N_ENTITY]; grid_l: [B, N_SKILL, N_GRID*N_GRID]
    if resources is not None:
        skill_l = apply_resource_mask(skill_l, resources, ws, agent_id)
    entity_l = apply_entity_mask(entity_l, obs_t)
    val = net.value(obs_t).item()

    skill_mask_bool = (skill_l[0] <= -1e8)            # [N_SKILL]
    entity_mask_bool = (entity_l[0, 0] <= -1e8)       # [N_ENTITY]

    # Slot 0 = "end" is decided by end_head, not by skill argmax/sample.
    # Mask it out before constructing the skill distribution.
    skill_l_play = skill_l.clone()
    skill_l_play[..., 0] = -1e9
    forced_end = (skill_l_play[0] <= -1e8).all().item()

    end_logit = end_l[0]
    end_dist = torch.distributions.Bernoulli(logits=end_logit)
    if forced_end:
        # No playable skill — turn must end. log_prob taken at end_a=1 so
        # gradient still flows through end_head (training it to recognise
        # these states).
        end_a = torch.ones((), device=end_logit.device)
    else:
        end_a = end_dist.sample()
    end_lp = end_dist.log_prob(end_a)

    if end_a.item() > 0.5 or forced_end:
        actions = torch.tensor([0, 0, 0], dtype=torch.long,
                                device=end_logit.device)
        zero = torch.zeros((), device=end_logit.device)
        log_probs = torch.stack([end_lp, zero, zero, zero])
        return actions, log_probs, val, skill_mask_bool, entity_mask_bool

    # Not ending — sample skill (slot 0 masked) + entity/grid conditionals.
    skill_dist = torch.distributions.Categorical(logits=skill_l_play.squeeze(0))
    skill_a = skill_dist.sample()
    skill_lp = skill_dist.log_prob(skill_a)

    ent_dist = torch.distributions.Categorical(logits=entity_l[0, skill_a, :])
    ent_a = ent_dist.sample()
    ent_lp = ent_dist.log_prob(ent_a)

    grid_dist = torch.distributions.Categorical(logits=grid_l[0, skill_a, :])
    grid_a = grid_dist.sample()
    grid_lp = grid_dist.log_prob(grid_a)

    actions   = torch.stack([skill_a, ent_a, grid_a])
    log_probs = torch.stack([end_lp, skill_lp, ent_lp, grid_lp])
    return actions, log_probs, val, skill_mask_bool, entity_mask_bool


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
        nv       = next_value if t == n - 1 else values[t + 1]
        delta    = rewards[t] + gamma * nv * (1.0 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


# ── Worker function for parallel rollout ─────────────────────────────────────
# Must be module-level (not a closure) for multiprocessing spawn to pickle it.

def _worker_collect(args: tuple) -> dict:
    """Run a sub-rollout on CPU in a worker process."""
    state_dict_bytes, n_steps, seed, gamma, gae_lambda, use_heuristic, opp_state_bytes = args
    import torch, numpy as np

    # Recreate model on CPU from serialised weights
    from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask
    from trpg.rl.env_v2 import CombatEnvV2, _AGENT_ID

    net = CombatPolicyNet()
    net.load_state_dict(torch.load(io.BytesIO(state_dict_bytes), map_location="cpu"))
    net.eval()

    env = CombatEnvV2(seed=seed)
    # Set opponent override BEFORE reset so it applies to the first episode too.
    if use_heuristic:
        env.use_heuristic_opponent()
    elif opp_state_bytes is not None:
        opp_net = CombatPolicyNet()
        opp_net.load_state_dict(torch.load(io.BytesIO(opp_state_bytes), map_location="cpu"))
        opp_net.eval()
        env.use_self_play_opponent(opp_net)
    obs, _ = env.reset()
    obs_list, action_list, lp_list, reward_list, value_list, done_list = [], [], [], [], [], []
    skill_mask_list: list[np.ndarray] = []
    entity_mask_list: list[np.ndarray] = []

    for _ in range(n_steps):
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            end_l, skill_l, entity_l, grid_l = net(obs_t)
            skill_l  = apply_resource_mask(skill_l, env.resources, env.ws, _AGENT_ID)
            entity_l = apply_entity_mask(entity_l, obs_t)
            val      = net.value(obs_t).item()
            sk_mask_bool  = (skill_l[0] <= -1e8)
            ent_mask_bool = (entity_l[0, 0] <= -1e8)
            # Match pick_action's factored policy: end first via end_head,
            # then skill (slot 0 masked) + entity/grid conditional.
            skill_l_play = skill_l.clone()
            skill_l_play[..., 0] = -1e9
            forced_end = (skill_l_play[0] <= -1e8).all().item()
            end_dist = torch.distributions.Bernoulli(logits=end_l[0])
            if forced_end:
                end_a_t = torch.ones(())
            else:
                end_a_t = end_dist.sample()
            end_lp = end_dist.log_prob(end_a_t)
            if end_a_t.item() > 0.5 or forced_end:
                action = torch.tensor([0, 0, 0], dtype=torch.long)
                zero = torch.zeros(())
                log_prob = torch.stack([end_lp, zero, zero, zero])
            else:
                skill_dist = torch.distributions.Categorical(logits=skill_l_play.squeeze(0))
                skill_a    = skill_dist.sample()
                ent_dist   = torch.distributions.Categorical(logits=entity_l[0, skill_a, :])
                ent_a      = ent_dist.sample()
                grid_dist  = torch.distributions.Categorical(logits=grid_l[0, skill_a, :])
                grid_a     = grid_dist.sample()
                action     = torch.stack([skill_a, ent_a, grid_a])
                log_prob   = torch.stack([end_lp,
                                           skill_dist.log_prob(skill_a),
                                           ent_dist.log_prob(ent_a),
                                           grid_dist.log_prob(grid_a)])
        a_np   = action.numpy().tolist()
        obs2, reward, term, trunc, _ = env.step(a_np)

        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.numpy())
        skill_mask_list.append(sk_mask_bool.numpy())
        entity_mask_list.append(ent_mask_bool.numpy())
        reward_list.append(float(reward))
        value_list.append(val)
        done_list.append(bool(term or trunc))

        obs = env.reset()[0] if (term or trunc) else obs2

    # Bootstrap and GAE computed per-worker so trajectory boundaries are correct
    with torch.no_grad():
        obs_t      = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        next_value = net.value(obs_t).item() if not done_list[-1] else 0.0

    rewards = np.array(reward_list, dtype=np.float32)
    values  = np.array(value_list,  dtype=np.float32)
    dones   = np.array(done_list,   dtype=np.float32)
    advantages, returns = _compute_gae(rewards, values, dones, next_value,
                                       gamma, gae_lambda)
    return {
        "obs":          {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]},
        "actions":      np.array(action_list, dtype=np.int64),
        "log_probs":    np.array(lp_list,     dtype=np.float32),
        "skill_masks":  np.stack(skill_mask_list, axis=0).astype(np.bool_),
        "entity_masks": np.stack(entity_mask_list, axis=0).astype(np.bool_),
        "rewards":      rewards,
        "values":       values,
        "returns":      returns,
        "advantages":   advantages,
        "dones":        dones,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def collect_ppo_rollout(net: CombatPolicyNet, n_steps: int = 1024,
                        n_envs: int = 1,
                        device: str = "cuda", seed: int = 0,
                        gamma: float = 0.99,
                        gae_lambda: float = 0.95,
                        use_heuristic_opponent: bool = False,
                        opponent_net: CombatPolicyNet | None = None) -> dict:
    """Collect an on-policy rollout, optionally using parallel workers.

    n_envs > 1  spawns worker processes (one per env). Each worker gets
    n_steps // n_envs steps. Total data = n_steps regardless of n_envs.

    If `opponent_net` is provided (and use_heuristic_opponent is False), the
    opponent in each env is driven by that network — self-play mode.
    """
    if n_envs > 1:
        return _collect_parallel(net, n_steps, n_envs, device, seed, gamma,
                                 gae_lambda, use_heuristic_opponent, opponent_net)
    return _collect_sequential(net, n_steps, device, seed, gamma, gae_lambda,
                               use_heuristic_opponent, opponent_net)


def _collect_sequential(net, n_steps, device, seed, gamma, gae_lambda,
                         use_heuristic=False, opponent_net=None) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).eval()
    env = CombatEnvV2(seed=seed)
    if use_heuristic:
        env.use_heuristic_opponent()
    elif opponent_net is not None:
        env.use_self_play_opponent(opponent_net)
    obs, _ = env.reset()
    obs_list, action_list, lp_list, reward_list, value_list, done_list = [], [], [], [], [], []
    skill_mask_list: list[np.ndarray] = []
    entity_mask_list: list[np.ndarray] = []

    for _ in range(n_steps):
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, log_prob, val, sk_mask, ent_mask = _sample_action(
                net, obs_t, env.resources, env.ws, _AGENT_ID)
        a_np   = action.cpu().numpy().tolist()
        obs2, reward, term, trunc, _ = env.step(a_np)

        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.cpu().numpy())
        skill_mask_list.append(sk_mask.cpu().numpy())
        entity_mask_list.append(ent_mask.cpu().numpy())
        reward_list.append(float(reward))
        value_list.append(val)
        done_list.append(bool(term or trunc))
        obs = env.reset()[0] if (term or trunc) else obs2

    with torch.no_grad():
        obs_t      = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        next_value = net.value(obs_t).item() if not done_list[-1] else 0.0

    rewards = np.array(reward_list, dtype=np.float32)
    values  = np.array(value_list,  dtype=np.float32)
    dones   = np.array(done_list,   dtype=np.float32)
    adv, ret = _compute_gae(rewards, values, dones, next_value, gamma, gae_lambda)
    return {
        "obs":          _stack_obs(obs_list),
        "actions":      np.array(action_list, dtype=np.int64),
        "log_probs":    np.array(lp_list,     dtype=np.float32),
        "skill_masks":  np.stack(skill_mask_list, axis=0).astype(np.bool_),
        "entity_masks": np.stack(entity_mask_list, axis=0).astype(np.bool_),
        "rewards":      rewards, "values": values,
        "returns":      ret, "advantages": adv, "dones": dones,
    }


_POOL: "mp.pool.Pool | None" = None
_POOL_N: int = 0


def _get_pool(n_envs: int):
    """Return a persistent worker pool, creating it once on first call."""
    import multiprocessing as mp
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != n_envs:
        if _POOL is not None:
            _POOL.terminate()
            _POOL.join()
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(processes=n_envs)
        _POOL_N = n_envs
    return _POOL


def _collect_parallel(net, n_steps, n_envs, device, seed, gamma, gae_lambda,
                       use_heuristic=False, opponent_net=None) -> dict:
    # Serialise weights — workers deserialise on CPU
    buf = io.BytesIO()
    torch.save(net.to("cpu").state_dict(), buf)
    net.to(device)
    state_bytes = buf.getvalue()

    opp_bytes = None
    if opponent_net is not None:
        ob = io.BytesIO()
        torch.save(opponent_net.to("cpu").state_dict(), ob)
        opp_bytes = ob.getvalue()

    steps_per_env = n_steps // n_envs
    worker_args = [
        (state_bytes, steps_per_env, seed + i, gamma, gae_lambda, use_heuristic, opp_bytes)
        for i in range(n_envs)
    ]

    pool    = _get_pool(n_envs)
    results = pool.map(_worker_collect, worker_args)

    obs_keys = list(results[0]["obs"].keys())
    combined = {
        "obs": {ok: np.concatenate([r["obs"][ok] for r in results], axis=0)
                for ok in obs_keys}
    }
    for k in results[0]:
        if k != "obs":
            combined[k] = np.concatenate([r[k] for r in results], axis=0)
    return combined


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

    n       = batch["actions"].shape[0]
    actions = torch.from_numpy(batch["actions"]).long().to(device)
    old_lp  = torch.from_numpy(batch["log_probs"]).to(device)
    returns = torch.from_numpy(batch["returns"]).to(device)
    adv     = torch.from_numpy(batch["advantages"]).to(device)
    adv     = (adv - adv.mean()) / (adv.std() + 1e-8)
    # Stored action masks — used to put new_lp into the *same* probability
    # space as the rollout-time old_lp. Without these the ratio is computed
    # against unmasked distributions while old_lp came from masked ones, so
    # ratio < 1 systematically and PPO pushes policy in a random direction.
    sk_mask  = torch.from_numpy(batch["skill_masks"]).to(device)   # [N, N_SKILL]
    ent_mask = torch.from_numpy(batch["entity_masks"]).to(device)  # [N, N_ENTITY]

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
            sk_mask_b  = sk_mask[sel]                 # [B, N_SKILL]
            ent_mask_b = ent_mask[sel]                # [B, N_ENTITY]

            end_l, skill_l, entity_l, grid_l = net(obs_b)
            # Re-apply rollout-time masks so the dist matches the one that
            # produced old_lp.
            skill_l  = skill_l.masked_fill(sk_mask_b, -1e9)
            entity_l = entity_l.masked_fill(ent_mask_b.unsqueeze(1), -1e9)
            # Slot 0 masked for skill sampling — matches _sample_action.
            skill_l_play = skill_l.clone()
            skill_l_play[..., 0] = -1e9
            B_now = skill_l.shape[0]
            bidx = torch.arange(B_now, device=skill_l.device)
            sk_chosen = acts_b[:, 0]
            cond_ent  = entity_l[bidx, sk_chosen]    # [B, N_ENTITY]
            cond_grid = grid_l[bidx, sk_chosen]      # [B, N_GRID*N_GRID]

            # Recompute the factored log_probs in the SAME structure as
            # _sample_action: [end, skill, entity, grid]. For "ended" steps
            # (acts_b[:,0] == 0) only end_lp is meaningful; skill/ent/grid
            # had no policy choice so their log_probs are 0, matching the
            # rollout-time placeholder so the ratio is 1 → 0 gradient.
            ended_steps = (acts_b[:, 0] == 0)
            end_dist   = torch.distributions.Bernoulli(logits=end_l)
            skill_dist = torch.distributions.Categorical(logits=skill_l_play)
            ent_dist   = torch.distributions.Categorical(logits=cond_ent)
            grid_dist  = torch.distributions.Categorical(logits=cond_grid)

            end_lp = end_dist.log_prob(ended_steps.float())
            # Use safe indices for ended steps (action=0 is masked, log_prob
            # would be -inf); mask them to 0 below.
            sk_safe = acts_b[:, 0].clamp(min=1)
            skill_lp = skill_dist.log_prob(sk_safe).masked_fill(ended_steps, 0.0)
            ent_lp   = ent_dist.log_prob(acts_b[:, 1]).masked_fill(ended_steps, 0.0)
            grid_lp  = grid_dist.log_prob(acts_b[:, 2]).masked_fill(ended_steps, 0.0)

            new_lp = torch.stack([end_lp, skill_lp, ent_lp, grid_lp], dim=-1)
            # Entropy across all four heads — skill/ent/grid contribute 0
            # for ended steps (no choice was made).
            ent_skill = skill_dist.entropy().masked_fill(ended_steps, 0.0)
            ent_ent   = ent_dist.entropy().masked_fill(ended_steps, 0.0)
            ent_grid  = grid_dist.entropy().masked_fill(ended_steps, 0.0)
            entropy = torch.stack([end_dist.entropy(),
                                   ent_skill, ent_ent, ent_grid], dim=-1).mean()
            ratio     = (new_lp - old_b).exp()
            adv_exp   = adv_b.unsqueeze(-1)
            policy_loss = -torch.min(ratio * adv_exp,
                                     torch.clamp(ratio, 1-clip, 1+clip) * adv_exp).mean()
            value_loss  = F.mse_loss(net.value(obs_b), ret_b)
            loss        = policy_loss + vf_coef * value_loss - ent_coef * entropy

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
