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

from .env_v2 import CombatEnvV2
from .model import CombatPolicyNet, apply_resource_mask, apply_entity_mask


def _sample_action(net: CombatPolicyNet, obs_t: dict,
                   resources: dict | None = None,
                   ws=None, agent_id: str = "",
                   mask_immune_null: bool = True
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
        skill_l = apply_resource_mask(skill_l, resources, ws, agent_id,
                                      mask_immune_null=mask_immune_null)
    entity_l = apply_entity_mask(entity_l, obs_t, ws, agent_id,
                                 mask_immune_null=mask_immune_null)
    val = net.value(obs_t).item()

    skill_mask_bool = (skill_l[0] <= -1e8)            # [N_SKILL]
    # Placeholder for end steps (no entity decision; lp is masked to 0 in the
    # update). For act steps this is overwritten with the SAMPLED skill row's
    # mask below: apply_entity_mask is per-row for ally-target skills, and the
    # update re-applies the saved mask to the chosen row — saving row 0 here
    # would put the new_lp in a different probability space than old_lp.
    entity_mask_bool = (entity_l[0, 0] <= -1e8)       # [N_ENTITY]
    # Grid legality mask placeholder — filled for POINT-family skills below.
    grid_mask_bool = torch.zeros(grid_l.shape[-1], dtype=torch.bool,
                                 device=grid_l.device)

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
        return (actions, log_probs, val, skill_mask_bool, entity_mask_bool,
                grid_mask_bool)

    # Not ending — sample skill (slot 0 masked) + entity/grid conditionals.
    skill_dist = torch.distributions.Categorical(logits=skill_l_play.squeeze(0))
    skill_a = skill_dist.sample()
    skill_lp = skill_dist.log_prob(skill_a)

    entity_mask_bool = (entity_l[0, skill_a] <= -1e8)   # chosen row's mask
    ent_dist = torch.distributions.Categorical(logits=entity_l[0, skill_a, :])
    ent_a = ent_dist.sample()
    ent_lp = ent_dist.log_prob(ent_a)

    # Grid legality mask (POINT/LINE/CONE skills): exclude aim cells that
    # decode_action would reject (blocked / no LoS → silent end-turn no-op).
    # Stored and re-applied in ppo_update so old/new log-probs share one
    # probability space — same contract as the entity mask.
    grid_row = grid_l[0, skill_a, :]
    if ws is not None and agent_id:
        from ..engine.skill import available_skills, TargetType
        from .action import point_validity_mask
        sk_i = int(skill_a.item())
        skills = available_skills(ws.characters[agent_id], ws)
        if sk_i < len(skills) and skills[sk_i].features.target_type in (
                TargetType.POINT, TargetType.LINE, TargetType.CONE):
            inv = point_validity_mask(ws, agent_id, skills[sk_i])
            if inv.any() and not inv.all():
                grid_mask_bool = torch.from_numpy(inv).to(grid_l.device)
                grid_row = grid_row.masked_fill(grid_mask_bool, -1e9)

    grid_dist = torch.distributions.Categorical(logits=grid_row)
    grid_a = grid_dist.sample()
    grid_lp = grid_dist.log_prob(grid_a)

    actions   = torch.stack([skill_a, ent_a, grid_a])
    log_probs = torch.stack([end_lp, skill_lp, ent_lp, grid_lp])
    return (actions, log_probs, val, skill_mask_bool, entity_mask_bool,
            grid_mask_bool)


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
    (state_dict_bytes, n_steps, seed, gamma, gae_lambda, use_heuristic,
     opp_state_bytes, n_agents, n_opps, agent_arch) = args
    import torch, numpy as np

    # Recreate model on CPU from serialised weights
    from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask
    from trpg.rl.env_v2 import CombatEnvV2

    net = CombatPolicyNet()
    net.load_state_dict(torch.load(io.BytesIO(state_dict_bytes), map_location="cpu"))
    net.eval()

    env = CombatEnvV2(seed=seed, n_agents=n_agents, n_opps=n_opps)
    # Set opponent override BEFORE reset so it applies to the first episode too.
    if use_heuristic:
        env.use_heuristic_opponent()
    elif opp_state_bytes is not None:
        opp_net = CombatPolicyNet()
        opp_net.load_state_dict(torch.load(io.BytesIO(opp_state_bytes), map_location="cpu"))
        opp_net.eval()
        env.use_self_play_opponent(opp_net)
    obs, _ = env.reset(agent_archs=[agent_arch] if agent_arch else None)
    obs_list, action_list, lp_list, reward_list, value_list, done_list = [], [], [], [], [], []
    skill_mask_list: list[np.ndarray] = []
    entity_mask_list: list[np.ndarray] = []
    last_transition_idx: dict[str, int] = {}

    for _ in range(n_steps):
        agent_id = env.current_agent_id
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            end_l, skill_l, entity_l, grid_l = net(obs_t)
            skill_l  = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
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

        cur_idx = len(reward_list)
        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.numpy())
        skill_mask_list.append(sk_mask_bool.numpy())
        entity_mask_list.append(ent_mask_bool.numpy())
        reward_list.append(float(reward))
        value_list.append(val)
        done_list.append(bool(term or trunc))

        last_transition_idx[agent_id] = cur_idx
        if term or trunc:
            for aid, idx in last_transition_idx.items():
                if aid != agent_id:
                    reward_list[idx] = float(reward)
            last_transition_idx = {}

        obs = (env.reset(agent_archs=[agent_arch] if agent_arch else None)[0]
               if (term or trunc) else obs2)

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
                        opponent_net: CombatPolicyNet | None = None,
                        n_agents: int | None = None,
                        n_opps: int | None = None,
                        expert_mix: float = 0.0,
                        agent_arch: str | None = None) -> dict:
    """Collect an on-policy rollout, optionally using parallel workers.

    ``agent_arch`` (optional): pin every agent to this archetype — used to train
    a single-class specialist that can devote full capacity to discovering a
    better-than-expert policy for one archetype (PPO already beats the scripted
    expert on evocation 58% vs 31% in the shared model; a specialist can do the
    same for the classes the shared model can't fit).

    n_envs > 1  spawns worker processes (one per env). Each worker gets
    n_steps // n_envs steps. Total data = n_steps regardless of n_envs.

    If `opponent_net` is provided (and use_heuristic_opponent is False), the
    opponent in each env is driven by that network — self-play mode.

    `expert_mix` (0..1): fraction of workers that use the scripted archetype
    expert instead of `opponent_net`. e.g. 0.5 with n_envs=8 → 4 workers vs
    scripted experts, 4 workers vs self-play snapshot. Adds OOD diversity to
    the opponent pool — keeps wizards (and other ranged classes) from
    converging to "stand-still" equilibria that only work against equally
    static self-play opponents.

    n_agents / n_opps: fixed team sizes; None = sample from weighted distribution.
    """
    if n_envs > 1:
        return _collect_parallel(net, n_steps, n_envs, device, seed, gamma,
                                 gae_lambda, use_heuristic_opponent, opponent_net,
                                 n_agents, n_opps, expert_mix, agent_arch)
    return _collect_sequential(net, n_steps, device, seed, gamma, gae_lambda,
                               use_heuristic_opponent, opponent_net, n_agents,
                               n_opps, agent_arch)


def _collect_sequential(net, n_steps, device, seed, gamma, gae_lambda,
                         use_heuristic=False, opponent_net=None,
                         n_agents=None, n_opps=None, agent_arch=None) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).eval()
    env = CombatEnvV2(seed=seed, n_agents=n_agents, n_opps=n_opps)
    if use_heuristic:
        env.use_heuristic_opponent()
    elif opponent_net is not None:
        env.use_self_play_opponent(opponent_net)
    obs, _ = env.reset(agent_archs=[agent_arch] if agent_arch else None)
    obs_list, action_list, lp_list, reward_list, value_list, done_list = [], [], [], [], [], []
    skill_mask_list: list[np.ndarray] = []
    entity_mask_list: list[np.ndarray] = []
    grid_mask_list: list[np.ndarray] = []
    last_transition_idx: dict[str, int] = {}

    for _ in range(n_steps):
        agent_id = env.current_agent_id
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
        with torch.no_grad():
            action, log_prob, val, sk_mask, ent_mask, gr_mask = _sample_action(
                net, obs_t, env.resources, env.ws, agent_id)
        a_np   = action.cpu().numpy().tolist()
        obs2, reward, term, trunc, _ = env.step(a_np)

        cur_idx = len(reward_list)
        obs_list.append(obs)
        action_list.append(a_np)
        lp_list.append(log_prob.cpu().numpy())
        skill_mask_list.append(sk_mask.cpu().numpy())
        entity_mask_list.append(ent_mask.cpu().numpy())
        grid_mask_list.append(gr_mask.cpu().numpy())
        reward_list.append(float(reward))
        value_list.append(val)
        done_list.append(bool(term or trunc))

        last_transition_idx[agent_id] = cur_idx
        if term or trunc:
            for aid, idx in last_transition_idx.items():
                if aid != agent_id:
                    reward_list[idx] = float(reward)
            last_transition_idx = {}

        obs = (env.reset(agent_archs=[agent_arch] if agent_arch else None)[0]
               if (term or trunc) else obs2)

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
        "grid_masks":   np.stack(grid_mask_list, axis=0).astype(np.bool_),
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
                       use_heuristic=False, opponent_net=None,
                       n_agents=None, n_opps=None, expert_mix: float = 0.0,
                       agent_arch=None) -> dict:
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
    # Deterministic round-robin: first `n_expert` workers use scripted experts
    # (opp_bytes=None → env falls back to make_archetype_policy), the rest
    # use the self-play snapshot. expert_mix is ignored when use_heuristic is
    # True (curriculum phase forces HeuristicCombatPolicy on all workers).
    n_expert = int(round(expert_mix * n_envs)) if not use_heuristic else 0
    worker_args = [
        (state_bytes, steps_per_env, seed + i, gamma, gae_lambda,
         use_heuristic,
         None if i < n_expert else opp_bytes,
         n_agents, n_opps, agent_arch)
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


def _pcgrad_project(grads: list[torch.Tensor]) -> torch.Tensor:
    """PCGrad surgery (Yu et al. 2020). Given a list of per-task flat
    gradient vectors, project each onto the normal of any other task it
    conflicts with (negative cosine), then return the mean.

    Conflict removal cures the negative-cosine cross-class gradients we
    measured on this codebase's PPO heads (~55% of archetype pairs had
    cos < 0); without it strong-signal archetypes overwrite the encoder
    in directions that crater weak-signal ones (v16: battle_master 49→4%).
    """
    import random as _rnd
    projected = [g.clone() for g in grads]
    for i in range(len(grads)):
        order = [j for j in range(len(grads)) if j != i]
        _rnd.shuffle(order)
        for j in order:
            gj = grads[j]
            dot = (projected[i] * gj).sum()
            if dot < 0:
                gj_sq = (gj * gj).sum() + 1e-12
                projected[i] = projected[i] - (dot / gj_sq) * gj
    return torch.stack(projected).mean(0)


def _flatten_grads(params) -> torch.Tensor:
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
        for p in params
    ])


def _unflatten_to_grads(params, flat: torch.Tensor) -> None:
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n


def ppo_update(net: CombatPolicyNet, batch: dict,
               optim: torch.optim.Optimizer, *,
               n_epochs: int = 4, batch_size: int = 64,
               clip: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.01, max_grad_norm: float = 0.5,
               device: str = "cuda",
               value_only: bool = False,
               reference_net: CombatPolicyNet | None = None,
               kl_coef: float = 0.0,
               pcgrad: bool = False) -> dict:
    """One PPO minibatch update with value loss + entropy bonus.

    ``value_only=True`` zeroes out the policy-gradient and entropy terms
    so the optimizer only fits the value head against the GAE returns —
    used for the first few updates to warm up V(s) before its outputs
    start driving advantage estimates. Without warmup, V(s) is at random
    init from BC (which never trained value_head), so early advantages
    are pure noise and the policy drifts in a meaningless direction
    before V(s) catches up.

    ``pcgrad=True`` applies PCGrad surgery to the per-archetype policy
    gradients before optimiser step. Mitigates the cross-class gradient
    conflict measured on this codebase (~55% of archetype pairs had
    cos < 0 on shared trunk/heads under v19).
    """
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
    # Grid legality masks (optional — older collectors don't store them).
    gr_mask = (torch.from_numpy(batch["grid_masks"]).to(device)
               if "grid_masks" in batch else None)                 # [N, N_GRID²]

    # Per-sample archetype id from the self entity's one-hot at indices
    # 7..7+N_ARCHETYPES-1. Needed for PCGrad to group samples by archetype
    # so we can compute per-archetype policy gradients before surgery.
    from .obs import N_ARCHETYPES
    arch_ids_np = batch["obs"]["entities"][:, 0, 7:7 + N_ARCHETYPES].argmax(axis=1)
    arch_ids    = torch.from_numpy(arch_ids_np).to(device)
    trainable_params = [p for p in net.parameters() if p.requires_grad]

    p_losses, v_losses, entropies, kl_vals = [], [], [], []
    if reference_net is not None:
        reference_net.to(device).eval()
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
            if gr_mask is not None:
                # Same probability space as rollout sampling (legality mask
                # on the chosen skill's aim row) — the entity-mask contract.
                cond_grid = cond_grid.masked_fill(gr_mask[sel], -1e9)

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
            # JOINT-action ratio (was per-head, a PPO correctness bug). The
            # factored policy is conditionally independent given the state, so
            # logπ(a|s) = Σ_heads logπ_head and the importance ratio is
            # exp(Σ_heads Δlogp), clipped ONCE on the joint. Clipping each head
            # independently let the product ratio drift far outside [1-clip,
            # 1+clip] (e.g. 4 heads each at 1.2 → joint 2.07), defeating the
            # trust region — which is why PPO couldn't refine the precise melee
            # sequences of champion/battle_master (it thrashed instead of taking
            # small reliable steps). Placeholders are equal (0) on new and old
            # for ended/inapplicable heads, so the per-head Δ there is 0 and the
            # sum is the true joint log-prob delta for both step types.
            ratio = (new_lp.sum(-1) - old_b.sum(-1)).exp()       # [B]
            policy_loss = -torch.min(ratio * adv_b,
                                     torch.clamp(ratio, 1-clip, 1+clip) * adv_b).mean()
            # Detach encoder for the value path so vf_coef * MSE doesn't
            # dominate encoder gradients. value_head still gets its full
            # signal; encoder is updated only by policy gradient.
            value_loss  = F.mse_loss(net.value(obs_b, detach_encoder=True), ret_b)

            # Optional KL anchor against a frozen reference policy (typically
            # the BC checkpoint we started from). Without it, PPO is free to
            # wander arbitrarily far from BC over many updates — small
            # policy_loss values like +0.005 still accumulate to a different
            # policy after 100 updates, which is what destroyed earlier runs.
            kl_total = torch.zeros((), device=end_l.device)
            if reference_net is not None and kl_coef > 0.0:
                with torch.no_grad():
                    ref_end_l, ref_skill_l, ref_entity_l, ref_grid_l = reference_net(obs_b)
                    ref_skill_l  = ref_skill_l.masked_fill(sk_mask_b, -1e9)
                    ref_entity_l = ref_entity_l.masked_fill(ent_mask_b.unsqueeze(1), -1e9)
                    ref_skill_l_play = ref_skill_l.clone()
                    ref_skill_l_play[..., 0] = -1e9
                    ref_cond_ent  = ref_entity_l[bidx, sk_chosen]
                    ref_cond_grid = ref_grid_l[bidx, sk_chosen]
                    ref_end_dist   = torch.distributions.Bernoulli(logits=ref_end_l)
                    ref_skill_dist = torch.distributions.Categorical(logits=ref_skill_l_play)
                    ref_ent_dist   = torch.distributions.Categorical(logits=ref_cond_ent)
                    ref_grid_dist  = torch.distributions.Categorical(logits=ref_cond_grid)
                kl_end   = torch.distributions.kl_divergence(end_dist, ref_end_dist)
                kl_skill = torch.distributions.kl_divergence(skill_dist, ref_skill_dist)
                kl_ent   = torch.distributions.kl_divergence(ent_dist, ref_ent_dist)
                kl_grid  = torch.distributions.kl_divergence(grid_dist, ref_grid_dist)
                # Ended steps had no policy choice on skill/ent/grid — mask out.
                kl_skill = kl_skill.masked_fill(ended_steps, 0.0)
                kl_ent   = kl_ent.masked_fill(ended_steps, 0.0)
                kl_grid  = kl_grid.masked_fill(ended_steps, 0.0)
                kl_total = (kl_end + kl_skill + kl_ent + kl_grid).mean()

            if value_only:
                loss = vf_coef * value_loss
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optim.step()
            elif pcgrad:
                # PCGrad: per-archetype policy-loss gradients → surgery
                # → set on .grad → add aux (value + entropy + KL) →
                # clip + step. Single forward already done above; we
                # reuse the cached ratio/adv tensors for per-arch slicing.
                mb_arch = arch_ids[sel]
                unique_archs = mb_arch.unique().tolist()
                per_arch_grads: list[torch.Tensor] = []
                ratio_full = (new_lp - old_b).exp()                        # [B, 4]
                adv_full   = adv_b.unsqueeze(-1)                           # [B, 1]
                for arch in unique_archs:
                    mask = (mb_arch == arch)
                    if mask.sum() < 2:
                        # Too few samples for a stable per-arch gradient
                        # — skip and let other archetypes drive update.
                        continue
                    r_a = ratio_full[mask]
                    a_a = adv_full[mask]
                    pl_a = -torch.min(r_a * a_a,
                                       torch.clamp(r_a, 1 - clip, 1 + clip) * a_a).mean()
                    optim.zero_grad()
                    pl_a.backward(retain_graph=True)
                    per_arch_grads.append(_flatten_grads(trainable_params).detach().clone())
                if len(per_arch_grads) >= 2:
                    proj = _pcgrad_project(per_arch_grads)
                elif len(per_arch_grads) == 1:
                    proj = per_arch_grads[0]
                else:
                    proj = None
                if proj is not None:
                    _unflatten_to_grads(trainable_params, proj)
                else:
                    optim.zero_grad()
                # Add auxiliary losses (value + entropy + optional KL)
                # without PCGrad: these don't cross-archetype-conflict by
                # the same mechanism (value is per-sample MSE, entropy is
                # a bonus). Accumulates onto existing .grad from surgery.
                aux = (vf_coef * value_loss
                       - ent_coef * entropy
                       + kl_coef * kl_total)
                aux.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optim.step()
            else:
                loss = (policy_loss
                        + vf_coef * value_loss
                        - ent_coef * entropy
                        + kl_coef * kl_total)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optim.step()

            p_losses.append(policy_loss.item())
            v_losses.append(value_loss.item())
            entropies.append(entropy.item())
            kl_vals.append(float(kl_total.item()))

    return {
        "policy_loss": float(np.mean(p_losses)),
        "value_loss":  float(np.mean(v_losses)),
        "entropy":     float(np.mean(entropies)),
        "kl":          float(np.mean(kl_vals)),
    }
