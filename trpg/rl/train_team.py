"""Team-mode PPO collection: train ONE slot of a 3v3 template party.

The learner controls agent_0 (its archetype is fixed); the two teammates are
driven by FROZEN nets (greedy pick_action, exactly like deployment routing);
opponents are the env's scripted experts. Each episode samples teammates and
the opponent comp from the role template (1 front + 1 striker + 1 support).

Only the learner's decisions enter the PPO batch. Reward stream: every
env.step reward that occurs between two learner decisions (teammate turns,
opponent turns, terminal bonus) is accumulated onto the learner's most recent
transition — this keeps the team-PBRS telescoping sum intact (the learner is
credited with all Φ changes that follow its action until it acts again, which
is the standard semi-MDP construction for turn-based games).

No new reward terms here: team PBRS + WASTED_MOVE_COST from env_v2 as-is.
"""
from __future__ import annotations
import random

import numpy as np
import torch

from .env_v2 import CombatEnvV2
from .model import CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action
from .train_ppo import _sample_action, _stack_obs, _compute_gae
from ..scenarios.archetypes import ARCHETYPE_ROLES


# Cross-side level-gap sampling for asymmetric-encounter training. Weighted
# toward symmetric (the regression-guarded regime stays the majority diet);
# the ±1..±3 tails expose the critic/policy to ahead/behind states so the
# learning signal in unbalanced fights is calibrated instead of being averaged
# into the symmetric baseline. Same philosophy as env_v2._TEAM_CONFIGS.
LEVEL_DELTA_TABLE: tuple[tuple[int, float], ...] = (
    (0, 4.0), (-1, 2.0), (1, 2.0), (-2, 1.0), (2, 1.0), (-3, 0.5), (3, 0.5),
)


def sample_opp_level(rng: random.Random, agent_level: int) -> int:
    """Opponent level = agent level + weighted ΔL, clamped to the engine's
    supported range (factories are exercised at L2..L8)."""
    deltas = [d for d, _ in LEVEL_DELTA_TABLE]
    weights = [w for _, w in LEVEL_DELTA_TABLE]
    dl = rng.choices(deltas, weights=weights, k=1)[0]
    return max(2, min(8, agent_level + dl))


def sample_comp(rng: random.Random, fixed: dict[int, str] | None = None) -> list[str]:
    """Random template comp [front, striker, support]; ``fixed`` pins slots.

    Buckets filter to STANDARD_ARCHETYPES: ARCHETYPE_ROLES is a LIVE registry
    that runtime registration (monsters/synths) extends, and team comps must
    stay the standard 12 regardless of what else the process registered."""
    from ..scenarios.archetypes import STANDARD_ARCHETYPES
    buckets = (
        sorted(a for a, r in ARCHETYPE_ROLES.items()
               if r == "front" and a in STANDARD_ARCHETYPES),
        sorted(a for a, r in ARCHETYPE_ROLES.items()
               if r == "striker" and a in STANDARD_ARCHETYPES),
        sorted(a for a, r in ARCHETYPE_ROLES.items()
               if r == "support" and a in STANDARD_ARCHETYPES),
    )
    comp = [rng.choice(b) for b in buckets]
    for slot, arch in (fixed or {}).items():
        comp[slot] = arch
    return comp


def _mate_action(net: CombatPolicyNet, obs: dict, env: CombatEnvV2,
                 agent_id: str) -> list[int]:
    """Greedy frozen-teammate action — identical to the eval/deployment path."""
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, agent_id)
    e = apply_entity_mask(e, ot, env.ws, agent_id)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=agent_id))


def collect_team_rollout(net: CombatPolicyNet,
                         mate_nets: dict[str, CombatPolicyNet],
                         trained_arch: str,
                         trained_slot: int,
                         n_steps: int = 2048,
                         device: str = "cpu",
                         seed: int = 0,
                         gamma: float = 0.99,
                         gae_lambda: float = 0.95,
                         asym_levels: bool = False) -> dict:
    """Collect ``n_steps`` LEARNER transitions of 3v3 template-party play.

    ``mate_nets``: archetype -> frozen net for every non-trained archetype
    that can appear in a comp. ``trained_slot``: 0=front 1=striker 2=support
    (which template slot the trained archetype occupies).
    ``asym_levels``: sample a cross-side level gap per episode from
    LEVEL_DELTA_TABLE (symmetric-majority); False keeps the historical
    symmetric-levels behaviour.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    net.to(device).eval()
    rng = random.Random(seed)

    env = CombatEnvV2(seed=seed, n_agents=3, n_opps=3)

    def fresh_episode():
        comp = sample_comp(rng, fixed={trained_slot: trained_arch})
        opp_comp = sample_comp(rng)
        if asym_levels:
            lvl = rng.randint(3, 8)
            return env.reset(agent_archs=comp, opp_archs=opp_comp,
                             level=lvl,
                             opp_level=sample_opp_level(rng, lvl))[0]
        return env.reset(agent_archs=comp, opp_archs=opp_comp)[0]

    obs = fresh_episode()
    learner_id = env.agent_ids[trained_slot]

    obs_list, action_list, lp_list, reward_list, value_list, done_list = \
        [], [], [], [], [], []
    skill_mask_list: list[np.ndarray] = []
    entity_mask_list: list[np.ndarray] = []
    grid_mask_list: list[np.ndarray] = []
    open_idx: int | None = None       # learner's latest transition this episode

    while len(reward_list) < n_steps:
        actor = env.current_agent_id
        if actor == learner_id:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                action, log_prob, val, sk_mask, ent_mask, gr_mask = \
                    _sample_action(net, obs_t, env.resources, env.ws, actor)
            a_np = action.cpu().numpy().tolist()
            obs2, reward, term, trunc, _ = env.step(a_np)

            open_idx = len(reward_list)
            obs_list.append(obs)
            action_list.append(a_np)
            lp_list.append(log_prob.cpu().numpy())
            skill_mask_list.append(sk_mask.cpu().numpy())
            entity_mask_list.append(ent_mask.cpu().numpy())
            grid_mask_list.append(gr_mask.cpu().numpy())
            reward_list.append(float(reward))
            value_list.append(val)
            done_list.append(bool(term or trunc))
        else:
            arch = dict(zip(env.agent_ids, env.agent_archs))[actor]
            mate = mate_nets[arch]
            a_np = _mate_action(mate, obs, env, actor)
            obs2, reward, term, trunc, _ = env.step(a_np)
            if open_idx is not None:
                # Credit the learner's last decision with everything that
                # happened since (keeps PBRS telescoping intact).
                reward_list[open_idx] += float(reward)
                if term or trunc:
                    done_list[open_idx] = True

        if term or trunc:
            obs = fresh_episode()
            learner_id = env.agent_ids[trained_slot]
            open_idx = None
        else:
            obs = obs2

    with torch.no_grad():
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                 for k, v in obs.items()}
        next_value = net.value(obs_t).item() if not done_list[-1] else 0.0

    rewards = np.array(reward_list, dtype=np.float32)
    values  = np.array(value_list, dtype=np.float32)
    dones   = np.array(done_list, dtype=np.float32)
    adv, ret = _compute_gae(rewards, values, dones, next_value, gamma, gae_lambda)
    return {
        "obs":          _stack_obs(obs_list),
        "actions":      np.array(action_list, dtype=np.int64),
        "log_probs":    np.array(lp_list, dtype=np.float32),
        "skill_masks":  np.stack(skill_mask_list, axis=0).astype(np.bool_),
        "entity_masks": np.stack(entity_mask_list, axis=0).astype(np.bool_),
        "grid_masks":   np.stack(grid_mask_list, axis=0).astype(np.bool_),
        "rewards":      rewards, "values": values,
        "returns":      ret, "advantages": adv, "dones": dones,
    }
