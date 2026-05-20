"""Collect (obs, action) rollouts for behaviour cloning.

For each archetype, run the matching expert policy as the agent and record
every (obs, encoded_action) the policy emits. Opponent stays scripted
(another archetype) — what we want to clone is the agent-side decision.
"""
from __future__ import annotations
import numpy as np

from ..engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
)
from ..engine.combat_policy import make_archetype_policy
from ..engine.status import tick_status_effects
from .env_v2 import (
    CombatEnvV2, ARCHETYPE_LIST,
    _AGENT_ID, _OPPONENT_ID, _MAX_SUB_ACTIONS_PER_TURN,
)
from .obs import build_obs
from .action import encode_action
from ..engine.skill import available_skills, TargetType


# Sentinel target_type for pairs that don't correspond to a skill choice
# (end-of-turn pairs). Distinct from any TargetType enum value.
TT_END = -1


def collect_bc_rollout(agent_arch: str, opponent_arch: str | None = None,
                        level: int = 5, seed: int = 0,
                        max_rounds: int = 30,
                        ) -> list[tuple[dict, tuple, int]]:
    """Run one BC episode. Returns list of (obs_dict, action_triplet, target_type).

    ``target_type`` is the chosen skill's target_type (or TT_END for
    end-of-turn pairs). Callers use it to mask per-head losses during
    training — entity_head only sees entity-targeted skills, grid_head
    only sees POINT/LINE/CONE skills.

    End pairs (obs when expert returns action=None) are always recorded as
    ``(obs, (0, 0, 0), TT_END)``. The multi-head architecture's end_head
    needs them to learn when to stop, and the skill/entity/grid heads route
    around them via target_type masking — so they no longer pollute skill
    learning the way they did before the heads were split.
    """
    env = CombatEnvV2(seed=seed)
    env.reset(agent_arch=agent_arch, opponent_arch=opponent_arch, level=level)
    expert = make_archetype_policy(agent_arch)
    pairs: list[tuple[dict, tuple, int]] = []

    while env.ws.combat.round_number <= max_rounds:
        agent = env.ws.characters[_AGENT_ID]
        opp = env.ws.characters[_OPPONENT_ID]
        if not agent.is_alive() or not opp.is_alive():
            break

        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            obs_now = build_obs(env.ws, _AGENT_ID, env.resources)
            decision = expert.decide(
                _AGENT_ID, agent, env.ws, env.resources,
                env.ws.combat.round_number,
            )
            if decision.action is None or decision.fled:
                if not decision.fled:
                    pairs.append((obs_now, (0, 0, 0), TT_END))
                break
            enc = encode_action(decision.action, env.ws, _AGENT_ID)
            if enc[0] >= 0:
                # Look up target_type from the chosen skill so the training
                # loss can route entity/grid supervision only to relevant
                # samples.
                skills = available_skills(agent, env.ws)
                tt = int(skills[enc[0]].features.target_type)
                pairs.append((obs_now, enc, tt))
            if decision.ended:
                break
            r = execute_action(decision.action, env.ws)
            if r.get("type") != "ERROR":
                consume_resources(env.resources, decision.action, r)
            if (env.resources["action"] <= 0
                and env.resources["bonus_action"] <= 0
                and env.resources["movement"] <= 1e-6):
                break

        # End-of-agent-turn
        tick_status_effects(agent, "self_turn_end", env.ws.combat.round_number)
        if opp.is_alive():
            env._run_opponent_turn()
        env._end_of_round_tick()
        if agent.is_alive():
            env.resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
            agent.reaction_used = False
            agent.leveled_spell_cast_this_turn = False
            tick_status_effects(agent, "self_turn_start", env.ws.combat.round_number)
            tick_terrain_damage(agent, env.ws.combat.battlefield)
    return pairs


def collect_bc_dataset(n_episodes_per_arch: int = 50, seed: int = 0) -> dict:
    """Roll out every archetype against every other; return a flat dataset.

    Returns dict with keys:
      obs:           {feature_name: ndarray}
      actions:       (N, 3) int64 — [skill_idx, entity_idx, grid_cell]
      target_types:  (N,)   int64 — TargetType of chosen skill, or TT_END (-1)
    """
    rng = np.random.default_rng(seed)
    all_pairs: list[tuple[dict, tuple, int]] = []
    for agent_arch in ARCHETYPE_LIST:
        for ep in range(n_episodes_per_arch):
            opp_arch = ARCHETYPE_LIST[rng.integers(len(ARCHETYPE_LIST))]
            level = int(rng.integers(3, 9))
            ep_seed = int(rng.integers(0, 2**31))
            all_pairs.extend(collect_bc_rollout(
                agent_arch, opp_arch, level, ep_seed,
            ))

    # Stack into arrays
    obs_keys = list(all_pairs[0][0].keys())
    obs_stacked = {k: np.stack([p[0][k] for p in all_pairs], axis=0) for k in obs_keys}
    actions = np.array([p[1] for p in all_pairs], dtype=np.int64)
    target_types = np.array([p[2] for p in all_pairs], dtype=np.int64)
    return {"obs": obs_stacked, "actions": actions, "target_types": target_types}
