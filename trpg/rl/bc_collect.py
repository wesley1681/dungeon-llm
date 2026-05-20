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


def collect_bc_rollout(agent_arch: str, opponent_arch: str | None = None,
                        level: int = 5, seed: int = 0,
                        max_rounds: int = 30) -> list[tuple[dict, tuple]]:
    """Run one BC episode. Returns list of (obs_dict, action_triplet)."""
    env = CombatEnvV2(seed=seed)
    env.reset(agent_arch=agent_arch, opponent_arch=opponent_arch, level=level)
    expert = make_archetype_policy(agent_arch)
    pairs: list[tuple[dict, tuple]] = []

    while env.ws.combat.round_number <= max_rounds:
        agent = env.ws.characters[_AGENT_ID]
        opp = env.ws.characters[_OPPONENT_ID]
        if not agent.is_alive() or not opp.is_alive():
            break

        # Record up to _MAX_SUB_ACTIONS_PER_TURN agent sub-actions. Skip the
        # turn-terminating pair: when expert returns action=None (or ended=True
        # with no action), it's signaling "I'm done", not making a skill
        # choice. Recording those collapses BC onto skill_idx=0 (the "end"
        # skill) — measured 84% end-spam in bc_v4. Engine-side masking still
        # lets the policy end its turn when no other skill is feasible.
        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            obs_now = build_obs(env.ws, _AGENT_ID, env.resources)
            decision = expert.decide(
                _AGENT_ID, agent, env.ws, env.resources,
                env.ws.combat.round_number,
            )
            if decision.action is None or decision.fled:
                break
            enc = encode_action(decision.action, env.ws, _AGENT_ID)
            # Drop unmatched encodings (skill_idx < 0) — better no signal
            # than wrong signal pointing the policy at skill 0 (end-turn).
            if enc[0] >= 0:
                pairs.append((obs_now, enc))
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
            tick_status_effects(agent, "self_turn_start", env.ws.combat.round_number)
            tick_terrain_damage(agent, env.ws.combat.battlefield)
    return pairs


def collect_bc_dataset(n_episodes_per_arch: int = 50, seed: int = 0) -> dict:
    """Roll out every archetype against every other; return a flat dataset."""
    rng = np.random.default_rng(seed)
    all_pairs: list[tuple[dict, tuple]] = []
    for agent_arch in ARCHETYPE_LIST:
        for ep in range(n_episodes_per_arch):
            opp_arch = ARCHETYPE_LIST[rng.integers(len(ARCHETYPE_LIST))]
            level = int(rng.integers(3, 9))
            ep_seed = int(rng.integers(0, 2**31))
            all_pairs.extend(collect_bc_rollout(agent_arch, opp_arch, level, ep_seed))

    # Stack into arrays
    obs_keys = list(all_pairs[0][0].keys())
    obs_stacked = {k: np.stack([p[0][k] for p in all_pairs], axis=0) for k in obs_keys}
    actions = np.array([p[1] for p in all_pairs], dtype=np.int64)
    return {"obs": obs_stacked, "actions": actions}
