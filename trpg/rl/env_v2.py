
"""Phase 2 RL combat environment.

Dict observation, MultiDiscrete action. See
docs/superpowers/specs/2026-05-19-rl-env-v2-design.md for the full spec.

Phase 1 (env.py) stays untouched so we can A/B compare baselines.
"""
from __future__ import annotations
import random
from typing import Any

import numpy as np

from ..engine.world_state import WorldState, CombatState
from ..engine.combat import (
    execute_action, consume_resources, setup_combat_positions,
    MOVE_BUDGET_M, tick_terrain_damage,
)
from ..engine.combat_policy import make_archetype_policy
from ..engine.status import tick_status_effects
from ..scenarios.archetypes import ARCHETYPE_FACTORIES
from .obs import build_obs
from .action import decode_action, ACTION_DIMS
from ..engine.skill import available_skills
from ..engine.abilities import CLASS_ABILITIES


ARCHETYPE_LIST: tuple[str, ...] = tuple(ARCHETYPE_FACTORIES.keys())
LAYOUTS = ("open", "open", "walls", "difficult")

_MAX_SUB_ACTIONS_PER_TURN = 5
_MAX_AGENT_STEPS_PER_EPISODE = 50

_AGENT_ID = "agent"
_OPPONENT_ID = "opponent"


class CombatEnvV2:
    """Single-agent combat env with Dict obs and skill-based action space."""

    action_dims = ACTION_DIMS

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.ws: WorldState | None = None
        self.resources: dict = {}
        self._step_count = 0
        self.agent_arch: str = ""
        self.opponent_arch: str = ""
        self._opponent_policy = None
        # Persistent opponent-policy preference across reset(). One of:
        #   None    — use the archetype-specific expert policy (default)
        #   "heuristic"  — use HeuristicCombatPolicy
        #   net (callable) — drive opponent with this network (self-play)
        self._opponent_override = None

    def reset(self, *, agent_arch: str | None = None,
              opponent_arch: str | None = None,
              level: int | None = None,
              layout: str | None = None,
              seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = random.Random(seed)
        self.agent_arch = agent_arch or self._rng.choice(ARCHETYPE_LIST)
        self.opponent_arch = opponent_arch or self._rng.choice(ARCHETYPE_LIST)
        lvl = level if level is not None else self._rng.randint(3, 8)
        layout = layout or self._rng.choice(LAYOUTS)

        agent_char = ARCHETYPE_FACTORIES[self.agent_arch](level=lvl)
        agent_char.char_id = _AGENT_ID
        agent_char.is_npc = False

        opp_char = ARCHETYPE_FACTORIES[self.opponent_arch](level=lvl)
        opp_char.char_id = _OPPONENT_ID
        opp_char.is_npc = True
        opp_char.attitude = 0

        self.ws = WorldState(
            characters={_AGENT_ID: agent_char, _OPPONENT_ID: opp_char},
            scene="rl_phase2", pc_ids=[_AGENT_ID], party_ids=[_AGENT_ID],
        )
        self.ws.combat = CombatState(
            active=True, initiative_order=[_AGENT_ID, _OPPONENT_ID],
            round_number=1,
        )
        setup_combat_positions(self.ws, self.ws.combat)
        self._apply_layout(layout)

        if self._opponent_override is None:
            self._opponent_policy = make_archetype_policy(self.opponent_arch)
        elif self._opponent_override == "heuristic":
            from ..engine.combat_policy import HeuristicCombatPolicy
            self._opponent_policy = HeuristicCombatPolicy()
        else:
            # Network checkpoint — wrap it for the engine
            from .neural_policy import NeuralCombatPolicy
            self._opponent_policy = NeuralCombatPolicy(self._opponent_override, device="cpu")
        self.resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
        self._step_count = 0

        # Start-of-turn ticks for the agent
        agent_char.reaction_used = False
        tick_status_effects(agent_char, "self_turn_start", 1)
        tick_terrain_damage(agent_char, self.ws.combat.battlefield)

        return build_obs(self.ws, _AGENT_ID, self.resources), {}

    def use_heuristic_opponent(self) -> None:
        """Switch to weak HeuristicCombatPolicy across resets (curriculum)."""
        from ..engine.combat_policy import HeuristicCombatPolicy
        self._opponent_override = "heuristic"
        self._opponent_policy = HeuristicCombatPolicy()

    def use_self_play_opponent(self, net) -> None:
        """Drive the opponent with a network across resets (self-play)."""
        from .neural_policy import NeuralCombatPolicy
        self._opponent_override = net
        self._opponent_policy = NeuralCombatPolicy(net, device="cpu")

    def _apply_layout(self, layout: str) -> None:
        if layout == "open":
            return
        from ..engine.vec2 import TerrainType
        bf = self.ws.combat.battlefield
        if layout == "walls":
            bf.add_rect_obstacle(7.5, 11.5, 8.5, 13.5)
            bf.add_rect_obstacle(7.5, 16.5, 8.5, 18.5)
        elif layout == "difficult":
            bf.add_rect_terrain(8.0, 0.0, 10.0, 30.0, TerrainType.DIFFICULT)
        elif layout == "lava":
            bf.add_rect_terrain(13.5, 13.5, 16.5, 16.5, TerrainType.DANGEROUS)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self.ws is None:
            raise RuntimeError("call reset() before step()")
        self._step_count += 1

        agent = self.ws.characters[_AGENT_ID]
        opp = self.ws.characters[_OPPONENT_ID]

        # Snapshot the skill list BEFORE executing — execute_action may mutate
        # state (e.g. drain lay_on_hands_pool) and change which skills are
        # available, so skill_idx → skill_id must be resolved against the
        # pre-action list.
        skill_id_used: str | None = None
        skill_idx_val = int(action[0])
        skills_before = available_skills(agent, self.ws)
        if 0 <= skill_idx_val < len(skills_before):
            skill_id_used = skills_before[skill_idx_val].skill_id

        action_dict = decode_action(action, self.ws, _AGENT_ID)
        result: dict[str, Any] | None = None
        if action_dict is not None:
            result = execute_action(action_dict, self.ws)
            # Any ERROR here is a masking bug — the model picked an action the
            # network should have masked. Fail loudly so we fix it instead of
            # silently consuming resources and continuing.
            if result.get("type") == "ERROR":
                raise RuntimeError(
                    f"Action mask leak: skill_id={skill_id_used!r} "
                    f"action_type={action_dict.get('type')!r} "
                    f"msg={result.get('message')!r} "
                    f"resources={self.resources} "
                    f"action_triplet={list(action)}"
                )
            consume_resources(self.resources, action_dict, result)
            # If MOVE went nowhere (stuck) or remaining budget is sub-meter
            # (can't reach a different grid cell anyway), drain it. Otherwise
            # the resource mask `movement > 1e-6` lets MOVE stay selectable
            # with a useless 0.04 m budget, wasting the step.
            if (action_dict.get("type") == "MOVE"
                    and (result.get("distance", 0) < 0.01
                         or self.resources["movement"] < 0.5)):
                self.resources["movement"] = 0.0
            # Decrement use count for limited-use class abilities. The expert
            # parse_command path does this; decode_action skips it, so we deduct
            # here to keep both paths consistent.
            if skill_id_used is not None:
                ab = CLASS_ABILITIES.get(skill_id_used)
                if ab is not None and ab.max_uses > 0:
                    agent.ability_uses[skill_id_used] = (
                        agent.ability_uses.get(skill_id_used, ab.max_uses) - 1
                    )

        turn_done = (
            action_dict is None
            or self.resources["action"] <= 0
        )
        if turn_done:
            tick_status_effects(agent, "self_turn_end", self.ws.combat.round_number)
            if opp.is_alive():
                self._run_opponent_turn()
            self._end_of_round_tick()
            if agent.is_alive():
                self.resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
                agent.reaction_used = False
                tick_status_effects(agent, "self_turn_start", self.ws.combat.round_number)
                tick_terrain_damage(agent, self.ws.combat.battlefield)

        # Pure terminal reward — no per-step shaping.
        # +5 win, -5 loss, -10 on step-limit truncation (2× loss). Self-play
        # eval showed half of dodge-vs-dodge episodes hit the step cap, so the
        # truncation penalty is the dominant gradient signal. Making it worse
        # than losing forces the policy to decide the fight rather than stall.
        reward = 0.0
        terminated = (not agent.is_alive()) or (not opp.is_alive())
        if terminated:
            reward = 5.0 if not opp.is_alive() else -5.0
        truncated = self._step_count >= _MAX_AGENT_STEPS_PER_EPISODE
        if truncated and not terminated:
            reward = -10.0

        return (build_obs(self.ws, _AGENT_ID, self.resources),
                float(reward), terminated, truncated,
                {"action_result": result})

    def _run_opponent_turn(self) -> None:
        opp = self.ws.characters[_OPPONENT_ID]
        if not opp.is_alive():
            return
        opp.reaction_used = False
        tick_status_effects(opp, "self_turn_start", self.ws.combat.round_number)
        tick_terrain_damage(opp, self.ws.combat.battlefield)
        resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            if not opp.is_alive():
                break
            decision = self._opponent_policy.decide(
                _OPPONENT_ID, opp, self.ws, resources,
                self.ws.combat.round_number,
            )
            if decision.fled or decision.action is None:
                break
            r = execute_action(decision.action, self.ws)
            if r.get("type") != "ERROR":
                consume_resources(resources, decision.action, r)
            if decision.ended:
                break
            if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                    and resources["movement"] <= 1e-6):
                break
        tick_status_effects(opp, "self_turn_end", self.ws.combat.round_number)

    def _end_of_round_tick(self) -> None:
        cs = self.ws.combat
        # Tick round_end on all alive with the ROUND THAT JUST FINISHED
        for cid in cs.initiative_order:
            c = self.ws.characters.get(cid)
            if c and c.is_alive():
                tick_status_effects(c, "round_end", cs.round_number)
        cs.round_number += 1

