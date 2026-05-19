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


ARCHETYPE_LIST: tuple[str, ...] = tuple(ARCHETYPE_FACTORIES.keys())
LAYOUTS = ("open", "open", "walls", "difficult", "lava")

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

        self._opponent_policy = make_archetype_policy(self.opponent_arch)
        self.resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
        self._step_count = 0

        # Start-of-turn ticks for the agent
        agent_char.reaction_used = False
        tick_status_effects(agent_char, "self_turn_start", 1)
        tick_terrain_damage(agent_char, self.ws.combat.battlefield)

        return build_obs(self.ws, _AGENT_ID, self.resources), {}

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
            bf.add_rect_terrain(9.0, 13.5, 12.0, 16.5, TerrainType.DANGEROUS)

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self.ws is None:
            raise RuntimeError("call reset() before step()")
        self._step_count += 1

        agent = self.ws.characters[_AGENT_ID]
        opp = self.ws.characters[_OPPONENT_ID]
        prev_agent_hp = agent.hp
        prev_opp_hp = opp.hp

        action_dict = decode_action(action, self.ws, _AGENT_ID)
        result: dict[str, Any] | None = None
        if action_dict is not None:
            result = execute_action(action_dict, self.ws)
            if result.get("type") != "ERROR":
                consume_resources(self.resources, action_dict, result)

        turn_done = (
            action_dict is None
            or (self.resources["action"] <= 0
                and self.resources["bonus_action"] <= 0
                and self.resources["movement"] <= 1e-6)
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

        reward = self._compute_reward(prev_agent_hp, prev_opp_hp,
                                       action_dict, result)
        terminated = (not agent.is_alive()) or (not opp.is_alive())
        if terminated:
            reward += 5.0 if not opp.is_alive() else -5.0
        truncated = self._step_count >= _MAX_AGENT_STEPS_PER_EPISODE

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

    def _compute_reward(self, prev_agent_hp: int, prev_opp_hp: int,
                        action_dict: dict | None, result: dict | None) -> float:
        agent = self.ws.characters[_AGENT_ID]
        opp = self.ws.characters[_OPPONENT_ID]
        dmg_dealt = max(0, prev_opp_hp - opp.hp)
        dmg_taken = max(0, prev_agent_hp - agent.hp)
        reward = dmg_dealt / max(1, opp.max_hp) - dmg_taken / max(1, agent.max_hp)
        reward -= 0.01

        # Skill-use bonus
        if action_dict is not None and result is not None:
            t = action_dict.get("type", "")
            if t not in ("MOVE", "ERROR") and result.get("type") != "ERROR":
                reward += 0.05   # flat bonus per non-trivial action
        return reward
