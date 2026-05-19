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
