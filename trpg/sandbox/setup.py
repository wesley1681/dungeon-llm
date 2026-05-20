"""World setup helpers for the sparring sandbox."""
from __future__ import annotations

from ..engine.combat import setup_combat_positions
from ..engine.vec2 import Vec2, Battlefield
from ..engine.world_state import WorldState
from ..engine.character import CombatState
from ..scenarios.archetypes import ARCHETYPE_FACTORIES
from .terrain import TERRAIN_PRESETS


_DEFAULT_BATTLEFIELD = (30.0, 30.0, 0.5)   # width m, height m, grid m


def build_world_state(*, agent_arch: str, opponent_arch: str, level: int,
                      agent_pos: Vec2, opp_pos: Vec2, terrain: str
                      ) -> WorldState:
    """Construct a ready-to-fight WorldState with custom positions + terrain.

    Mirrors env_v2.reset()'s wiring (char_id, is_npc, initiative_order) so the
    model sees the same shape it was trained on. Differences:
      - agent is a PC (is_npc=False) — the user controls them
      - positions come from caller, not setup_combat_positions defaults
      - terrain is a named preset
    """
    if agent_arch not in ARCHETYPE_FACTORIES:
        raise KeyError(f"unknown agent archetype: {agent_arch!r}")
    if opponent_arch not in ARCHETYPE_FACTORIES:
        raise KeyError(f"unknown opponent archetype: {opponent_arch!r}")
    if terrain not in TERRAIN_PRESETS:
        raise KeyError(f"unknown terrain preset: {terrain!r}")

    agent = ARCHETYPE_FACTORIES[agent_arch](level=level)
    agent.char_id = "agent"
    agent.is_npc = False

    opp = ARCHETYPE_FACTORIES[opponent_arch](level=level)
    opp.char_id = "opponent"
    opp.is_npc = True
    opp.attitude = 0

    ws = WorldState(
        characters={"agent": agent, "opponent": opp},
        scene="sparring", pc_ids=["agent"], party_ids=["agent"],
    )
    w, h, g = _DEFAULT_BATTLEFIELD
    bf = Battlefield(width=w, height=h, grid_resolution=g)
    ws.combat = CombatState(
        active=True, initiative_order=["agent", "opponent"], round_number=1,
        battlefield=bf,
    )
    # Run the engine's standard placement so other state (e.g. reaction_used)
    # initialises, then overwrite with user-specified positions.
    setup_combat_positions(ws, ws.combat)
    agent.position = agent_pos
    opp.position = opp_pos
    TERRAIN_PRESETS[terrain](bf)
    return ws
