"""Single source of truth for the trained combat model that drives NPCs
(allies + monsters) in the narrative game.

The narrative game's combat is fully mechanical (engine/combat.py) and decides
each combatant's turn through a `CombatPolicy`. `NeuralCombatPolicy` already
conforms to that protocol, so plugging the general self-play model in is just
"make it the default policy for every non-human combatant". This module is the
ONE place that names the checkpoint path — front-ends call `load_combat_policy()`
and pass the result to `GameSession(default_combat_policy=...)`.
"""
from __future__ import annotations
from pathlib import Path

# The ONE physical location naming the combat model. Change here, everywhere.
DEFAULT_COMBAT_MODEL = "models/general_selfplay/mt_u0300.pt"


def load_combat_policy(path: str | None = None):
    """Load the general combat net as ONE shared `NeuralCombatPolicy` that drives
    every NPC (it is stateless per-actor — `decide(actor_id, ...)` — so a single
    instance serves all allies and monsters).

    Returns None if the checkpoint is absent, so the game falls back gracefully
    to the scripted heuristic instead of crashing.
    """
    from .architectures import load_net
    from .neural_policy import NeuralCombatPolicy
    from ..scenarios.monsters import register_monsters

    register_monsters()   # ensure monster factories/kits exist for obs encoding
    p = Path(path or DEFAULT_COMBAT_MODEL)
    if not p.exists():
        return None
    return NeuralCombatPolicy(load_net(str(p)))
