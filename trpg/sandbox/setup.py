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
    model sees the same shape it was trained on. Differences from env_v2.reset():
      - positions come from caller, not setup_combat_positions defaults
      - terrain is a named preset (env_v2 uses a hardcoded LAYOUTS list)
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


# ── Catalog + loadout helpers ─────────────────────────────────────────────────

from ..engine.abilities import CLASS_ABILITIES
from ..engine.items import WEAPON_DEFS


def list_catalog() -> dict[str, list[str]]:
    """Return the full skill catalog grouped by kind.

    - weapons:   keys of WEAPON_DEFS
    - abilities: keys of CLASS_ABILITIES restricted to engine_ready & not is_reaction
                 (reactions can't be triggered from the action menu in the sandbox).
                 Spells are ClassAbilities — they appear here alongside other abilities.
    """
    return {
        "weapons": sorted(WEAPON_DEFS.keys()),
        "abilities": sorted(
            sid for sid, ab in CLASS_ABILITIES.items()
            if ab.engine_ready and not ab.is_reaction
        ),
    }


def apply_loadout(char, *, add: list[str], remove: list[str]) -> None:
    """Mutate ``char`` to add and remove skills by string id.

    Resolution rules:
      - if id is in WEAPON_DEFS → weapon
      - elif id is in CLASS_ABILITIES → ability (also seeds ability_uses if max_uses > 0)
      - else → ValueError

    Spells are now ClassAbility entries (e.g. fireball_ev, cure_wounds), so use
    the ability id, not the spell display name.

    add and remove are processed in that order (remove last, so you can swap
    a weapon by name without a 0-length intermediate state).
    """
    for sid in add:
        if sid in WEAPON_DEFS:
            char.weapons.append(WEAPON_DEFS[sid])
        elif sid in CLASS_ABILITIES:
            ab = CLASS_ABILITIES[sid]
            if sid not in char.known_abilities:
                char.known_abilities.append(sid)
            if ab.max_uses > 0:
                char.ability_uses[sid] = ab.max_uses
        else:
            raise ValueError(f"unknown skill id: {sid!r}")

    for sid in remove:
        if sid in WEAPON_DEFS and any(w.name == sid for w in char.weapons):
            char.weapons[:] = [w for w in char.weapons if w.name != sid]
        elif sid in CLASS_ABILITIES and sid in char.known_abilities:
            char.known_abilities.remove(sid)
            char.ability_uses.pop(sid, None)
        else:
            raise ValueError(f"unknown skill id: {sid!r}")
