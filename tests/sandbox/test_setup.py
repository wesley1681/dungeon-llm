import pytest
from trpg.engine.vec2 import Vec2, TerrainType
from trpg.sandbox.setup import build_world_state


def test_build_world_state_default_positions():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker",
        level=5, agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
        terrain="empty",
    )
    assert "agent" in ws.characters
    assert "opponent" in ws.characters
    assert ws.characters["agent"].position == Vec2(5.0, 5.0)
    assert ws.characters["opponent"].position == Vec2(25.0, 25.0)
    assert ws.combat is not None
    assert ws.combat.active is True
    assert ws.combat.initiative_order == ["agent", "opponent"]


def test_agent_is_pc_opponent_is_npc():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker",
        level=5, agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
        terrain="empty",
    )
    assert ws.characters["agent"].is_npc is False
    assert ws.characters["opponent"].is_npc is True
    assert ws.characters["opponent"].attitude == 0  # hostile


def test_terrain_preset_applied():
    ws = build_world_state(
        agent_arch="evocation", opponent_arch="berserker",
        level=5, agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
        terrain="lava_strip",
    )
    bf = ws.combat.battlefield
    assert bf.terrain_at(Vec2(15.0, 15.0)) == TerrainType.DANGEROUS


def test_unknown_terrain_raises():
    with pytest.raises(KeyError):
        build_world_state(
            agent_arch="evocation", opponent_arch="berserker",
            level=5, agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
            terrain="quicksand",
        )


def test_unknown_archetype_raises():
    with pytest.raises(KeyError):
        build_world_state(
            agent_arch="dragon", opponent_arch="berserker",
            level=5, agent_pos=Vec2(5.0, 5.0), opp_pos=Vec2(25.0, 25.0),
            terrain="empty",
        )
