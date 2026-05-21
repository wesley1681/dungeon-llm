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


from trpg.sandbox.setup import list_catalog, apply_loadout
from trpg.engine.abilities import ABILITY_REGISTRY


def test_list_catalog_returns_two_buckets():
    cat = list_catalog()
    assert set(cat) == {"weapons", "abilities"}
    # Spells live in the abilities bucket now (their Ability ids)
    assert "fireball_ev" in cat["abilities"]
    assert any(w == "長劍" for w in cat["weapons"])
    # Only engine-ready, non-reaction abilities should appear
    for sid in cat["abilities"]:
        ab = ABILITY_REGISTRY[sid]
        assert ab.engine_ready and not ab.is_reaction


def test_apply_loadout_adds_spell_ability():
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    char = ARCHETYPE_FACTORIES["berserker"](level=5)
    initial = list(char.known_abilities)
    apply_loadout(char, add=["fireball_ev"], remove=[])
    assert "fireball_ev" in char.known_abilities
    assert len(char.known_abilities) == len(initial) + 1


def test_apply_loadout_removes_weapon():
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    char = ARCHETYPE_FACTORIES["champion"](level=5)
    # Champion has at least one weapon by default
    assert char.weapons
    first_name = char.weapons[0].name
    apply_loadout(char, add=[], remove=[first_name])
    assert all(w.name != first_name for w in char.weapons)


def test_apply_loadout_adds_ability_with_uses():
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    char = ARCHETYPE_FACTORIES["evocation"](level=5)
    apply_loadout(char, add=["rage"], remove=[])
    assert "rage" in char.known_abilities
    # rage has finite max_uses → should be seeded in ability_uses
    rage = ABILITY_REGISTRY["rage"]
    if rage.max_uses > 0:
        assert char.ability_uses.get("rage") == rage.max_uses


def test_apply_loadout_unknown_skill_raises():
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    char = ARCHETYPE_FACTORIES["evocation"](level=5)
    with pytest.raises(ValueError, match="unknown skill"):
        apply_loadout(char, add=["wibblefrobnitz"], remove=[])


def test_apply_loadout_remove_absent_weapon_raises():
    from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
    char = ARCHETYPE_FACTORIES["evocation"](level=5)  # wizard, no longsword
    # 長劍 (longsword) is in WEAPON_DEFS but not on a wizard
    with pytest.raises(ValueError, match="unknown skill"):
        apply_loadout(char, add=[], remove=["長劍"])
