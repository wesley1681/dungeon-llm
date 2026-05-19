import numpy as np
import pytest
from trpg.engine.character import Character, Stats
from trpg.engine.world_state import WorldState, CombatState
from trpg.engine.combat import setup_combat_positions
from trpg.engine.items import WEAPON_DEFS
from trpg.rl.action import decode_action, ACTION_DIMS


def _world():
    a = Character(name="a", race="人類", class_="戰士", level=3,
                  stats=Stats(STR=14, DEX=12, CON=14),
                  hp=24, max_hp=24, ac=14,
                  weapons=[WEAPON_DEFS["長劍"]], is_npc=False)
    b = Character(name="b", race="哥布林", class_="戰士", level=1,
                  stats=Stats(STR=10, DEX=12),
                  hp=10, max_hp=10, ac=12,
                  weapons=[WEAPON_DEFS["短劍"]], is_npc=True, attitude=0)
    ws = WorldState(characters={"a": a, "b": b}, scene="test",
                    pc_ids=["a"], party_ids=["a"])
    ws.combat = CombatState(active=True, initiative_order=["a","b"], round_number=1)
    setup_combat_positions(ws, ws.combat)
    return ws


def test_action_dims_shape():
    assert ACTION_DIMS == (20, 6, 400)


def test_decode_skill_0_returns_none_end_turn():
    ws = _world()
    out = decode_action([0, 0, 0], ws, "a")
    assert out is None   # end turn


def test_decode_skill_out_of_range_returns_none():
    ws = _world()
    # slot 19 is past available_skills() length → invalid
    out = decode_action([19, 0, 0], ws, "a")
    assert out is None


def test_decode_attack_targets_entity_slot_3():
    # slot 3 in entities is enemy_1. We need to find the weapon skill index.
    from trpg.engine.skill import available_skills
    ws = _world()
    skills = available_skills(ws.characters["a"], ws)
    weapon_idx = next(
        i for i, s in enumerate(skills) if s.skill_id.startswith("weapon:")
    )
    out = decode_action([weapon_idx, 3, 0], ws, "a")
    assert out is not None
    assert out["type"] == "ATTACK"
    assert out["attacker"] == "a"
    assert out["target"] == "b"
