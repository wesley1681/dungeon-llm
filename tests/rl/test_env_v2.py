import numpy as np
import pytest
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.obs import OBS_KEYS, N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.engine.skill import SKILL_FEATURE_DIM


def test_archetype_list_has_12_entries():
    assert len(ARCHETYPE_LIST) == 12


def test_reset_returns_dict_obs():
    env = CombatEnvV2(seed=0)
    obs, info = env.reset()
    for k in OBS_KEYS:
        assert k in obs


def test_reset_obs_shapes():
    env = CombatEnvV2(seed=0)
    obs, _ = env.reset()
    assert obs["skills"].shape == (N_SKILL_SLOTS, SKILL_FEATURE_DIM)
    assert obs["skill_mask"].shape == (N_SKILL_SLOTS,)
    assert obs["entities"].shape == (N_ENTITY_SLOTS, ENTITY_DIM)
    assert obs["resources"].shape == (4,)
    assert obs["terrain"].shape == (N_GRID, N_GRID)


def test_reset_with_fixed_archetype():
    env = CombatEnvV2(seed=0)
    env.reset(agent_arch="battle_master", opponent_arch="evocation", level=5)
    assert env.ws.characters["agent"].archetype_id == "battle_master"
    assert env.ws.characters["opponent"].archetype_id == "evocation"


def test_reset_seed_is_reproducible():
    env1 = CombatEnvV2(seed=42)
    env2 = CombatEnvV2(seed=42)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    np.testing.assert_array_equal(obs1["entities"], obs2["entities"])
