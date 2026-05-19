import numpy as np
import torch
from trpg.rl.model import CombatPolicyNet
from trpg.rl.obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.engine.skill import SKILL_FEATURE_DIM


def _fake_obs_batch(batch=2):
    return {
        "skills":     torch.zeros(batch, N_SKILL_SLOTS, SKILL_FEATURE_DIM),
        "skill_mask": torch.ones(batch, N_SKILL_SLOTS),
        "entities":   torch.zeros(batch, N_ENTITY_SLOTS, ENTITY_DIM),
        "resources":  torch.zeros(batch, 4),
        "terrain":    torch.zeros(batch, N_GRID, N_GRID),
    }


def test_model_forward_output_shapes():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=4)
    skill_logits, entity_logits, grid_logits = net(obs)
    assert skill_logits.shape == (4, N_SKILL_SLOTS)
    assert entity_logits.shape == (4, N_ENTITY_SLOTS)
    assert grid_logits.shape == (4, N_GRID * N_GRID)


def test_model_skill_mask_zeros_padding():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=2)
    obs["skill_mask"][:, 5:] = 0   # only first 5 valid
    skill_logits, _, _ = net(obs)
    # Padded slots should have very negative logits
    assert (skill_logits[:, 5:] < -1e8).all()
