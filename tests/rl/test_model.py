import numpy as np
import torch
from trpg.rl.model import CombatPolicyNet
from trpg.rl.obs import N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID
from trpg.rl.obs import N_ENTITY_GRID_CHANNELS
from trpg.engine.skill import SKILL_FEATURE_DIM


def _fake_obs_batch(batch=2):
    return {
        "skills":      torch.zeros(batch, N_SKILL_SLOTS, SKILL_FEATURE_DIM),
        "skill_mask":  torch.ones(batch, N_SKILL_SLOTS),
        "entities":    torch.zeros(batch, N_ENTITY_SLOTS, ENTITY_DIM),
        "resources":   torch.zeros(batch, 4),
        "terrain":     torch.zeros(batch, N_GRID, N_GRID),
        "entity_grid": torch.zeros(batch, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID),
    }


def test_model_forward_output_shapes():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=4)
    end_logit, skill_logits, entity_logits, grid_logits = net(obs)
    assert end_logit.shape == (4,)
    assert skill_logits.shape == (4, N_SKILL_SLOTS)
    assert entity_logits.shape == (4, N_SKILL_SLOTS, N_ENTITY_SLOTS)
    assert grid_logits.shape == (4, N_SKILL_SLOTS, N_GRID * N_GRID)


def test_model_skill_mask_zeros_padding():
    net = CombatPolicyNet()
    obs = _fake_obs_batch(batch=2)
    obs["skill_mask"][:, 5:] = 0   # only first 5 valid
    _, skill_logits, _, _ = net(obs)
    # Padded slots should have very negative logits
    assert (skill_logits[:, 5:] < -1e8).all()


def test_skill_ent_attn_proj_exists():
    net = CombatPolicyNet()
    assert hasattr(net, "skill_ent_attn_proj"), "skill_ent_attn_proj layer missing"
    assert net.skill_ent_attn_proj.in_features == 64
    assert net.skill_ent_attn_proj.out_features == 32


def test_skill_head_input_size():
    """skill_head must accept sk_emb(64) + h(128) + sk_ent_ctx(32) = 224 dims."""
    net = CombatPolicyNet()
    assert net.skill_head.in_features == 64 + 128 + 32, (
        f"Expected 224, got {net.skill_head.in_features}"
    )
