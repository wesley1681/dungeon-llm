import numpy as np
import torch
from trpg.rl.train_bc import train_bc, bc_loss_step
from trpg.rl.model import CombatPolicyNet
from trpg.engine.skill import SKILL_FEATURE_DIM
from trpg.rl.obs import (ENTITY_DIM, N_ENTITY_SLOTS, N_GRID,
                          N_ENTITY_GRID_CHANNELS,
                          N_DISTANCE_GRID_CHANNELS, END_FEATURES_DIM,
                          N_DECISION_CTX)


def test_bc_loss_step_returns_scalar():
    net = CombatPolicyNet()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Tiny fake batch
    obs = {
        "skills":        torch.randn(4, 20, SKILL_FEATURE_DIM),
        "skill_mask":    torch.ones(4, 20),
        "entities":      torch.randn(4, N_ENTITY_SLOTS, ENTITY_DIM),
        "resources":     torch.randn(4, 4),
        "terrain":       torch.randn(4, N_GRID, N_GRID),
        "entity_grid":   torch.randn(4, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID),
        "distance_grid": torch.randn(4, N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID),
        "end_features":  torch.randn(4, END_FEATURES_DIM),
        "decision_context": torch.zeros(4, N_DECISION_CTX),
    }
    actions = torch.tensor([[1, 3, 50], [2, 0, 100], [0, 3, 0], [3, 4, 200]])
    # All four target_types: end (-1), entity-targeted (1), grid-targeted (5), self (0)
    target_types = torch.tensor([1, 5, -1, 0])
    loss, acc = bc_loss_step(net, obs, actions, target_types, optim)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert set(acc.keys()) == {"end", "skill", "entity", "grid"}


def test_train_bc_returns_history():
    # Tiny dataset
    obs_batch = {
        "skills":        np.zeros((10, 20, SKILL_FEATURE_DIM), dtype=np.float32),
        "skill_mask":    np.ones((10, 20), dtype=np.float32),
        "entities":      np.zeros((10, N_ENTITY_SLOTS, ENTITY_DIM), dtype=np.float32),
        "resources":     np.zeros((10, 4), dtype=np.float32),
        "terrain":       np.zeros((10, N_GRID, N_GRID), dtype=np.float32),
        "entity_grid":   np.zeros((10, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32),
        "distance_grid": np.zeros((10, N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID), dtype=np.float32),
        "end_features":  np.zeros((10, END_FEATURES_DIM), dtype=np.float32),
        "decision_context": np.zeros((10, N_DECISION_CTX), dtype=np.float32),
    }
    actions = np.random.randint(0, 5, (10, 3)).astype(np.int64)
    target_types = np.random.randint(-1, 6, (10,)).astype(np.int64)
    dataset = {"obs": obs_batch, "actions": actions, "target_types": target_types}
    history = train_bc(dataset, epochs=2, batch_size=4, lr=1e-3, device="cpu")
    assert "losses" in history
    assert len(history["losses"]) >= 2
