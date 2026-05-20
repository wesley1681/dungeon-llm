import numpy as np
import torch
from trpg.rl.train_bc import train_bc, bc_loss_step
from trpg.rl.model import CombatPolicyNet
from trpg.rl.obs import ENTITY_DIM


def test_bc_loss_step_returns_scalar():
    net = CombatPolicyNet()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Tiny fake batch
    obs = {
        "skills":     torch.randn(4, 20, 53),
        "skill_mask": torch.ones(4, 20),
        "entities":   torch.randn(4, 6, ENTITY_DIM),
        "resources":  torch.randn(4, 4),
        "terrain":    torch.randn(4, 20, 20),
    }
    actions = torch.tensor([[1, 3, 50], [2, 0, 100], [0, 3, 0], [3, 4, 200]])
    loss, acc = bc_loss_step(net, obs, actions, optim)
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert set(acc.keys()) == {"end", "skill", "entity", "grid"}


def test_train_bc_returns_history():
    # Tiny dataset
    obs_batch = {
        "skills":     np.zeros((10, 20, 53), dtype=np.float32),
        "skill_mask": np.ones((10, 20), dtype=np.float32),
        "entities":   np.zeros((10, 6, ENTITY_DIM), dtype=np.float32),
        "resources":  np.zeros((10, 4), dtype=np.float32),
        "terrain":    np.zeros((10, 20, 20), dtype=np.float32),
    }
    actions = np.random.randint(0, 5, (10, 3)).astype(np.int64)
    dataset = {"obs": obs_batch, "actions": actions}
    history = train_bc(dataset, epochs=2, batch_size=4, lr=1e-3, device="cpu")
    assert "losses" in history
    assert len(history["losses"]) >= 2
