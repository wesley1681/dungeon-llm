import torch
from trpg.rl.train_ppo import collect_ppo_rollout, ppo_update
from trpg.rl.model import CombatPolicyNet


def test_collect_ppo_rollout_returns_batch():
    net = CombatPolicyNet()
    net.eval()
    batch = collect_ppo_rollout(net, n_steps=20, device="cpu", seed=0)
    assert "obs" in batch
    assert "actions" in batch
    assert "rewards" in batch
    assert "values" in batch
    assert "log_probs" in batch
    assert batch["actions"].shape[0] >= 1
    assert batch["actions"].shape[1] == 3


def test_ppo_update_runs():
    net = CombatPolicyNet()
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    batch = collect_ppo_rollout(net, n_steps=20, device="cpu", seed=0)
    info = ppo_update(net, batch, optim, n_epochs=1, batch_size=16, device="cpu")
    assert "policy_loss" in info
    assert "value_loss" in info
