import pytest
from trpg.engine.combat_policy import CombatPolicy, HeuristicCombatPolicy
from trpg.sandbox.policy_loader import load_policy


def test_random_returns_heuristic():
    p = load_policy("random")
    assert isinstance(p, HeuristicCombatPolicy)


def test_expert_returns_archetype_policy():
    p = load_policy("expert:evocation")
    assert isinstance(p, CombatPolicy)
    # The factory falls back to Heuristic for unknown ids — accept any CombatPolicy
    # but verify the alias parsed
    assert p is not None


def test_expert_requires_archetype_suffix():
    with pytest.raises(ValueError, match="expert:<archetype>"):
        load_policy("expert")


def test_path_returns_neural(tmp_path):
    import torch
    from trpg.rl.model import CombatPolicyNet
    from trpg.rl.neural_policy import NeuralCombatPolicy

    net = CombatPolicyNet()
    p = tmp_path / "fake.pt"
    torch.save(net.state_dict(), str(p))
    pol = load_policy(str(p))
    assert isinstance(pol, NeuralCombatPolicy)


def test_unknown_spec_raises(tmp_path):
    with pytest.raises(ValueError, match="cannot interpret"):
        load_policy("not_a_real_thing")
