import numpy as np
from trpg.rl.bc_collect import collect_bc_rollout, collect_bc_dataset


def test_collect_bc_rollout_returns_obs_action_pairs():
    pairs = collect_bc_rollout(agent_arch="champion", opponent_arch="champion",
                                level=5, seed=0)
    assert len(pairs) > 0
    obs, action = pairs[0]
    assert "skills" in obs
    assert len(action) == 3


def test_collect_bc_dataset_concatenates():
    ds = collect_bc_dataset(n_episodes_per_arch=1, seed=0)
    # obs is dict of stacked arrays
    assert "skills" in ds["obs"]
    assert ds["actions"].ndim == 2
    assert ds["actions"].shape[1] == 3
    assert ds["actions"].shape[0] == ds["obs"]["skills"].shape[0]
