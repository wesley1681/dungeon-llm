"""Map --model CLI spec to a CombatPolicy instance for the sandbox."""
from __future__ import annotations
import os

import torch

from ..engine.combat_policy import (
    CombatPolicy, HeuristicCombatPolicy, make_archetype_policy,
)
from ..rl.model import CombatPolicyNet
from ..rl.neural_policy import NeuralCombatPolicy


def load_policy(spec: str) -> CombatPolicy:
    """Parse a --model spec into a policy instance.

    Accepted forms:
      "random"                — HeuristicCombatPolicy (greedy melee baseline)
      "expert:<archetype>"    — ArchetypePolicy for that archetype
      "<path>.pt"             — NeuralCombatPolicy from a saved checkpoint
    """
    if spec == "random":
        return HeuristicCombatPolicy()
    if spec.startswith("expert:"):
        arch = spec[len("expert:"):].strip()
        if not arch:
            raise ValueError("expert:<archetype> — archetype name missing")
        return make_archetype_policy(arch)
    if spec.endswith(".pt") and os.path.exists(spec):
        net = CombatPolicyNet()
        net.load_state_dict(torch.load(spec, map_location="cpu"))
        net.eval()
        return NeuralCombatPolicy(net, device="cpu")
    raise ValueError(
        f"cannot interpret model spec {spec!r} — "
        f"expected 'random', 'expert:<archetype>', or path to a .pt file"
    )
