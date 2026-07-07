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
    if spec.endswith(".pt"):
        if not os.path.exists(spec):
            raise ValueError(f"model file not found: {spec!r}")
        # Route through the architecture registry's defensive loader: it rebuilds
        # the RIGHT architecture from the checkpoint's sidecar codename and STRICT-
        # loads, raising a clear diff if the weights don't match (instead of the
        # old strict=False that silently dropped mismatched keys → random weights).
        # migrate=True keeps the legacy shared-head→per-arch tiling bridge for old
        # checkpoints; a checkpoint with no sidecar is assumed 'uni' (with a warn).
        from ..rl.architectures import load_net
        net = load_net(spec, migrate=True)
        return NeuralCombatPolicy(net, device="cpu")
    raise ValueError(
        f"cannot interpret model spec {spec!r} — "
        f"expected 'random', 'expert:<archetype>', or path to a .pt file"
    )
