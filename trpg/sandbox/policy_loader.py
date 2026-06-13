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
        net = CombatPolicyNet()
        # adapt + strict=False so older shared-head checkpoints still load:
        # adapt_state_dict_for_perarch tiles a single skill_head/entity_head/
        # grid_query_proj into the 12 per-arch copies, and strict=False lets a
        # checkpoint that predates the MLP value head keep that head's fresh
        # init (the value head is unused at inference anyway).
        sd = torch.load(spec, map_location="cpu", weights_only=True)
        sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
        # Drop shape-mismatched keys (e.g. critic input dim changed across model
        # versions) — inference never uses the critic, so this is safe and lets
        # any checkpoint, old or new, load.
        msd = net.state_dict()
        sd = {k: v for k, v in sd.items()
              if not (k in msd and v.shape != msd[k].shape)}
        net.load_state_dict(sd, strict=False)
        net.eval()
        return NeuralCombatPolicy(net, device="cpu")
    raise ValueError(
        f"cannot interpret model spec {spec!r} — "
        f"expected 'random', 'expert:<archetype>', or path to a .pt file"
    )
