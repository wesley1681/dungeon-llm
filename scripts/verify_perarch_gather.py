"""Empirical check: do the per-archetype heads actually route by arch_id?

The architecture-review subagent claimed the gather on dim=1 with a length-1
index axis 'always reads a fixed archetype index'. That is a misreading of
torch.gather semantics: the *value* in the index tensor is arch_ids[b], which
varies per sample; expand only broadcasts that value along the slot axis.

This script proves it: build one net, feed two identical observations that
differ ONLY in the self archetype one-hot, and confirm the three per-arch
heads (skill / entity / grid) produce DIFFERENT logits. If the gather were
buggy (constant index) the outputs would be byte-identical.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trpg.rl.model import CombatPolicyNet, _ARCH_OH_START, _ARCH_OH_END
from trpg.rl.obs import (N_SKILL_SLOTS, N_ENTITY_SLOTS, ENTITY_DIM, N_GRID,
                         N_ENTITY_GRID_CHANNELS, N_DISTANCE_GRID_CHANNELS,
                         END_FEATURES_DIM, N_ARCHETYPES)
from trpg.engine.skill import SKILL_FEATURE_DIM

torch.manual_seed(0)


def make_obs(arch_id: int):
    ent = torch.zeros(1, N_ENTITY_SLOTS, ENTITY_DIM)
    # mark self row alive/self and set the archetype one-hot bit
    ent[0, 0, 5] = 1.0  # is_alive
    ent[0, 0, 6] = 1.0  # is_self
    ent[0, 0, _ARCH_OH_START + arch_id] = 1.0
    # one live enemy so masks have something to chew on
    ent[0, 3, 5] = 1.0
    ent[0, 3, 4] = 1.0  # is_enemy
    return {
        "skills":        torch.randn(1, N_SKILL_SLOTS, SKILL_FEATURE_DIM),
        "skill_mask":    torch.ones(1, N_SKILL_SLOTS),
        "entities":      ent,
        "resources":     torch.zeros(1, 4),
        # Non-zero spatial features: grid_logits = bmm(grid_query, spatial_flat),
        # so with zero spatial features ALL archetypes trivially get 0 (a false
        # 'identical' that hides per-arch routing). Use random spatial input.
        "terrain":       torch.randn(1, N_GRID, N_GRID),
        "entity_grid":   torch.randn(1, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID),
        "distance_grid": torch.randn(1, N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID),
        "end_features":  torch.zeros(1, END_FEATURES_DIM),
    }


def main():
    net = CombatPolicyNet()
    net.eval()

    # Same inputs for all three so ONLY arch_id differs.
    base_skills = torch.randn(1, N_SKILL_SLOTS, SKILL_FEATURE_DIM)
    base_terrain = torch.randn(1, N_GRID, N_GRID)
    base_egrid = torch.randn(1, N_ENTITY_GRID_CHANNELS, N_GRID, N_GRID)
    base_dgrid = torch.randn(1, N_DISTANCE_GRID_CHANNELS, N_GRID, N_GRID)

    outs = {}
    for arch_id in (0, 5, 11):
        obs = make_obs(arch_id)
        obs["skills"] = base_skills.clone()
        obs["terrain"] = base_terrain.clone()
        obs["entity_grid"] = base_egrid.clone()
        obs["distance_grid"] = base_dgrid.clone()
        with torch.no_grad():
            end_l, s, e, g = net(obs)
        outs[arch_id] = (s.clone(), e.clone(), g.clone())

    def report(name, idx):
        a0 = outs[0][idx]; a5 = outs[5][idx]; a11 = outs[11][idx]
        d05 = (a0 - a5).abs().max().item()
        d011 = (a0 - a11).abs().max().item()
        routed = d05 > 1e-6 and d011 > 1e-6
        print(f"{name:14s} |arch0-arch5|max={d05:.4f}  |arch0-arch11|max={d011:.4f}  "
              f"-> {'ROUTED (per-arch works)' if routed else 'IDENTICAL (BUG!)'}")
        return routed

    print("Per-archetype head routing check (different arch_id, same everything else):")
    ok_s = report("skill_logits", 0)
    ok_e = report("entity_logits", 1)
    ok_g = report("grid_logits", 2)

    print()
    if ok_s and ok_e and ok_g:
        print("VERDICT: per-arch gather is CORRECT. Subagent claim (P1/P2/P3) REFUTED.")
    else:
        print("VERDICT: some head is NOT routing by arch_id — subagent claim CONFIRMED.")


if __name__ == "__main__":
    main()
