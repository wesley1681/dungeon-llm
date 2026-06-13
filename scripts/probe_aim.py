"""Probe whether the model has learned to AIM AoE spells at enemies.

For TargetType.POINT/LINE/CONE skills, the grid head picks a cell — decoded to
(x,y) world coords, clamped to spell range. We record:
  - distance from chosen cell to nearest enemy
  - distance from chosen cell to self
  - "Random baseline": uniform draw from N_GRID*N_GRID grid

If the model learned aiming, chosen-cell-to-enemy should be much smaller than
the random baseline (and ideally close to 0).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.action import _grid_cell_to_xy
from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
from trpg.engine.skill import available_skills, TargetType


AOE_TARGETS = (TargetType.POINT, TargetType.LINE, TargetType.CONE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=33333)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"Model: {args.model}\n")

    chosen_to_enemy = []  # distance: chosen cell → enemy position
    chosen_to_self = []   # distance: chosen cell → agent position
    skill_names = []
    enemy_dist_at_action = []  # how far is enemy when AoE is cast?

    for ep in range(args.n):
        env = CombatEnvV2(seed=args.seed + ep, n_agents=1, n_opps=1)
        obs, _ = env.reset()
        agent_id = env.agent_ids[0]
        opp_id = env.opp_ids[0]
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, skill_l, entity_l, grid_l = net(obs_t)
            skill_l = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
            entity_l = apply_entity_mask(entity_l, obs_t)

            skill_idx, entity_idx, grid_cell = pick_action(
                end_l[0], skill_l[0], entity_l[0], grid_l[0],
                ws=env.ws, agent_id=agent_id,
            )

            agent = env.ws.characters[agent_id]
            skills = available_skills(agent, env.ws)
            tt = None
            sk_name = "?"
            if 0 <= skill_idx < len(skills):
                tt = skills[skill_idx].features.target_type
                sk_name = skills[skill_idx].skill_id

            if tt in AOE_TARGETS:
                cx, cy = _grid_cell_to_xy(grid_cell)
                opp = env.ws.characters[opp_id]
                d_enemy = ((cx - opp.position.x) ** 2 + (cy - opp.position.y) ** 2) ** 0.5
                d_self = ((cx - agent.position.x) ** 2 + (cy - agent.position.y) ** 2) ** 0.5
                d_a2e = agent.position.distance_to(opp.position)
                chosen_to_enemy.append(d_enemy)
                chosen_to_self.append(d_self)
                skill_names.append(sk_name)
                enemy_dist_at_action.append(d_a2e)

            obs, _, term, trunc, _ = env.step([skill_idx, entity_idx, grid_cell])
            done = term or trunc

    # Random baseline: uniform cell over grid
    rng = np.random.default_rng(0)
    n_rand = 10_000
    rand_cells = rng.integers(0, N_GRID * N_GRID, size=n_rand)
    # Random distances from grid centre to a random opponent point — use 15, 15 (battle centre)
    rand_to_centre = []
    cx_centre, cy_centre = 15.0, 15.0
    for c in rand_cells:
        x, y = _grid_cell_to_xy(int(c))
        rand_to_centre.append(((x - cx_centre) ** 2 + (y - cy_centre) ** 2) ** 0.5)
    rand_mean = np.mean(rand_to_centre)

    n_aoe = len(chosen_to_enemy)
    print(f"Total episodes:        {args.n}")
    print(f"AoE casts captured:    {n_aoe}")
    if n_aoe == 0:
        print("\n  Model never cast an AoE — can't probe aim.")
        return

    arr_e = np.array(chosen_to_enemy)
    arr_s = np.array(chosen_to_self)
    arr_d = np.array(enemy_dist_at_action)

    print(f"\n=== AIMING ACCURACY (chosen grid cell → enemy position) ===")
    print(f"  Mean distance: {arr_e.mean():.2f} m   (lower = better aim)")
    print(f"  Median:        {np.median(arr_e):.2f} m")
    print(f"  Min / Max:     {arr_e.min():.2f} / {arr_e.max():.2f} m")
    print(f"\n=== Self-damage risk (chosen cell → agent position) ===")
    print(f"  Mean: {arr_s.mean():.2f} m   (typical AoE radius ~3m; lower = self-hit risk)")
    print(f"  <3m  count: {(arr_s < 3).sum()}/{n_aoe}  ({100*(arr_s<3).sum()/n_aoe:.1f}%)")
    print(f"\n=== Random baseline (uniform grid sample → centre) ===")
    print(f"  Mean distance:  {rand_mean:.2f} m")
    print(f"\n=== Distance between agent and enemy when AoE cast ===")
    print(f"  Mean: {arr_d.mean():.2f} m")

    # Which skills?
    from collections import Counter
    cnt = Counter(skill_names)
    print(f"\n=== AoE skills cast ===")
    for sk, c in cnt.most_common():
        print(f"  {sk:30s}  {c:3d}")

    # Which grid cells does the model actually pick?
    # Re-collect with cells this time
    print(f"\n=== Grid cell histogram (most common picks) ===")
    cells_picked = Counter()
    for ep in range(args.n):
        env = CombatEnvV2(seed=args.seed + ep, n_agents=1, n_opps=1)
        obs, _ = env.reset()
        agent_id = env.agent_ids[0]
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, skill_l, entity_l, grid_l = net(obs_t)
            skill_l = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
            entity_l = apply_entity_mask(entity_l, obs_t)
            skill_idx, entity_idx, grid_cell = pick_action(
                end_l[0], skill_l[0], entity_l[0], grid_l[0],
                ws=env.ws, agent_id=agent_id,
            )
            # Record cell only when grid was meaningful (skill is grid-targeted)
            if grid_cell != 0:
                cells_picked[grid_cell] += 1
            obs, _, term, trunc, _ = env.step([skill_idx, entity_idx, grid_cell])
            done = term or trunc
    for cell, c in cells_picked.most_common(5):
        ix = cell // N_GRID
        iy = cell % N_GRID
        x, y = _grid_cell_to_xy(cell)
        print(f"  cell {cell:4d} (ix={ix},iy={iy}, x={x:.1f}, y={y:.1f}m): picked {c} times")


if __name__ == "__main__":
    main()
