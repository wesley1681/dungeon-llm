"""Probe whether the model has learned to target enemies WITHOUT the entity mask.

In 1v1 combat the entity mask reduces the entity head to a single valid slot
(the opponent), so any argmax is forced — we can't tell from gameplay whether
the model actually learned targeting. This script runs episodes and records
the entity head's RAW softmax (pre-mask) at every step, then reports the
distribution over slots: 0=self, 1-2=ally, 3-5=enemy.

If a trained model has learned to target enemies, slot 3 should dominate.
If it's only the mask doing the work, the distribution should be near-uniform.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import Counter
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import CombatPolicyNet, apply_resource_mask
from trpg.engine.skill import available_skills, TargetType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=11111)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"Model: {args.model}")

    # Pre-mask argmax distribution
    raw_argmax: Counter[int] = Counter()
    # Pre-mask softmax avg per slot
    sum_probs = np.zeros(6)
    n_steps = 0
    # Among steps where chosen skill targets an enemy: does argmax pick enemy slot?
    enemy_skill_steps = 0
    enemy_skill_correct = 0

    for ep in range(args.n):
        env = CombatEnvV2(seed=args.seed + ep, n_agents=1, n_opps=1)
        obs, _ = env.reset()
        agent_id = env.agent_ids[0]
        done = False
        while not done:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                _, skill_l, entity_l, grid_l = net(obs_t)
            # Use the masked skill head to pick the action (same as gameplay)
            skill_l = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
            skill_idx = int(skill_l[0].argmax())

            # RAW entity head for the chosen skill (NO mask) —
            # what would the model do unconstrained?
            raw_probs = torch.softmax(entity_l[0, skill_idx, :], dim=-1).cpu().numpy()
            raw_argmax[int(raw_probs.argmax())] += 1
            sum_probs += raw_probs
            n_steps += 1

            agent = env.ws.characters[agent_id]
            skills = available_skills(agent, env.ws)
            tt = None
            if 0 <= skill_idx < len(skills):
                tt = skills[skill_idx].features.target_type
            if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY):
                enemy_skill_steps += 1
                if raw_probs.argmax() >= 3:
                    enemy_skill_correct += 1

            # Step the env using the MASKED action (real gameplay)
            from trpg.rl.model import apply_entity_mask
            entity_masked = apply_entity_mask(entity_l, obs_t)
            action = [skill_idx, int(entity_masked[0].argmax()), int(grid_l.argmax())]
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    avg_probs = sum_probs / max(n_steps, 1)
    print(f"\nTotal steps analyzed: {n_steps}")
    print("\n=== Pre-mask entity-head argmax distribution ===")
    print("  slot 0 (self)  :", f"{raw_argmax[0]:5d} ({100*raw_argmax[0]/n_steps:5.1f}%)")
    print("  slot 1 (ally1) :", f"{raw_argmax[1]:5d} ({100*raw_argmax[1]/n_steps:5.1f}%)")
    print("  slot 2 (ally2) :", f"{raw_argmax[2]:5d} ({100*raw_argmax[2]/n_steps:5.1f}%)")
    print("  slot 3 (enemy1):", f"{raw_argmax[3]:5d} ({100*raw_argmax[3]/n_steps:5.1f}%)")
    print("  slot 4 (enemy2):", f"{raw_argmax[4]:5d} ({100*raw_argmax[4]/n_steps:5.1f}%)")
    print("  slot 5 (enemy3):", f"{raw_argmax[5]:5d} ({100*raw_argmax[5]/n_steps:5.1f}%)")
    print("\n=== Pre-mask mean softmax per slot ===")
    for i, p in enumerate(avg_probs):
        label = ["self", "ally1", "ally2", "enemy1", "enemy2", "enemy3"][i]
        bar = "#" * int(p * 60)
        print(f"  slot {i} ({label:6s}): {p:.3f}  {bar}")

    if enemy_skill_steps:
        acc = 100 * enemy_skill_correct / enemy_skill_steps
        print(f"\n=== Targeting accuracy (when skill needs enemy) ===")
        print(f"  Steps with enemy-targeted skill: {enemy_skill_steps}")
        print(f"  Of these, raw argmax picked enemy slot (3-5): "
              f"{enemy_skill_correct} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
