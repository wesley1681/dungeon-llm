"""Compare multiple model checkpoints on the same eval suite.

Runs each checkpoint against the same seeded episodes vs expert opponents
and emits a compact per-archetype win-rate table.

Usage:
    python scripts/compare_snapshots.py --models bc_v3.pt ppo_v13/ppo_u0100.pt ...
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from trpg.engine.skill import available_skills
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (
    CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action,
)


def eval_model(model_path: str, n: int, seed: int, device: str,
               layout: str | None = None) -> dict:
    """Returns {'wins', 'losses', 'trunc', 'by_arch', 'end_rate'}."""
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    by_arch: dict[str, list] = defaultdict(list)
    end_steps = 0
    total_steps = 0

    for i in range(n):
        env = CombatEnvV2(seed=seed + i, n_agents=1, n_opps=1)
        obs, _ = env.reset(layout=layout)
        agent_id = env.agent_ids[0]
        opp_id = env.opp_ids[0]
        agent = env.ws.characters[agent_id]
        opp = env.ws.characters[opp_id]
        done = False
        while not done:
            total_steps += 1
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            s = apply_resource_mask(s, env.resources, env.ws, agent_id)
            e = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                      ws=env.ws, agent_id=agent_id))

            skills_before = available_skills(agent, env.ws)
            skill_id = (skills_before[action[0]].skill_id
                        if 0 <= action[0] < len(skills_before) else "end")
            had_resources = (env.resources.get("action", 0) > 0
                             or env.resources.get("bonus_action", 0) > 0)
            obs, _, term, trunc, _ = env.step(action)
            if skill_id == "end" and had_resources:
                end_steps += 1
            done = term or trunc

        if not opp.is_alive():
            outcome = "win"
        elif not agent.is_alive():
            outcome = "loss"
        else:
            outcome = "truncated"
        by_arch[env.agent_archs[0]].append(outcome)

    wins = sum(1 for outs in by_arch.values() for o in outs if o == "win")
    losses = sum(1 for outs in by_arch.values() for o in outs if o == "loss")
    trunc = sum(1 for outs in by_arch.values() for o in outs if o == "truncated")
    return {
        "wins": wins, "losses": losses, "trunc": trunc, "total": n,
        "by_arch": by_arch,
        "end_rate": 100.0 * end_steps / total_steps if total_steps else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="checkpoints to compare")
    parser.add_argument("--n", type=int, default=120,
                        help="episodes per model")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--layout", type=str, default=None,
                        help="lock layout (open|walls|difficult|lava). "
                             "Default: random per env")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Eval: {args.n} episodes per model, seed={args.seed}, "
          f"layout={args.layout or 'random'}, device={device}")

    results = {}
    for m in args.models:
        print(f"  evaluating {m} ...")
        results[m] = eval_model(m, args.n, args.seed, device, layout=args.layout)
        r = results[m]
        wr = 100.0 * r["wins"] / r["total"]
        print(f"    win={wr:5.1f}%  end={r['end_rate']:.1f}%")

    # --- Aggregate table ---
    names = [Path(m).stem for m in args.models]
    print("\n=== Win-rate by archetype ===")
    header = "  arch              " + " ".join(f"{n:>13s}" for n in names)
    print(header)
    for arch in ARCHETYPE_LIST:
        row = [f"  {arch:18s}"]
        for m in args.models:
            outs = results[m]["by_arch"].get(arch, [])
            if not outs:
                row.append(f"{'-':>13s}")
                continue
            w = sum(1 for o in outs if o == "win")
            pct = 100.0 * w / len(outs)
            row.append(f"  {w:2d}/{len(outs):2d}({pct:4.0f}%)")
        print(" ".join(row))

    print("\n=== Overall ===")
    print("  metric           " + " ".join(f"{n:>13s}" for n in names))
    line = "  win rate         "
    for m in args.models:
        r = results[m]
        wr = 100.0 * r["wins"] / r["total"]
        line += f" {wr:12.1f}%"
    print(line)
    line = "  truncation       "
    for m in args.models:
        r = results[m]
        tr = 100.0 * r["trunc"] / r["total"]
        line += f" {tr:12.1f}%"
    print(line)
    line = "  end-spam rate    "
    for m in args.models:
        line += f" {results[m]['end_rate']:12.1f}%"
    print(line)


if __name__ == "__main__":
    main()
