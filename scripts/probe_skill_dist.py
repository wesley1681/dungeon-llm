"""CLI wrapper around trpg.rl.skill_probe.probe_and_report.

For each archetype, runs the model and the scripted expert side-by-side and
prints their skill distributions. Flags category collapses where the expert
uses a category at least 15% of the time but the model uses it < 3% — these
are the failure modes BC point-wise accuracy hides.

Usage:
    python scripts/probe_skill_dist.py --model models/bc_v9_perskill.pt --n 15
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import torch

from trpg.rl.model import CombatPolicyNet
from trpg.rl.skill_probe import probe_and_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="checkpoint to probe (.pt)")
    parser.add_argument("--n", type=int, default=15,
                        help="episodes per archetype per side")
    parser.add_argument("--seed", type=int, default=99999)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device,
                                    weights_only=True))
    net.eval()
    print(f"Model: {args.model}  device={device}")
    print(f"Episodes per archetype: {args.n} (each side)\n")
    probe_and_report(net, n_episodes=args.n, device=device, seed=args.seed)


if __name__ == "__main__":
    main()
