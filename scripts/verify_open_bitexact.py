"""Prove the LoS-row fine-tune is BIT-EXACT vs base in open fields.

The 2v2-safety claim of train_grid_los_only rests on one mathematical fact:
in an open layout the per-cell LoS channel (los_grid) is all-ones, so the
grid-head's LoS-query contributes the SAME additive constant to every cell
logit -> argmax over cells is invariant to that weight, and every other head
(skill/entity/end) is frozen. Hence open-field action selection must be
identical to base, regardless of how the LoS row moved during training.

This script falsifies that empirically: roll out open-layout episodes (1v1 AND
2v2) driving with BASE, and at every decision state recompute pick_action with
BOTH base and the trained net. Any single mismatch breaks the guarantee (and
would mean 2v2 could regress). Zero mismatches = the open game is untouched and
only walled behaviour changed.

Usage:
  python scripts/verify_open_bitexact.py --base models/pop_mon/pop_u0005.pt \
         --ckpt models/pop_los_row/pop_u0040.pt --games 40
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _act(net, ot, env, actor):
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return tuple(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def compare(base, net, na, no_, games):
    steps = mism = 0
    examples = []
    for g in range(games):
        k = crc32(f"open|{na}v{no_}|{g}".encode()); random.seed(k)
        a_archs = [STANDARD_IDS[(g + i) % len(STANDARD_IDS)] for i in range(na)]
        o_archs = [STANDARD_IDS[(g + 3 + i) % len(STANDARD_IDS)] for i in range(no_)]
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=na, n_opps=no_)
        obs, _ = env.reset(agent_archs=a_archs, opp_archs=o_archs,
                           level=5, opp_level=5, layout="open")
        done = False
        while not done:
            actor = env.current_agent_id
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            a_base = _act(base, ot, env, actor)
            a_net = _act(net, ot, env, actor)
            steps += 1
            if a_base != a_net:
                mism += 1
                if len(examples) < 5:
                    examples.append((actor, a_base, a_net))
            obs, _, term, trunc, _ = env.step(list(a_base))  # drive with base
            done = term or trunc
    return steps, mism, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()

    base = load_student(args.base); base.eval()
    net = load_student(args.ckpt); net.eval()
    print(f"base={args.base}\nckpt={args.ckpt}\nopen-field action match "
          f"({args.games} games/arrangement):\n")
    total_m = 0
    for na, no_ in [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)]:
        steps, mism, ex = compare(base, net, na, no_, args.games)
        total_m += mism
        tag = "OK bit-exact" if mism == 0 else f"!! {mism} MISMATCH"
        print(f"  {na}v{no_}: {steps:5d} steps, {mism:4d} mismatches  [{tag}]")
        for actor, ab, an in ex:
            print(f"      {actor}: base={ab} net={an}")
    print(f"\n{'PASS — open game untouched' if total_m == 0 else 'FAIL'}: "
          f"{total_m} total mismatches", flush=True)


if __name__ == "__main__":
    main()
