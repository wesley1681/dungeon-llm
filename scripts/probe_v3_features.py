"""Probe whether a checkpoint actually USES the obs-v3 threat features.

For a fixed full-HP opening state (mirror archetype, agent L5), sweep the
OPPONENT level 2..8 and report, per opp level:
  - critic V(s)            — a level-blind critic prints one flat line
                              (this was the cross-difficulty miscalibration);
  - policy first-action     — skill picked on the very first sub-action;
  - end_p                   — end-head sigmoid on the opening state.

A freshly MIGRATED pre-v3 checkpoint must show V(s) and the policy exactly
constant across opp levels (its v3 feature columns are zero-padded) — that is
the "before" baseline. After fine-tuning on the asym level distribution, V(s)
should decrease with opp level if the critic learned the threat feature.

CAVEAT (measured 2026-06-12, devotion spread 0.617 on a migrated file): the
opening state is captured AFTER any opponent turns that precede the agent's
first turn, so an EXPERT opponent whose opening actions are level-gated
changes LEGACY-visible columns across the sweep (devotion L3+ opens
sacred_weapon → status bit; L2 cannot). A migrated checkpoint reading that
bit legitimately shows nonzero spread. "spread = 0 proves level-blind" only
holds when opening statuses are identical across levels (berserker mirror:
rage at every level → flat 0.0000; bm pilot baseline: flat 0.0000).

Usage: python scripts/probe_v3_features.py <ckpt> [arch] [agent_level]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_routed import load_net, stable_seed


def main():
    ckpt = sys.argv[1]
    arch = sys.argv[2] if len(sys.argv) > 2 else "battle_master"
    agent_level = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    net = load_net(ckpt)

    print(f"ckpt={ckpt}  arch={arch}  agent L{agent_level}, opp = mirror archetype")
    print(f"{'opp_lvl':>7s} {'V(s)':>8s} {'end_p':>7s}  first_action")
    rows = []
    for opp_lvl in range(2, 9):
        # Same seed for every opp level → identical layout/positions; the ONLY
        # thing that varies is the opponent's level (and its derived stats).
        env = CombatEnvV2(seed=stable_seed(f"probe_v3_{arch}"), n_agents=1, n_opps=1)
        obs, _ = env.reset(agent_archs=[arch], opp_archs=[arch],
                           level=agent_level, opp_level=opp_lvl)
        aid = env.agent_ids[0]
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            v = float(net.value(ot)[0])
            el, s, e, g = net(ot)
        end_p = float(torch.sigmoid(el[0]))
        s = apply_resource_mask(s, env.resources, env.ws, aid)
        e = apply_entity_mask(e, ot, env.ws, aid)
        act = pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=aid)
        skills = available_skills(env.ws.characters[aid], env.ws)
        name = skills[act[0]].skill_id if act[0] < len(skills) else "end"
        rows.append(v)
        print(f"{opp_lvl:>7d} {v:>8.3f} {end_p:>7.3f}  {name}")

    spread = max(rows) - min(rows)
    print(f"\nV(s) spread across opp levels: {spread:.4f} "
          f"({'LEVEL-BLIND (flat)' if spread < 1e-6 else 'level-sensitive'})")


if __name__ == "__main__":
    main()
