"""MEASURE-FIRST (CLAUDE.md 先量再開刀): is there headroom to TRAIN a
model-controlled reaction, and does the current model fail to capture it?

For a Shield-capable wizard vs a hard-hitting melee attacker, compare the
wizard's win-rate under four reaction policies for ITS OWN reaction only
(opponents always keep the engine greedy default):

  never   — always decline the reaction (baseline floor)
  greedy  — engine default (Shield fires only when +5 AC flips a hit→miss)
  always  — always spend the reaction on the first legal option (ceiling-ish)
  model   — NeuralReactionDecider on the given checkpoint (untrained on reactions)

Reads:
  - greedy/always ≫ never  → the reaction MATTERS (headroom exists).
  - model ≈ never (and ≪ greedy) → the model IGNORES the reaction = a real gap
    that training (M4) must close. model ≈ greedy → already captured, no need.

Global RNG is reseeded per arm so every arm sees the same dice (the std12 /
seeded-fingerprint lesson: cross-arm WR is only fair under identical seeds).

Usage:
  python scripts/probe_reaction_headroom.py --games 300 \
      --agent evocation --opp battle_master --level 5 --ckpt models/pop_mon/pop_u0005.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.combat import greedy_reaction_decider
from trpg.scenarios.monsters import register_monsters
register_monsters()


class _CountingDecider:
    """Wraps a base decider; counts how often the AGENT actually reacts AND how
    often it FACED a legal reaction decision (so the fire-rate denominator is the
    number of real chances, not games). Reaction fires inside the opponent's
    turn, invisible to env.step's result — counting at the decider is the only
    correct place."""
    def __init__(self, base, ids):
        self.base = base
        self.ids = set(ids)
        self.fires = 0          # agent chose a reaction
        self.chances = 0        # agent had ≥1 legal option

    def __call__(self, ctx):
        choice = (self.base(ctx) if self.base is not None
                  else greedy_reaction_decider(ctx))
        if ctx.reactor_id in self.ids and ctx.options:
            self.chances += 1
            if choice in ctx.options:
                self.fires += 1
        return choice


def _make_base(kind, net, ids, device):
    def never(ctx):
        return None if ctx.reactor_id in ids else greedy_reaction_decider(ctx)

    def always(ctx):
        if ctx.reactor_id not in ids:
            return greedy_reaction_decider(ctx)
        return ctx.options[0] if ctx.options else None

    if kind == "never":
        return never
    if kind == "always":
        return always
    if kind == "greedy":
        return None                      # engine default everywhere
    if kind == "model":
        from trpg.rl.reaction_policy import NeuralReactionDecider
        return NeuralReactionDecider(net, ids, device=device)
    raise ValueError(kind)


def _agent_act(net, env, agent_id, device):
    from trpg.rl.obs import build_obs
    obs = build_obs(env.ws, agent_id, env.resources)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        end_l, skill_l, ent_l, grid_l = net(obs_t)
        skill_l = apply_resource_mask(skill_l, env.resources, env.ws, agent_id)
        ent_l = apply_entity_mask(ent_l, obs_t, env.ws, agent_id)
        return list(pick_action(end_l[0], skill_l[0], ent_l[0], grid_l[0],
                                ws=env.ws, agent_id=agent_id))


def play(net, kind, agent, opp, level, seed, device):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent], opp_archs=[opp], level=level, seed=seed)
    dec = _CountingDecider(_make_base(kind, net, set(env.agent_ids), device),
                           env.agent_ids)
    env.ws.reaction_decider = dec
    aid = env.agent_ids[0]
    for _ in range(env._max_steps):
        act = _agent_act(net, env, aid, device)
        _o, _r, term, trunc, info = env.step(act)
        if term or trunc:
            break
    win = (all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
           and any(env.ws.characters[a].is_alive() for a in env.agent_ids))
    return win, dec.fires, dec.chances


def run_arm(net, kind, agent, opp, level, games, device):
    wins = fires = chances = 0
    for g in range(games):
        random.seed(1000 + g); np.random.seed(1000 + g); torch.manual_seed(1000 + g)
        w, sf, ch = play(net, kind, agent, opp, level, seed=1000 + g, device=device)
        wins += int(w); fires += sf; chances += ch
    take = fires / chances if chances else 0.0
    return wins / games, fires / games, take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--agent", default="evocation")
    ap.add_argument("--opp", default="battle_master")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from trpg.sandbox.policy_loader import load_policy
    net = load_policy(args.ckpt).net
    net.eval()

    print(f"reaction headroom: {args.agent} (shield) vs {args.opp} @L{args.level}, "
          f"{args.games} games/arm, ckpt={args.ckpt}")
    print(f"{'arm':>8} | {'WR':>7} | {'react/game':>10} | {'take-rate':>9}")
    print("-" * 46)
    for kind in ("never", "greedy", "always", "model"):
        wr, fr, take = run_arm(net, kind, args.agent, args.opp, args.level,
                               args.games, args.device)
        print(f"{kind:>8} | {wr*100:6.1f}% | {fr:10.2f} | {take*100:8.1f}%")


if __name__ == "__main__":
    main()
