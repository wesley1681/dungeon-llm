"""MEASURE-FIRST: does COUNTERSPELL have a clean, WINNABLE discrimination regime
that reads decision_context.spell_level? (Shield was proven to have none in 1v1
— slot is free or crippling; see project_reaction_legendary.)

Counterspell ALWAYS costs a 3rd+ slot (min_slot = max(3, spell_level)), so it is
inherently costly — wasting it on the enemy's weak L1 spell burns a slot the
wizard needs for Fireball, while countering a Fireball/Ice Storm is a great
trade. The OPTIMAL policy is conditional on the incoming spell_level (carried in
decision_context). This probe checks:
  (1) the opponent casts a SPREAD of spell levels (else the choice never arises),
  (2) headroom: 'smart' (counter iff level>=thresh) beats 'always' and 'never',
  (3) the current model's take-rate by level bucket (does it already read level?).

Arms (agent's counterspell only; opponent keeps greedy):
  never  — never counter
  always — counter every eligible spell (greedy default for counterspell)
  smart  — counter iff incoming spell_level >= --thresh (the conditional oracle)
  model  — NeuralReactionDecider on the checkpoint

Usage:
  python scripts/probe_counterspell_headroom.py --games 200 \
      --agent evocation --opp evocation --level 7 --thresh 3
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse, random
from collections import Counter
import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs
from trpg.engine.combat import greedy_reaction_decider
from trpg.scenarios.monsters import register_monsters
register_monsters()


class _CSDecider:
    """Agent counterspell decider for one arm. Records (spell_level, took) for
    every counterspell CHANCE the agent faced (analysis / take-rate by level)."""
    def __init__(self, kind, net, ids, thresh=3, device="cpu"):
        self.kind, self.net, self.ids = kind, net, set(ids)
        self.thresh, self.device = thresh, device
        self.events = []          # (spell_level, took)
        if kind == "model":
            from trpg.rl.reaction_policy import NeuralReactionDecider
            self._model = NeuralReactionDecider(net, self.ids, device=device)

    def __call__(self, ctx):
        if ctx.reactor_id not in self.ids:
            return greedy_reaction_decider(ctx)
        if ctx.trigger == "spell" and ctx.options:
            took = False
            if self.kind == "never":
                choice = None
            elif self.kind == "always":
                choice = "counterspell"
            elif self.kind == "smart":
                choice = "counterspell" if ctx.spell_level >= self.thresh else None
            else:  # model
                choice = self._model(ctx)
            took = choice == "counterspell"
            self.events.append((ctx.spell_level, took))
            return choice
        # non-spell triggers (shield etc.): keep greedy so we isolate counterspell
        return greedy_reaction_decider(ctx)


def _grant_counterspell(env, agent_id):
    c = env.ws.characters[agent_id]
    if "counterspell" not in (c.reactions or []):
        c.reactions = list(c.reactions or []) + ["counterspell"]


def _agent_act(net, env, aid, device):
    obs = build_obs(env.ws, aid, env.resources)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
    with torch.no_grad():
        e, sk, en, gr = net(obs_t)
        sk = apply_resource_mask(sk, env.resources, env.ws, aid)
        en = apply_entity_mask(en, obs_t, env.ws, aid)
        return list(pick_action(e[0], sk[0], en[0], gr[0], ws=env.ws, agent_id=aid))


def play(net, kind, agent, opp, level, thresh, seed, device):
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(agent_archs=[agent], opp_archs=[opp], level=level, seed=seed)
    aid = env.agent_ids[0]
    _grant_counterspell(env, aid)
    dec = _CSDecider(kind, net, {aid}, thresh, device)
    env.ws.reaction_decider = dec
    for _ in range(env._max_steps):
        env.step(_agent_act(net, env, aid, device))
        if (all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
                or all(not env.ws.characters[a].is_alive() for a in env.agent_ids)):
            break
    win = (all(not env.ws.characters[o].is_alive() for o in env.opp_ids)
           and any(env.ws.characters[a].is_alive() for a in env.agent_ids))
    return win, dec.events


def run_arm(net, kind, agent, opp, level, thresh, games, device):
    wins = 0
    lvl_take = Counter(); lvl_seen = Counter()
    for g in range(games):
        s = 4000 + g
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        win, evs = play(net, kind, agent, opp, level, thresh, s, device)
        wins += int(win)
        for lv, took in evs:
            lvl_seen[lv] += 1
            if took:
                lvl_take[lv] += 1
    return wins / games, lvl_seen, lvl_take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--agent", default="evocation")
    ap.add_argument("--opp", default="evocation")
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--thresh", type=int, default=3)
    ap.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from trpg.sandbox.policy_loader import load_policy
    net = load_policy(a.ckpt).net; net.eval()

    print(f"counterspell headroom: {a.agent}(+counterspell) vs {a.opp} @L{a.level}, "
          f"thresh={a.thresh}, {a.games} games/arm, ckpt={a.ckpt}")
    print(f"{'arm':>7} | {'WR':>6} | spell-levels faced → take-rate")
    print("-" * 60)
    for kind in ("never", "always", "smart", "model"):
        wr, seen, take = run_arm(net, kind, a.agent, a.opp, a.level, a.thresh,
                                 a.games, a.device)
        dist = "  ".join(f"L{lv}:{take[lv]}/{seen[lv]}"
                         for lv in sorted(seen)) or "(no counterspell chances)"
        print(f"{kind:>7} | {wr*100:5.1f}% | {dist}")


if __name__ == "__main__":
    main()
