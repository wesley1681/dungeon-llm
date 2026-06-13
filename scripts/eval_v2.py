"""Comprehensive policy evaluation: 12x12 archetype matchups + tactical metrics.

For every (agent_arch, opp_arch) pair, plays N games and reports:
  - win_rate
  - avg_agent_hp_end_on_win    (how much HP left when winning)
  - avg_opp_hp_end_on_loss     (how close was the loss)
  - avg_rounds                 (slow games hint at stalemate/stall)
  - avg_self_damage            (damage the agent took during own sub-actions —
                                 captures self-blast on own AOE)

Init order is recorded per game so we can detect matchups where the result
depends on who goes first (e.g. mirror match always loses second).

Action source can be a neural net checkpoint or "expert:<archetype>" (the
scripted policy that produced BC labels) — the latter is a sanity baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse, json
from pathlib import Path

import numpy as np
import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                            apply_entity_mask, pick_action)
from trpg.engine.combat_policy import make_archetype_policy


def _load_net(path: str, hidden: int) -> CombatPolicyNet:
    net = CombatPolicyNet(hidden=hidden)
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    net.eval()
    return net


def _neural_action(net, env, actor_id):
    obs = env._cached_obs if hasattr(env, "_cached_obs") else None
    if obs is None:
        from trpg.rl.obs import build_obs
        obs = build_obs(env.ws, actor_id, env.resources)
    obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
    with torch.no_grad():
        end_l, s, e, g = net(obs_t)
    s = apply_resource_mask(s, env.resources, env.ws, actor_id)
    e = apply_entity_mask(e, obs_t)
    return list(pick_action(end_l[0], s[0], e[0], g[0],
                             ws=env.ws, agent_id=actor_id))


def play_one_game(net, agent_arch: str, opp_arch: str, seed: int,
                   level: int = 5) -> dict:
    """Play one 1v1 game. Returns dict of per-game stats.

    ``net`` may be a CombatPolicyNet (neural) or ``None`` to mean "use the
    archetype's scripted expert as the agent" (a BC-label baseline).
    """
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[agent_arch], opp_archs=[opp_arch], level=level)
    aid = env.agent_ids[0]
    oid = env.opp_ids[0]
    init_first = env.ws.combat.initiative_order[0]
    init_label = "agent" if init_first == aid else "opp"

    expert_pol = make_archetype_policy(agent_arch) if net is None else None

    self_damage = 0.0
    done = False
    while not done:
        actor_id = env.current_agent_id
        agent_hp_before = env.ws.characters[aid].hp if actor_id == aid else None

        if net is not None:
            obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            s = apply_resource_mask(s, env.resources, env.ws, actor_id)
            e = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                       ws=env.ws, agent_id=actor_id))
        else:
            # Expert path: only the agent uses the scripted policy; the
            # opponent is run inside env. So we just need to produce one
            # action for the agent here.
            actor = env.ws.characters[actor_id]
            decision = expert_pol.decide(actor_id, actor, env.ws,
                                          env.resources,
                                          env.ws.combat.round_number)
            if decision.action is None or decision.fled:
                action = [0, 0, 0]
            else:
                from trpg.rl.action import encode_action
                action = list(encode_action(decision.action, env.ws, actor_id))

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

        if agent_hp_before is not None:
            agent_hp_after = env.ws.characters[aid].hp
            if agent_hp_after < agent_hp_before:
                self_damage += (agent_hp_before - agent_hp_after)

    agent_alive = env.ws.characters[aid].is_alive()
    opp_alive = env.ws.characters[oid].is_alive()
    won = (not opp_alive) and agent_alive

    return {
        "win":              won,
        "init":             init_label,
        "agent_hp_end":     env.ws.characters[aid].hp,
        "agent_max_hp":     env.ws.characters[aid].max_hp,
        "opp_hp_end":       env.ws.characters[oid].hp,
        "opp_max_hp":       env.ws.characters[oid].max_hp,
        "rounds":           env.ws.combat.round_number if env.ws.combat else 0,
        "self_damage":      self_damage,
    }


def eval_matchup(net, agent_arch: str, opp_arch: str, n_games: int,
                  seed_base: int = 0) -> dict:
    games = [play_one_game(net, agent_arch, opp_arch, seed=seed_base + i)
             for i in range(n_games)]
    wins = [g for g in games if g["win"]]
    losses = [g for g in games if not g["win"]]
    n = len(games)
    nw = len(wins)
    nl = len(losses)
    return {
        "n": n,
        "wins": nw,
        "win_rate": nw / n if n else 0.0,
        "win_rate_agent_first": (
            sum(1 for g in games if g["win"] and g["init"] == "agent")
            / max(1, sum(1 for g in games if g["init"] == "agent"))),
        "win_rate_opp_first": (
            sum(1 for g in games if g["win"] and g["init"] == "opp")
            / max(1, sum(1 for g in games if g["init"] == "opp"))),
        "avg_agent_hp_frac_on_win": (
            np.mean([g["agent_hp_end"] / max(g["agent_max_hp"], 1) for g in wins])
            if wins else 0.0),
        "avg_opp_hp_frac_on_loss": (
            np.mean([g["opp_hp_end"] / max(g["opp_max_hp"], 1) for g in losses])
            if losses else 0.0),
        "avg_rounds":       float(np.mean([g["rounds"] for g in games])),
        "avg_self_damage":  float(np.mean([g["self_damage"] for g in games])),
    }


def eval_all(net, n_games: int, label: str,
              archs: list[str] | None = None) -> dict:
    archs = archs or list(ARCHETYPE_LIST)
    result = {}
    total = len(archs) * len(archs)
    done = 0
    for a in archs:
        result[a] = {}
        for o in archs:
            seed_base = (hash(f"{a}_{o}") & 0xFFFFFF)
            result[a][o] = eval_matchup(net, a, o, n_games, seed_base)
            done += 1
            if done % 24 == 0:
                print(f"  [{label}] {done}/{total} matchups done")
    return result


def summary_table(result: dict, archs: list[str]) -> None:
    """Print per-archetype averages across all opponents."""
    print(f"\n{'archetype':20s}  {'wr':>5s}  {'wr_1st':>6s}  {'wr_2nd':>6s}  "
          f"{'hp_win':>6s}  {'opp_hp_loss':>11s}  {'rounds':>6s}  {'self_dmg':>8s}")
    for a in archs:
        wr_list, wr1, wr2, hpw, oph, rds, sd = [], [], [], [], [], [], []
        for o in archs:
            r = result[a][o]
            wr_list.append(r["win_rate"])
            wr1.append(r["win_rate_agent_first"])
            wr2.append(r["win_rate_opp_first"])
            hpw.append(r["avg_agent_hp_frac_on_win"])
            oph.append(r["avg_opp_hp_frac_on_loss"])
            rds.append(r["avg_rounds"])
            sd.append(r["avg_self_damage"])
        print(f"{a:20s}  {np.mean(wr_list):>4.0%}  "
              f"{np.mean(wr1):>5.0%}  {np.mean(wr2):>5.0%}  "
              f"{np.mean(hpw):>5.0%}  {np.mean(oph):>10.0%}  "
              f"{np.mean(rds):>6.1f}  {np.mean(sd):>8.2f}")
    avg = np.mean([result[a][o]["win_rate"]
                    for a in archs for o in archs])
    print(f"{'OVERALL avg':20s}  {avg:>4.0%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, required=True,
                        help="path/to/checkpoint.pt OR 'expert' (scripted baseline)")
    parser.add_argument("--hidden", type=int, default=128,
                        help="model hidden dim (must match checkpoint)")
    parser.add_argument("--games",  type=int, default=50,
                        help="games per matchup (50 → 7200 total games / model)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Optional path to write JSON result")
    parser.add_argument("--label",  type=str, default="MODEL")
    args = parser.parse_args()

    if args.model.lower() == "expert":
        net = None
        label = args.label or "expert"
    else:
        net = _load_net(args.model, args.hidden)
        label = args.label or Path(args.model).name

    print(f"Evaluating {label}: {len(ARCHETYPE_LIST)}x{len(ARCHETYPE_LIST)} "
          f"matchups, {args.games} games each")
    result = eval_all(net, args.games, label)

    print(f"\n=== {label} ===")
    summary_table(result, list(ARCHETYPE_LIST))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"label": label, "games_per_matchup": args.games,
                       "result": result}, f, indent=2)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
