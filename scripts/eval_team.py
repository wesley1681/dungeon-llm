"""2v2 team evaluation — expert-as-judge mode.

The model controls both allies; opponents use archetype expert policies.
This tests whether the model has learned genuine team strategy (focus fire,
healer prioritisation, coordinated approach) rather than self-play exploits.

Usage:
    python scripts/eval_team.py --model models/ppo_best.pt --n 200
    python scripts/eval_team.py --model models/ppo_best.pt --n 200 --vs_self  # model vs model
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import Counter, defaultdict

import torch

from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action


def run_episode(net, env: CombatEnvV2, device: str,
                agent_archs=None, opp_archs=None) -> dict:
    obs, _ = env.reset(agent_archs=agent_archs, opp_archs=opp_archs)
    done = False
    total_steps = 0

    # Strategic pattern counters
    focus_fire_attacks = 0     # both agents attacked same target on same round
    round_targets: dict[int, set] = defaultdict(set)   # round → set of targets attacked

    while not done:
        total_steps += 1
        agent_id = env.current_agent_id
        round_num = env.ws.combat.round_number if env.ws.combat else 0

        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                 for k, v in obs.items()}
        with torch.no_grad():
            end_l, s, e, g = net(obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, agent_id)
        e = apply_entity_mask(e, obs_t)
        action = list(pick_action(end_l[0], s[0], e[0], g[0],
                                  ws=env.ws, agent_id=agent_id))

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        # Track which enemy was targeted this round (for focus-fire metric)
        result = info.get("action_result") or {}
        target = result.get("target") or result.get("defender")
        if target and target in env.opp_ids:
            round_targets[round_num].add(target)

    # Focus-fire: rounds where both agents attacked and targeted the same enemy
    for targets in round_targets.values():
        if len(targets) == 1:   # single target focused
            focus_fire_attacks += 1

    opps_dead = all(not env.ws.characters[oid].is_alive() for oid in env.opp_ids)
    team_dead = all(not env.ws.characters[aid].is_alive() for aid in env.agent_ids)
    if opps_dead:
        outcome = "win"
    elif team_dead:
        outcome = "loss"
    else:
        outcome = "truncated"

    return {
        "outcome": outcome,
        "agent_archs": tuple(env.agent_archs),
        "opp_archs":   tuple(env.opp_archs),
        "steps": total_steps,
        "rounds": env.ws.combat.round_number if env.ws.combat else 0,
        "focus_fire_rounds": focus_fire_attacks,
        "attack_rounds": len(round_targets),
    }


def print_report(episodes: list[dict], model_path: str):
    n = len(episodes)
    wins   = sum(1 for e in episodes if e["outcome"] == "win")
    losses = sum(1 for e in episodes if e["outcome"] == "loss")
    trunc  = sum(1 for e in episodes if e["outcome"] == "truncated")

    print(f"\n=== Team Eval: {model_path} ===")
    print(f"Episodes: {n}  (model=both allies, expert=both opponents)")
    print(f"  Win:       {wins:4d} ({100.0*wins/n:5.1f}%)")
    print(f"  Loss:      {losses:4d} ({100.0*losses/n:5.1f}%)")
    print(f"  Truncated: {trunc:4d} ({100.0*trunc/n:5.1f}%)")

    # Focus-fire metric
    total_attack_rounds = sum(e["attack_rounds"] for e in episodes) or 1
    ff_rounds = sum(e["focus_fire_rounds"] for e in episodes)
    print(f"\n=== Strategic patterns ===")
    print(f"  Focus-fire rate: {100.0*ff_rounds/total_attack_rounds:5.1f}%"
          f"  ({ff_rounds}/{total_attack_rounds} rounds both attacked same target)")

    # Per-archetype: win rate when this archetype is on agent team.
    # Each archetype's appearance is counted once per episode regardless of
    # how many copies the team has — measures "having archetype X on my side".
    agent_arch_wins: dict[str, list] = defaultdict(list)
    opp_arch_wins:   dict[str, list] = defaultdict(list)
    for e in episodes:
        for arch in set(e["agent_archs"]):
            agent_arch_wins[arch].append(e["outcome"])
        for arch in set(e["opp_archs"]):
            opp_arch_wins[arch].append(e["outcome"])

    print(f"\n=== Per-archetype win rate (archetype is on agent team) ===")
    rows = []
    for arch in ARCHETYPE_LIST:
        outcomes = agent_arch_wins.get(arch, [])
        if not outcomes:
            continue
        w = sum(1 for o in outcomes if o == "win")
        rows.append((arch, w, len(outcomes), 100.0 * w / len(outcomes)))
    rows.sort(key=lambda r: -r[3])
    for arch, w, tot, pct in rows:
        bar = "#" * int(pct / 5)
        print(f"  {arch:18s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    print(f"\n=== Hardest opponent archetypes (lowest win-rate when they are on opp team) ===")
    rows = []
    for arch in ARCHETYPE_LIST:
        outcomes = opp_arch_wins.get(arch, [])
        if not outcomes:
            continue
        w = sum(1 for o in outcomes if o == "win")
        rows.append((arch, w, len(outcomes), 100.0 * w / len(outcomes)))
    rows.sort(key=lambda r: r[3])
    for arch, w, tot, pct in rows:
        bar = "#" * int(pct / 5)
        print(f"  vs {arch:15s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    # Per-agent-team-archetype win rates (sorted by archetype pair)
    pair_wins: dict[tuple, list] = defaultdict(list)
    for e in episodes:
        pair_wins[e["agent_archs"]].append(e["outcome"])

    print(f"\n=== Agent team win rates (top combos) ===")
    rows = []
    for pair, outcomes in pair_wins.items():
        w = sum(1 for o in outcomes if o == "win")
        rows.append((pair, w, len(outcomes), 100.0 * w / len(outcomes)))
    rows.sort(key=lambda r: -r[3])
    for pair, w, tot, pct in rows[:12]:
        label = f"{pair[0]} + {pair[1]}"
        bar = "#" * int(pct / 5)
        print(f"  {label:36s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    # Per-opponent-team-archetype win rates
    opp_pair_wins: dict[tuple, list] = defaultdict(list)
    for e in episodes:
        opp_pair_wins[e["opp_archs"]].append(e["outcome"])

    print(f"\n=== Hardest opponent combos ===")
    rows = []
    for pair, outcomes in opp_pair_wins.items():
        w = sum(1 for o in outcomes if o == "win")
        rows.append((pair, w, len(outcomes), 100.0 * w / len(outcomes)))
    rows.sort(key=lambda r: r[3])
    for pair, w, tot, pct in rows[:8]:
        label = f"vs {pair[0]} + {pair[1]}"
        bar = "#" * int(pct / 5)
        print(f"  {label:38s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    avg_rounds = sum(e["rounds"] for e in episodes) / n
    avg_steps  = sum(e["steps"] for e in episodes) / n
    print(f"\n=== Episode stats ===")
    print(f"  Avg rounds/episode: {avg_rounds:.1f}")
    print(f"  Avg steps/episode:  {avg_steps:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, required=True)
    parser.add_argument("--n",       type=int, default=200)
    parser.add_argument("--seed",    type=int, default=88888)
    parser.add_argument("--vs_self", action="store_true",
                        help="opponents also use the model (model vs model)")
    parser.add_argument("--n_agents", type=int, default=None,
                        help="fixed number of model-controlled agents (1-3); "
                             "omit to sample from weighted distribution")
    parser.add_argument("--n_opps",   type=int, default=None,
                        help="fixed number of opponents (1-3); "
                             "omit to sample from weighted distribution")
    parser.add_argument("--agent_archs", type=str, default=None,
                        help="comma-separated archetype names for agents (e.g. life,fighter)")
    parser.add_argument("--opp_archs",   type=str, default=None,
                        help="comma-separated archetype names for opponents")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    agent_archs = args.agent_archs.split(",") if args.agent_archs else None
    opp_archs   = args.opp_archs.split(",")   if args.opp_archs   else None

    episodes = []
    for i in range(args.n):
        env = CombatEnvV2(seed=args.seed + i,
                          n_agents=args.n_agents, n_opps=args.n_opps)
        if args.vs_self:
            env.use_self_play_opponent(net)
        episodes.append(run_episode(net, env, device,
                                     agent_archs=agent_archs,
                                     opp_archs=opp_archs))
        if (i + 1) % 50 == 0:
            wins = sum(1 for e in episodes if e["outcome"] == "win")
            print(f"  ... {i+1}/{args.n}  running win-rate: {100.0*wins/(i+1):5.1f}%")

    print_report(episodes, args.model)


if __name__ == "__main__":
    main()
