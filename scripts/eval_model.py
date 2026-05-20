"""Behavior evaluation for a trained policy.

Runs N episodes with random matchups and reports:
  - Overall win / loss / truncated rates
  - Per-archetype win rate (as agent and as opponent)
  - Action family distribution (move / weapon / spell / ability / defense / end)
  - End-spam rate (steps that voluntarily ended turn with resources left)
  - Average rounds per episode
  - Resource usage (slots spent, lay_on_hands consumed, concentration uptime)

Usage:
    python scripts/eval_model.py --model models/ppo_best.pt --n 200
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from trpg.engine.skill import available_skills
from trpg.engine.abilities import CLASS_ABILITIES
from trpg.rl.env_v2 import CombatEnvV2, _AGENT_ID, _OPPONENT_ID, ARCHETYPE_LIST
from trpg.rl.model import CombatPolicyNet, apply_resource_mask, apply_entity_mask


def categorize(skill_id: str) -> str:
    """Bucket a skill_id into an action family for aggregate stats."""
    if skill_id == "end":
        return "end"
    if skill_id == "move":
        return "move"
    if skill_id.startswith("weapon:"):
        return "weapon"
    if skill_id.startswith("spell:"):
        return "spell"
    if skill_id in ("dodge", "hide", "disengage"):
        return "defense"
    if skill_id in CLASS_ABILITIES:
        return "ability"
    return "other"


def aggregate(episodes: list[dict]) -> dict:
    """Aggregate per-episode stats into a report dict."""
    n = len(episodes)
    wins = sum(1 for e in episodes if e["outcome"] == "win")
    losses = sum(1 for e in episodes if e["outcome"] == "loss")
    trunc = sum(1 for e in episodes if e["outcome"] == "truncated")

    by_agent: dict[str, list] = defaultdict(list)
    by_opp: dict[str, list] = defaultdict(list)
    for e in episodes:
        by_agent[e["agent_arch"]].append(e["outcome"])
        by_opp[e["opp_arch"]].append(e["outcome"])

    action_total: Counter[str] = Counter()
    for e in episodes:
        action_total.update(e["actions"])
    total_steps = sum(action_total.values()) or 1

    end_with_resources = sum(e["end_with_resources"] for e in episodes)
    avg_rounds = sum(e["rounds"] for e in episodes) / n
    avg_slots = sum(e["slots_used"] for e in episodes) / n
    avg_loh = sum(e["loh_used"] for e in episodes) / n
    conc_total = sum(e["concentration_steps"] for e in episodes)

    return {
        "n": n, "wins": wins, "losses": losses, "trunc": trunc,
        "by_agent": by_agent, "by_opp": by_opp,
        "action_pct": {k: 100.0 * v / total_steps for k, v in action_total.items()},
        "end_spam_rate": 100.0 * end_with_resources / total_steps,
        "avg_rounds": avg_rounds,
        "avg_slots_used": avg_slots,
        "avg_loh_used": avg_loh,
        "concentration_pct": 100.0 * conc_total / total_steps,
        "total_steps": total_steps,
    }


def print_report(report: dict, model_path: str):
    n = report["n"]
    wr = 100.0 * report["wins"] / n
    print(f"\n=== Model: {model_path} ===")
    print(f"Episodes: {n}")
    print(f"  Win:       {report['wins']:4d} ({wr:5.1f}%)")
    print(f"  Loss:      {report['losses']:4d} ({100.0*report['losses']/n:5.1f}%)")
    print(f"  Truncated: {report['trunc']:4d} ({100.0*report['trunc']/n:5.1f}%)")

    print(f"\n=== Per-archetype win-rate (as agent) ===")
    rows = []
    for arch in ARCHETYPE_LIST:
        outcomes = report["by_agent"].get(arch, [])
        if not outcomes:
            continue
        wins = sum(1 for o in outcomes if o == "win")
        rows.append((arch, wins, len(outcomes), 100.0 * wins / len(outcomes)))
    rows.sort(key=lambda r: -r[3])
    for arch, w, tot, pct in rows:
        bar = "#" * int(pct / 5)
        print(f"  {arch:18s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    print(f"\n=== Hardest opponents (lowest win-rate against) ===")
    rows = []
    for arch in ARCHETYPE_LIST:
        outcomes = report["by_opp"].get(arch, [])
        if not outcomes:
            continue
        wins = sum(1 for o in outcomes if o == "win")
        rows.append((arch, wins, len(outcomes), 100.0 * wins / len(outcomes)))
    rows.sort(key=lambda r: r[3])
    for arch, w, tot, pct in rows:
        bar = "#" * int(pct / 5)
        print(f"  vs {arch:15s}  {w:3d}/{tot:<3d}  {pct:5.1f}%  {bar}")

    print(f"\n=== Action distribution ({report['total_steps']} total steps) ===")
    items = sorted(report["action_pct"].items(), key=lambda x: -x[1])
    for family, pct in items:
        bar = "#" * int(pct / 2)
        print(f"  {family:10s}  {pct:5.1f}%  {bar}")

    print(f"\n=== Behavior diagnostics ===")
    print(f"  End-spam rate:        {report['end_spam_rate']:5.2f}%  (voluntary end with resources)")
    print(f"  Concentration uptime: {report['concentration_pct']:5.2f}%  (steps while concentrating)")
    print(f"  Avg rounds/episode:   {report['avg_rounds']:5.1f}")
    print(f"  Avg spell slots used: {report['avg_slots_used']:5.2f}")
    print(f"  Avg lay-on-hands HP:  {report['avg_loh_used']:5.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n", type=int, default=200, help="number of episodes")
    parser.add_argument("--seed", type=int, default=88888, help="base seed")
    parser.add_argument("--agent", type=str, default=None,
                        help="lock agent archetype (default: random)")
    parser.add_argument("--opponent", type=str, default=None,
                        help="lock opponent archetype (default: random)")
    parser.add_argument("--opponent_model", type=str, default=None,
                        help="if set, opponent uses this network checkpoint "
                             "(self-play eval) instead of the ArchetypePolicy")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = CombatPolicyNet().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    opp_net = None
    if args.opponent_model:
        opp_net = CombatPolicyNet()
        opp_net.load_state_dict(torch.load(args.opponent_model, map_location="cpu"))
        opp_net.eval()
        print(f"Self-play eval: agent={args.model}, opponent={args.opponent_model}")

    episodes = []
    for i in range(args.n):
        env = CombatEnvV2(seed=args.seed + i)
        if opp_net is not None:
            env.use_self_play_opponent(opp_net)
        episodes.append(run_episode(net, env, device, args.agent, args.opponent))
        if (i + 1) % 50 == 0:
            wins = sum(1 for e in episodes if e["outcome"] == "win")
            print(f"  ... {i+1}/{args.n}  running win-rate: {100.0*wins/(i+1):5.1f}%")

    report = aggregate(episodes)
    print_report(report, args.model)


def run_episode(net, env, device, agent_arch=None, opp_arch=None) -> dict:
    """Run one episode, return per-episode stats. Locked archetypes optional."""
    obs, _ = env.reset(agent_arch=agent_arch, opponent_arch=opp_arch)
    agent = env.ws.characters[_AGENT_ID]
    opp = env.ws.characters[_OPPONENT_ID]
    agent_max_hp = agent.max_hp
    opp_max_hp = opp.max_hp
    starting_slots = sum(agent.spell_slots.values())
    starting_loh = getattr(agent, "lay_on_hands_pool", 0)

    action_counts: Counter[str] = Counter()
    end_with_resources = 0
    total_steps = 0
    concentration_steps = 0

    done = False
    while not done:
        total_steps += 1
        obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                 for k, v in obs.items()}
        with torch.no_grad():
            s, e, g = net(obs_t)
        s = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
        e = apply_entity_mask(e, obs_t)
        action = [int(s[0].argmax()), int(e[0].argmax()), int(g.argmax())]

        skills_before = available_skills(agent, env.ws)
        if 0 <= action[0] < len(skills_before):
            skill_id = skills_before[action[0]].skill_id
        else:
            skill_id = "end"
        action_counts[categorize(skill_id)] += 1
        had_action = env.resources.get("action", 0) > 0
        had_bonus = env.resources.get("bonus_action", 0) > 0

        if agent.concentrating_on:
            concentration_steps += 1

        obs, _, term, trunc, _ = env.step(action)
        if skill_id == "end" and (had_action or had_bonus):
            end_with_resources += 1
        done = term or trunc

    rounds = env.ws.combat.round_number if env.ws.combat else 0
    if not opp.is_alive():
        outcome = "win"
    elif not agent.is_alive():
        outcome = "loss"
    else:
        outcome = "truncated"

    return {
        "outcome": outcome,
        "agent_arch": env.agent_arch,
        "opp_arch": env.opponent_arch,
        "rounds": rounds,
        "steps": total_steps,
        "agent_hp_final": agent.hp,
        "opp_hp_final": opp.hp,
        "agent_max_hp": agent_max_hp,
        "opp_max_hp": opp_max_hp,
        "actions": action_counts,
        "end_with_resources": end_with_resources,
        "slots_used": starting_slots - sum(agent.spell_slots.values()),
        "loh_used": starting_loh - getattr(agent, "lay_on_hands_pool", 0),
        "concentration_steps": concentration_steps,
    }


if __name__ == "__main__":
    main()
