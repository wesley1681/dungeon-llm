"""Probe a model's per-archetype skill-choice distribution.

For each archetype, runs:
  - the MODEL as that archetype for N episodes vs expert opponents
  - the EXPERT for that archetype for N episodes vs the same opponents
and prints both skill distributions side-by-side.

Catches policy collapse the BC training metrics hide — e.g., 'wizard never
casts spells' shows up as model.spell_ratio ≈ 0% while expert.spell_ratio
≈ 60%, even though the BC loss looked fine.

Usage:
    python scripts/probe_skill_dist.py --model models/bc_v9_perskill.pt --n 20
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from collections import Counter

import torch

from trpg.engine.abilities import CLASS_ABILITIES
from trpg.engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.skill import available_skills
from trpg.engine.status import tick_status_effects
from trpg.rl.env_v2 import (
    CombatEnvV2, ARCHETYPE_LIST,
    _AGENT_ID, _OPPONENT_ID, _MAX_SUB_ACTIONS_PER_TURN,
)
from trpg.rl.model import (
    CombatPolicyNet, apply_resource_mask, apply_entity_mask, pick_action,
)


def _category(skill_id: str) -> str:
    """Bucket a skill_id into a coarse action family."""
    if skill_id == "end":  return "end"
    if skill_id == "move": return "move"
    if skill_id.startswith("weapon:"): return "weapon"
    if skill_id.startswith("spell:"):  return "spell"
    if skill_id in ("dodge", "hide", "disengage"): return "defense"
    if skill_id in CLASS_ABILITIES:    return "ability"
    return "other"


def run_model_episodes(net, agent_arch: str, n: int, device: str,
                        seed: int) -> Counter:
    """Drive the model as agent for n episodes; return Counter[skill_id]."""
    counter: Counter = Counter()
    for ep in range(n):
        env = CombatEnvV2(seed=seed + ep)
        obs, _ = env.reset(agent_arch=agent_arch)
        done = False
        while not done:
            agent = env.ws.characters[_AGENT_ID]
            skills_now = available_skills(agent, env.ws)
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device)
                     for k, v in obs.items()}
            with torch.no_grad():
                end_l, s, e, g = net(obs_t)
            s_m = apply_resource_mask(s, env.resources, env.ws, _AGENT_ID)
            e_m = apply_entity_mask(e, obs_t)
            action = list(pick_action(end_l[0], s_m[0], e_m[0], g[0],
                                       ws=env.ws, agent_id=_AGENT_ID))
            skill_idx = action[0]
            if 0 <= skill_idx < len(skills_now):
                counter[skills_now[skill_idx].skill_id] += 1
            else:
                counter["?"] += 1
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return counter


def run_expert_episodes(agent_arch: str, n: int, seed: int) -> Counter:
    """Drive the scripted expert as agent for n episodes; return Counter[skill_id].

    Mirrors env_v2's turn loop manually because env_v2 expects a model-style
    action triplet on the agent side, not a CombatDecision from a policy.
    """
    counter: Counter = Counter()
    for ep in range(n):
        env = CombatEnvV2(seed=seed + ep)
        env.reset(agent_arch=agent_arch)
        expert = make_archetype_policy(agent_arch)
        max_rounds = 20
        while env.ws.combat.round_number <= max_rounds:
            agent = env.ws.characters[_AGENT_ID]
            opp = env.ws.characters[_OPPONENT_ID]
            if not agent.is_alive() or not opp.is_alive():
                break
            # Agent turn (expert)
            agent.reaction_used = False
            tick_status_effects(agent, "self_turn_start", env.ws.combat.round_number)
            tick_terrain_damage(agent, env.ws.combat.battlefield)
            env.resources = {"action": 1, "bonus_action": 1,
                             "movement": MOVE_BUDGET_M}
            for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
                if not agent.is_alive():
                    break
                decision = expert.decide(
                    _AGENT_ID, agent, env.ws, env.resources,
                    env.ws.combat.round_number,
                )
                if decision.action is None:
                    counter["end"] += 1
                    break
                counter[decision.action.get("skill_id", "?")] += 1
                if decision.ended:
                    break
                r = execute_action(decision.action, env.ws)
                if r.get("type") != "ERROR":
                    consume_resources(env.resources, decision.action, r)
                if (env.resources["action"] <= 0
                        and env.resources["bonus_action"] <= 0
                        and env.resources["movement"] <= 1e-6):
                    break
            tick_status_effects(agent, "self_turn_end",
                                 env.ws.combat.round_number)
            if not opp.is_alive():
                break
            # Opp turn — use env's stock opponent loop
            env._run_opponent_turn()
            env._end_of_round_tick()
    return counter


def _print_breakdown(label: str, counter: Counter) -> None:
    total = sum(counter.values()) or 1
    cats: Counter = Counter()
    for sid, n in counter.items():
        cats[_category(sid)] += n
    cat_str = "  ".join(f"{c}={100*cats[c]/total:.0f}%"
                        for c in ("weapon", "spell", "ability", "move", "defense", "end")
                        if cats[c] > 0)
    print(f"  {label} ({total} steps): {cat_str}")
    # Top-5 skill_ids
    top = counter.most_common(5)
    top_str = "  ".join(f"{sid}={100*n/total:.0f}%" for sid, n in top)
    print(f"    top: {top_str}")


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

    flagged: list[str] = []
    for arch in ARCHETYPE_LIST:
        print(f"=== {arch} ===")
        model_dist = run_model_episodes(net, arch, args.n, device, args.seed)
        expert_dist = run_expert_episodes(arch, args.n, args.seed)
        _print_breakdown("MODEL ", model_dist)
        _print_breakdown("EXPERT", expert_dist)
        # Flag big collapses: any category where expert > 15% but model < 3%
        m_total = sum(model_dist.values()) or 1
        e_total = sum(expert_dist.values()) or 1
        m_cats = Counter(); e_cats = Counter()
        for sid, n in model_dist.items():
            m_cats[_category(sid)] += n
        for sid, n in expert_dist.items():
            e_cats[_category(sid)] += n
        for c in ("weapon", "spell", "ability"):
            e_pct = 100 * e_cats[c] / e_total
            m_pct = 100 * m_cats[c] / m_total
            if e_pct >= 15 and m_pct < 3:
                flagged.append(f"  {arch}: expert {c} {e_pct:.0f}% → model {m_pct:.0f}%")
        print()

    if flagged:
        print("\n!!! COLLAPSE FLAGS (expert ≥ 15% but model < 3%):")
        for line in flagged:
            print(line)
    else:
        print("\nNo collapses detected (model ≥ 3% on all categories where expert ≥ 15%).")


if __name__ == "__main__":
    main()
