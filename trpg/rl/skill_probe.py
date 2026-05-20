"""Per-archetype skill-distribution probe.

Drop-in behavioural check that runs after BC training (or against any saved
checkpoint) and reveals collapses that point-wise per-head accuracy hides:

    - model uses skill_id X heavily, expert doesn't (over-imitation)
    - expert uses skill_id Y heavily, model never picks it (collapse)
    - whole categories missing (e.g. spell ratio 0% on wizards, or end ratio
      0% across the board — the symptom that motivated the multi-head fix)

Usage from a script:

    from trpg.rl.skill_probe import probe_and_report
    flags = probe_and_report(net, n_episodes=10, device="cuda")
    if flags:
        ...

The function prints a side-by-side per-archetype report and returns a list
of flag strings; an empty list means no major collapses were detected.
"""
from __future__ import annotations
from collections import Counter

import torch

from ..engine.abilities import CLASS_ABILITIES
from ..engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
)
from ..engine.combat_policy import make_archetype_policy
from ..engine.skill import available_skills
from ..engine.status import tick_status_effects
from .env_v2 import (
    CombatEnvV2, ARCHETYPE_LIST,
    _AGENT_ID, _OPPONENT_ID, _MAX_SUB_ACTIONS_PER_TURN,
)
from .model import apply_resource_mask, apply_entity_mask, pick_action


# Categories used to bucket skill_ids for high-level reporting + collapse
# flags. Note: caster archetypes mostly use class-ability shortcuts like
# `fireball_ev`, `hold_person` which land in `ability`. The bare `spell:`
# prefix covers the generic spell catalog and is rarely picked.
_CATEGORIES = ("weapon", "spell", "ability", "move", "defense", "end")


def _category(skill_id: str) -> str:
    if skill_id == "end":  return "end"
    if skill_id == "move": return "move"
    if skill_id.startswith("weapon:"): return "weapon"
    if skill_id.startswith("spell:"):  return "spell"
    if skill_id in ("dodge", "hide", "disengage"): return "defense"
    if skill_id in CLASS_ABILITIES:    return "ability"
    return "other"


def run_model_episodes(net, agent_arch: str, n: int, device: str,
                        seed: int) -> Counter:
    """Drive ``net`` as agent for n episodes; return Counter[skill_id]."""
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

    Drives the engine manually because env_v2.step expects a model-style
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
            agent.reaction_used = False
            agent.leveled_spell_cast_this_turn = False
            tick_status_effects(agent, "self_turn_start",
                                 env.ws.combat.round_number)
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
            env._run_opponent_turn()
            env._end_of_round_tick()
    return counter


def _category_counts(c: Counter) -> Counter:
    out: Counter = Counter()
    for sid, n in c.items():
        out[_category(sid)] += n
    return out


def _format_breakdown(label: str, counter: Counter) -> list[str]:
    total = sum(counter.values()) or 1
    cats = _category_counts(counter)
    cat_str = "  ".join(f"{c}={100*cats[c]/total:.0f}%"
                        for c in _CATEGORIES if cats[c] > 0)
    top = counter.most_common(5)
    top_str = "  ".join(f"{sid}={100*n/total:.0f}%" for sid, n in top)
    return [f"  {label} ({total} steps): {cat_str}",
            f"    top: {top_str}"]


def probe_and_report(net, *, n_episodes: int = 10, device: str = "cpu",
                     seed: int = 99999,
                     expert_threshold_pct: float = 15.0,
                     model_threshold_pct: float = 3.0,
                     verbose: bool = True) -> list[str]:
    """Run the per-archetype skill probe and return flagged category collapses.

    A collapse is flagged when, for some archetype + category, the expert
    uses ``category`` at least ``expert_threshold_pct``% of the time but the
    model uses it less than ``model_threshold_pct``%. The ``end`` category
    is included — a 0% end rate against a 30-50% expert is exactly the bug
    the multi-head architecture was supposed to fix.
    """
    net.eval()
    flags: list[str] = []
    for arch in ARCHETYPE_LIST:
        if verbose:
            print(f"=== {arch} ===")
        m = run_model_episodes(net, arch, n_episodes, device, seed)
        e = run_expert_episodes(arch, n_episodes, seed)
        if verbose:
            for line in _format_breakdown("MODEL ", m):
                print(line)
            for line in _format_breakdown("EXPERT", e):
                print(line)
            print()
        m_total = sum(m.values()) or 1
        e_total = sum(e.values()) or 1
        m_cats = _category_counts(m)
        e_cats = _category_counts(e)
        for c in _CATEGORIES:
            e_pct = 100 * e_cats[c] / e_total
            m_pct = 100 * m_cats[c] / m_total
            if e_pct >= expert_threshold_pct and m_pct < model_threshold_pct:
                flags.append(
                    f"  {arch}: expert {c} {e_pct:.0f}% -> model{m_pct:.0f}%"
                )
    if verbose:
        if flags:
            print(f"\n!!! COLLAPSE FLAGS "
                  f"(expert >= {expert_threshold_pct:.0f}% but model "
                  f"< {model_threshold_pct:.0f}%):")
            for f in flags:
                print(f)
        else:
            print(f"\nNo collapses detected "
                  f"(model >= {model_threshold_pct:.0f}% wherever "
                  f"expert >= {expert_threshold_pct:.0f}%).")
    return flags
