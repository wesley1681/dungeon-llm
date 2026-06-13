"""Diagnose BC encoding correctness.

Runs the BC data collection logic but counts:
  - Total expert decisions (action != None)
  - End-turn signals (action == None)
  - Decisions that match a real skill (skill_idx >= 0)
  - Decisions that fail to match (skill_idx == -1, dropped)
  - Per-action-type breakdown of matched vs dropped

Also verifies that after the fix, no decision gets silently coded as
skill_idx=0 (the end-turn slot).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter, defaultdict

from trpg.engine.combat import (
    execute_action, consume_resources, MOVE_BUDGET_M, tick_terrain_damage,
)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.engine.status import tick_status_effects
from trpg.rl.env_v2 import (
    CombatEnvV2, ARCHETYPE_LIST,
    _MAX_SUB_ACTIONS_PER_TURN,
)
from trpg.rl.obs import build_obs
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills


def diagnose_archetype(agent_arch: str, opp_arch: str, level: int, seed: int) -> dict:
    env = CombatEnvV2(seed=seed, n_agents=1, n_opps=1)
    env.reset(
        agent_archs=[agent_arch] if agent_arch else None,
        opp_archs=[opp_arch] if opp_arch else None,
        level=level,
    )
    expert = make_archetype_policy(agent_arch)
    agent_id = env.agent_ids[0]
    opp_id = env.opp_ids[0]

    counts = Counter()
    matched_by_type = Counter()
    dropped_by_type = Counter()
    matched_skill_distribution = Counter()

    while env.ws.combat.round_number <= 30:
        agent = env.ws.characters[agent_id]
        opp = env.ws.characters[opp_id]
        if not agent.is_alive() or not opp.is_alive():
            break
        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
            decision = expert.decide(agent_id, agent, env.ws,
                                     env.resources, env.ws.combat.round_number)
            if decision.action is None or decision.fled:
                counts["end_or_fled"] += 1
                break
            counts["decisions"] += 1
            a_type = decision.action.get("type", "?")
            enc = encode_action(decision.action, env.ws, agent_id)
            if enc[0] < 0:
                counts["dropped"] += 1
                dropped_by_type[a_type] += 1
            else:
                counts["matched"] += 1
                matched_by_type[a_type] += 1
                matched_skill_distribution[enc[0]] += 1
                # Critical sanity check: skill_idx=0 means we labeled the
                # decision as the end-turn skill. Should never happen for a
                # real expert action.
                if enc[0] == 0:
                    skills = available_skills(agent, env.ws)
                    counts["matched_as_end"] += 1
                    print(f"  [BUG] {agent_arch} action {a_type} matched to skill 0 "
                          f"(skill_id={skills[0].skill_id})")
            if decision.ended:
                break
            r = execute_action(decision.action, env.ws)
            if r.get("type") != "ERROR":
                consume_resources(env.resources, decision.action, r)
            if (env.resources["action"] <= 0
                and env.resources["bonus_action"] <= 0
                and env.resources["movement"] <= 1e-6):
                break
        tick_status_effects(agent, "self_turn_end", env.ws.combat.round_number)
        if opp.is_alive():
            env._run_opponent_turn()
        env._end_of_round_tick()
        if agent.is_alive():
            env.resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
            agent.reaction_used = False
            tick_status_effects(agent, "self_turn_start", env.ws.combat.round_number)
            tick_terrain_damage(agent, env.ws.combat.battlefield)

    return {
        "counts": counts,
        "matched_by_type": matched_by_type,
        "dropped_by_type": dropped_by_type,
        "matched_skill_distribution": matched_skill_distribution,
    }


def main():
    total = Counter()
    matched_by_type = Counter()
    dropped_by_type = Counter()
    skill_distribution = Counter()
    per_archetype: dict[str, dict] = {}

    import numpy as np
    rng = np.random.default_rng(42)
    n_eps_per_arch = 30

    for agent_arch in ARCHETYPE_LIST:
        arch_counts = Counter()
        for ep in range(n_eps_per_arch):
            opp_arch = ARCHETYPE_LIST[rng.integers(len(ARCHETYPE_LIST))]
            level = int(rng.integers(3, 9))
            ep_seed = int(rng.integers(0, 2**31))
            r = diagnose_archetype(agent_arch, opp_arch, level, ep_seed)
            total.update(r["counts"])
            arch_counts.update(r["counts"])
            matched_by_type.update(r["matched_by_type"])
            dropped_by_type.update(r["dropped_by_type"])
            skill_distribution.update(r["matched_skill_distribution"])
        per_archetype[agent_arch] = arch_counts

    n_dec = total["decisions"]
    n_match = total["matched"]
    n_drop = total["dropped"]
    print(f"\n=== Overall encoding stats over {n_eps_per_arch} eps/archetype ===")
    print(f"Total expert decisions:  {n_dec:,}")
    print(f"  Matched (kept):        {n_match:,}  ({100*n_match/n_dec:.1f}%)")
    print(f"  Unmatched (dropped):   {n_drop:,}  ({100*n_drop/n_dec:.1f}%)")
    print(f"  Mis-labeled as skill 0: {total.get('matched_as_end', 0)}  "
          "(should be 0)")
    print(f"End-turn signals (action=None): {total['end_or_fled']:,}  "
          "(correctly dropped from dataset)")

    print(f"\n=== Matched action types ===")
    for t, c in matched_by_type.most_common():
        print(f"  {t:14s}  {c:5d}")

    print(f"\n=== Dropped action types (unmatched, dropped from dataset) ===")
    if dropped_by_type:
        for t, c in dropped_by_type.most_common():
            print(f"  {t:14s}  {c:5d}")
    else:
        print("  (none — every expert action was matched)")

    print(f"\n=== Per-archetype drop rate ===")
    for arch, c in per_archetype.items():
        nd = c["decisions"] or 1
        drop_pct = 100 * c["dropped"] / nd
        n0 = c.get("matched_as_end", 0)
        print(f"  {arch:18s}  decisions={c['decisions']:4d}  "
              f"dropped={c['dropped']:4d} ({drop_pct:4.1f}%)  "
              f"mislabeled_as_end={n0}")

    print(f"\n=== Final dataset skill_idx distribution (top 15) ===")
    for sid, c in skill_distribution.most_common(15):
        print(f"  skill_idx {sid:3d}  {c:5d}")
    if 0 in skill_distribution:
        print(f"\n  WARNING: skill_idx 0 appears {skill_distribution[0]} times in matched data")
        print(f"  (skill 0 is the end-turn slot — these would teach BC to end-spam)")
    else:
        print(f"\n  OK: skill_idx 0 never appears in matched data — no end-spam pollution.")


if __name__ == "__main__":
    main()
