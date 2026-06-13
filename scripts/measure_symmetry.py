"""Decisive test: is the 'agent' role disadvantaged by its EXECUTION PATH?

measure_draws showed expert-vs-expert gives agent_win 33.7% / opp_win 64.8%
despite IDENTICAL policies. Two candidate causes:
  (A) execution path: agent goes decide->encode->decode->step, and step() ends
      the turn the moment `action` is spent (leftover bonus/move wasted);
      the opponent runs a full multi-subaction turn via _run_opponent_turn
      and executes decisions DIRECTLY.
  (B) spawn/positioning: party vs enemy spawn asymmetry.

This script removes (A): it builds the same world as env.reset(), then drives
EVERY character (agent and opp alike) through the IDENTICAL direct full-turn
loop (a copy of _run_opponent_turn). If agent_win climbs to ~50%, the
asymmetry was the agent's step() execution path, not spawning.

Run also a 'env-path' control that uses the real env.step for the agent, so
the two numbers are directly comparable on the same matchups/seeds.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from trpg.engine.world_state import WorldState, CombatState
from trpg.engine.combat import (execute_action, consume_resources,
                                 setup_combat_positions, MOVE_BUDGET_M,
                                 tick_terrain_damage, tick_aura_damage)
from trpg.engine.status import tick_status_effects
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
from trpg.rl.env_v2 import CombatEnvV2, ARCHETYPE_LIST, _MAX_SUB_ACTIONS_PER_TURN
import random


from trpg.rl.action import encode_action, decode_action

# Module-level switch for how the AGENT side executes its turn.
#   "full"        : opponent-identical full-turn loop, direct execute
#   "roundtrip"   : full-turn loop BUT route each action through
#                   encode_action->decode_action (isolates round-trip loss)
#   "truncated"   : full-turn loop BUT end the turn as soon as `action`<=0,
#                   mimicking env.step's turn_done (isolates resource waste)
AGENT_MODE = "full"


def run_full_turn(cid, ws, policy, agent_side=False):
    """Identical to env._run_opponent_turn, applied to ANY character.

    When agent_side and AGENT_MODE != 'full', inject the candidate
    step()-path defects to isolate which one costs the agent its win rate.
    """
    mode = AGENT_MODE if agent_side else "full"
    c = ws.characters[cid]
    c.reaction_used = False
    c.leveled_spell_cast_this_turn = False
    tick_status_effects(c, "self_turn_start", ws.combat.round_number)
    tick_terrain_damage(c, ws.combat.battlefield)
    tick_aura_damage(c, ws, ws.combat.round_number)
    if any(c.has_status(s) for s in ("paralyzed", "asleep", "stunned")):
        tick_status_effects(c, "self_turn_end", ws.combat.round_number)
        return
    resources = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
    for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
        if not c.is_alive():
            break
        decision = policy.decide(cid, c, ws, resources, ws.combat.round_number)
        if decision.fled or decision.action is None:
            break
        act = decision.action
        if mode == "roundtrip":
            triplet = encode_action(act, ws, cid)
            act = decode_action(list(triplet), ws, cid)
            if act is None:
                break
        r = execute_action(act, ws)
        if r.get("type") != "ERROR":
            consume_resources(resources, act, r)
        if decision.ended:
            break
        if mode == "truncated" and resources["action"] <= 0:
            break   # env.step's turn_done: leftover bonus/move wasted
        if (resources["action"] <= 0 and resources["bonus_action"] <= 0
                and resources["movement"] <= 1e-6):
            break
    tick_status_effects(c, "self_turn_end", ws.combat.round_number)


def play_symmetric(agent_arch, opp_arch, seed):
    """Both sides driven by run_full_turn — identical execution path."""
    rng = random.Random(seed)
    lvl = 5
    chars = {}
    a = ARCHETYPE_FACTORIES[agent_arch](level=lvl); a.char_id = "agent_0"
    a.is_npc = False; chars["agent_0"] = a
    o = ARCHETYPE_FACTORIES[opp_arch](level=lvl); o.char_id = "opp_0"
    o.is_npc = True; o.attitude = 0; chars["opp_0"] = o
    all_ids = ["agent_0", "opp_0"]; rng.shuffle(all_ids)
    ws = WorldState(characters=chars, scene="rl_team",
                    pc_ids=["agent_0"], party_ids=["agent_0"])
    ws.combat = CombatState(active=True, initiative_order=all_ids, round_number=1)
    setup_combat_positions(ws, ws.combat, rng=rng)
    policies = {"agent_0": make_archetype_policy(agent_arch),
                "opp_0":   make_archetype_policy(opp_arch)}
    max_rounds = 50
    for _ in range(max_rounds):
        for cid in all_ids:
            if ws.characters[cid].is_alive():
                run_full_turn(cid, ws, policies[cid], agent_side=(cid == "agent_0"))
            if not chars["agent_0"].is_alive() or not chars["opp_0"].is_alive():
                break
        for cid in all_ids:
            cc = ws.characters.get(cid)
            if cc and cc.is_alive():
                tick_status_effects(cc, "round_end", ws.combat.round_number)
        ws.combat.round_number += 1
        if not chars["agent_0"].is_alive() or not chars["opp_0"].is_alive():
            break
    aa = chars["agent_0"].is_alive(); oa = chars["opp_0"].is_alive()
    if not oa and aa:
        return "agent_win"
    if not aa and oa:
        return "opp_win"
    if not aa and not oa:
        return "both_dead"
    return "timeout"


def main():
    global AGENT_MODE
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    archs = list(ARCHETYPE_LIST)
    AGENT_MODE = "full"   # clean expert ceiling (no env handicaps)
    import numpy as np
    per_arch = {a: {"agent_win": 0, "total": 0} for a in archs}
    for ag in archs:
        for op in archs:
            base = hash(f"{ag}_{op}") & 0xFFFFFF
            for i in range(games):
                r = play_symmetric(ag, op, base + i)
                per_arch[ag]["total"] += 1
                if r == "agent_win":
                    per_arch[ag]["agent_win"] += 1
    print("=== CLEAN EXPERT ceiling, per agent-archetype (run_full_turn, both sides) ===")
    for a in archs:
        wr = per_arch[a]["agent_win"] / max(1, per_arch[a]["total"])
        print(f"  {a:18s} {wr:5.1%}")
    overall = sum(p["agent_win"] for p in per_arch.values()) / sum(p["total"] for p in per_arch.values())
    print(f"  {'OVERALL':18s} {overall:5.1%}")


if __name__ == "__main__":
    main()
