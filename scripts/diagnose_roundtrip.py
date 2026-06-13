"""Diagnose encode_action -> decode_action fidelity loss.

BC stores encode_action(expert_decision) as the label but advances the world
with the ORIGINAL action. At eval the model emits a triplet that decode_action
turns back into an engine action. If encode->decode is not identity, BC clones
a degraded action (measured ~7 win-rate points in measure_symmetry).

This script runs expert turns across all archetypes and classifies, per
decision, whether decode_action(encode_action(a)) reproduces `a`:
  - OK            : same skill_id, same target, position within 0.6m
  - SKILL_CHANGED : skill_id differs (or decode returns None / wrong skill)
  - TARGET_CHANGED: target char differs
  - POS_DRIFT     : same skill but target position moved > 0.6m
  - DROPPED       : encode returned (-1,..) -> action lost entirely
Reports the breakdown by target_type so we know which action classes suffer.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter, defaultdict
import random

from trpg.engine.world_state import WorldState, CombatState
from trpg.engine.combat import (execute_action, consume_resources,
                                 setup_combat_positions, MOVE_BUDGET_M)
from trpg.engine.combat_policy import make_archetype_policy
from trpg.scenarios.archetypes import ARCHETYPE_FACTORIES
from trpg.rl.env_v2 import ARCHETYPE_LIST, _MAX_SUB_ACTIONS_PER_TURN
from trpg.rl.action import encode_action, decode_action

CLOSE_M = 0.6


def classify(orig, rt):
    if rt is None:
        return "SKILL_CHANGED"
    if orig.get("skill_id") != rt.get("skill_id"):
        return "SKILL_CHANGED"
    ot = orig.get("target") or orig.get("character")
    rtt = rt.get("target") or rt.get("character")
    if ot != rtt:
        return "TARGET_CHANGED"
    op = orig.get("target_position")
    rp = rt.get("target_position")
    if op is not None and rp is not None:
        d = ((op[0] - rp[0]) ** 2 + (op[1] - rp[1]) ** 2) ** 0.5
        if d > CLOSE_M:
            return "POS_DRIFT"
    return "OK"


def run(games=4):
    by_tt = defaultdict(Counter)
    overall = Counter()
    skill_examples = Counter()
    for ag in ARCHETYPE_LIST:
        for op in ARCHETYPE_LIST:
            base = hash(f"{ag}_{op}") & 0xFFFFFF
            for g in range(games):
                rng = random.Random(base + g)
                chars = {}
                a = ARCHETYPE_FACTORIES[ag](level=5); a.char_id = "agent_0"
                a.is_npc = False; chars["agent_0"] = a
                o = ARCHETYPE_FACTORIES[op](level=5); o.char_id = "opp_0"
                o.is_npc = True; o.attitude = 0; chars["opp_0"] = o
                ids = ["agent_0", "opp_0"]; rng.shuffle(ids)
                ws = WorldState(characters=chars, scene="rl",
                                pc_ids=["agent_0"], party_ids=["agent_0"])
                ws.combat = CombatState(active=True, initiative_order=ids,
                                        round_number=1)
                setup_combat_positions(ws, ws.combat, rng=rng)
                pol = {"agent_0": make_archetype_policy(ag),
                       "opp_0": make_archetype_policy(op)}
                for _round in range(30):
                    for cid in ids:
                        c = ws.characters[cid]
                        if not c.is_alive():
                            continue
                        res = {"action": 1, "bonus_action": 1,
                               "movement": MOVE_BUDGET_M}
                        for _ in range(_MAX_SUB_ACTIONS_PER_TURN):
                            if not c.is_alive():
                                break
                            d = pol[cid].decide(cid, c, ws, res,
                                                ws.combat.round_number)
                            if d.fled or d.action is None:
                                break
                            if cid == "agent_0":
                                enc = encode_action(d.action, ws, cid)
                                if enc[0] < 0:
                                    overall["DROPPED"] += 1
                                    sid = d.action.get("skill_id", "?")
                                    skill_examples[("DROPPED", sid)] += 1
                                else:
                                    rt = decode_action(list(enc), ws, cid)
                                    verdict = classify(d.action, rt)
                                    overall[verdict] += 1
                                    sid = d.action.get("skill_id", "?")
                                    tt = "?"
                                    try:
                                        from trpg.engine.skill import available_skills
                                        sk = available_skills(c, ws)[enc[0]]
                                        tt = sk.features.target_type.name
                                    except Exception:
                                        pass
                                    by_tt[tt][verdict] += 1
                                    if verdict != "OK":
                                        skill_examples[(verdict, sid)] += 1
                            r = execute_action(d.action, ws)
                            if r.get("type") != "ERROR":
                                consume_resources(res, d.action, r)
                            if d.ended:
                                break
                            if (res["action"] <= 0 and res["bonus_action"] <= 0
                                    and res["movement"] <= 1e-6):
                                break
                    if not chars["agent_0"].is_alive() or not chars["opp_0"].is_alive():
                        break
                    ws.combat.round_number += 1

    tot = sum(overall.values())
    print(f"total agent decisions analysed: {tot}")
    for k in ("OK", "POS_DRIFT", "TARGET_CHANGED", "SKILL_CHANGED", "DROPPED"):
        print(f"  {k:16s}: {overall[k]:6d}  {overall[k]/max(tot,1):6.1%}")
    print("\nby target_type (non-OK fraction):")
    for tt, c in sorted(by_tt.items(), key=lambda kv: -sum(kv[1].values())):
        t = sum(c.values())
        nonok = t - c["OK"]
        print(f"  {tt:16s} n={t:5d}  non-OK={nonok/max(t,1):5.1%}  "
              f"{dict(c)}")
    print("\ntop offending (verdict, skill_id):")
    for (v, sid), n in skill_examples.most_common(15):
        print(f"  {v:16s} {sid:24s} {n}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
