"""Is the BC student's WR-gap vs the expert a COVARIATE-SHIFT problem (the
documented-but-never-run DAgger lever)? Measure policy agreement between the
student and the scripted expert on two state distributions:

  EXPERT-visited : expert drives the seat; at each expert decision, does the
                   student's GREEDY skill slot match the expert's?
  STUDENT-visited: student drives the seat (greedy); at each student decision,
                   does the expert (queried on the SAME state) agree?

If agreement is high on expert-visited but LOW on student-visited states, the
student imitates the expert's marginal action distribution but drifts into
states the demos never covered and errs there -> classic covariate shift ->
DAgger (collect expert labels on student-visited states) is the fix.

General: agreement is a raw skill-slot match, NO skill-name list. Same seats/
dice as the other 1v2 probes.

Usage: python scripts/probe_covshift.py --ckpt models/seed_kit/kit_s1000.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from zlib import crc32
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _expert_slot(script, env, actor):
    """The expert's chosen skill slot at the current state (0 = end/none)."""
    ch = env.ws.characters[actor]
    dec = script.decide(actor, ch, env.ws, env.resources,
                        env.ws.combat.round_number)
    if dec.action is None or getattr(dec, "fled", False):
        return 0
    try:
        return list(encode_action(dec.action, env.ws, actor))[0]
    except Exception:
        return 0


def _student_act(net, env, actor, obs):
    """The student's greedy full action at the current state."""
    ob = blind_np_single(obs)
    ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
    with torch.no_grad():
        el, s, e, g = net(ot)
    s = apply_resource_mask(s, env.resources, env.ws, actor)
    e = apply_entity_mask(e, ot, env.ws, actor)
    return list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))


def run(net, ident, games):
    # [agree, total] on each state distribution (agent decisions only)
    exp_visited = [0, 0]
    stu_visited = [0, 0]
    for gi in range(games):
        # --- EXPERT drives ---
        key = f"{ident}|2|{gi}"; k = crc32(key.encode()); random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(2)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]; script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            e_slot = _expert_slot(script, env, actor)
            if actor == aid:
                s_act = _student_act(net, env, actor, obs)
                exp_visited[1] += 1
                exp_visited[0] += int(s_act[0] == e_slot)
            # step the env with the EXPERT action (expert drives this rollout)
            ch = env.ws.characters[actor]
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]
            obs, _, term, trunc, _ = env.step(act); done = term or trunc

        # --- STUDENT drives (same seed) ---
        random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=2)
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]; script = make_archetype_policy(ident)
        done = False
        while not done:
            actor = env.current_agent_id
            s_act = _student_act(net, env, actor, obs)
            if actor == aid:
                e_slot = _expert_slot(script, env, actor)
                stu_visited[1] += 1
                stu_visited[0] += int(s_act[0] == e_slot)
            obs, _, term, trunc, _ = env.step(s_act); done = term or trunc
    return exp_visited, stu_visited


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed_kit/kit_s1000.pt")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--idents", nargs="*",
                   default=["battle_master", "war", "arcane_trickster",
                            "assassin", "champion"])
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  student<->expert skill-slot agreement, 1v2 weak\n")
    print(f"  {'ident':<18} {'expert-visited':>16} {'student-visited':>17}   gap")
    for ident in args.idents:
        ev, sv = run(net, ident, args.games)
        ea = ev[0] / max(1, ev[1]); sa = sv[0] / max(1, sv[1])
        print(f"  {ident:<18} {ea:>14.0%} (n={ev[1]:>4}) "
              f"{sa:>13.0%} (n={sv[1]:>4})   {ea - sa:+.0%}", flush=True)
    print("\n  high expert-visited but low student-visited => covariate shift "
          "=> DAgger lever")


if __name__ == "__main__":
    main()
