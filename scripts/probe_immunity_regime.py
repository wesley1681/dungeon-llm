"""Does the fire-immunity scenario actually REQUIRE reading the descriptor?

The fire_lab PPO runs all stayed channel-blind (火% on==off). Before redesigning
the training, verify the PREREQUISITE the user's goal needs: is reading the
enemy fire-immunity both NECESSARY (no safe hedge wins both halves) and
BENEFICIAL (the right action per half beats the wrong one)?

For agent=evocation at several levels vs a weak base enemy (orc), measure the
WIN RATE of two FORCED policies in each half:
  force-fire : every damaging turn use the 火 option (fireball)
  force-mm   : every damaging turn use the 力場 option (magic_missile)
Forcing is done by picking, among available skills, the one whose built-action
damage_type matches the target type (no hardcoded skill ids); non-damaging
turns fall back to the greedy net so positioning/utility is unchanged.

Reading is NECESSARY iff neither fixed policy wins both halves, i.e.
  immune half : WR(fire) ≈ 0  and  WR(mm) high
  normal half : WR(fire) high and  WR(mm) clearly LOWER   (else always-mm hedges)
The clean regime is the (level) where each half has a unique winning action and
the wrong action loses — that is where pure reward can force conditioning.

Usage: python scripts/probe_immunity_regime.py --games 40
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import torch

from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import build_obs
from trpg.rl.action import encode_action
from trpg.engine.skill import available_skills
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from eval_routed import stable_seed
from probe_descriptor_ab import blind_self_one_hot
from probe_fire_switch import action_damage_type


def forced_skill_index(env, actor_id, oid, want_type, sks):
    """Index of an available damaging skill whose built action is want_type,
    else None (caller then defers to the net)."""
    actor = env.ws.characters[actor_id]
    o = env.ws.characters[oid]
    for i, sk in enumerate(sks):
        if i == 0 or sk.features.expected_damage <= 0:
            continue
        built = sk.build_action(actor_id, oid, (o.position.x, o.position.y))
        if action_damage_type(built, actor) == want_type:
            return i
    return None


def run(net, enemy, level, immune, want_type, games):
    """Force the agent to the want_type damage option each damaging turn."""
    wins = forced = ndmg = 0
    olvl = MONSTER_DEFS[enemy].natural_level
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"reg_{enemy}_{immune}_{want_type}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=["evocation"], opp_archs=[enemy],
                  level=level, opp_level=olvl)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
        if immune:
            env.ws.characters[oid].damage_multipliers["火"] = 0.0
        done = False
        while not done:
            actor_id = env.current_agent_id
            ob = blind_self_one_hot(build_obs(env.ws, actor_id, env.resources))
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor_id)
            e = apply_entity_mask(e, ot, env.ws, actor_id)
            ai = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                  agent_id=actor_id))
            if actor_id == aid:
                sks = available_skills(env.ws.characters[aid], env.ws)
                fi = forced_skill_index(env, aid, oid, want_type, sks)
                if fi is not None:
                    # rebuild the action dict for the forced skill, greedy target
                    o = env.ws.characters[oid]
                    forced_act = sks[fi].build_action(aid, oid,
                                                      (o.position.x, o.position.y))
                    enc = encode_action(forced_act, env.ws, aid)
                    if enc[0] >= 0:
                        ai = list(enc); forced += 1; ndmg += 1
            obs2, _, term, trunc, _ = env.step(ai)
            done = term or trunc
        if (env.ws.characters[oid].is_dead()
                and env.ws.characters[aid].is_alive()):
            wins += 1
    return wins / games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--enemy", default="orc")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--levels", default="4,5,6,7,8")
    args = p.parse_args()
    register_monsters()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt} agent=evocation vs {args.enemy} "
          f"(natL={MONSTER_DEFS[args.enemy].natural_level}), {args.games} g/cell\n")
    print(f"{'lvl':>3} | {'IMMUNE fire':>11} {'IMMUNE mm':>10} | "
          f"{'normal fire':>11} {'normal mm':>10}   verdict")
    for lvl in (int(x) for x in args.levels.split(",")):
        imm_f = run(net, args.enemy, lvl, True, "火", args.games)
        imm_m = run(net, args.enemy, lvl, True, "力場", args.games)
        nor_f = run(net, args.enemy, lvl, False, "火", args.games)
        nor_m = run(net, args.enemy, lvl, False, "力場", args.games)
        # necessary+beneficial: immune→mm wins & fire loses; normal→fire wins & mm lower
        clean = (imm_m - imm_f > 0.4) and (nor_f - nor_m > 0.2)
        verdict = "<- clean flip" if clean else ""
        print(f"{lvl:>3} | {imm_f:>11.0%} {imm_m:>10.0%} | "
              f"{nor_f:>11.0%} {nor_m:>10.0%}   {verdict}")
    print("\n判讀：免疫半→mm 勝、fire 0；正常半→fire 勝、mm 明顯較低"
          "＝讀 descriptor 必要且有益＝純獎勵可逼出條件化的乾淨 regime。")


if __name__ == "__main__":
    main()
