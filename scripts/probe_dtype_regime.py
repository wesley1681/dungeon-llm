"""Find a CLEAN dtype-switch regime: two AT-WILL damage options, enemies whose
natural resist flips the winner, both winning options reliably winnable.

fire_lab failed partly on a confound: evocation's non-fire option (magic_missile)
is slot-limited, so the fire-immune half is barely winnable even with perfect
play (probe_immunity_regime: forced-mm WR 8-18%) → no strong switch→win gradient.

The life cleric avoids that: 長劍 (斬擊, at-will weapon) and sacred_flame
(光耀, at-will cantrip) are BOTH unlimited. Monsters' natural typed_resist flips
which dominates (shadow: 斬擊×0.5, 光耀×2; skeleton: 鈍擊 vuln; most: neutral →
斬擊 higher base). This probe forces each at-will option and measures WR per enemy
(natural resists, NO injection), to locate enemies where:
  type_a wins & type_b clearly lower   (reading not needed → use a)
  type_b wins & type_a clearly lower   (reading flips the answer)
A training distribution mixing BOTH kinds (identity-blind) makes descriptor-
reading necessary AND beneficial, with both right-answers reliably winnable.

Usage: python scripts/probe_dtype_regime.py --agent life --games 40
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
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS, onev1_viable_monsters

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from eval_routed import stable_seed
from probe_descriptor_ab import blind_self_one_hot
from probe_fire_switch import action_damage_type


def forced_idx(env, aid, oid, want_type, sks):
    actor = env.ws.characters[aid]; o = env.ws.characters[oid]
    for i, sk in enumerate(sks):
        if i == 0 or sk.features.expected_damage <= 0:
            continue
        built = sk.build_action(aid, oid, (o.position.x, o.position.y))
        if action_damage_type(built, actor) == want_type:
            return i
    return None


def run(net, agent, enemy, level, want_type, games):
    wins = 0
    olvl = MONSTER_DEFS[enemy].natural_level
    for gi in range(games):
        env = CombatEnvV2(seed=stable_seed(f"dr_{agent}_{enemy}_{want_type}_{gi}"),
                          n_agents=1, n_opps=1)
        env.reset(agent_archs=[agent], opp_archs=[enemy],
                  level=level, opp_level=olvl)
        aid = env.agent_ids[0]; oid = env.opp_ids[0]
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
                fi = forced_idx(env, aid, oid, want_type, sks)
                if fi is not None:
                    o = env.ws.characters[oid]
                    enc = encode_action(
                        sks[fi].build_action(aid, oid,
                                             (o.position.x, o.position.y)),
                        env.ws, aid)
                    if enc[0] >= 0:
                        ai = list(enc)
            obs2, _, term, trunc, _ = env.step(ai)
            done = term or trunc
        if (env.ws.characters[oid].is_dead()
                and env.ws.characters[aid].is_alive()):
            wins += 1
    return wins / games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/seed_dtype1/seed_s1600.pt")
    p.add_argument("--agent", default="life")
    p.add_argument("--type_a", default="斬擊")
    p.add_argument("--type_b", default="光耀")
    p.add_argument("--level", type=int, default=6)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--enemies", default="",
                   help="comma list; default = onev1-viable with a 斬擊/光耀 mult")
    args = p.parse_args()
    register_monsters()
    net = load_student(args.ckpt); net.eval()

    def enemy_mult(m):
        env = CombatEnvV2(seed=1, n_agents=1, n_opps=1)
        env.reset(agent_archs=[args.agent], opp_archs=[m], level=args.level,
                  opp_level=MONSTER_DEFS[m].natural_level)
        return env.ws.characters[env.opp_ids[0]].damage_multipliers or {}

    if args.enemies:
        enemies = args.enemies.split(",")
    else:
        enemies = []
        for m in onev1_viable_monsters(8.0):
            mult = enemy_mult(m)
            # keep enemies that resist/vuln either type (the flip cases) plus a
            # few neutral controls
            if (mult.get(args.type_a, 1) != 1 or mult.get(args.type_b, 1) != 1):
                enemies.append(m)
        # add neutral controls
        for c in ("orc", "bandit", "ogre"):
            if c in MONSTER_DEFS and c not in enemies:
                enemies.append(c)

    print(f"ckpt={args.ckpt} agent={args.agent}@L{args.level} "
          f"A={args.type_a} B={args.type_b}, {args.games} g/cell\n")
    print(f"{'enemy':16s} {'A.mult':>6} {'B.mult':>6} | "
          f"{'WR(A)':>6} {'WR(B)':>6}  winner / reading-pays")
    for m in enemies:
        mult = enemy_mult(m)
        ma, mb = mult.get(args.type_a, 1), mult.get(args.type_b, 1)
        wa = run(net, args.agent, m, args.level, args.type_a, args.games)
        wb = run(net, args.agent, m, args.level, args.type_b, args.games)
        win = args.type_a if wa > wb + 0.1 else (
            args.type_b if wb > wa + 0.1 else "tie")
        pays = abs(wa - wb)
        print(f"{m:16s} {ma:>6} {mb:>6} | {wa:>6.0%} {wb:>6.0%}  "
              f"{win}  Δ={pays:.0%}")
    print("\n判讀：找到一組『A 勝 B 輸』與『B 勝 A 輸』都成立、且贏的那邊 WR 高"
          "（at-will 無耗盡）的敵人 = 讀 descriptor 必要且有益的乾淨訓練分布。")


if __name__ == "__main__":
    main()
