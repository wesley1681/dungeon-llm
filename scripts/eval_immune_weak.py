"""FIX metric for the immune-x-weak-enemy blind spot (maxHP-gate dodge collapse).

A fire-caster vs a WEAK, low-HP enemy that is IMMUNE to the caster's primary
(fire) damage. Correct play = fall back to the secondary non-immune attack and
kill it. The bug = dodge/retreat, 0 damage. Enemy is ACTIVE (env scripted) so
this is realistic combat, not the stationary probe. Reports damage-dealt
fraction + win rate per (kit, enemy). Reusable before/after training.

  python scripts/eval_immune_weak.py models/unified/uni_v9.pt --games 6
"""
from __future__ import annotations
import sys, os, argparse
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import random, torch
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.scenarios.monsters import register_monsters, MONSTER_DEFS
register_monsters()
from distill_routed import load_student
from train_population import blind_np_single

KITS = ["evocation", "divination"]          # fire-primary casters
WEAK = ["orc", "goblin", "kobold", "bandit"]  # low-HP enemies
IMMUNE = "火"


def episode(net, kit, enemy, level, key):
    k = crc32(key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x51, n_agents=1, n_opps=1)
    env.reset(agent_archs=[kit], opp_archs=[enemy], level=level,
              opp_level=MONSTER_DEFS[enemy].natural_level, layout="open")
    aid = env.agent_ids[0]; oid = env.opp_ids[0]
    env.ws.characters[oid].damage_multipliers[IMMUNE] = 0.0
    hp0 = env.ws.characters[oid].hp
    from trpg.rl.obs import build_obs
    obs = blind_np_single(build_obs(env.ws, env.current_agent_id, env.resources))
    done = False
    while not done:
        actor = env.current_agent_id
        if actor == aid:
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            obs2, _, term, trunc, _ = env.step(act)
        else:
            obs2, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
        if not done:
            obs = blind_np_single(build_obs(env.ws, env.current_agent_id, env.resources))
    dmg = hp0 - max(0, env.ws.characters[oid].hp)
    win = (not env.ws.characters[oid].is_alive()) and env.ws.characters[aid].is_alive()
    return dmg / max(1.0, hp0), int(win)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--level", type=int, default=8)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  fire-caster vs WEAK 火-immune enemy (active), "
          f"L{args.level}, {args.games}g\n")
    print(f"{'kit':12s} {'enemy':10s} {'dmg%':>7s} {'win%':>6s}")
    tot_d = tot_w = tot_n = 0.0
    for kit in KITS:
        for enemy in WEAK:
            ds = ws = 0.0
            for gi in range(args.games):
                d, w = episode(net, kit, enemy, args.level, f"iw_{kit}_{enemy}_{gi}")
                ds += d; ws += w
            print(f"{kit:12s} {enemy:10s} {ds/args.games:7.1%} {ws/args.games:6.0%}")
            tot_d += ds; tot_w += ws; tot_n += args.games
    print(f"\nOVERALL  dmg%={tot_d/tot_n:.1%}  win%={tot_w/tot_n:.0%}  (n={int(tot_n)})")


if __name__ == "__main__":
    main()
