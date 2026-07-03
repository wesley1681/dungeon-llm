"""Does the training distribution actually CONTAIN must-flank states?

The LoS-row fine-tune can only learn to flank if the rollout puts the agent in
fully-blocked states (no enemy in line of sight) often enough, AND those states
persist long enough that MOVING is what restores LoS (not the enemy walking
around the wall on its own). Scripted experts path toward the agent, so a
blocked LoS may resolve itself -> near-zero gradient for flanking.

This measures, over wall/pillar-layout episodes vs scripted experts (the train
condition), per agent decision-turn:
    blocked%      agent-turns with NO enemy in LoS (the state that needs flank)
    resolved-by   of blocked runs, did LoS return because the ENEMY moved into
                  view (enemy-LoS handed to agent for free) or the AGENT moved?
    persist       mean consecutive blocked agent-turns (1 = transient)
If blocked% is tiny and runs resolve by enemy-move, the curriculum gives the
LoS row almost nothing to learn from -> explains a flat/noisy flanking result.

Usage:
  python scripts/diag_blocked_freq.py --ckpt models/pop_mon/pop_u0005.pt \
         --layouts pillar walls --games 40
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
from trpg.engine.vec2 import Vec2  # noqa
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.rl.obs import partition_entities

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _enemy_los(ws, aid):
    bf = ws.combat.battlefield if ws.combat else None
    if bf is None:
        return True
    a = ws.characters[aid]
    _, enemies = partition_entities(ws, aid)
    for e in enemies:
        ec = ws.characters[e]
        if ec.is_alive() and bf.has_line_of_sight(a.position, ec.position):
            return True
    return False


def run(net, ident, opp, layout, ep_key):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x9, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=[opp],
                       level=5, opp_level=5, layout=layout)
    aid = env.agent_ids[0]
    agent_turns = blocked_turns = 0
    runs_enemy = runs_agent = 0       # how a blocked run ended
    run_lens = []
    in_run = False; run_len = 0; blk_pos = None
    done = False
    while not done:
        actor = env.current_agent_id
        if actor == aid:
            agent_turns += 1
            blk = not _enemy_los(env.ws, aid)
            if blk:
                blocked_turns += 1
                if not in_run:
                    in_run = True; run_len = 0
                run_len += 1
                blk_pos = env.ws.characters[aid].position
            else:
                if in_run:
                    # how did it resolve? compare agent displacement
                    moved = (blk_pos is not None and
                             env.ws.characters[aid].position.distance_to(blk_pos) > 0.5)
                    if moved:
                        runs_agent += 1
                    else:
                        runs_enemy += 1
                    run_lens.append(run_len); in_run = False
        ob = blind_np_single(obs)
        ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        s = apply_resource_mask(s, env.resources, env.ws, actor)
        e = apply_entity_mask(e, ot, env.ws, actor)
        act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    if in_run:
        run_lens.append(run_len); runs_enemy += 1  # ended by truncation/death
    return agent_turns, blocked_turns, runs_enemy, runs_agent, run_lens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--layouts", nargs="*", default=["pillar", "walls", "open"])
    p.add_argument("--games", type=int, default=40)
    args = p.parse_args()
    net = load_student(args.ckpt); net.eval()
    print(f"ckpt={args.ckpt}  games/layout={args.games}\n")
    print(f"{'layout':<10} {'blocked%':>9} {'persist':>8} "
          f"{'end-by-enemy':>13} {'end-by-agent':>13}")
    for layout in args.layouts:
        AT = BT = re_ = ra = 0; RL = []
        for g in range(args.games):
            ident = STANDARD_IDS[g % len(STANDARD_IDS)]
            opp = STANDARD_IDS[(g + 4) % len(STANDARD_IDS)]
            at, bt, r_e, r_a, rl = run(net, ident, opp, layout,
                                       f"blk|{layout}|{g}")
            AT += at; BT += bt; re_ += r_e; ra += r_a; RL += rl
        blkpct = BT / max(1, AT)
        persist = sum(RL) / len(RL) if RL else 0.0
        tot = re_ + ra
        print(f"{layout:<10} {blkpct:>9.1%} {persist:>8.2f} "
              f"{re_/max(1,tot):>13.0%} {ra/max(1,tot):>13.0%}", flush=True)


if __name__ == "__main__":
    main()
