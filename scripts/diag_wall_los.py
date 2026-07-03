"""躲牆後不發呆 — controlled line-of-sight diagnostic.

Goal requirement: "不能敵人躲在牆後就發呆". The engine fully supports it —
walls (BLOCKED terrain) block movement AND ranged attacks (attack_range_check
-> has_line_of_sight), terrain is in obs, and apply_resource_mask masks every
enemy-target skill when no enemy is in LoS (so the turn can't silently no-op
into the wall). So the open question is purely BEHAVIOURAL: when an enemy sits
behind a wall, does the policy MOVE to peek around it and re-establish LoS, or
does it dither / stall until truncation?

Setup (clean, deterministic): a single wall pillar at mid-field; agent on one
side, a STATIONARY enemy on the other, placed so the straight agent->enemy
segment is initially BLOCKED. The enemy never acts (injected stationary
policy) so we measure ONLY the agent's peeking behaviour. Per identity we run
the model arm and the script arm and report:
    los@t0      LoS blocked at start? (sanity — must be False/blocked)
    regain%     episodes where the agent re-established LoS at all
    turns2los   mean agent-turns until first LoS (inf = never)
    move/blkT   moves taken per blocked turn (0 ≈ 發呆; >0 = trying)
    win%        episodes won (stationary enemy => regain+kill is achievable)

A LoS-blind WASTED_MOVE_COST (env_v2: penalises any move that doesn't change
straight-line distance to nearest enemy by >0.5m) would specifically punish
the lateral peek, so a low regain%/high turns2los on the MODEL arm but not the
SCRIPT arm is the signature of that reward bug biting.

Usage:
  python scripts/diag_wall_los.py --ckpt models/pop_mon/pop_u0005.pt \
         --idents assassin evocation champion --games 8 --max_turns 16
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

import argparse
import random
from types import SimpleNamespace
from zlib import crc32

import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()

from trpg.engine.vec2 import Vec2, TerrainType
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.action import encode_action
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills
from trpg.engine.combat_policy import make_archetype_policy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single


class StationaryPolicy:
    """Opponent that always ends its turn immediately (never moves/attacks)."""
    def decide(self, opp_id, opp, ws, resources, round_number):
        return SimpleNamespace(action=None, fled=False, ended=True)


def _paint_pillar(bf, cx: float, cy: float, half_w: float, half_h: float):
    """A solid wall pillar centred at (cx, cy)."""
    bf.add_rect_obstacle(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def run_wall_episode(ident: str, ep_key: str, net, max_turns: int) -> dict:
    """One controlled peek episode. net=None -> script arm."""
    k = crc32(ep_key.encode())
    random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x33, n_agents=1, n_opps=1)
    obs, _ = env.reset(agent_archs=[ident], opp_archs=["champion"],
                       level=5, opp_level=5, layout="open")
    aid = env.agent_ids[0]
    oid = env.opp_ids[0]
    env._opp_policies[oid] = StationaryPolicy()      # freeze the enemy

    bf = env.ws.combat.battlefield
    W, H = bf.width, bf.height
    # fresh clean field + one central pillar
    nx = len(bf.cells[0]); ny = len(bf.cells)
    bf.cells = [[int(TerrainType.NORMAL)] * nx for _ in range(ny)]
    cx, cy = W * 0.5, H * 0.5
    _paint_pillar(bf, cx, cy, half_w=0.75, half_h=3.0)
    agent = env.ws.characters[aid]
    enemy = env.ws.characters[oid]
    agent.position = Vec2(W * 0.20, cy)
    enemy.position = Vec2(W * 0.80, cy)              # straight line crosses pillar

    los0 = bf.has_line_of_sight(agent.position, enemy.position)
    script = None if net is not None else make_archetype_policy(ident)

    turns = 0
    moves_blocked = 0
    blocked_turns = 0
    turns2los = None
    ddist_blocked: list[float] = []   # Δeuclidean dist per blocked move (<0=closing)
    lat_end: list[float] = []         # |agent.y - enemy.y| at end of blocked turns
    done = False
    prev_actor = None
    while not done and turns < max_turns:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        is_agent_turn = (actor == aid)
        # count agent TURNS (not sub-actions)
        if is_agent_turn and actor != prev_actor:
            turns += 1
            has_los_now = bf.has_line_of_sight(agent.position, enemy.position)
            if has_los_now and turns2los is None:
                turns2los = turns - 1   # regained before this turn started
            if not has_los_now:
                blocked_turns += 1
        prev_actor = actor

        blocked_before = (is_agent_turn
                          and not bf.has_line_of_sight(agent.position,
                                                       enemy.position))
        pos_before = agent.position

        if net is not None:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
        else:
            dec = script.decide(actor, ch, env.ws, env.resources,
                                env.ws.combat.round_number)
            if dec.action is None or getattr(dec, "fled", False):
                act = [0, 0, 0]
            else:
                try:
                    act = list(encode_action(dec.action, env.ws, actor))
                except Exception:
                    act = [0, 0, 0]

        d_before_enemy = pos_before.distance_to(enemy.position)
        obs, _, term, trunc, _ = env.step(act)
        if blocked_before and agent.position.distance_to(pos_before) > 1e-6:
            moves_blocked += 1
            ddist_blocked.append(
                agent.position.distance_to(enemy.position) - d_before_enemy)
        if blocked_before:
            lat_end.append(abs(agent.position.y - enemy.position.y))
        done = term or trunc

    # final LoS check (might have regained on the very last sub-action)
    if turns2los is None and bf.has_line_of_sight(agent.position,
                                                  enemy.position):
        turns2los = turns
    won = (agent.is_alive() and not enemy.is_alive())
    return {
        "los0": los0,
        "regained": turns2los is not None,
        "turns2los": turns2los if turns2los is not None else float("inf"),
        "moves_blocked": moves_blocked,
        "blocked_turns": max(1, blocked_turns),
        "won": won,
        "ddist_blocked": ddist_blocked,
        "lat_end": lat_end,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_mon/pop_u0005.pt")
    p.add_argument("--idents", nargs="*",
                   default=["assassin", "evocation", "champion", "war",
                            "arcane_trickster", "battle_master"])
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=16)
    args = p.parse_args()

    net = load_student(args.ckpt)
    net.eval()
    print(f"ckpt={args.ckpt}  enemy=STATIONARY behind pillar  "
          f"games={args.games}  max_turns={args.max_turns}\n")
    print(f"{'identity':<18} {'arm':<7} {'los@0':>6} {'regain%':>8} "
          f"{'turns2los':>10} {'mv/blkT':>8} {'Δdist/mv':>9} {'latEnd':>7} "
          f"{'win%':>6}")
    print("  (Δdist/mv<0 = blocked moves CLOSE distance=hug wall; "
          "latEnd≈0 = never goes lateral off the sightline)")

    for ident in args.idents:
        for arm, drv in (("model", net), ("script", None)):
            R = [run_wall_episode(ident, f"wall|{ident}|{arm}|{k}", drv,
                                  args.max_turns) for k in range(args.games)]
            n = len(R)
            regain = sum(r["regained"] for r in R) / n
            fin = [r["turns2los"] for r in R if r["regained"]]
            t2l = (sum(fin) / len(fin)) if fin else float("inf")
            mv = sum(r["moves_blocked"] for r in R) / max(
                1, sum(r["blocked_turns"] for r in R))
            win = sum(r["won"] for r in R) / n
            los0 = sum(r["los0"] for r in R) / n
            dd = [x for r in R for x in r["ddist_blocked"]]
            la = [x for r in R for x in r["lat_end"]]
            dd_m = sum(dd) / len(dd) if dd else float("nan")
            la_m = sum(la) / len(la) if la else float("nan")
            t2l_s = f"{t2l:.1f}" if fin else "  inf"
            print(f"{ident:<18} {arm:<7} {los0:>6.0%} {regain:>8.0%} "
                  f"{t2l_s:>10} {mv:>8.2f} {dd_m:>+9.2f} {la_m:>7.2f} "
                  f"{win:>6.0%}", flush=True)


if __name__ == "__main__":
    main()
