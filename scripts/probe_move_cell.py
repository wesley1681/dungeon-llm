"""Why do melee classes pick LATERAL (non-closing) move cells?

distance_grid channel 1 already gives every cell its distance to the nearest
enemy, so a pointwise-linear grid head CAN prefer a closing cell. Yet champion /
battle_master move laterally ~78% of the time. This probe instruments every
MOVE decision to separate three hypotheses:

  H1 mask/movement bug — MOVE chosen while movement is ~0 (engine displaces 0 →
     counts as lateral, and the cost fires on a move the policy couldn't avoid).
  H2 head picks FAR cells — a distance-closing cell was LEGAL but the grid head
     ranked a non-closing cell above it (policy/weighting failure).
  H3 engine clamp — the chosen cell IS near-optimal (low dist-to-enemy) but the
     post-move displacement still doesn't close (straight-walk / range clamp).

Per MOVE decision we log:
  mv_avail        resources['movement'] at decision time
  chosen_de       chosen cell's distance-to-nearest-enemy feature (0..1, /diag)
  best_legal_de   min distance-to-enemy over the LEGAL move cells (mask-passed)
  de_gap          chosen_de - best_legal_de  (0 = picked an optimal closing cell)
  rank_pct        percentile of chosen cell among legal cells by dist-to-enemy
                  (0% = closest-to-enemy legal cell; 100% = farthest)
  closed          actual (d_before - d_after) >= 0.5  (did distance drop?)

Usage:
  python scripts/probe_move_cell.py --ckpt models/pop_fix2/pop_u0040.pt
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import argparse, random
from zlib import crc32
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.obs import distance_grid_obs, N_GRID
from trpg.rl.action import decode_action, point_validity_mask
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_routed import load_student
from train_population import blind_np_single
from synth_identity import STANDARD_IDS


def _move_skill_idx(ws, aid):
    for i, sk in enumerate(available_skills(ws.characters[aid], ws)):
        if sk.skill_id == "move":
            return i, sk
    return None, None


def probe(net, ident, n_opp, games):
    rows = []
    for gi in range(games):
        key = f"{ident}|{n_opp}|{gi}"
        k = crc32(key.encode())
        random.seed(k)
        env = CombatEnvV2(seed=k ^ 0x5, n_agents=1, n_opps=n_opp)
        opps = [STANDARD_IDS[(k + i) % len(STANDARD_IDS)] for i in range(n_opp)]
        obs, _ = env.reset(agent_archs=[ident], opp_archs=opps, level=5,
                           opp_level=2, layout="open")
        aid = env.agent_ids[0]
        oids = list(env.opp_ids)
        done = False
        while not done:
            actor = env.current_agent_id
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws,
                                   agent_id=actor))
            row = None
            if actor == aid:
                ad = decode_action(act, env.ws, aid)
                is_move = ad is not None and str(ad.get("type")) == "MOVE"
                if is_move:
                    # env.resources is the CURRENT agent's flat dict; actor==aid
                    # here, so read movement directly (NOT keyed by id).
                    mv_avail = env.resources.get("movement", 0.0)
                    dg = distance_grid_obs(env.ws, aid)[1]   # [N_GRID,N_GRID]
                    de_flat = dg.reshape(-1)                 # cell = ix*N_GRID+iy
                    cell = int(act[2])
                    chosen_de = float(de_flat[cell])
                    mi, msk = _move_skill_idx(env.ws, aid)
                    inv = point_validity_mask(env.ws, aid, msk) if msk is not None \
                        else np.ones(N_GRID * N_GRID, bool)
                    legal = ~inv.astype(bool)
                    if legal.any():
                        legal_de = de_flat[legal]
                        best_legal = float(legal_de.min())
                        rank_pct = float((legal_de < chosen_de).mean())
                    else:
                        best_legal, rank_pct = chosen_de, 0.0
                    d_before = env._nearest_enemy_dist(env.ws.characters[aid])
                    act_avail = env.resources.get("action", 0)
                    bonus_avail = env.resources.get("bonus_action", 0)
                    row = dict(mv_avail=mv_avail, chosen_de=chosen_de,
                               best_legal=best_legal, de_gap=chosen_de - best_legal,
                               rank_pct=rank_pct, d_before=d_before,
                               act_avail=act_avail, bonus_avail=bonus_avail)
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc
            if row is not None:
                d_after = env._nearest_enemy_dist(env.ws.characters[aid])
                row["closed"] = (row["d_before"] is not None and d_after is not None
                                 and (row["d_before"] - d_after) >= 0.5)
                rows.append(row)
    return rows


def summarize(label, rows):
    if not rows:
        print(f"  {label:<26} (no moves)")
        return
    n = len(rows)
    no_mv = sum(r["mv_avail"] < 0.5 for r in rows) / n
    closed = sum(r["closed"] for r in rows) / n
    mean_gap = sum(r["de_gap"] for r in rows) / n
    mean_rank = sum(r["rank_pct"] for r in rows) / n
    mean_chosen = sum(r["chosen_de"] for r in rows) / n
    mean_best = sum(r["best_legal"] for r in rows) / n
    # among moves WITH movement available, did they close / pick optimal?
    mv = [r for r in rows if r["mv_avail"] >= 0.5]
    closed_mv = (sum(r["closed"] for r in mv) / len(mv)) if mv else float("nan")
    db = [r["d_before"] for r in rows if r["d_before"] is not None]
    mean_db = sum(db) / len(db) if db else float("nan")
    adj_rows = [r for r in rows if r["d_before"] is not None and r["d_before"] <= 1.6]
    adj = len(adj_rows) / len(db) if db else float("nan")  # already in melee reach
    # THE decisive number: of moves made WHILE ALREADY ADJACENT, how many still
    # had an attack action available (=> chose a no-op move OVER a legal attack).
    adj_act = (sum(r["act_avail"] > 0 for r in adj_rows) / len(adj_rows)
               if adj_rows else float("nan"))
    print(f"  {label:<26} n={n:>3} | closed={closed:>4.0%} | d_before={mean_db:>4.1f}m "
          f"adj={adj:>4.0%} | ADJ-moves-with-ATTACK-still-up={adj_act:>4.0%} "
          f"| gap={mean_gap:.3f} rank%={mean_rank:>4.0%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/pop_fix2/pop_u0040.pt")
    p.add_argument("--base", default="models/pop_los_final/pop_u0044.pt")
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--idents", nargs="*", default=["champion", "battle_master", "war"])
    args = p.parse_args()
    print("de = distance-to-nearest-enemy feature (0..1, /diag). Lower = closer.")
    print("H1 if no-mvmt high; H2 if gap/rank high (closing cell was legal, not "
          "picked); H3 if gap~0 but closed low (picked good cell, engine clamps).\n")
    for tag, ck in (("FIX2", args.ckpt), ("BASE", args.base)):
        net = load_student(ck); net.eval()
        print(f"== {tag}  {ck} ==")
        for ident in args.idents:
            for n_opp in (1, 2):
                rows = probe(net, ident, n_opp, args.games)
                summarize(f"{ident} 1v{n_opp}", rows)
        print()


if __name__ == "__main__":
    main()
