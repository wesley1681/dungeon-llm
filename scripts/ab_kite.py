"""量測『遠程身分不風箏』是否吃勝率(=是否真 WR-BUG)。

A=原模型；B=同模型+測量用風箏 gate(僅測量)：當
  - 身分有遠程攻擊(range_m>2.5 的鎖敵/POINT 技)
  - 最近敵在 2.5m 內(被貼臉)
  - 還有移動力(>=2m)
時,先強制 MOVE 到『保有對最近敵視線 + 離敵最遠』的可達格(風箏),其餘決策不變。
同批 seed 配對比 WR。顯著上升=不風箏是真 WR-BUG;持平=非 WR-BUG。

用法: python scripts/ab_kite.py [模型] [--games N]
"""
from __future__ import annotations
import sys, os, argparse, random
from zlib import crc32
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters, EQUIV_LEVEL_1V1
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask, apply_entity_mask, pick_action
from trpg.engine.skill import available_skills, TargetType
from trpg.rl.obs import N_GRID, GRID_CELL_SIZE_M
from trpg.engine.vec2 import Vec2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single

try:
    from chimera_defs import register_chimeras
    register_chimeras()
except Exception:
    pass


def load(path):
    from distill_routed import load_student
    return load_student(path)


def _ranged_range(ch, ws):
    """此身分最遠的鎖敵/POINT 攻擊射程(m);無遠程回 0。"""
    best = 0.0
    for s in available_skills(ch, ws):
        tt = s.features.target_type
        if s.skill_id == "move":
            continue
        if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                  TargetType.POINT, TargetType.LINE, TargetType.CONE):
            rng = float(getattr(s.features, "range_m", 0.0) or 0.0)
            if getattr(s.features, "expected_damage", 0) > 0:
                best = max(best, rng)
    return best


def _cell_center(idx):
    return Vec2((idx // N_GRID + 0.5) * GRID_CELL_SIZE_M,
                (idx % N_GRID + 0.5) * GRID_CELL_SIZE_M)


def _kite_cell(ch, ws, mv_budget, foe_pos, rng):
    """找『可達(<=移動力) + 有對敵視線 + 離敵最遠 + 仍在射程內』的格,回 grid idx 或 None。"""
    bf = ws.combat.battlefield if ws.combat else None
    cur = ch.position
    best_idx, best_d = None, cur.distance_to(foe_pos)
    for idx in range(N_GRID * N_GRID):
        c = _cell_center(idx)
        if cur.distance_to(c) > mv_budget + 1e-6:
            continue
        d = c.distance_to(foe_pos)
        if d <= best_d + 1e-6:      # 要更遠
            continue
        if d > rng:                 # 別退出射程
            continue
        if bf is not None:
            if bf.is_blocked(c) if hasattr(bf, "is_blocked") else False:
                continue
            if not bf.has_line_of_sight(c, foe_pos):
                continue
        best_idx, best_d = idx, d
    return best_idx


def run_combat(net, ident, opp_archs, lvl, opp_lvl, ep_key, gate):
    k = crc32(ep_key.encode()); random.seed(k)
    env = CombatEnvV2(seed=k ^ 0x5A5A5A, n_agents=1, n_opps=len(opp_archs))
    obs, _ = env.reset(agent_archs=[ident], opp_archs=list(opp_archs),
                       level=lvl, opp_level=opp_lvl)
    aid = env.agent_ids[0]
    done = False
    while not done:
        actor = env.current_agent_id
        ch = env.ws.characters[actor]
        if actor in env.agent_ids:
            ob = blind_np_single(obs)
            ot = {kk: torch.from_numpy(v).unsqueeze(0) for kk, v in ob.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            s = apply_resource_mask(s, env.resources, env.ws, actor)
            e = apply_entity_mask(e, ot, env.ws, actor)
            act = list(pick_action(el[0], s[0], e[0], g[0], ws=env.ws, agent_id=actor))
            if gate:
                rng = _ranged_range(ch, env.ws)
                mv = env.resources.get("movement", 0.0)
                live = [env.ws.characters[o] for o in env.opp_ids
                        if env.ws.characters[o].is_alive()]
                if rng > 2.5 and mv >= 2.0 and live:
                    foe = min(live, key=lambda o: ch.position.distance_to(o.position))
                    nd = ch.position.distance_to(foe.position)
                    sks = available_skills(ch, env.ws)
                    chose_move = 0 < act[0] < len(sks) and sks[act[0]].skill_id == "move"
                    if nd <= 2.5 and not chose_move:
                        mi = next((i for i, sk in enumerate(sks)
                                   if sk.skill_id == "move"), None)
                        if mi is not None:
                            cell = _kite_cell(ch, env.ws, mv, foe.position, rng)
                            if cell is not None:
                                act = [mi, 0, cell]
            obs, _, term, trunc, _ = env.step(act)
        else:
            obs, _, term, trunc, _ = env.step([0, 0, 0])
        done = term or trunc
    return (env.ws.characters[aid].is_alive()
            and not any(env.ws.characters[o].is_alive() for o in env.opp_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="models/unified/uni_v8.pt")
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()
    net = load(args.ckpt)
    # 遠程身分: 法師/遠程怪/遠程流氓。對手用近戰逼近者。
    IDENTS = ["evocation", "divination", "arcane_trickster", "assassin",
              "mage_npc", "manticore", "kobold"]
    OPP = ["champion", "berserker", "totem_bear", "vengeance"]
    print(f"模型={os.path.basename(args.ckpt)} games={args.games} (遠程vs近戰逼近)\n")
    print(f"{'身分':16s} | 射程 | A原始 | B+kite | Δ")
    print("-" * 52)
    tA = tB = tn = 0
    for ident in IDENTS:
        eq = EQUIV_LEVEL_1V1.get(ident, 5)
        lvl = 8 if eq == float("inf") else max(1, int(round(eq)))
        # 探一次射程
        env = CombatEnvV2(n_agents=1, n_opps=1)
        env.reset(agent_archs=[ident], opp_archs=["champion"], level=lvl, opp_level=lvl)
        rng = _ranged_range(env.ws.characters["agent_0"], env.ws)
        a = b = 0
        for gi in range(args.games):
            opp = [OPP[gi % len(OPP)]]
            key = f"{ident}|{gi}"
            a += int(run_combat(net, ident, opp, lvl, lvl, key, gate=False))
            b += int(run_combat(net, ident, opp, lvl, lvl, key, gate=True))
        tA += a; tB += b; tn += args.games
        d = b - a
        flag = " ⚠" if d != 0 else ""
        print(f"{ident:16s} | {rng:4.1f} | {a:3d}/{args.games} | {b:3d}/{args.games} | {d:+d}{flag}")
    print("-" * 52)
    print(f"{'總計':16s} |      | {tA:3d}/{tn} | {tB:3d}/{tn} | {tB-tA:+d}"
          f"  (A={tA/tn*100:.1f}% B={tB/tn*100:.1f}%)")


if __name__ == "__main__":
    main()
