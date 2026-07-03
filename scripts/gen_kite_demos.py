"""合成「遠程身分被貼臉 → 一次跳到最遠安全格」示範,教 grid_query 一跳到位(非小碎步)。

根因(已查):風箏停手已由 threat 修好,但 move-head 每次只跳 ~3m,敵 move 9m 立刻再貼上→白風箏。
gate(+43)是「一次跳到最遠保 LoS 的可達格」。grid_query 吃 h(情境)故能依「遠程+被貼臉」條件化跳遠,
只是沒學到。本檔直接造 demo:遠程 ident、敵貼臉 → label = move 到「離敵最遠 + 可達(<=移動力) + 有 LoS +
仍在射程內」的格(= gate 的 _kite_cell)。teach 一跳到位;radius=3 的 end-head 跳完(已 >3m 安全)就停。

輸出 kite_demos.npz,與 stuck_demos 一起用 train_dagger --switch_demos 共訓(只訓 end_head+grid_query)。
"""
from __future__ import annotations
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, MOVE_BUDGET_M
from trpg.rl.obs import build_obs, N_GRID, GRID_CELL_SIZE_M
from trpg.engine.skill import available_skills, TargetType
from trpg.engine.vec2 import Vec2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_population import blind_np_single
try:
    from chimera_defs import register_chimeras; register_chimeras()
except Exception:
    pass


def _cell(idx):
    return Vec2((idx // N_GRID + 0.5) * GRID_CELL_SIZE_M,
               (idx % N_GRID + 0.5) * GRID_CELL_SIZE_M)


def _ranged_range(ch, ws):
    best = 0.0
    for s in available_skills(ch, ws):
        tt = s.features.target_type
        if s.skill_id == "move":
            continue
        if tt in (TargetType.SINGLE_ENEMY, TargetType.MULTI_ENEMY,
                  TargetType.POINT, TargetType.LINE, TargetType.CONE):
            if getattr(s.features, "expected_damage", 0) > 0:
                best = max(best, float(getattr(s.features, "range_m", 0.0) or 0.0))
    return best


def _kite_cell(ch, ws, mv, foe_pos, rng_m):
    bf = ws.combat.battlefield if ws.combat else None
    cur = ch.position
    best_idx, best_d = None, cur.distance_to(foe_pos)
    for idx in range(N_GRID * N_GRID):
        c = _cell(idx)
        if cur.distance_to(c) > mv + 1e-6:
            continue
        d = c.distance_to(foe_pos)
        if d <= best_d + 1e-6 or d > rng_m:
            continue
        if bf is not None:
            if hasattr(bf, "is_blocked") and bf.is_blocked(c):
                continue
            if not bf.has_line_of_sight(c, foe_pos):
                continue
        best_idx, best_d = idx, d
    return best_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/unified_kite9/kite_demos.npz")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(0)
    IDENTS = ["evocation", "divination", "mage_npc", "manticore"]
    O, A, T = [], [], []
    ep = 0
    while len(A) < args.n:
        ident = rng.choice(IDENTS)
        env = CombatEnvV2(seed=ep * 2654435761 % (2**31), n_agents=1, n_opps=1)
        try:
            env.reset(agent_archs=[ident], opp_archs=["champion"],
                      level=8, opp_level=8, layout="open")
        except Exception:
            ep += 1; continue
        ch = env.ws.characters["agent_0"]; foe = env.ws.characters["opp_0"]
        rngm = _ranged_range(ch, env.ws)
        if rngm < 5.0:                      # 非遠程,跳過
            ep += 1; continue
        # 全場各處(含貼牆/角落)— 真實戰鬥常在邊角,kite_cell 會算出該位置的最遠格
        cx, cy = rng.uniform(1.5, 28.5), rng.uniform(1.5, 28.5)
        ch.position = Vec2(cx, cy)
        ang = rng.uniform(0, 6.2832)
        d = rng.uniform(1.0, 2.4)            # 貼臉
        foe.position = Vec2(cx + d * np.cos(ang), cy + d * np.sin(ang))
        mv = MOVE_BUDGET_M
        sks = available_skills(ch, env.ws)
        mi = next((i for i, s in enumerate(sks) if s.skill_id == "move"), None)
        if mi is None:
            ep += 1; continue
        cell = _kite_cell(ch, env.ws, mv, foe.position, rngm)
        if cell is None:
            ep += 1; continue
        res = {"action": rng.choice([0, 1]), "bonus_action": rng.choice([0, 1]),
               "movement": mv}
        ob = blind_np_single(build_obs(env.ws, "agent_0", res))
        O.append(ob); A.append([mi, 0, cell]); T.append(int(sks[mi].features.target_type))
        ep += 1
    obs_np = {k: np.stack([o[k] for o in O]) for k in O[0]}
    save = {f"obs_{k}": v for k, v in obs_np.items()}
    save["actions"] = np.array(A, np.int64)
    save["target_types"] = np.array(T, np.int64)
    np.savez_compressed(args.out, **save)
    print(f"寫出 {len(A)} 筆遠程跳遠風箏 demo → {args.out}")


if __name__ == "__main__":
    main()
