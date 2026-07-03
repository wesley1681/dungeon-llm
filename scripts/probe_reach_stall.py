"""直接插樁 reach-stall：在審計重現的「近戰怪停 1.51m 然後空轉」情境下，
把每個 MOVE 決策的完整合法格資訊 dump 出來，一翻兩瞪眼分清三種根因：

  GRID-QUANT  : reach 內根本沒有合法格(min_legal_dist > reach) → 引擎/網格量化，
                訓練救不了(模型已挑最近格)。
  MODEL-PICK  : reach 內有合法格但模型沒挑(chosen_dist > reach 且 best_legal < reach)
                → 模型選格失誤，可訓練。
  ENGINE-CLAMP: 模型挑了 reach 內的格(chosen cell 在 reach 內)但移動後位移被夾/沒到位。

用法: python scripts/probe_reach_stall.py [模型路徑] [身分] [dist] [angle_deg]
"""
from __future__ import annotations
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2, MOVE_BUDGET_M, _MAX_SUB_ACTIONS_PER_TURN
import trpg.rl.env_v2 as E
from trpg.rl.model import (CombatPolicyNet, apply_resource_mask,
                           apply_entity_mask, pick_action)
from trpg.rl.obs import build_obs, N_GRID, GRID_CELL_SIZE_M, N_ARCHETYPES
from trpg.rl.neural_policy import _ENT_ARCH_START, _END_ARCH_START
from trpg.rl.action import decode_action, point_validity_mask
from trpg.engine.vec2 import Vec2
from trpg.engine.skill import available_skills, TargetType


def load(path):
    net = CombatPolicyNet(hidden=128)
    sd = torch.load(path, map_location="cpu")
    sd = CombatPolicyNet.adapt_state_dict_for_perarch(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def cell_center(idx):
    ix, iy = idx // N_GRID, idx % N_GRID
    return Vec2((ix + 0.5) * GRID_CELL_SIZE_M, (iy + 0.5) * GRID_CELL_SIZE_M)


def reach_of(ch):
    try:
        return float(ch.get_weapon().range_normal)
    except Exception:
        return 1.5


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v7.pt"
    ident = sys.argv[2] if len(sys.argv) > 2 else "kobold"
    dist = float(sys.argv[3]) if len(sys.argv) > 3 else 1.6
    ang = float(sys.argv[4]) if len(sys.argv) > 4 else 27.0
    net = load(path)

    env = CombatEnvV2(n_agents=1, n_opps=1)
    env.reset(level=5, opp_level=5, layout="open",
              agent_archs=["battle_master"], opp_archs=[ident], seed=7)
    dummy = env.ws.characters["agent_0"]
    mob = env.ws.characters["opp_0"]
    cx, cy = 15.0, 15.3
    dummy.position = Vec2(cx, cy)
    a = math.radians(ang)
    mob.position = Vec2(cx + dist * math.cos(a), cy + dist * math.sin(a))
    reach = reach_of(mob)
    print(f"模型={os.path.basename(path)} 身分={ident} d={dist}@{ang:.0f}° reach={reach}")
    print(f"假人@({dummy.position.x:.2f},{dummy.position.y:.2f})  "
          f"怪@({mob.position.x:.2f},{mob.position.y:.2f})\n")

    res = {"action": 1, "bonus_action": 1, "movement": MOVE_BUDGET_M}
    for step in range(_MAX_SUB_ACTIONS_PER_TURN):
        d0 = mob.position.distance_to(dummy.position)
        obs = build_obs(env.ws, "opp_0", res)
        obs["entities"][0, _ENT_ARCH_START:_ENT_ARCH_START + N_ARCHETYPES] = 0.0
        obs["end_features"][_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
        obs_t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            end_l, skill_l, entity_l, grid_l = net(obs_t)
        skill_l = apply_resource_mask(skill_l, res, env.ws, "opp_0")
        entity_l = apply_entity_mask(entity_l, obs_t, env.ws, "opp_0")
        triplet = pick_action(end_l[0], skill_l[0], entity_l[0], grid_l[0],
                              ws=env.ws, agent_id="opp_0")
        skills = available_skills(mob, env.ws)
        sidx = triplet[0]
        sid = skills[sidx].skill_id if sidx < len(skills) else "END"
        act = decode_action(list(triplet), env.ws, "opp_0")
        if act is None:
            print(f"[step{step}] d={d0:.2f}m → END"); break
        atype = act.get("type")

        if atype == "MOVE" and sidx < len(skills):
            sk = skills[sidx]
            row = grid_l[0][sidx].clone()
            inv = point_validity_mask(env.ws, "opp_0", sk)
            legal = ~inv if (inv.any() and not inv.all()) else np.ones(N_GRID * N_GRID, bool)
            dists = np.array([cell_center(i).distance_to(dummy.position)
                              for i in range(N_GRID * N_GRID)])
            legal_idx = np.where(legal)[0]
            chosen = triplet[2]
            chosen_pos = cell_center(chosen)
            chosen_d = chosen_pos.distance_to(dummy.position)
            in_reach_legal = legal_idx[dists[legal_idx] <= reach]
            min_legal = dists[legal_idx].min()
            best_in_reach = (dists[in_reach_legal].min()
                             if len(in_reach_legal) else None)
            # chosen cell 的 logit 排名(在合法格中)
            lg = row.numpy()
            chosen_rank = int((lg[legal_idx] > lg[chosen]).sum())
            if len(in_reach_legal):
                best_ir = in_reach_legal[np.argmax(lg[in_reach_legal])]
                print(f"[step{step}] MOVE d={d0:.2f}→chosen格@({chosen_pos.x:.1f},"
                      f"{chosen_pos.y:.1f}) dist={chosen_d:.2f}m"
                      f" | 合法格={len(legal_idx)} reach內合法格={len(in_reach_legal)}"
                      f" min_legal={min_legal:.2f}m")
                print(f"          chosen logit={lg[chosen]:.3f}(在合法格排第{chosen_rank})"
                      f" | reach內最高logit格@dist={dists[best_ir]:.2f}m logit={lg[best_ir]:.3f}")
            else:
                print(f"[step{step}] MOVE d={d0:.2f}→chosen dist={chosen_d:.2f}m"
                      f" | 合法格={len(legal_idx)} reach內合法格=0"
                      f" min_legal={min_legal:.2f}m ← GRID-QUANT(reach內無合法格)")
        else:
            print(f"[step{step}] d={d0:.2f}m → {sid} ({atype})")

        # 執行
        r = E.execute_action(act, env.ws)
        if r.get("type") != "ERROR":
            E.consume_resources(res, act, r)
        if atype == "MOVE":
            moved = r.get("distance", 0.0)
            d1 = mob.position.distance_to(dummy.position)
            print(f"          → 實際位移 {moved:.3f}m, 移動後 d={d1:.2f}m, "
                  f"movement剩 {res['movement']:.1f}")
            if moved < 0.01 or res["movement"] < 0.5:
                res["movement"] = 0.0


if __name__ == "__main__":
    main()
