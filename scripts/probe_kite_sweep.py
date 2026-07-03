"""第二層根因:訓練後的法師為何在遠距(該守)也後退?
掃描敵距 1→12m,post-cast(action=0),dump end_prob + (不結束時)move argmax 方向。
直接回答:end-head/move-head 有沒有『依距離』條件化,還是全距離一律不結束/一律後退。

  若 end_prob 隨距離上升而升(遠距→高end=會停手) = 有條件化,過度風箏來自別處。
  若 end_prob 全距離一律低 = end-head 沒學會依距離,這才是過度風箏根因。
用法: python scripts/probe_kite_sweep.py <ckpt> [ident...]
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from trpg.scenarios.monsters import register_monsters
register_monsters()
from trpg.rl.env_v2 import CombatEnvV2
from trpg.rl.model import apply_resource_mask
from trpg.rl.obs import build_obs, N_ARCHETYPES, N_GRID, GRID_CELL_SIZE_M
from trpg.rl.neural_policy import _ENT_ARCH_START, _END_ARCH_START
from trpg.rl.action import point_validity_mask
from trpg.engine.skill import available_skills
from trpg.engine.vec2 import Vec2


def load(path):
    from distill_routed import load_student
    return load_student(path)


def cc(idx):
    return Vec2((idx // N_GRID + 0.5) * GRID_CELL_SIZE_M,
               (idx % N_GRID + 0.5) * GRID_CELL_SIZE_M)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v8.pt"
    idents = sys.argv[2:] or ["evocation"]
    net = load(path)
    print(f"模型={os.path.basename(path)}  (post-cast: action=0,bonus=0,mv=9)\n")
    for ident in idents:
        env = CombatEnvV2(n_agents=1, n_opps=1)
        env.reset(level=8, opp_level=8, layout="open",
                  agent_archs=[ident], opp_archs=["champion"], seed=11)
        ch = env.ws.characters["agent_0"]; foe = env.ws.characters["opp_0"]
        ch.position = Vec2(15.0, 15.0)
        bonus = int(os.environ.get("PROBE_BONUS", "1"))
        print(f"== {ident}  (bonus={bonus}) ==")
        print(f"{'敵距':>5s} | end_prob | 不結束時 move方向")
        for d in [1.3, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0]:
            foe.position = Vec2(15.0 + d, 15.0)
            res = {"action": 0, "bonus_action": bonus, "movement": 9.0}
            obs = build_obs(env.ws, "agent_0", res)
            obs["entities"][0, _ENT_ARCH_START:_ENT_ARCH_START + N_ARCHETYPES] = 0.0
            obs["end_features"][_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
            ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            with torch.no_grad():
                el, s, e, g = net(ot)
            ep = torch.sigmoid(el[0]).item()
            s = apply_resource_mask(s, res, env.ws, "agent_0")
            sks = available_skills(ch, env.ws)
            mi = next((i for i, sk in enumerate(sks) if sk.skill_id == "move"), None)
            direc = "?"
            if mi is not None:
                row = g[0][mi].clone()
                inv = point_validity_mask(env.ws, "agent_0", sks[mi])
                if inv.any() and not inv.all():
                    row = row.masked_fill(torch.from_numpy(inv), -1e9)
                rmask = np.array([ch.position.distance_to(cc(i)) <= 9.0
                                  for i in range(N_GRID * N_GRID)])
                row[~torch.from_numpy(rmask)] = -1e9
                best = int(row.argmax().item())
                bd = cc(best).distance_to(foe.position)
                direc = ("離敵" if bd > d + 0.3 else "朝敵" if bd < d - 0.3 else "原地") + f"({bd:.1f}m)"
            tag = "→會停手" if ep > 0.5 else "→繼續動"
            print(f"{d:5.1f} |  {ep:.3f}  {tag} | {direc}")
        print()


if __name__ == "__main__":
    main()
