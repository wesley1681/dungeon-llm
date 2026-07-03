"""釘死『遠程身分不風箏』根因：法師被貼臉、主動作已用(只剩 move/end),
dump end-head 機率 + grid-head move 列的 argmax 格是朝敵(approach)還是離敵(kite)。

  END-BIAS    : end_prob>0.5 → 模型施法後直接結束,根本不嘗試移動(=不風箏主因)。
  MOVE-TOWARD : 不結束但 move argmax 朝敵 → 移動策略偏approach(melee偏置外溢)。
  MOVE-AWAY   : move argmax 離敵 → 其實會風箏(問題在別處)。
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
from trpg.rl.model import apply_resource_mask, apply_entity_mask
from trpg.rl.obs import build_obs, N_ARCHETYPES, N_GRID, GRID_CELL_SIZE_M
from trpg.rl.neural_policy import _ENT_ARCH_START, _END_ARCH_START
from trpg.rl.action import point_validity_mask
from trpg.engine.skill import available_skills
from trpg.engine.vec2 import Vec2


def load(path):
    from distill_routed import load_student
    return load_student(path)


def cell_center(idx):
    return Vec2((idx // N_GRID + 0.5) * GRID_CELL_SIZE_M,
                (idx % N_GRID + 0.5) * GRID_CELL_SIZE_M)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/unified/uni_v8.pt"
    idents = sys.argv[2:] or ["evocation", "divination", "mage_npc", "manticore"]
    net = load(path)
    print(f"模型={os.path.basename(path)} (法師貼臉、action已用→該風箏)\n")
    for ident in idents:
        env = CombatEnvV2(n_agents=1, n_opps=1)
        env.reset(level=8, opp_level=8, layout="open",
                  agent_archs=[ident], opp_archs=["champion"], seed=11)
        ch = env.ws.characters["agent_0"]
        foe = env.ws.characters["opp_0"]
        ch.position = Vec2(15.0, 15.0)
        foe.position = Vec2(16.3, 15.0)   # 貼臉 1.3m
        nd0 = ch.position.distance_to(foe.position)
        # action 已用、movement 滿、bonus 用掉(排除 misty_step,測純 move 風箏)
        res = {"action": 0, "bonus_action": 0, "movement": 9.0}
        obs = build_obs(env.ws, "agent_0", res)
        obs["entities"][0, _ENT_ARCH_START:_ENT_ARCH_START + N_ARCHETYPES] = 0.0
        obs["end_features"][_END_ARCH_START:_END_ARCH_START + N_ARCHETYPES] = 0.0
        ot = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            el, s, e, g = net(ot)
        end_prob = torch.sigmoid(el[0]).item()
        s = apply_resource_mask(s, res, env.ws, "agent_0")
        sks = available_skills(ch, env.ws)
        mi = next((i for i, sk in enumerate(sks) if sk.skill_id == "move"), None)
        verdict = ""
        if mi is None:
            verdict = "無move技?"
        else:
            row = g[0][mi].clone()
            inv = point_validity_mask(env.ws, "agent_0", sks[mi])
            if inv.any() and not inv.all():
                row = row.masked_fill(torch.from_numpy(inv), -1e9)
            # 只看可達格(<=movement)
            reach_mask = np.array([ch.position.distance_to(cell_center(i)) <= 9.0
                                   for i in range(N_GRID * N_GRID)])
            rr = row.clone()
            rr[~torch.from_numpy(reach_mask)] = -1e9
            best = int(rr.argmax().item())
            best_d = cell_center(best).distance_to(foe.position)
            direction = "離敵(kite)" if best_d > nd0 + 0.3 else (
                "朝敵(approach)" if best_d < nd0 - 0.3 else "原地")
            verdict = (f"move argmax→格dist={best_d:.1f}m (現{nd0:.1f}m) = {direction}")
        flag = "END-BIAS(施法後直接結束)" if end_prob > 0.5 else "會嘗試移動"
        print(f"{ident:14s}: end_prob={end_prob:.3f} [{flag}] | {verdict}")


if __name__ == "__main__":
    main()
